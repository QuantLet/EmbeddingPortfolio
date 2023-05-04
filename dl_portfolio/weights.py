import pandas as pd
import numpy as np

import riskparityportfolio as rp
import cvxpy as cp

from typing import Union
from sklearn.cluster import KMeans
from sklearn.covariance import shrunk_covariance

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models

from dl_portfolio.alm import fmin_ALM, _epsilon, test_principal_port
from dl_portfolio.logger import LOGGER
from dl_portfolio.cluster import get_cluster_labels
from dl_portfolio.constant import PORTFOLIOS
from prado.hrp import correlDist, getQuasiDiag, getRecBipart

import scipy
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize

from portfoliolab.clustering.herc import HierarchicalEqualRiskContribution


def portfolio_weights(
    returns,
    budget=None,
    loading=None,
    portfolio=["markowitz", "shrink_markowitz", "ivp", "aerp", "erc"],
    **kwargs,
):
    assert all([p in PORTFOLIOS for p in portfolio]), [
        p for p in portfolio if p not in PORTFOLIOS
    ]
    port_w = {}

    mu = returns.mean()
    S = returns.cov()

    if "markowitz" in portfolio:
        LOGGER.info("Computing Markowitz weights...")
        port_w["markowitz"] = markowitz_weights(mu, S)

    if "ivp" in portfolio:
        LOGGER.info("Computing IVP weights...")
        port_w["ivp"] = ivp_weights(S)

    if "erc" in portfolio:
        LOGGER.info("Computing ERC weights...")
        assert budget is not None
        port_w["erc"] = riskparity_weights(S, budget=budget["rc"].values)

    if "aerp" in portfolio:
        LOGGER.info("Computing AE Risk Parity weights...")
        assert loading is not None
        port_w["aerp"] = ae_rp_weights(returns, loading)

    if "hrp" in portfolio:
        port_w["hrp"] = getHRP(cov=S.values, corr=returns.corr().values,
                               index=returns.columns)

    if 'hcaa' in portfolio:
        LOGGER.info('Computing HCAA weights...')
        port_w['hcaa'] = herc_weights(returns,
                                      optimal_num_clusters=kwargs.get(
                                          'optimal_num_clusters'),
                                      risk_measure='equal_weighting')

    if "rb_factor" in portfolio:
        LOGGER.info('Computing RB factor weights...')
        port_w['rb_factor'] = get_rb_factor_weights(returns, loading)

    return port_w


def get_eigen_decomposition(cov, shrinkage=None):
    """

    :param cov:
    :param shrinkage:
    :return: eigvals, eigvecs
    """
    if shrinkage is not None:
        cov = shrunk_covariance(cov, shrinkage=shrinkage)
    eigvals, eigvecs = np.linalg.eig(cov)
    order_ = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order_]
    eigvecs = eigvecs[:, order_]

    return eigvals, eigvecs


def compute_factor_weight(a, W_tilde):
    """
    cf Roncalli, T., & Weisang, G. (2016). Risk parity portfolios with risk
    factors. Quantitative Finance, 16(3), 377-388.
    Appendix p.28

    :param a:
    :param W_tilde:
    :return:
    """
    V = W_tilde.T
    V_tilde = scipy.linalg.null_space(V).T

    return np.dot(V, a), np.dot(V_tilde, a)


def compute_factor_risk_contribution(a, W_tilde, Sigma):
    """
    Therom 2 in  Roncalli, T., & Weisang, G. (2016). Risk parity portfolios
    with risk factors. Quantitative Finance, 16(3), 377-388.

    :param a:
    :param W_tilde:
    :param Sigma:
    :return:
    """
    assert W_tilde.shape[-1] < Sigma.shape[0]
    sigma_a = np.sqrt(np.dot(a.T, np.dot(Sigma, a)))
    V = W_tilde.T
    V_tilde = scipy.linalg.null_space(V).T
    rc_z = np.dot(W_tilde.T, a) * np.dot(np.linalg.pinv(W_tilde),
                                         np.dot(Sigma, a)) / sigma_a
    rc_y = np.dot(V_tilde, a) * np.dot(V_tilde, np.dot(Sigma, a)) / sigma_a

    return rc_z, rc_y


def get_neg_entropy_from_weights_factor(a, W_tilde, Sigma):
    """
    cf: Lohre et al 2014,
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1974446
    =1974446

    """
    rc_z, rc_y = compute_factor_risk_contribution(a, W_tilde, Sigma)

    rc = np.concatenate([rc_z, rc_y])
    p = np.abs(rc) / np.sum(np.abs(rc))
    N_Ent = np.exp(-np.sum(p * np.log(p)))

    return -N_Ent


def get_drp_factor_weights(W_tilde, Sigma, w0, verbose=False):
    """
    Lohre et al 2014, Diversified Risk Parity weights with long-only and
    full investment constraints.
    cf: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1974446

    """
    n_assets = Sigma.shape[0]

    # Full investment constraints
    cons = [
        {"type": "eq", "fun": lambda x: np.array([1 - np.sum(x)])}
    ]
    # Long only contraints
    bnds = tuple((0 * x - 0.0, 0 * x + 1.0) for x in range(n_assets))
    res = minimize(get_neg_entropy_from_weights_factor, w0, args=(W_tilde,
                                                                  Sigma),
                   bounds=bnds, constraints=cons, options={"disp": verbose})
    weights = pd.Series(res.x, index=w0.index)

    return weights


def get_neg_entropy_from_weights_principal(w, cov, shrinkage=None):
    """
    cf: Lohre et al 2014,
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1974446
    =1974446

    """
    assert len(w.shape) == 2
    assert w.shape[-1] == 1
    eigvals, eigvecs = get_eigen_decomposition(cov, shrinkage=shrinkage)
    eigvals = eigvals.reshape(-1, 1)
    w_tilde = np.dot(eigvecs.T, w)
    var_i = w_tilde ** 2 * eigvals
    p = var_i / var_i.sum()
    assert p.shape == w_tilde.shape
    # sometimes eigen decomposition returns small complex numbers with real
    # part close to 0, just take real and abs
    p = np.abs(np.real(p))
    N_Ent = np.exp(-np.sum(p * np.log(p)))

    return -N_Ent


def effective_bets(w, cov, t):
    """
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2276632

    :param w: Portfolio weights
    :param cov: Covariance matrix
    :param t: decorrelation matrix
    :return:
    """
    cov = np.asmatrix(cov)
    w = np.asmatrix(w)
    p = np.asmatrix(
        np.asarray(np.linalg.inv(t.T) * w.T) * np.asarray(t * cov * w.T)
    ) / (w * cov * w.T)
    enb = np.exp(- p.T * np.log(p))
    return p, enb


def get_drp_weights(S: pd.DataFrame(), w0, n_components=None, shrinkage=None,
                    verbose=False):
    """
    Lohre et al 2014, Diversified Risk Parity weights with long-only and
    full investment constraints.
    cf: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1974446

    """
    n_assets = S.shape[0]

    # Full investment constraints
    cons = [
        {"type": "eq", "fun": lambda x: np.array([1 - np.sum(x)])}
    ]
    # Long only contraints
    bnds = tuple((0 * x - 0.0, 0 * x + 1.0) for x in range(n_assets))

    if n_components is not None:
        # Keep only eigenvectors with largest eigenvalues
        eigvals, eigvecs = get_eigen_decomposition(S.values,
                                                   shrinkage=shrinkage)
        to_drop = eigvecs[:, n_components:]

        # Add constraints that cancel exposure to small eigenvectors
        for i in range(to_drop.shape[-1]):
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.array([np.sum(x * to_drop[:, i])])
                }
            )

    res = minimize(get_neg_entropy_from_weights_principal, w0, args=(S.values,
                                                           shrinkage),
                   bounds=bnds,
                   constraints=cons, options={"disp": verbose})
    weights = pd.Series(res.x, index=S.index)

    return weights


def get_principal_port(S: pd.DataFrame(), x0, n_components,
                       verbose=False):
    assets = S.columns
    n_assets = len(assets)
    eigvals, eigvecs = get_eigen_decomposition(S.values)

    # Get G
    G = np.diag(np.sqrt(eigvals)) * np.linalg.inv(eigvecs)
    G = np.asmatrix(G)

    # Create constraints
    # Equality constraints
    eigen_const = np.concatenate([
        np.asarray(eigvecs[:, n_components:]),
        np.zeros(n_assets - n_components).reshape(1, -1)
    ]).T.tolist()
    eqParameters = [tuple(c) for c in eigen_const]
    # Budget constraint
    eqParameters += [tuple([1] * n_assets + [-1])]

    # Inequality constraints: long-only and no leverage
    ineqParameters = [tuple([1 if i == c else 0 for i in range(n_assets + 1)])
                      for c in range(n_assets)]
    max_1 = [[- 1 if i == c else 0 for i in range(n_assets + 1)] for c in
             range(n_assets)]
    for c in max_1:
        c[-1] = 1
    max_1 = [tuple(c) for c in max_1]
    ineqParameters += max_1

    conParaDict = {'ineq': ineqParameters, 'eq': eqParameters}

    def neg_entropy(x):
        v_ = np.dot(G, x)
        p = np.multiply(v_, v_)
        R_2 = p / np.sum(p)
        R_2[R_2 < 1e-10] = 1e-10  # Remove small values
        MinusEnt = - R_2.T * np.log(R_2)
        MinusEnt = - np.exp(np.asarray(MinusEnt)[0][0])

        return float(MinusEnt)

    res = fmin_ALM(neg_entropy, x0, test_principal_port,
                   eqConNum=len(eqParameters),
                   ineqConNum=len(ineqParameters),
                   funArgs=(), conArgs=(conParaDict,),
                   multiplierRange=(0.1, 1.),
                   gtol=1e-3, sigma=3., beta=0.9, alpha=1.3,
                   norm=2, epsilon=_epsilon, maxiter=1000, full_output=1,
                   disp=int(verbose), retall=1, callback=None)
    weights = res[0]
    weights[weights <= 0] = 0
    weights = weights / np.sum(weights)
    weights = pd.Series(weights, index=assets)
    return weights


def get_cluster_var(cov, cluster_items, weights=None):
    """

    Compute the variance per cluster

    :param cov: covariance matrix
    :type cov: np.ndarray
    :param cluster_items: tickers in the cluster
    :type cluster_items: list
    :param weights: portfolio weights. If None we will compute inverse variance weights
    :return:
    """
    cov_slice = cov.loc[cluster_items, cluster_items]
    if weights is None:
        weights = ivp_weights(cov_slice)
    return np.linalg.multi_dot((weights, cov_slice, weights))


def get_inner_cluster_weights(cov, loading, clusters, market_budget=None):
    weights = {}
    n_clusters = len(clusters)
    for c in clusters:
        cluster_items = clusters[c]
        if cluster_items:
            if len(cluster_items) == 1:
                weights[c] = pd.Series([1.0], index=cluster_items)
            else:
                if market_budget is not None:
                    budget = market_budget.loc[cluster_items, "rc"]
                else:
                    budget = loading.loc[cluster_items, c] ** 2 / np.sum(
                        loading.loc[cluster_items, c] ** 2
                    )
                cov_slice = cov.loc[cluster_items, cluster_items]
                weights[c] = pd.Series(
                    rp.RiskParityPortfolio(
                        covariance=cov_slice, budget=budget.values
                    ).weights,
                    index=cluster_items,
                )
    reorder_weights = {}
    i = 0
    for c in weights:
        reorder_weights[i] = weights[c]
        i += 1
    weights = {i: weights[c] for i, c in enumerate(list(weights.keys()))}
    return weights


def markowitz_weights(
    mu: Union[pd.Series, np.ndarray],
    S: pd.DataFrame,
    fix_cov: bool = False,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    if fix_cov:
        S = risk_models.fix_nonpositive_semidefinite(S, fix_method="spectral")
    weights = None
    try:
        LOGGER.info(f"Trying Markowitz with default 'ECOS' solver")
        ef = EfficientFrontier(mu, S, verbose=False)
        # ef.add_objective(objective_functions.L2_reg, gamma=0)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = pd.Series(weights, index=weights.keys())
        LOGGER.info("Success")
    except Exception as _exc:
        LOGGER.info(f"Error with max sharpe: {_exc}")
        try:
            LOGGER.info(f"Trying Markowitz with 'SCS' solver")
            ef = EfficientFrontier(mu, S, verbose=True, solver="SCS")
            # ef.add_objective(objective_functions.L2_reg, gamma=0)
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            weights = pd.Series(weights, index=weights.keys())
            LOGGER.info("Success")
        except Exception as _exc:
            LOGGER.info(f"Error with max sharpe: {_exc}")
            try:
                LOGGER.info(f"Trying Markowitz with 'OSQP' solver")
                ef = EfficientFrontier(mu, S, verbose=True, solver="OSQP")
                # ef.add_objective(objective_functions.L2_reg, gamma=0)
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = pd.Series(weights, index=weights.keys())
                LOGGER.info("Success")
            except Exception as _exc:
                LOGGER.info(f"Error with max sharpe: {_exc}")
                try:
                    LOGGER.info(f"Trying Markowitz with 'CVXOPT' solver")

                    # ef = EfficientFrontier(mu, S, verbose=True, solver=cp.CVXOPT, solver_options={'feastol': 1e-4})
                    ef = EfficientFrontier(mu, S, verbose=True, solver=cp.SCS)
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                    weights = pd.Series(weights, index=weights.keys())
                    LOGGER.info("Success")
                except Exception as _exc:
                    LOGGER.info(f"Error with max sharpe: {_exc}")

    if weights is None:
        raise _exc

    return weights


def ivp_weights(S: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
    # Compute the inverse-variance portfolio
    ivp = 1.0 / np.diag(S.values)
    weights = ivp / ivp.sum()

    if isinstance(S, pd.DataFrame):
        weights = pd.Series(weights, index=S.index)
    else:
        weights = pd.Series(weights)

    return weights


def ivolp_weights(S: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
    # Compute the inverse-volatility portfolio (simple ERC)
    ivp = 1.0 / np.sqrt(np.diag(S.values))
    weights = ivp / ivp.sum()

    if isinstance(S, pd.DataFrame):
        weights = pd.Series(weights, index=S.index)
    else:
        weights = pd.Series(weights)

    return weights


def riskparity_weights(S: pd.DataFrame(), budget: np.ndarray) -> pd.Series:
    weights = rp.RiskParityPortfolio(covariance=S, budget=budget).weights
    weights = pd.Series(weights, index=S.index)

    return weights


def rb_perfect_corr(
        sigmas,
        budget,
        corr
):
    """
    cf (4) and (5) in Bruder, B., & Roncalli, T. (2012). Managing risk
    exposures using the risk budgeting approach. Available at SSRN 2009778.

    :param sigmas:
    :param budget:
    :param corr: 0 ou 1
    :return:
    """
    assert sigmas.shape == budget.shape

    if corr == 0:
        budget = np.sqrt(budget)
    else:
        assert corr == 1

    weights = budget / sigmas
    weights /= np.sum(weights)

    return weights


def get_full_rb_factor_weights(returns, loading, threshold=1e-2):
    """
    Compute inner weights with rb_perfect_corr and apply inverse volatility
    at the factor level
    :param returns:
    :param loading:
    :param threshold:
    :return:
    """
    assets = returns.columns.tolist()
    n_components = loading.shape[-1]
    cluster_ind = loading[
        loading ** 2 >= threshold].idxmax(axis=1).dropna().astype(int)
    assert len(cluster_ind.unique()) == n_components
    budget = loading ** 2

    inner_weights = pd.DataFrame(0, columns=loading.columns, index=assets)
    for i in range(n_components):
        inner_weights.loc[:, i] = rb_perfect_corr(
            np.std(returns, axis=0), budget.iloc[:, i], corr=1)

    factor_returns = np.array(
        [np.dot(returns, inner_weights.values[:, i]) for i in range(
            n_components)])
    if factor_returns.shape[0] > 1:
        cluster_weights = rp.RiskParityPortfolio(
            covariance=np.cov(factor_returns),
            budget=np.ones(n_components)/n_components
        ).weights
    else:
        cluster_weights = np.array([1.])
    weights = inner_weights * cluster_weights
    weights = weights.sum(axis=1)

    return weights


def get_rb_factor_weights(returns, loading, threshold=1e-2,
                          erc_factor=False, simple=False):
    """
    Compute inner weights with rb_perfect_corr and apply inverse volatility
    at the factor level
    :param returns:
    :param loading:
    :param threshold:
    :return:
    """
    assets = returns.columns.tolist()
    n_components = loading.shape[-1]
    # reset columns
    loading.columns = range(n_components)
    cluster_ind = loading[
        loading ** 2 >= threshold].idxmax(axis=1).dropna().astype(int)

    if len(cluster_ind.unique()) != n_components:
        print(cluster_ind.unique())
        print(loading)
        print("here")

    assert len(cluster_ind.unique()) == n_components, (cluster_ind.unique(),
                                                       n_components)

    inner_weights = pd.DataFrame(0, columns=loading.columns, index=assets)
    for i in range(n_components):
        c_assets = cluster_ind[cluster_ind == i].index
        budget = loading.loc[c_assets, i].copy()
        if simple:
            budget.values[:] = 1
        budget = budget / np.linalg.norm(budget)
        budget = budget**2 # Must sum to 1, also is interpretable as
        # explained variance
        inner_weights.loc[c_assets, i] = rb_perfect_corr(
            np.std(returns[c_assets], axis=0), budget, corr=1)

    factor_returns = np.array(
        [np.dot(returns, inner_weights.values[:, i]) for i in range(
            n_components)])
    if erc_factor:
        cluster_weights = rp.RiskParityPortfolio(
            covariance=np.cov(factor_returns),
            budget=np.ones(n_components)/n_components
        ).weights

    else:
        assert len(factor_returns.shape) == 2
        if factor_returns.shape[0] > 1:
            cluster_sigmas = np.sqrt(np.diag(np.cov(factor_returns)))
            cluster_weights = 1 / cluster_sigmas / np.sum(1 / cluster_sigmas)
        else:
            cluster_weights = np.array([1.])

    weights = inner_weights * cluster_weights
    weights = weights.sum(axis=1)

    return weights


def ae_riskparity_weights(
    returns,
    embedding,
    loading,
    market_budget,
    risk_parity="budget",
    threshold=0.1,
):
    """

    :param returns:
    :param embedding: To get cluster assignment
    :param loading: To get inner cluster weights
    :param market_budget:
    :param risk_parity: if 'budget' then use budget for risk allocation, if 'cluster' use relative asset cluster
    importance from the embedding matrix
    :return:
    """
    assert risk_parity in ["budget", "cluster"]
    # Rename columns in case of previous renaming
    loading.columns = list(range(len(loading.columns)))
    embedding.columns = list(range(len(embedding.columns)))
    max_cluster = embedding.shape[-1] - 1
    # First get cluster allocation to forget about small contribution
    clusters, _ = get_cluster_labels(embedding, threshold=threshold)
    clusters = {c: clusters[c] for c in clusters if c <= max_cluster}

    # Now get weights of assets inside each cluster
    if risk_parity == "budget":
        inner_cluster_weights = get_inner_cluster_weights(
            returns.cov(), loading, clusters, market_budget=market_budget
        )
    elif risk_parity == "cluster":
        inner_cluster_weights = get_inner_cluster_weights(
            returns.cov(), loading, clusters
        )
    else:
        raise NotImplementedError(risk_parity)

    # Now compute return of each cluster
    cluster_returns = pd.DataFrame()
    for c in inner_cluster_weights:
        cret = (
            returns[inner_cluster_weights[c].index] * inner_cluster_weights[c]
        ).sum(1)
        cluster_returns = pd.concat([cluster_returns, cret], 1)
    cluster_returns.columns = list(inner_cluster_weights.keys())

    # Now get risk contribution of each cluster defined by user
    cluster_rc = {
        c: (inner_cluster_weights[c]).idxmax() for c in inner_cluster_weights
    }
    cluster_rc = {
        c: market_budget.loc[cluster_rc[c], "rc"] for c in cluster_rc
    }

    # Compute cluster weights with risk parity portfolio
    cov = cluster_returns.cov()
    budget = np.array(list(cluster_rc.values()))
    budget = budget / np.sum(budget)
    cluster_weight = rp.RiskParityPortfolio(
        covariance=cov, budget=budget
    ).weights
    # Compute asset weight inside global portfolio
    weights = pd.Series(dtype="float32")
    for c in inner_cluster_weights:
        weights = pd.concat(
            [weights, inner_cluster_weights[c] * cluster_weight[c]]
        )
    weights = weights.reindex(returns.columns)  # reorder
    weights.fillna(0.0, inplace=True)

    return weights


def kmaa_weights(returns: pd.DataFrame, n_clusters: int) -> pd.Series:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(returns.T)
    labels = pd.DataFrame(
        kmeans.labels_.reshape(1, -1), columns=returns.columns
    ).T
    labels.columns = ["label"]
    clusters = {
        i: list(labels[labels["label"] == i].index) for i in range(n_clusters)
    }

    # Now get weights of assets inside each cluster
    cluster_weights = {
        c: pd.Series(
            [1 / len(clusters[c])] * len(clusters[c]), index=clusters[c]
        )
        for c in clusters
    }
    # {asset: 1 / n_items for asset in clusters[c]}}

    # Compute asset weight inside global portfolio
    weights = pd.Series(dtype="float32")
    for c in cluster_weights:
        weights = pd.concat([weights, cluster_weights[c]])
    weights = weights / n_clusters  # Rescale each weightq
    weights = weights.reindex(returns.columns)  # rerorder
    weights.fillna(0.0, inplace=True)

    return weights


def aeaa_weights(
    returns: Union[np.ndarray, pd.DataFrame],
    embedding: Union[np.ndarray, pd.DataFrame],
) -> pd.Series:
    max_cluster = embedding.shape[-1] - 1
    # First get cluster allocation to forget about small contribution
    # Rename columns in case of previous renaming
    embedding.columns = list(range(len(embedding.columns)))
    clusters, _ = get_cluster_labels(embedding)
    clusters = {c: clusters[c] for c in clusters if c <= max_cluster}
    n_clusters = embedding.shape[-1]

    # Now get weights of assets inside each cluster
    cluster_weights = {
        c: pd.Series(
            [1 / len(clusters[c])] * len(clusters[c]), index=clusters[c]
        )
        for c in clusters
    }
    # {asset: 1 / n_items for asset in clusters[c]}}

    # Compute asset weight inside global portfolio
    weights = pd.Series(dtype="float32")
    for c in cluster_weights:
        weights = pd.concat([weights, cluster_weights[c]])
    weights = weights / n_clusters  # Rescale each weight
    weights = weights.reindex(returns.columns)  # rerorder
    weights.fillna(0.0, inplace=True)

    return weights


def equal_class_weights(market_budget: pd.DataFrame):
    market_class = np.unique(
        market_budget["market"].values, return_counts=True
    )
    inner_class_weight = {
        c: 1 / market_class[1][i] for i, c in enumerate(market_class[0])
    }
    weights = pd.Series(index=market_budget.index)
    for c in inner_class_weight:
        assets = market_budget.index[market_budget["market"] == c]
        weights.loc[assets] = inner_class_weight[c]
    weights /= np.sum(weights)

    return weights


def ae_rp_weights(returns, loading, erc=True, cluster_erc=False):
    max_cluster = loading.shape[-1] - 1
    # First get cluster allocation to forget about small contribution
    # Rename columns in case of previous renaming
    loading.columns = list(range(len(loading.columns)))
    clusters, _ = get_cluster_labels(loading)
    clusters = {c: clusters[c] for c in clusters if c <= max_cluster}

    # Now get weights of assets inside each cluster
    cluster_asset_weights = {}
    cluster_weights = {}
    cov = returns.cov()
    for c in clusters:
        cluster_items = clusters[c]
        if cluster_items:
            if erc:
                c_weights = ivolp_weights(
                    cov.loc[cluster_items, cluster_items])
            else:
                c_weights = ivp_weights(cov.loc[cluster_items, cluster_items])
            cluster_var = get_cluster_var(
                cov, cluster_items, weights=c_weights
            )
            cluster_asset_weights[c] = c_weights
            if cluster_erc:
                cluster_weights[c] = np.sqrt(cluster_var)
            else:
                cluster_weights[c] = cluster_var

    cluster_weights = {c: 1 / cluster_weights[c] for c in cluster_weights}
    cluster_weights = {
        c: cluster_weights[c] / np.sum(list(cluster_weights.values()))
        for c in cluster_weights
    }
    cluster_weights = {
        c: cluster_weights[c] * cluster_asset_weights[c]
        for c in cluster_weights
    }

    # Compute asset weight inside global portfolio
    weights = pd.Series(dtype="float32")
    for c in cluster_weights:
        weights = pd.concat([weights, cluster_weights[c]])
    weights = weights.reindex(returns.columns)  # rerorder
    weights.fillna(0.0, inplace=True)

    return weights


def getHRP(cov, corr, index):
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(dist, "single")
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    hrp = getRecBipart(cov, sortIx)
    hrp = hrp.sort_index()
    hrp.index = index
    return hrp


def herc_weights(returns: pd.DataFrame, linkage: str = 'single', risk_measure: str = 'equal_weighting',
                 covariance_matrix=None, optimal_num_clusters=None) -> pd.Series:
    hercEW_single = HierarchicalEqualRiskContribution()
    hercEW_single.allocate(asset_names=returns.columns,
                           asset_returns=returns,
                           covariance_matrix=covariance_matrix,
                           risk_measure=risk_measure,
                           optimal_num_clusters=optimal_num_clusters,
                           linkage=linkage)

    weights = hercEW_single.weights.T
    weights = weights[0]
    weights = weights[returns.columns]

    return weights

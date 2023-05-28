import pandas as pd
import numpy as np

import riskfolio as rf

from typing import Union, Optional, List
from sklearn.cluster import KMeans
from sklearn.covariance import shrunk_covariance


from dl_portfolio.alm import fmin_ALM, _epsilon, test_principal_port
from dl_portfolio.logger import LOGGER
from dl_portfolio.cluster import get_cluster_labels
from dl_portfolio.constant import PORTFOLIOS
from dl_portfolio.herc import HCPortfolio
from prado.hrp import correlDist, getQuasiDiag, getRecBipart

import scipy
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize


def portfolio_weights(
    returns,
    loading=None,
    portfolio: List[str] = ["markowitz", "shrink_markowitz", "ivp", "aerp",
                            "erc"],
    alpha: List[float] = [0.01, 0.025, 0.05],
):
    assert all([p in PORTFOLIOS for p in portfolio]), [
        p for p in portfolio if p not in PORTFOLIOS
    ]
    port_w = {}

    mu = returns.mean()
    S = returns.cov()

    if "ivp" in portfolio:
        LOGGER.info("Computing IVP weights...")
        port_w["ivp"] = ivp_weights(S)

    if "erc" in portfolio:
        LOGGER.info("Computing ERC weights...")
        port_w["erc"] = riskparity_weights(returns)

    if "erc_es" in portfolio:
        LOGGER.info("Computing ERC ES weights...")
        for a in alpha:
            port_w[f"erc_es_{a}"] = riskparity_weights(returns,
                                                       risk_measure="CVaR",
                                                       alpha=a)

    if "erc_cdar" in portfolio:
        LOGGER.info("Computing ERC CDaR weights...")
        for a in alpha:
            port_w[f"erc_cdar_{a}"] = riskparity_weights(returns,
                                                         risk_measure="CDaR",
                                                         alpha=a)

    if "aerp" in portfolio:
        LOGGER.info("Computing AE Risk Parity weights...")
        assert loading is not None
        port_w["aerp"] = ae_rp_weights(returns, loading)

    if "hrp" in portfolio:
        port_w["hrp"] = getHRP(cov=S.values, corr=returns.corr().values,
                               index=returns.columns)

    if 'hcaa' in portfolio:
        LOGGER.info('Computing HCAA weights...')
        port_w['hcaa'] = herc_weights(returns,  risk_measure="equal")

    if 'herc_vol' in portfolio:
        LOGGER.info('Computing HERC VOL weights...')
        port_w['herc_vol'] = herc_weights(returns,  risk_measure="vol")

    if 'herc_es' in portfolio:
        LOGGER.info('Computing HERC ES weights...')
        for a in alpha:
            port_w[f"herc_es_{a}"] = herc_weights(returns,
                                                  risk_measure="CVaR",
                                                  alpha=a)

    if 'herc_cdar' in portfolio:
        LOGGER.info('Computing HERC CDaR weights...')
        for a in alpha:
            port_w[f"herc_cdar_{a}"] = herc_weights(returns,
                                                    risk_measure="CDaR",
                                                    alpha=a)

    if "rb_factor" in portfolio:
        LOGGER.info('Computing RB factor weights...')
        port_w['rb_factor'] = get_rb_factor_weights(returns, loading)

    if "rb_factor_full_vol" in portfolio:
        LOGGER.info('Computing RB factor full Vol weights...')
        port_w['rb_factor_full_vol'] = get_rb_factor_full(
            returns, loading, risk_measure="MV")

    if "rb_factor_cdar" in portfolio:
        LOGGER.info('Computing RB factor CDaR weights...')
        for a in alpha:
            port_w[f"rb_factor_cdar_{a}"] = get_rb_factor_full(
                returns, loading, risk_measure="CDaR", alpha=a)

    if "rb_factor_es" in portfolio:
        LOGGER.info('Computing RB factor ES weights...')
        for a in alpha:
            port_w[f"rb_factor_es_{a}"] = get_rb_factor_full(
                returns, loading, risk_measure="CDaR", alpha=a)

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


def riskparity_weights(returns: pd.DataFrame(),
                       risk_measure: str = "MV",
                       budget: Optional[np.ndarray] = None,
                       alpha: float = 0.05,
                       **kwargs) -> pd.Series:
    if budget is not None and len(budget.shape) == 1:
        budget = budget.reshape(-1, 1)

    port = rf.Portfolio(returns=returns, alpha=alpha)
    port.assets_stats(method_mu="hist", method_cov="hist", d=0.94)
    weights = port.rp_optimization(model="Classic", rm=risk_measure,
                                   b=budget, **kwargs)

    return weights["weights"]


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


def expected_shortfall(returns, level: float = 0.05):
    VaR = np.quantile(returns, level)
    return np.mean(returns[returns <= VaR])


def compute_parity_cvar_weights(returns):
    weights = 1 / np.array([expected_shortfall(returns[c]) for c in
                            returns.columns])
    weights = weights / np.sum(weights)
    return weights


def get_rb_factor_full(returns, loading, threshold=1e-2, risk_measure="MV",
                       alpha=0.05, simple=False):
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

    assert len(cluster_ind.unique()) == n_components, (cluster_ind.unique(),
                                                       n_components)

    inner_weights = pd.DataFrame(0, columns=loading.columns, index=assets)

    for i in range(n_components):
        c_assets = cluster_ind[cluster_ind == i].index
        budget = loading.loc[c_assets, i].copy()
        if simple:
            budget.values[:] = 1
        budget = budget / np.linalg.norm(budget)
        budget = budget ** 2  # Must sum to 1, also is interpretable as

        port = rf.Portfolio(returns=returns[c_assets], alpha=alpha)
        port.assets_stats(method_mu="hist", method_cov="hist", d=0.94)
        w = port.rp_optimization(model="Classic", rm=risk_measure,
                                 b=budget.values.reshape(-1, 1))
        inner_weights.loc[c_assets, i] = w["weights"]

    # intra weights
    factor_returns = np.array(
        [np.dot(returns, inner_weights.values[:, i]) for i in range(
            n_components)])
    if factor_returns.shape[0] > 1:
        factor_returns = pd.DataFrame(factor_returns).T
        port = rf.Portfolio(returns=factor_returns, alpha=alpha)
        port.assets_stats(method_mu="hist", method_cov="hist", d=0.94)
        cluster_weights = port.rp_optimization(model="Classic", rm=risk_measure)
        cluster_weights = cluster_weights["weights"]
    else:
        cluster_weights = np.array([1.])

    weights = (inner_weights * cluster_weights).sum(axis=1)

    return weights


def get_rb_factor_weights(returns, loading, threshold=1e-2, simple=False):
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

    assert len(cluster_ind.unique()) == n_components, (cluster_ind.unique(),
                                                       n_components)

    inner_weights = pd.DataFrame(0, columns=loading.columns, index=assets)
    for i in range(n_components):
        c_assets = cluster_ind[cluster_ind == i].index
        budget = loading.loc[c_assets, i].copy()
        if simple:
            budget.values[:] = 1
        budget = budget / np.linalg.norm(budget)
        budget = budget**2  # Must sum to 1, also is interpretable as
        # explained variance
        inner_weights.loc[c_assets, i] = rb_perfect_corr(
            np.std(returns[c_assets], axis=0), budget, corr=1)

    factor_returns = np.array(
        [np.dot(returns, inner_weights.values[:, i]) for i in range(
            n_components)])

    assert len(factor_returns.shape) == 2
    if factor_returns.shape[0] > 1:
        cluster_sigmas = np.sqrt(np.diag(np.cov(factor_returns)))
        cluster_weights = 1 / cluster_sigmas / np.sum(1 / cluster_sigmas)
    else:
        cluster_weights = np.array([1.])

    weights = inner_weights * cluster_weights
    weights = weights.sum(axis=1)

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


def herc_weights(returns: pd.DataFrame, linkage: str = 'single',
                 risk_measure: str = "equal", alpha: float = 0.05, **kwargs):
    port = HCPortfolio(returns=returns, alpha=alpha)
    weights = port.optimization(model="HERC", rm=risk_measure,
                                linkage=linkage, **kwargs)
    weights = weights.loc[returns.columns, "weights"]

    return weights

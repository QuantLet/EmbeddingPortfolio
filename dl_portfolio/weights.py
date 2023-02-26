import pandas as pd
import numpy as np

import riskparityportfolio as rp
import cvxpy as cp

from typing import Union
from sklearn.cluster import KMeans

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models

from dl_portfolio.logger import LOGGER
from dl_portfolio.cluster import get_cluster_labels
from dl_portfolio.constant import PORTFOLIOS
from prado.hrp import correlDist, getQuasiDiag, getRecBipart
import scipy.cluster.hierarchy as sch

from portfoliolab.clustering.herc import HierarchicalEqualRiskContribution


def portfolio_weights(
    returns,
    shrink_cov=None,
    budget=None,
    embedding=None,
    loading=None,
    portfolio=["markowitz", "shrink_markowitz", "ivp", "aerp", "rp", "aeerc"],
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

    if "shrink_markowitz" in portfolio:
        assert shrink_cov is not None
        LOGGER.info("Computing shrinked Markowitz weights...")
        port_w["shrink_markowitz"] = markowitz_weights(mu, shrink_cov)

    if "ivp" in portfolio:
        LOGGER.info("Computing IVP weights...")
        port_w["ivp"] = ivp_weights(S)

    if "rp" in portfolio:
        LOGGER.info("Computing Riskparity weights...")
        assert budget is not None
        port_w["rp"] = riskparity_weights(S, budget=budget["rc"].values)

    if "kmaa" in portfolio:
        LOGGER.info("Computing KMeans Asset Allocation weights...")
        assert embedding is not None
        port_w["kmaa"] = kmaa_weights(returns, n_clusters=embedding.shape[-1])

    if "aerp" in portfolio:
        LOGGER.info("Computing AE Risk Parity weights...")
        assert embedding is not None
        port_w["aerp"] = ae_ivp_weights(returns, embedding)

    if "aeerc" in portfolio:
        LOGGER.info("Computing AE Risk Contribution weights...")
        assert budget is not None
        assert embedding is not None
        port_w["aeerc"] = ae_riskparity_weights(
            returns, embedding, loading, budget, risk_parity="budget"
        )

    if "ae_rp_c" in portfolio:
        LOGGER.info("Computing AE Risk Contribution Cluster weights...")
        assert budget is not None
        assert embedding is not None
        port_w["ae_rp_c"] = ae_riskparity_weights(
            returns, embedding, loading, budget, risk_parity="cluster"
        )

    if "sector_erc" in portfolio:
        LOGGER.info("Computing Sector ERC weights...")
        assert budget is not None
        embedding = pd.DataFrame(columns=budget["market"].unique(),
                                 index=budget.index)
        for c in embedding.columns:
            embedding.loc[
                budget.index[budget["market"] == c], c] = 1
        embedding.fillna(0, inplace=True)
        loading = embedding.copy()
        port_w["sector_erc"] = ae_riskparity_weights(
            returns,
            embedding,
            loading,
            budget,
            risk_parity="budget",
        )

    if "aeaa" in portfolio:
        LOGGER.info("Computing AE Asset Allocation weights...")
        port_w["aeaa"] = aeaa_weights(returns, embedding)

    if "hrp" in portfolio:
        port_w["hrp"] = getHRP(cov=S.values, corr=returns.corr().values,
                               index=returns.columns)

    if 'hcaa' in portfolio:
        LOGGER.info('Computing HCAA weights...')
        port_w['hcaa'] = herc_weights(returns,
                                      optimal_num_clusters=kwargs.get(
                                          'optimal_num_clusters'),
                                      risk_measure='equal_weighting')

    return port_w


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


def riskparity_weights(S: pd.DataFrame(), budget: np.ndarray) -> pd.Series:
    weights = rp.RiskParityPortfolio(covariance=S, budget=budget).weights
    weights = pd.Series(weights, index=S.index)

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
    weights = weights / n_clusters  # Rescale each weight
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


def ae_ivp_weights(returns, embedding):
    max_cluster = embedding.shape[-1] - 1
    # First get cluster allocation to forget about small contribution
    # Rename columns in case of previous renaming
    embedding.columns = list(range(len(embedding.columns)))
    clusters, _ = get_cluster_labels(embedding)
    clusters = {c: clusters[c] for c in clusters if c <= max_cluster}

    # Now get weights of assets inside each cluster
    cluster_asset_weights = {}
    cluster_weights = {}
    cov = returns.cov()
    for c in clusters:
        cluster_items = clusters[c]
        if cluster_items:
            c_weights = ivp_weights(cov.loc[cluster_items, cluster_items])
            cluster_var = get_cluster_var(
                cov, cluster_items, weights=c_weights
            )
            cluster_asset_weights[c] = c_weights
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

import pandas as pd
import numpy as np

import riskparityportfolio as rp
import cvxpy as cp

from typing import Union
from sklearn.cluster import KMeans

from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from portfoliolab.clustering.hrp import HierarchicalRiskParity
from portfoliolab.clustering.herc import HierarchicalEqualRiskContribution

from dl_portfolio.logger import LOGGER
from dl_portfolio.cluster import get_cluster_labels
from dl_portfolio.constant import PORTFOLIOS


def portfolio_weights(returns, shrink_cov=None, budget=None, embedding=None, loading=None,
                      portfolio=['markowitz', 'shrink_markowitz', 'ivp', 'ae_ivp', 'hrp', 'rp', 'ae_rp', 'herc'],
                      **kwargs):
    assert all([p in PORTFOLIOS for p in portfolio]), [p for p in portfolio if p not in PORTFOLIOS]
    port_w = {}

    mu = returns.mean()
    S = returns.cov()

    if 'markowitz' in portfolio:
        LOGGER.info('Computing Markowitz weights...')
        port_w['markowitz'] = markowitz_weights(mu, S)

    if 'shrink_markowitz' in portfolio:
        assert shrink_cov is not None
        LOGGER.info('Computing shrinked Markowitz weights...')
        port_w['shrink_markowitz'] = markowitz_weights(mu, shrink_cov)

    if 'ivp' in portfolio:
        LOGGER.info('Computing IVP weights...')
        port_w['ivp'] = ivp_weights(S)

    if 'hrp' in portfolio:
        LOGGER.info('Computing HRP weights...')
        port_w['hrp'] = hrp_weights(S)

    if 'herc' in portfolio:
        LOGGER.info('Computing HERC weights with variance as risk measure...')
        port_w['herc'] = herc_weights(returns, optimal_num_clusters=kwargs.get('optimal_num_clusters'),
                                      risk_measure='variance')

    if 'hcaa' in portfolio:
        LOGGER.info('Computing HCAA weights...')
        port_w['hcaa'] = herc_weights(returns, optimal_num_clusters=kwargs.get('optimal_num_clusters'),
                                      risk_measure='equal_weighting')

    if 'rp' in portfolio:
        LOGGER.info('Computing Riskparity weights...')
        assert budget is not None
        port_w['rp'] = riskparity_weights(S, budget=budget['rc'].values)

    if 'kmaa' in portfolio:
        LOGGER.info('Computing KMeans Asset Allocation weights...')
        assert embedding is not None
        port_w['kmaa'] = kmaa_weights(returns, n_clusters=embedding.shape[-1])

    if 'ae_ivp' in portfolio:
        LOGGER.info('Computing AE IVP weights...')
        assert embedding is not None
        port_w['ae_ivp'] = ae_ivp_weights(returns, embedding)

    if 'ae_rp' in portfolio:
        LOGGER.info('Computing AE Riskparity weights...')
        assert budget is not None
        assert embedding is not None
        port_w['ae_rp'] = ae_riskparity_weights(returns, embedding, loading, budget, risk_parity='budget')

    if 'ae_rp_c' in portfolio:
        LOGGER.info('Computing AE Riskparity Cluster weights...')
        assert budget is not None
        assert embedding is not None
        port_w['ae_rp_c'] = ae_riskparity_weights(returns, embedding, loading, budget, risk_parity='cluster')

    if 'aeaa' in portfolio:
        LOGGER.info('Computing AE Asset Allocation weights...')
        port_w['aeaa'] = aeaa_weights(returns, embedding)

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
    if weights is not None:
        weights = ivp_weights(cov_slice)
    return np.linalg.multi_dot((weights, cov_slice, weights))


def get_inner_cluster_weights(cov, loading, clusters, market_budget=None):
    weights = {}
    n_clusters = len(clusters)
    for c in clusters:
        cluster_items = clusters[c]

        if cluster_items:
            if market_budget is not None:
                budget = market_budget.loc[cluster_items, 'rc']
            else:
                budget = loading.loc[cluster_items, c] ** 2 / np.sum(loading.loc[cluster_items, c] ** 2)
            cov_slice = cov.loc[cluster_items, cluster_items]
            weights[c] = pd.Series(
                rp.RiskParityPortfolio(covariance=cov_slice, budget=budget.values).weights,
                index=cluster_items
            )
    weights = {i: weights.get(i) for i in range(n_clusters) if weights.get(i) is not None}  # reorder dict

    return weights


def cluster_portfolio(returns, features_cov, embedding, no_leverage=True):
    n_clusters = features_cov.shape[0]
    cluster_weights = ivp_weights(features_cov)
    if no_leverage:
        asset_weights = (embedding / embedding.sum())
    else:
        asset_weights = embedding.copy()

    port_returns = pd.Series(np.zeros(len(returns)), index=returns.index)
    for i in range(n_clusters):
        cluster_ret = ((cluster_weights[i] * asset_weights.iloc[:, i]) * returns).sum(1)
        port_returns = port_returns + cluster_ret.values

    return port_returns, cluster_weights, asset_weights


def get_mdd(performance: [pd.Series, np.ndarray]):
    assert len(performance.shape) == 1
    dd = performance / performance.cummax() - 1.0
    mdd = dd.cummin()
    mdd = abs(min(mdd))
    return mdd


def calmar_ratio(performance: [pd.Series, np.ndarray]):
    assert len(performance.shape) == 1
    annual_return = performance[-1] / performance[0] - 1
    mdd = get_mdd(performance)
    return annual_return / mdd


def markowitz_weights(mu: Union[pd.Series, np.ndarray], S: pd.DataFrame, fix_cov: bool = False,
                      risk_free_rate: float = 0.) -> pd.Series:
    if fix_cov:
        S = risk_models.fix_nonpositive_semidefinite(S, fix_method='spectral')
    weights = None
    try:
        LOGGER.info(f"Trying Markowitz with default 'ECOS' solver")
        ef = EfficientFrontier(mu, S, verbose=False)
        # ef.add_objective(objective_functions.L2_reg, gamma=0)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = pd.Series(weights, index=weights.keys())
        LOGGER.info("Success")
    except Exception as _exc:
        LOGGER.info(f'Error with max sharpe: {_exc}')
        try:
            LOGGER.info(f"Trying Markowitz with 'SCS' solver")
            ef = EfficientFrontier(mu, S, verbose=True, solver='SCS')
            # ef.add_objective(objective_functions.L2_reg, gamma=0)
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            weights = pd.Series(weights, index=weights.keys())
            LOGGER.info("Success")
        except Exception as _exc:
            LOGGER.info(f'Error with max sharpe: {_exc}')
            try:
                LOGGER.info(f"Trying Markowitz with 'OSQP' solver")
                ef = EfficientFrontier(mu, S, verbose=True, solver='OSQP')
                # ef.add_objective(objective_functions.L2_reg, gamma=0)
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                weights = pd.Series(weights, index=weights.keys())
                LOGGER.info("Success")
            except Exception as _exc:
                LOGGER.info(f'Error with max sharpe: {_exc}')
                try:
                    LOGGER.info(f"Trying Markowitz with 'CVXOPT' solver")

                    # ef = EfficientFrontier(mu, S, verbose=True, solver=cp.CVXOPT, solver_options={'feastol': 1e-4})
                    ef = EfficientFrontier(mu, S, verbose=True, solver=cp.SCS)
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
                    weights = pd.Series(weights, index=weights.keys())
                    LOGGER.info("Success")
                except Exception as _exc:
                    LOGGER.info(f'Error with max sharpe: {_exc}')

    if weights is None:
        raise _exc

    return weights


def hrp_weights_old(S: pd.DataFrame) -> pd.Series:
    hrp = HRPOpt(cov_matrix=S)
    weights = hrp.optimize()
    weights = pd.Series(weights, index=S.index)
    return weights


def hrp_weights(S: pd.DataFrame, linkage: str = 'single') -> pd.Series:
    # constructing our Single Linkage portfolio
    hrp_single = HierarchicalRiskParity()
    hrp_single.allocate(asset_names=S.columns,
                        covariance_matrix=S,
                        linkage=linkage)

    weights = hrp_single.weights.T
    weights = weights[0]
    weights = weights[S.columns]

    return weights


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


def ivp_weights(S: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(S.values)
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


def ae_riskparity_weights(returns, embedding, loading, market_budget, risk_parity='budget'):
    """

    :param returns:
    :param embedding: To get cluster assignment
    :param loading: To get inner cluster weights
    :param market_budget:
    :param risk_parity: if 'budget' then use budget for risk allocation, if 'cluster' use relative asset cluster
    importance from the embedding matrix
    :return:
    """
    assert risk_parity in ['budget', 'cluster']
    max_cluster = embedding.shape[-1] - 1
    # First get cluster allocation to forget about small contribution
    clusters, _ = get_cluster_labels(embedding)
    clusters = {c: clusters[c] for c in clusters if c <= max_cluster}

    # Now get weights of assets inside each cluster
    if risk_parity == 'budget':
        inner_cluster_weights = get_inner_cluster_weights(returns.cov(),
                                                          loading,
                                                          clusters,
                                                          market_budget=market_budget)
    elif risk_parity == 'cluster':
        inner_cluster_weights = get_inner_cluster_weights(returns.cov(),
                                                          loading,
                                                          clusters)
    else:
        raise NotImplementedError(risk_parity)

    # Now compute return of each cluster
    cluster_returns = pd.DataFrame()
    for c in inner_cluster_weights:
        cret = (returns[inner_cluster_weights[c].index] * inner_cluster_weights[c]).sum(1)
        cluster_returns = pd.concat([cluster_returns, cret], 1)
    cluster_returns.columns = list(inner_cluster_weights.keys())

    # Now get risk contribution of each cluster defined by user
    cluster_rc = {c: (inner_cluster_weights[c]).idxmax() for c in inner_cluster_weights}
    cluster_rc = {c: market_budget.loc[cluster_rc[c], 'rc'] for c in cluster_rc}

    # Compute cluster weights with risk parity portfolio
    cov = cluster_returns.cov()
    budget = np.array(list(cluster_rc.values()))
    budget = budget / np.sum(budget)
    cluster_weight = rp.RiskParityPortfolio(covariance=cov, budget=budget).weights
    # Compute asset weight inside global portfolio
    weights = pd.Series(dtype='float32')
    for c in inner_cluster_weights:
        weights = pd.concat([weights, inner_cluster_weights[c] * cluster_weight[c]])
    weights = weights.reindex(returns.columns)  # rerorder
    weights.fillna(0., inplace=True)

    return weights


def kmaa_weights(returns: pd.DataFrame, n_clusters: int) -> pd.Series:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(returns.T)
    labels = pd.DataFrame(kmeans.labels_.reshape(1, -1), columns=returns.columns).T
    labels.columns = ['label']
    clusters = {i: list(labels[labels['label'] == i].index) for i in range(n_clusters)}

    # Now get weights of assets inside each cluster
    cluster_weights = {c: pd.Series([1 / len(clusters[c])] * len(clusters[c]), index=clusters[c]) for c in clusters}
    # {asset: 1 / n_items for asset in clusters[c]}}

    # Compute asset weight inside global portfolio
    weights = pd.Series(dtype='float32')
    for c in cluster_weights:
        weights = pd.concat([weights, cluster_weights[c]])
    weights = weights / n_clusters  # Rescale each weight
    weights = weights.reindex(returns.columns)  # rerorder
    weights.fillna(0., inplace=True)

    return weights


def aeaa_weights(returns: Union[np.ndarray, pd.DataFrame], embedding: Union[np.ndarray, pd.DataFrame]) -> pd.Series:
    max_cluster = embedding.shape[-1] - 1
    # First get cluster allocation to forget about small contribution
    clusters, _ = get_cluster_labels(embedding)
    clusters = {c: clusters[c] for c in clusters if c <= max_cluster}
    n_clusters = embedding.shape[-1]

    # Now get weights of assets inside each cluster
    cluster_weights = {c: pd.Series([1 / len(clusters[c])] * len(clusters[c]), index=clusters[c]) for c in clusters}
    # {asset: 1 / n_items for asset in clusters[c]}}

    # Compute asset weight inside global portfolio
    weights = pd.Series(dtype='float32')
    for c in cluster_weights:
        weights = pd.concat([weights, cluster_weights[c]])
    weights = weights / n_clusters  # Rescale each weight
    weights = weights.reindex(returns.columns)  # rerorder
    weights.fillna(0., inplace=True)

    return weights


def equal_class_weights(market_budget: pd.DataFrame):
    market_class = np.unique(market_budget['market'].values, return_counts=True)
    inner_class_weight = {c: 1 / market_class[1][i] for i, c in enumerate(market_class[0])}
    weights = pd.Series(index=market_budget.index)
    for c in inner_class_weight:
        assets = market_budget.index[market_budget['market'] == c]
        weights.loc[assets] = inner_class_weight[c]
    weights /= np.sum(weights)

    return weights


def ae_ivp_weights(returns, embedding):
    max_cluster = embedding.shape[-1] - 1
    # First get cluster allocation to forget about small contribution
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
            cluster_var = get_cluster_var(cov, cluster_items, weights=cluster_weights)
            cluster_asset_weights[c] = c_weights
            cluster_weights[c] = cluster_var

    cluster_weights = {c: 1 / cluster_weights[c] for c in cluster_weights}
    cluster_weights = {c: cluster_weights[c] / np.sum(list(cluster_weights.values())) for c in cluster_weights}
    cluster_weights = {c: cluster_weights[c] * cluster_asset_weights[c] for c in cluster_weights}

    # Compute asset weight inside global portfolio
    weights = pd.Series(dtype='float32')
    for c in cluster_weights:
        weights = pd.concat([weights, cluster_weights[c]])
    weights = weights.reindex(returns.columns)  # rerorder
    weights.fillna(0., inplace=True)

    return weights

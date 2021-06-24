import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import riskparityportfolio as rp
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from dl_portfolio.logger import LOGGER
from typing import Union, Dict
import pickle

def cv_portfolio_perf(cv_results: Dict,
                      portfolios=['equal', 'markowitz', 'shrink_markowitz', 'ivp', 'hrp', 'rp', 'ae_rp']):
    port_perf = {}
    for p in portfolios:
        port_perf[p] = {}
        port_perf[p]['total'] = pd.DataFrame()

    for cv in cv_results:
        for p in portfolios:
            if p == 'equal':
                port_perf['equal'][cv] = (cv_results[cv]['returns']).mean(1)
            else:
                port_perf[p][cv] = (cv_results[cv]['returns'] * cv_results[cv]['port'][p]).sum(1)

            port_perf[p]['total'] = pd.concat([port_perf[p]['total'], port_perf[p][cv]])

    return port_perf


def get_cv_results(base_dir, test_set, n_folds, market_budget):
    assert test_set in ['val', 'test']
    cv_results = {cv: {} for cv in range(n_folds)}
    for cv in range(n_folds):
        scaler = pickle.load(open(f'{base_dir}/{cv}/scaler.p', 'rb'))
        embedding = pd.read_pickle(f'{base_dir}/{cv}/encoder_weights.p')
        train_returns = pd.read_pickle(f'{base_dir}/{cv}/train_returns.p')
        returns = pd.read_pickle(f'{base_dir}/{cv}/{test_set}_returns.p')
        pred = pd.read_pickle(f'{base_dir}/{cv}/{test_set}_prediction.p')
        train_features = pd.read_pickle(f'{base_dir}/{cv}/train_features.p')

        residuals = returns - pred
        std = np.sqrt(scaler['attributes']['var_'])
        scaled_residuals = residuals * std
        scaled_embedding = np.dot(np.diag(std, k=0), embedding)
        assets = train_returns.columns

        cv_results[cv]['embedding'] = embedding
        cv_results[cv]['scaled_embedding'] = scaled_embedding
        cv_results[cv]['Sf'] = train_features.cov()
        cv_results[cv]['Su'] = scaled_residuals.cov()
        cv_results[cv]['H'] = pd.DataFrame(
            np.dot(scaled_embedding, np.dot(cv_results[cv]['Sf'], scaled_embedding.T)) + cv_results[cv]['Su'],
            index=embedding.index,
            columns=embedding.index)
        cv_results[cv]['w'] = embedding
        cv_results[cv]['returns'] = returns
        cv_results[cv]['port'] = portfolio_weights(train_returns,
                                                   shrink_cov=cv_results[cv]['H'],
                                                   budget=market_budget.loc[assets],
                                                   embedding=embedding
                                                   )
    return cv_results


def portfolio_weights(returns, shrink_cov=None, budget=None, embedding=None,
                      portfolio=['markowitz', 'shrink_markowitz', 'ivp', 'hrp', 'rp', 'ae_rp']):
    port_w = {}

    mu = returns.mean()
    S = returns.cov()

    if 'markowitz' in portfolio:
        LOGGER.info('Computing Markowitz weights...')
        markowitz_weights(mu, S)
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

    if 'rp' in portfolio:
        LOGGER.info('Computing Riskparity weights...')
        assert budget is not None
        port_w['rp'] = riskparity_weights(S, budget=budget['rc'].values)

    if 'ae_rp' in portfolio:
        LOGGER.info('Computing AE Riskparity weights...')
        assert budget is not None
        assert embedding is not None
        port_w['ae_rp'] = ae_riskparity_weights(returns, embedding, budget)

    return port_w


def markowitz_weights(mu: Union[pd.Series, np.ndarray], S: pd.DataFrame, fix_cov: bool = False,
                      risk_free_rate: float = 0.):
    if fix_cov:
        S = risk_models.fix_nonpositive_semidefinite(S, fix_method='spectral')

    try:
        ef = EfficientFrontier(mu, S, verbose=False)
        # ef.add_objective(objective_functions.L2_reg, gamma=0)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = pd.Series(weights, index=weights.keys())

    except Exception as _exc:
        LOGGER.info(f'Error with max sharpe: {_exc}')
        weights = None

    return weights


def hrp_weights(S: pd.DataFrame):
    hrp = HRPOpt(cov_matrix=S)
    weights = hrp.optimize()
    weights = pd.Series(weights, index=S.index)
    return weights


def ivp_weights(S: Union[pd.DataFrame, np.ndarray]):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(S.values)
    weights = ivp / ivp.sum()

    if isinstance(S, pd.DataFrame):
        weights = pd.Series(weights, index=S.index)
    else:
        weights = pd.Series(weights)

    return weights


def riskparity_weights(S: pd.DataFrame(), budget: np.ndarray):
    weights = rp.RiskParityPortfolio(covariance=S, budget=budget).weights
    weights = pd.Series(weights, index=S.index)

    return weights


def ae_riskparity_weights(returns, embedding, market_budget):
    # First get cluster allocation to forget about small contribution
    clusters = get_cluster_labels(embedding)

    # Now get weights of assets inside each cluster
    cluster_weights = get_cluster_weights(returns.cov(),
                                          embedding,
                                          clusters)
    # Now compute return of each cluster
    cluster_returns = pd.DataFrame()
    for c in cluster_weights:
        cret = (returns[cluster_weights[c].index] * cluster_weights[c]).sum(1)
        cluster_returns = pd.concat([cluster_returns, cret], 1)
    cluster_returns.columns = list(cluster_weights.keys())

    # Now get risk contribution of each cluster defined by user
    cluster_rc = {c: (cluster_weights[c]).idxmax() for c in cluster_weights}
    cluster_rc = {c: market_budget.loc[cluster_rc[c], 'rc'] for c in cluster_rc}

    # Compute cluster weights with risk parity portfolio
    cov = cluster_returns.cov()
    budget = np.array(list(cluster_rc.values()))
    budget = budget / np.sum(budget)
    c_weights = rp.RiskParityPortfolio(covariance=cov, budget=budget).weights

    # Compute asset weight inside global portfolio
    weights = pd.Series(dtype='float32')
    for c in cluster_weights:
        weights = pd.concat([weights, cluster_weights[c] * c_weights[c]])
    weights = weights.loc[returns.columns]  # rerorder

    return weights


def get_cluster_var(cov, cluster_items, weights=None):
    """
    Compute the variance per cluster

    :param cov: covariance matrix
    :type cov: np.ndarray
    :param cluster_items: tickers in the cluster
    :type cluster_items: list
    :return: the variance per cluster
    :rtype: float
    """
    if weights is not None:
        weights = get_cluster_inverse_var_weights(cov, cluster_items)
    cov_slice = cov.loc[cluster_items, cluster_items]

    return np.linalg.multi_dot((weights, cov_slice, weights))


def get_cluster_inverse_var_weights(cov, cluster_items):
    """
    Compute the inverse variance weights of each asset in cluster_items

    :param cov: covariance matrix
    :type cov: np.ndarray
    :param cluster_items: tickers in the cluster
    :type cluster_items: list
    :return: the variance per cluster
    :rtype: float
    """
    # Compute variance per cluster
    cov_slice = cov.loc[cluster_items, cluster_items]
    weights = 1 / np.diag(cov_slice)  # Inverse variance weights
    weights /= weights.sum()

    weights = pd.Series(weights.astype(np.float32), index=cluster_items)
    return weights


def get_cluster_labels(embedding):
    n_clusters = embedding.shape[-1]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding.values)
    labels = pd.DataFrame(kmeans.labels_, index=embedding.index, columns=['label'])
    clusters = {}
    for i in range(n_clusters):
        assets = list(labels.loc[labels['label'] == i].index)
        label = np.argmax(embedding.loc[assets, :].sum())
        clusters[label] = assets
    clusters = {i: clusters[i] for i in range(n_clusters)}  # reorder dict
    return clusters


def get_cluster_weights(cov, embedding, clusters):
    cluster_weights = {}
    n_clusters = len(clusters)
    for c in clusters:
        cluster_items = clusters[c]
        budget = embedding.loc[cluster_items, c] / np.sum(embedding.loc[cluster_items, c])
        cov_slice = cov.loc[cluster_items, cluster_items]

        cluster_weights[c] = pd.Series(
            rp.RiskParityPortfolio(covariance=cov_slice, budget=budget.values).weights,
            index=cluster_items
        )
    cluster_weights = {i: cluster_weights[i] for i in range(n_clusters)}  # reorder dict

    return cluster_weights


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


def get_portfolio_perf(train_returns, returns, features, embedding, fx_levrage=1.):
    # AE port
    features_cov = features.cov()
    port_returns, cluster_weights, asset_weights = cluster_portfolio(returns, features_cov, embedding)

    ae = {
        'returns': port_returns,
        'cluster_weights': cluster_weights,
        'asset_weights': asset_weights
    }
    # ivp
    cov = np.cov(train_returns, rowvar=False)
    weights = ivp_weights(cov)
    port_returns = (returns * weights).sum(1)
    ivp = {
        'returns': port_returns,
        'weights': weights
    }
    # equally
    port_returns = returns.mean(1)
    equally_weighted = {
        'returns': port_returns
    }
    return equally_weighted, ivp, ae

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import riskparityportfolio as rp
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from dl_portfolio.logger import LOGGER
from typing import Union, Dict, Optional
import pickle
from joblib import Parallel, delayed


def get_timeseries_weights(cv_results):
    weights = {}
    portfolios = cv_results[0]['port'].keys()
    for p in portfolios:
        pweights = pd.DataFrame()
        dates = []
        for cv in cv_results.keys():
            dates.append(cv_results[cv]['returns'].index[0])
            w = cv_results[cv]['port'][p]
            if w is None:
                assets = cv_results[cv]['returns'].columns
                w = pd.Series([None] * len(assets), index=assets)
            pweights = pd.concat([pweights, w], 1)
        pweights = pweights.T
        pweights.index = dates
        weights[p] = pweights
    return weights


def cv_portfolio_perf(cv_results: Dict,
                      portfolios=['equal', 'markowitz', 'shrink_markowitz', 'ivp', 'ae_ivp', 'hrp', 'rp', 'ae_rp']):
    port_perf = {}
    for p in portfolios:
        port_perf[p] = {}
        port_perf[p]['total'] = pd.DataFrame()

    for cv in cv_results:
        for p in portfolios:
            if p == 'equal':
                port_perf['equal'][cv] = (cv_results[cv]['returns']).mean(1)
            else:
                if cv_results[cv]['port'][p] is not None:
                    port_perf[p][cv] = (cv_results[cv]['returns'] * cv_results[cv]['port'][p]).sum(1)
                else:
                    port_perf[p][cv] = cv_results[cv]['returns'] * np.nan

            port_perf[p]['total'] = pd.concat([port_perf[p]['total'], port_perf[p][cv]])

    return port_perf


def get_cv_results(base_dir, test_set, n_folds, market_budget=None, compute_weights=True, window: Optional[int] = None,
                   n_jobs: int = None):
    assert test_set in ['val', 'test']

    def run(cv):
        LOGGER.info(f'CV {cv}')
        res = {}
        scaler = pickle.load(open(f'{base_dir}/{cv}/scaler.p', 'rb'))
        embedding = pd.read_pickle(f'{base_dir}/{cv}/encoder_weights.p')
        train_returns = pd.read_pickle(f'{base_dir}/{cv}/train_returns.p')
        returns = pd.read_pickle(f'{base_dir}/{cv}/{test_set}_returns.p')
        pred = pd.read_pickle(f'{base_dir}/{cv}/{test_set}_prediction.p')
        train_features = pd.read_pickle(f'{base_dir}/{cv}/train_features.p')
        test_features = pd.read_pickle(f'{base_dir}/{cv}/{test_set}_features.p')

        residuals = returns - pred

        std = np.sqrt(scaler['attributes']['var_'])
        scaled_residuals = residuals * std
        scaled_embedding = np.dot(np.diag(std, k=0), embedding)
        assets = train_returns.columns

        res['embedding'] = embedding
        res['scaled_embedding'] = scaled_embedding
        res['train_features'] = train_features
        res['test_features'] = test_features
        res['test_pred'] = pred
        res['Sf'] = train_features.cov()
        res['Su'] = scaled_residuals.cov()
        res['H'] = pd.DataFrame(
            np.dot(scaled_embedding, np.dot(res['Sf'], scaled_embedding.T)) + res['Su'],
            index=embedding.index,
            columns=embedding.index)
        res['w'] = embedding
        res['returns'] = returns
        if compute_weights:
            assert market_budget is not None
            if window is not None:
                assert isinstance(window, int)
                res['port'] = portfolio_weights(train_returns.iloc[-window:],
                                                shrink_cov=res['H'],
                                                budget=market_budget.loc[assets],
                                                embedding=embedding
                                                )
            else:
                res['port'] = portfolio_weights(train_returns,
                                                shrink_cov=res['H'],
                                                budget=market_budget.loc[assets],
                                                embedding=embedding
                                                )
        else:
            res['port'] = None
        res['mean_mse'] = np.mean((residuals ** 2).mean(1))
        res['mse'] = np.sum((residuals ** 2).mean(1))

        return cv, res

    if n_jobs:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            cv_results = _parallel_pool(
                delayed(run)(cv) for cv in range(n_folds)
            )

        # Build dictionary
        cv_results = {cv_results[i][0]: cv_results[i][1] for i in range(len(cv_results))}
        # Reorder dictionary
        cv_results = {cv: cv_results[cv] for cv in range(n_folds)}
    else:
        cv_results = {}
        for cv in range(n_folds):
            _, cv_results[cv] = run(cv)

    return cv_results


def portfolio_weights(returns, shrink_cov=None, budget=None, embedding=None,
                      portfolio=['markowitz', 'shrink_markowitz', 'ivp', 'ae_ivp', 'hrp', 'rp', 'ae_rp']):
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

    if 'ae_ivp' in portfolio:
        LOGGER.info('Computing AE IVP weights...')
        assert embedding is not None
        port_w['ae_ivp'] = ae_ivp_weights(returns, embedding)

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
                                          clusters,
                                          market_budget=market_budget)
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
    weights = weights.reindex(returns.columns)  # rerorder
    weights.fillna(0., inplace=True)

    return weights


def ae_ivp_weights(returns, embedding):
    # First get cluster allocation to forget about small contribution
    clusters = get_cluster_labels(embedding)

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
    cov_slice = cov.loc[cluster_items, cluster_items]
    if weights is not None:
        weights = ivp_weights(cov_slice)
    return np.linalg.multi_dot((weights, cov_slice, weights))


def get_cluster_labels(embedding, threshold=0.1):
    n_clusters = embedding.shape[-1]
    if threshold is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding.values)
        labels = pd.DataFrame(kmeans.labels_, index=embedding.index, columns=['label'])
    else:
        mask = embedding >= threshold
        # assert all(mask.sum(1) <= 1)
        labels = pd.DataFrame(mask.idxmax(axis=1), columns=['label'])
    clusters = {}
    for i in range(n_clusters):
        assets = list(labels.loc[labels['label'] == i].index)
        if len(assets) > 0:
            label = np.argmax(embedding.loc[assets, :].sum())
            clusters[label] = assets
    clusters = {i: clusters.get(i) for i in range(n_clusters) if clusters.get(i) is not None}  # reorder dict

    return clusters


def get_cluster_weights(cov, embedding, clusters, market_budget=None):
    cluster_weights = {}
    n_clusters = len(clusters)
    for c in clusters:
        cluster_items = clusters[c]

        if cluster_items:
            if market_budget is not None:
                budget = market_budget.loc[cluster_items, 'rc']
            else:
                budget = embedding.loc[cluster_items, c] / np.sum(embedding.loc[cluster_items, c])

            cov_slice = cov.loc[cluster_items, cluster_items]

            cluster_weights[c] = pd.Series(
                rp.RiskParityPortfolio(covariance=cov_slice, budget=budget.values).weights,
                index=cluster_items
            )
    cluster_weights = {i: cluster_weights.get(i) for i in range(n_clusters) if
                       cluster_weights.get(i) is not None}  # reorder dict

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

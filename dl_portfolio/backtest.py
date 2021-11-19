import pickle
import scipy

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import riskparityportfolio as rp

from typing import Union, Dict, Optional, List
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models

from portfoliolab.clustering.hrp import HierarchicalRiskParity
from portfoliolab.clustering.herc import HierarchicalEqualRiskContribution

from dl_portfolio.logger import LOGGER
from dl_portfolio.cluster import get_cluster_labels
from dl_portfolio.ae_data import load_data
from dl_portfolio.constant import CRYPTO_ASSETS
from dl_portfolio.ae_data import get_features
from dl_portfolio.pca_ae import build_model
from dl_portfolio.utils import load_result

PORTFOLIOS = ['equal', 'equal_class', 'markowitz', 'shrink_markowitz', 'ivp', 'ae_ivp', 'hrp', 'rp', 'ae_rp', 'herc',
              'hcaa', 'ae_rp_c', 'kmaa', 'aeaa']


def get_ts_weights(cv_results, port) -> pd.DataFrame:
    dates = [cv_results[0][cv]['returns'].index[0] for cv in cv_results[0]]
    weights = pd.DataFrame()
    for cv in cv_results[0]:
        if 'ae' in port:
            avg_weights_cv = pd.DataFrame()
            for i in cv_results:
                w = pd.DataFrame(cv_results[i][cv]['port'][port]).T
                avg_weights_cv = pd.concat([avg_weights_cv, w])
            avg_weights_cv = avg_weights_cv.mean()
            avg_weights_cv = pd.DataFrame(avg_weights_cv).T
            weights = pd.concat([weights, avg_weights_cv])
        else:
            w = pd.DataFrame(cv_results[0][cv]['port'][port]).T
            weights = pd.concat([weights, w])
    try:
        weights.index = dates
    except Exception as _exc:
        raise Exception(f"Probblem with portfolio '{port}':\n{_exc}")

    return weights


def get_average_perf(port_perf: Dict, port: str) -> pd.DataFrame:
    perf = pd.DataFrame()
    for i in port_perf:
        perf = pd.concat([perf, port_perf[i][port]['total']], 1)
    perf = perf.mean(1)
    return perf


def plot_perf(perf, strategies=['aerp'], save_path=None, show=False, legend=True):
    plt.figure(figsize=(20, 10))
    for s in strategies:
        plt.plot(np.cumprod(perf[s] + 1) - 1, label=s)
    if legend:
        plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def bar_plot_weights(weights, show=False, legend=False, save_path=None):
    labels = [str(d.date()) for d in weights.index]
    fig, ax = plt.subplots(figsize=(20, 10))
    NUM_COLORS = len(weights.columns)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    for i, c in enumerate(list(weights.columns)):
        ax.bar(labels, weights[c],
               label=c, width=1, align='edge',
               bottom=weights.iloc[:, :i].sum(1))
    if legend:
        plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()


def backtest_stats(perf: Dict, weights: Dict, period: int = 252, format: bool = True, **kwargs):
    """

    :param perf:
    :param weights:
    :param period:
    :param format:
    :return:
    """
    strats = list(perf.keys())
    stats = pd.DataFrame(index=strats,
                         columns=['Return', 'Volatility', 'Skewness', 'Excess kurtosis', 'VaR-5%',
                                  'ES-5%', 'SR', 'ASR', 'MDD', 'CR', 'CEQ', 'SSPW', 'TTO'],
                         dtype=np.float32)
    assets = weights['hrp'].columns
    n_assets = weights['hrp'].shape[-1]
    for strat in strats:
        if strat == 'equal':
            weights['equal'] = pd.DataFrame([1 / n_assets] * n_assets).T
            weights['equal'].columns = assets

        elif strat == 'equal_class':
            market_budget = kwargs.get('market_budget')
            assert market_budget is not None
            weights['equal_class'] = pd.DataFrame(equal_class_weights(market_budget)).T

        stats.loc[strat] = [perf[strat].mean(),
                            annualized_volatility(perf[strat], period=period),
                            scipy.stats.skew(perf[strat], axis=0),
                            scipy.stats.kurtosis(perf[strat], axis=0) - 3,
                            hist_VaR(perf[strat], level=0.05),
                            hist_ES(perf[strat], level=0.05),
                            sharpe_ratio(perf[strat], period=period),
                            adjusted_sharpe_ratio(perf[strat], period=period),
                            get_mdd(np.cumprod(perf[strat] + 1)),
                            calmar_ratio(np.cumprod(perf[strat] + 1)),
                            ceq(perf[strat]),
                            sspw(weights[strat]),
                            total_average_turnover(weights[strat]) if strat != 'equal' else 0.
                            ]
    if format:
        stats['Return'] = stats['Return'] * 100
        stats['VaR-5%'] = stats['VaR-5%'] * 100
        stats['ES-5%'] = stats['ES-5%'] * 100
        stats['MDD'] = stats['MDD'] * 100
        stats['CEQ'] = stats['CEQ'] * 100
        stats = np.round(stats, 2)

    return stats


def annualized_volatility(perf, period: int = 1):
    return perf.std() * np.sqrt(period)


def hist_VaR(perf, level: float = 0.05):
    return - np.quantile(perf, level)


def hist_ES(perf, level: float = 0.05):
    VaR = np.quantile(perf, level)
    return - np.mean(perf[perf <= VaR])


def ceq(returns, gamma: float = 1., risk_free=0.):
    """

    :param perf: returns
    :param gamma: risk aversion
    :param risk_free: risk free rate
    :return:
    """

    return (returns.mean() - risk_free) - gamma / 2 * returns.var()


def sspw(weights):
    """
    compute averge sum of squared portfolio weights SSPW
    :param weights: pd.DataFrame
    :return:
    """
    return (weights ** 2).sum(1).mean()


def average_turnover(weights):
    """
    compute average turnover per rebalancing
    :param weights: pd.DataFrame
    :return:
    """
    return weights.diff().abs().dropna().mean()


def total_average_turnover(weights):
    """
    compute total average turnover per rebalancing
    :param weights: pd.DataFrame
    :return:
    """
    return weights.diff().abs().dropna().mean().sum()


def adjusted_sharpe_ratio(perf, period: int = 1):
    # check (Pezier and White (2008))
    # SR x [1 + (S/6) x SR - ((K-3) / 24) x SR^2]

    # annualized ASR = sqrt(period) * ASR
    # Indeed annualized skew = skew / sqrt(period) and annualized kurtosis = kurtosis / period

    sr = sharpe_ratio(perf)
    skew = scipy.stats.skew(perf, axis=0)
    kurtosis = scipy.stats.kurtosis(perf, axis=0)

    return sr * (1 + (skew / 6) * sr - ((kurtosis - 3) / 24) * (sr ** 2)) * np.sqrt(period)


def sharpe_ratio(perf, period: int = 1):
    return perf.mean() / perf.std() * np.sqrt(period)


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
                      portfolios: List = ['equal', 'markowitz', 'shrink_markowitz', 'ivp', 'ae_ivp', 'hrp', 'rp',
                                          'ae_rp'],
                      annualized: bool = True,
                      fee: float = 2e-4,
                      **kwargs):
    """

    :param cv_results:
    :param portfolios:
    :param annualized:
    :param fee: 2 bps = 0.02 %
    :return:
    """
    assert all([p in PORTFOLIOS for p in portfolios])
    N = cv_results[0]['returns'].shape[-1]
    port_perf = {}
    for p in portfolios:
        port_perf[p] = {}
        port_perf[p]['total'] = pd.DataFrame()
        port_perf[p]['cost'] = []

    for cv in cv_results:
        for p in portfolios:
            if p == 'equal':
                port_perf['equal'][cv] = portfolio_return(cv_results[cv]['returns'], weights=1 / N)
            elif p == 'equal_class':
                market_budget = kwargs.get('market_budget')
                assert market_budget is not None
                weights = equal_class_weights(market_budget)
                port_perf['equal_class'][cv] = portfolio_return(cv_results[cv]['returns'], weights=weights)
            else:
                if cv_results[cv]['port'][p] is not None:
                    port_perf[p][cv] = portfolio_return(cv_results[cv]['returns'], weights=cv_results[cv]['port'][p])
                else:
                    LOGGER.info(f'Warning: No weight for {p} portfolio at cv {cv}. Setting to NaN')
                    port_perf[p][cv] = cv_results[cv]['returns'] * np.nan
            # Volatility target weights
            if annualized:
                if p == 'equal':
                    train_port_perf = portfolio_return(cv_results[cv]['train_returns'], weights=1 / N)
                    cost = 0
                elif p == 'equal_class':
                    market_budget = kwargs.get('market_budget')
                    assert market_budget is not None
                    weights = equal_class_weights(market_budget)
                    train_port_perf = portfolio_return(cv_results[cv]['train_returns'], weights=weights)

                else:
                    train_port_perf = portfolio_return(cv_results[cv]['train_returns'],
                                                       weights=cv_results[cv]['port'][p])
                    if cv == 0:
                        cost = fee * np.sum(np.abs(np.ones_like(cv_results[cv]['port'][p]) - cv_results[cv]['port'][p]))
                    else:
                        cost = fee * np.sum(np.abs(cv_results[cv - 1]['port'][p] - cv_results[cv]['port'][p]))

                # Check Jaeger et al 2021
                base_vol = np.max((np.std(train_port_perf[-20:]), np.std(train_port_perf[-60:]))) * np.sqrt(252)
                leverage = 0.05 / base_vol
                cost = cost * leverage
                port_perf[p][cv] = leverage * port_perf[p][cv]
                port_perf[p][cv].iloc[0] = port_perf[p][cv].iloc[0] - cost

            port_perf[p]['total'] = pd.concat([port_perf[p]['total'], port_perf[p][cv]])

    return port_perf


def portfolio_return(returns, weights: Optional[Union[float, np.ndarray]] = None):
    if weights is None:
        port_perf = returns.mean(1) * np.nan
    else:
        port_perf = (returns * weights).sum(1)
    return port_perf


def one_cv(data, assets, base_dir, cv, test_set, portfolios, market_budget=None, compute_weights=True,
           window: Optional[int] = None,
           dataset='global', **kwargs):
    ae_config = kwargs.get('ae_config')
    res = {}

    scaler, dates, test_data, test_features, pred, embedding = load_result(test_set, data, assets, base_dir, cv,
                                                                           ae_config)

    std = np.sqrt(scaler['attributes']['var_'])
    data = data.pct_change(1).dropna()
    data = data[assets]
    assert np.sum(data.isna().sum()) == 0
    train_returns = data.loc[dates['train']]
    returns = data.loc[dates[test_set]]

    if window is not None:
        assert isinstance(window, int)
        train_returns = train_returns.iloc[-window:]

    residuals = returns - pred
    scaled_residuals = residuals * std
    scaled_embedding = np.dot(np.diag(std, k=0), embedding)

    res['embedding'] = embedding
    res['scaler'] = scaler
    res['scaled_embedding'] = scaled_embedding
    # res['train_features'] = train_features
    res['test_features'] = test_features
    res['test_pred'] = pred
    # res['Sf'] = train_features.cov()
    res['Su'] = scaled_residuals.cov()
    # res['H'] = pd.DataFrame(np.dot(scaled_embedding, np.dot(res['Sf'], scaled_embedding.T)) + res['Su'],
    #                         index=embedding.index, columns=embedding.index)
    res['w'] = embedding
    res['train_returns'] = train_returns
    res['returns'] = returns
    if compute_weights:
        assert market_budget is not None
        res['port'] = portfolio_weights(train_returns,
                                        # shrink_cov=res['H'],
                                        budget=market_budget.loc[assets],
                                        embedding=embedding,
                                        portfolio=portfolios,
                                        **kwargs
                                        )
    else:
        res['port'] = None
    res['mean_mse'] = np.mean((residuals ** 2).mean(1))
    res['mse'] = np.sum((residuals ** 2).mean(1))

    return cv, res


def get_cv_results(base_dir, test_set, n_folds, portfolios=None, market_budget=None, compute_weights=True,
                   window: Optional[int] = None, n_jobs: int = None, dataset='global', **kwargs):
    assert test_set in ['val', 'test']

    ae_config = kwargs.get('ae_config')
    data, assets = load_data(dataset=dataset, assets=ae_config.assets, freq=ae_config.freq, crix=ae_config.crix,
                             crypto_assets=ae_config.crypto_assets)
    if n_jobs:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            cv_results = _parallel_pool(
                delayed(one_cv)(data, assets, base_dir, cv, test_set, portfolios, market_budget=market_budget,
                                compute_weights=compute_weights,
                                window=window, dataset=dataset, **kwargs)
                for cv in range(n_folds)
            )
        # Build dictionary
        cv_results = {cv_results[i][0]: cv_results[i][1] for i in range(len(cv_results))}
        # Reorder dictionary
        cv_results = {cv: cv_results[cv] for cv in range(n_folds)}
    else:
        cv_results = {}
        for cv in range(n_folds):
            _, cv_results[cv] = one_cv(data, assets, base_dir, cv, test_set, portfolios, market_budget=market_budget,
                                       compute_weights=compute_weights,
                                       window=window, dataset=dataset, **kwargs)

    return cv_results


def portfolio_weights(returns, shrink_cov=None, budget=None, embedding=None,
                      portfolio=['markowitz', 'shrink_markowitz', 'ivp', 'ae_ivp', 'hrp', 'rp', 'ae_rp', 'herc'],
                      **kwargs):
    assert all([p in PORTFOLIOS for p in portfolio]), [p for p in portfolio if p not in PORTFOLIOS]
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
        port_w['ae_rp'] = ae_riskparity_weights(returns, embedding, budget, risk_parity='budget')

    if 'ae_rp_c' in portfolio:
        LOGGER.info('Computing AE Riskparity Cluster weights...')
        assert budget is not None
        assert embedding is not None
        port_w['ae_rp_c'] = ae_riskparity_weights(returns, embedding, budget, risk_parity='cluster')

    if 'aeaa' in portfolio:
        LOGGER.info('Computing AE Asset Allocation weights...')
        port_w['aeaa'] = aeaa_weights(returns, embedding)

    return port_w


def markowitz_weights(mu: Union[pd.Series, np.ndarray], S: pd.DataFrame, fix_cov: bool = False,
                      risk_free_rate: float = 0.) -> pd.Series:
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


def ae_riskparity_weights(returns, embedding, market_budget, risk_parity='budget'):
    """

    :param returns:
    :param embedding:
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
                                                          embedding,
                                                          clusters,
                                                          market_budget=market_budget)
    elif risk_parity == 'cluster':
        inner_cluster_weights = get_inner_cluster_weights(returns.cov(),
                                                          embedding,
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


def get_inner_cluster_weights(cov, embedding, clusters, market_budget=None):
    weights = {}
    n_clusters = len(clusters)
    for c in clusters:
        cluster_items = clusters[c]

        if cluster_items:
            if market_budget is not None:
                budget = market_budget.loc[cluster_items, 'rc']
            else:
                budget = embedding.loc[cluster_items, c] ** 2 / np.sum(embedding.loc[cluster_items, c] ** 2)
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

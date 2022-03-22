import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy import stats as scipy_stats
from typing import Union, Dict, Optional, List
from joblib import Parallel, delayed

from dl_portfolio.logger import LOGGER
from dl_portfolio.hedge import hedged_portfolio_weights
from dl_portfolio.data import load_data
from dl_portfolio.utils import load_result
from dl_portfolio.constant import DATA_SPECS_BOND, DATA_SPECS_MULTIASSET_TRADITIONAL
from dl_portfolio.probabilistic_sr import probabilistic_sharpe_ratio, min_track_record_length
from dl_portfolio.weights import portfolio_weights, equal_class_weights
from dl_portfolio.constant import PORTFOLIOS


def backtest_stats(perf: pd.DataFrame, weights: Dict, period: int = 250, format: bool = True, sspw_tto=True,
                   **kwargs):
    """

    :param perf:
    :param weights:
    :param period:
    :param format:
    :return:
    """
    bench_names = ['SP500', 'Russel2000', 'EuroStoxx50']
    benchmark, _ = load_data(dataset='raffinot_bloomberg_comb_update_2021', assets=None)
    benchmark = benchmark.pct_change().dropna()
    benchmark = benchmark.loc[perf.index, bench_names]
    benchmark = benchmark * 0.05 / (benchmark.std() * np.sqrt(252))

    perf = pd.concat([perf, benchmark], 1)

    strats = list(perf.keys())
    if sspw_tto:
        cols = ['Return', 'Volatility', 'Skewness', 'Excess kurtosis', 'VaR-5%', 'ES-5%', 'SR', 'PSR', 'minTRL', 'MDD',
                'CR', 'CEQ', 'SSPW', 'TTO']
    else:
        cols = ['Return', 'Volatility', 'Skewness', 'Excess kurtosis', 'VaR-5%', 'ES-5%', 'SR', 'PSR', 'minTRL', 'MDD',
                'CR', 'CEQ']

    stats = pd.DataFrame(index=strats,
                         columns=cols,
                         dtype=np.float32)
    ports = list(weights.keys())
    assets = weights[ports[0]].columns
    n_assets = weights[ports[0]].shape[-1]
    for strat in strats:
        if strat == 'equal':
            weights['equal'] = pd.DataFrame([1 / n_assets] * n_assets).T
            weights['equal'].columns = assets

        elif strat == 'equal_class':
            market_budget = kwargs.get('market_budget')
            market_budget = market_budget.loc[assets, :]
            assert market_budget is not None
            weights['equal_class'] = pd.DataFrame(equal_class_weights(market_budget)).T

        if sspw_tto:
            stats.loc[strat] = [perf[strat].mean() * period,
                                annualized_volatility(perf[strat], period=period),
                                scipy_stats.skew(perf[strat], axis=0),
                                scipy_stats.kurtosis(perf[strat], axis=0) - 3,
                                hist_VaR(perf[strat], level=0.05),
                                hist_ES(perf[strat], level=0.05),
                                sharpe_ratio(perf[strat], period=period),
                                probabilistic_sharpe_ratio(perf[strat], sr_benchmark=0),
                                min_track_record_length(perf[strat], sr_benchmark=0),
                                get_mdd(np.cumprod(perf[strat] + 1)),
                                calmar_ratio(np.cumprod(perf[strat] + 1)),
                                ceq(perf[strat], period=period),
                                sspw(weights[strat]) if strat not in bench_names else np.nan,
                                total_average_turnover(weights[strat]) if strat not in bench_names + ['equal'] else 0.
                                ]
        else:
            stats.loc[strat] = [perf[strat].mean() * period,
                                annualized_volatility(perf[strat], period=period),
                                scipy_stats.skew(perf[strat], axis=0),
                                scipy_stats.kurtosis(perf[strat], axis=0) - 3,
                                hist_VaR(perf[strat], level=0.05),
                                hist_ES(perf[strat], level=0.05),
                                sharpe_ratio(perf[strat], period=period),
                                probabilistic_sharpe_ratio(perf[strat], sr_benchmark=0),
                                min_track_record_length(perf[strat], sr_benchmark=0),
                                get_mdd(np.cumprod(perf[strat] + 1)),
                                calmar_ratio(np.cumprod(perf[strat] + 1)),
                                ceq(perf[strat], period=period)
                                ]

    if format:
        print("Formatting table")
        stats['Return'] = stats['Return'] * 100
        stats['VaR-5%'] = stats['VaR-5%'] * 100
        stats['ES-5%'] = stats['ES-5%'] * 100
        stats['MDD'] = stats['MDD'] * 100
        stats['CEQ'] = stats['CEQ'] * 100
        if sspw_tto:
            stats['TTO'] = stats['TTO'] * 100
    if kwargs.get("round", False):
        stats = np.round(stats, 2)

    return stats


def get_target_vol_other_weights(portfolio: str, window_size=250):
    # Load pyrobustm results
    if portfolio == 'GMV_robust_bond':
        dataset = 'bond'
        crypto_assets = ['BTC', 'DASH', 'ETH', 'LTC', 'XRP']
        weights = pd.read_csv('./run_7_global_bond_dl_portfolio_20220122_151211/weights_GMV_robust.csv', index_col=0)
        data_specs = DATA_SPECS_BOND
    elif portfolio == "MeanVar_bond":
        dataset = 'bond'
        crypto_assets = ['BTC', 'DASH', 'ETH', 'LTC', 'XRP']
        weights = pd.read_csv('./run_7_global_bond_dl_portfolio_20220122_151211/weights_MeanVar_long.csv', index_col=0)
        data_specs = DATA_SPECS_BOND
    elif portfolio == 'GMV_robust_raffinot':
        dataset = 'raffinot_bloomberg_comb_update_2021'
        crypto_assets = None
        data_specs = DATA_SPECS_MULTIASSET_TRADITIONAL
        weights = pd.read_csv('./run_6_multiasset_traditional_dl_portfolio_20211206_173539/weights_GMV_robust.csv',
                              index_col=0)
    elif portfolio == 'MeanVar_raffinot':
        dataset = 'raffinot_bloomberg_comb_update_2021'
        crypto_assets = None
        data_specs = DATA_SPECS_MULTIASSET_TRADITIONAL
        weights = pd.read_csv('./run_6_multiasset_traditional_dl_portfolio_20211206_173539/weights_MeanVar_long.csv',
                              index_col=0)
    else:
        raise NotImplementedError()

    weights.index = pd.to_datetime(weights.index)
    # Load data
    data, assets = load_data(dataset=dataset,
                             assets=None,
                             freq='1D',
                             crix=False,
                             crypto_assets=crypto_assets)

    returns = data.pct_change(1).dropna()

    port_perf = pd.DataFrame()

    leverage = []

    for cv in data_specs:
        test_start = data_specs[cv]['test_start']
        test_end = data_specs[cv]['end']
        w = {'other': weights.loc[test_start:test_end]}
        if cv == 0:
            prev_w = {"other": pd.DataFrame(np.ones_like(w["other"]), columns=assets)}
        else:
            prev_w = {"other": weights.loc[data_specs[cv - 1]['test_start']:data_specs[cv - 1]['end']]}

        train_returns = returns.loc[:test_start].iloc[-window_size - 1:-1]
        test_returns = returns.loc[test_start:test_end]
        one_cv_perf, l = get_portfolio_perf_wrapper(train_returns, test_returns, w, portfolios=['other'],
                                                    prev_weights=prev_w,
                                                    train_weights={"other": w["other"].iloc[0, :]})
        leverage.append(l['other'])
        prev_w = w.copy()
        port_perf = pd.concat([port_perf, one_cv_perf['other']])
    leverage = pd.DataFrame(leverage, columns=['other'])
    return port_perf, leverage


def get_ts_weights_from_cv_results(cv_results, port) -> pd.DataFrame:
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


def get_ts_weights(port_weights: Dict) -> Dict:
    portfolios = port_weights[0].keys()
    all_weights = {}
    for p in portfolios:
        weights = pd.DataFrame()
        for cv in port_weights.keys():
            weights = pd.concat([weights, port_weights[cv][p]])
        weights.index = pd.to_datetime(weights.index)
        all_weights[p] = weights
    return all_weights


def get_dl_average_weights(cv_results):
    port_weights = {cv: {} for cv in cv_results[0]}
    portfolios = cv_results[0][0]['port'].keys()
    for cv in cv_results[0]:
        date = cv_results[0][cv]['returns'].index[0]
        for port in portfolios:
            if 'ae' in port:
                weights = pd.DataFrame()
                for i in cv_results:
                    w = pd.DataFrame(cv_results[i][cv]['port'][port]).T
                    weights = pd.concat([weights, w])
                weights = weights.mean()
                weights = pd.DataFrame(weights).T
                weights.index = [date]
                port_weights[cv][port] = weights

    return port_weights


def get_average_perf(port_perf: Dict, port: str) -> pd.DataFrame:
    perf = pd.DataFrame()
    for i in port_perf:
        perf = pd.concat([perf, port_perf[i][port]['total']], 1)
    perf = perf.mean(1)
    return perf


def plot_perf(perf, strategies=['aerp'], save_path=None, show=False, legend=True, figsize=(20, 10)):
    plt.figure(figsize=figsize)
    for s in strategies:
        plt.plot(np.cumprod(perf[s] + 1) - 1, label=s)
    plt.plot(perf[s] - perf[s], "--", c="lightgrey")
    if legend:
        plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
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

    ax.tick_params(axis='x', labelrotation=45)
    if len(weights) > 45:
        _ = ax.set_xticks(list(range(0, len(weights), 6)))

    if legend:
        plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    if show:
        plt.show()


def annualized_volatility(perf, period: int = 1):
    return perf.std() * np.sqrt(period)


def hist_VaR(perf, level: float = 0.05):
    return - np.quantile(perf, level)


def hist_ES(perf, level: float = 0.05):
    VaR = np.quantile(perf, level)
    return - np.mean(perf[perf <= VaR])


def ceq(returns, gamma: float = 1., risk_free=0., period: int = 1):
    """
    See Raffinot paper or DeMiguel et al. [2009]
    :param perf: returns
    :param gamma: risk aversion
    :param risk_free: risk free rate
    :return:
    """

    return (returns.mean() * period - risk_free) - gamma / 2 * returns.var() * period


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
    skew = scipy_stats.skew(perf, axis=0)
    kurtosis = scipy_stats.kurtosis(perf, axis=0)

    return sr * (1 + (skew / 6) * sr - ((kurtosis - 3) / 24) * (sr ** 2))  # * np.sqrt(period)


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


def compute_balance(price, weights, prev_weights, prev_K, fee=2e-4, leverage=1):
    returns = price.pct_change(1).dropna()
    port_return = leverage * (returns * weights).sum(1)

    # Compute trading cost
    cost = leverage * fee * np.sum(np.abs(prev_weights - weights))
    # Calculate capital with trading cost at the begining of period
    K0 = prev_K * (1 - cost)
    # Compute balance on the period
    K = K0 * (port_return + 1).cumprod()
    # Calculate number of shares at beginning of period
    N = leverage * K0 * weights / price.loc[price.index[0]]

    return K, N, cost



def get_portfolio_perf_wrapper(train_returns: pd.DataFrame, returns: pd.DataFrame, weights: Dict, portfolios: List,
                               train_weights: Optional[Dict] = None, prev_weights: Optional[Dict] = None,
                               fee: float = 2e-4, volatility_target: Optional[float] = 0.05, **kwargs):
    """

    Logic:
    - Train weights are the portfolio weights on train set.
    If train weights is not given (it corresponds to the usual case where weights are constant on test set),
    then use same weights as on test set
    - prev weights is previous cv weights or vector of 1s for the first cv
    - weights is current weights for the test period

    :param portfolio: one of  ['equal', 'markowitz', 'shrink_markowitz', 'ivp', 'aerp', 'hrp', 'rp', 'aeerc']
    :param train_returns:
    :param returns:
    :param weights: Dict with portfolio keys and corresponding weight
    :param prev_weights: Dict with portfolio keys and corresponding weight for the previous period (to compute fees)
    :return:
    """
    N = returns.shape[-1]
    port_perfs = {}
    leverages = {}
    for portfolio in portfolios:
        if portfolio == 'equal':
            port_perf = portfolio_return(returns, weights=1 / N)
        elif portfolio == 'equal_class':
            market_budget = kwargs.get('market_budget')
            assert market_budget is not None
            market_budget = market_budget.loc[returns.columns, :]
            w = equal_class_weights(market_budget)
            port_perf = portfolio_return(returns, weights=w)
        else:
            if weights[portfolio] is not None:
                port_perf = portfolio_return(returns, weights=weights[portfolio])
            else:
                LOGGER.info(f'Warning: No weight for {portfolio} portfolio... Setting to NaN')
                port_perf = returns * np.nan
        # Volatility target weights
        if volatility_target:
            if portfolio == 'equal':
                train_port_perf = portfolio_return(train_returns, weights=1 / N)
                cost = 0
            elif portfolio == 'equal_class':
                market_budget = kwargs.get('market_budget')
                assert market_budget is not None
                market_budget = market_budget.loc[returns.columns, :]
                w = equal_class_weights(market_budget)
                train_port_perf = portfolio_return(train_returns, weights=w)
                cost = 0
            else:
                assert train_weights is not None
                train_port_perf = portfolio_return(train_returns, weights=train_weights[portfolio])

                if weights[portfolio].shape[0] > 1:
                    assert isinstance(weights[portfolio], pd.DataFrame)
                    assert isinstance(prev_weights[portfolio], pd.DataFrame)
                    cost = pd.concat([prev_weights[portfolio].iloc[-1:, :], weights[portfolio]])
                    cost = fee * np.abs(cost.diff().dropna()).sum(1)
                    cost = pd.Series(cost, index=returns.index)
                    assert cost.isna().sum() == 0
                else:
                    mu = np.abs(prev_weights[portfolio] - weights[portfolio])
                    cost = fee * np.sum(mu)
                    assert not np.isnan(cost)

            # Check Jaeger et al 2021
            base_vol = np.max((np.std(train_port_perf[-20:]), np.std(train_port_perf[-60:]))) * np.sqrt(252)
            assert not np.isinf(base_vol)
            assert not np.isnan(base_vol)

            leverage = 0.05 / base_vol
            cost = cost * leverage
            port_perf = leverage * port_perf
            if portfolio not in ["equal", "equal_class"]:
                if weights[portfolio].shape[0] > 1:
                    assert isinstance(cost, pd.Series)
                    port_perf -= cost
                else:
                    assert isinstance(cost, float)
                    port_perf.iloc[0] = port_perf.iloc[0] - cost

        port_perfs[portfolio] = port_perf
        leverages[portfolio] = leverage

    return port_perfs, leverages


def cv_portfolio_perf(cv_results: Dict,
                      portfolios: List = ['equal', 'markowitz', 'shrink_markowitz', 'ivp', 'aerp', 'hrp', 'rp',
                                          'aeerc'],
                      **kwargs) -> Union[Dict, pd.DataFrame]:
    """

    :param cv_results: Dictionary with keys:
     - first key is cv fold
     - for each cv:
        - "port" for portfolio weights with each strategy as key: [cv]["port"]["hrp"], [cv]["port"]["rp"], etc.
        - "train_returns"
        - "returns"
    :param portfolios:
    :return:
    """
    assert all([p in PORTFOLIOS for p in portfolios])
    port_perf = {}
    leverage = {}
    for p in portfolios:
        port_perf[p] = {}
        leverage[p] = []
        port_perf[p]['total'] = pd.DataFrame()

    for cv in cv_results:
        weights = cv_results[cv]['port'].copy()
        if cv == 0:
            prev_weights = {p: np.ones_like(cv_results[cv]['port'][p]) for p in portfolios if
                            p not in ['equal', 'equal_class']}
        else:
            prev_weights = cv_results[cv - 1]['port'].copy()

        one_cv_perf, one_cv_leverage = get_portfolio_perf_wrapper(cv_results[cv]['train_returns'],
                                                                  cv_results[cv]['returns'],
                                                                  weights,
                                                                  portfolios,
                                                                  prev_weights=prev_weights,
                                                                  **kwargs)
        for p in portfolios:
            port_perf[p]['total'] = pd.concat([port_perf[p]['total'], one_cv_perf[p]])
            leverage[p].append(one_cv_leverage[p])
    leverage = pd.DataFrame(leverage)

    return port_perf, leverage


def cv_portfolio_perf_df(cv_portfolio: Dict, portfolios: List[str] = ['ae_rp_c', 'aeaa', 'aeerc'],
                         train_weights: Optional[Dict] = None, **kwargs):
    """
    Logic:
    - Train weights are the portfolio weights on train set.
    If train weights is not given (it corresponds to the usual case where weights are constant on test set),
    then use same weights as on test set
    - prev weights is previous cv weights or vector of 1s for the first cv

    :param cv_portfolio: Dictionary with keys:
     - first key is cv fold
     - for each cv:
        - "port" for portfolio weights with each strategy as key: [cv]["port"]["hrp"], [cv]["port"]["rp"], etc.
        - "train_returns"
        - "returns"
    :poram portfolios: List of portfolio on which to compute weights
    :param train_weights: Dictionary with keys (cv: 0, 1, 2, ...), each containing a pd.DataFrame of weights
    :return:
    """
    assert all([p in PORTFOLIOS for p in portfolios])
    port_perf = {}
    leverage = {}
    for p in portfolios:
        port_perf[p] = {}
        leverage[p] = []
        port_perf[p]['total'] = pd.DataFrame()

    assets = cv_portfolio[0]['train_returns'].columns
    for cv in cv_portfolio:
        weights = cv_portfolio[cv]['port'].copy()
        if cv == 0:
            prev_weights = {p: pd.DataFrame(np.ones_like(cv_portfolio[cv]['port']["ae_rp_c"]), columns=assets) for p in
                            portfolios if p not in ['equal', 'equal_class']}
        else:
            prev_weights = cv_portfolio[cv - 1]['port']

        if train_weights is not None:
            train_w = train_weights[cv]
        else:
            train_w = {p: weights[p].iloc[0, :] for p in weights}

        one_cv_perf, one_cv_leverage = get_portfolio_perf_wrapper(cv_portfolio[cv]['train_returns'],
                                                                  cv_portfolio[cv]['returns'],
                                                                  weights,
                                                                  portfolios,
                                                                  train_weights=train_w,
                                                                  prev_weights=prev_weights,
                                                                  **kwargs)

        for p in portfolios:
            port_perf[p]['total'] = pd.concat([port_perf[p]['total'], one_cv_perf[p]])
            leverage[p].append(one_cv_leverage[p])
    leverage = pd.DataFrame(leverage)

    return port_perf, leverage


def portfolio_return(returns, weights: Optional[Union[float, np.ndarray]] = None):
    if weights is None:
        port_perf = returns.mean(1) * np.nan
    else:
        port_perf = (returns * weights).sum(1)
    return port_perf


def one_cv(data, assets, base_dir, cv, test_set, portfolios, market_budget=None, compute_weights=True,
           window: Optional[int] = 250, **kwargs):
    ae_config = kwargs.get('ae_config')
    res = {}

    model, scaler, dates, test_data, test_features, pred, embedding, decoding, _ = load_result(ae_config, test_set,
                                                                                               data,
                                                                                               assets,
                                                                                               base_dir, cv)

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

    res['embedding'] = embedding
    res['loading'] = decoding
    res['scaler'] = scaler
    res['test_features'] = test_features
    res['test_pred'] = pred
    res['Su'] = scaled_residuals.cov()
    res['w'] = decoding
    res['train_returns'] = train_returns
    res['returns'] = returns
    if compute_weights:
        assert market_budget is not None
        res['port'] = portfolio_weights(train_returns,
                                        # shrink_cov=res['H'],
                                        budget=market_budget.loc[assets],
                                        embedding=embedding,
                                        loading=decoding,
                                        portfolio=portfolios,
                                        **kwargs
                                        )
    else:
        res['port'] = None

    # clusters, _ = get_cluster_labels(embedding)
    res['mean_mse'] = np.mean((residuals ** 2).mean(1))
    res['mse'] = np.sum((residuals ** 2).mean(1))

    return cv, res


def get_cv_results(base_dir, test_set, n_folds, portfolios=None, market_budget=None, compute_weights=True,
                   window: Optional[int] = None, n_jobs: int = None, dataset='global', **kwargs):
    assert test_set in ['val', 'test']

    ae_config = kwargs.get('ae_config')

    if dataset == 'bond':
        data, assets = load_data(dataset=dataset, assets=ae_config.assets, dropnan=ae_config.dropnan,
                                 freq=ae_config.freq, crix=ae_config.crix, crypto_assets=ae_config.crypto_assets)
    else:
        data, assets = load_data(dataset=dataset, assets=ae_config.assets, dropnan=ae_config.dropnan,
                                 freq=ae_config.freq)

    if n_jobs:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            cv_results = _parallel_pool(
                delayed(one_cv)(data, assets, base_dir, cv, test_set, portfolios, market_budget=market_budget,
                                compute_weights=compute_weights, window=window, **kwargs)
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
                                       compute_weights=compute_weights, window=window, **kwargs)

    return cv_results


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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy import stats as scipy_stats
from typing import Union, Dict, Optional, List
from joblib import Parallel, delayed
import scipy

from dl_portfolio.logger import LOGGER
from dl_portfolio.data import (
    load_data,
    load_risk_free,
    impute_missing_risk_free,
)
from dl_portfolio.utils import load_result
from dl_portfolio.constant import (
    DATA_SPECS_AE_DATASET1,
    DATA_SPECS_AE_DATASET2,
)
from dl_portfolio.probabilistic_sr import (
    probabilistic_sharpe_ratio,
    min_track_record_length,
)
from dl_portfolio.weights import portfolio_weights, equal_class_weights, \
    compute_factor_weight, compute_factor_risk_contribution, \
    get_neg_entropy_from_weights_principal
from dl_portfolio.constant import PORTFOLIOS


def backtest_stats(
    perf: pd.DataFrame,
    weights: Dict,
    period: int = 250,
    format: bool = True,
    sspw_tto=True,
    volatility_target=0.05,
    **kwargs,
):
    """

    :param perf:
    :param weights:
    :param period:
    :param format:
    :return:
    """
    benchmark = pd.read_csv(
        "data/benchmarks.csv", index_col=0, parse_dates=True
    )
    bench_names = list(benchmark.columns)
    benchmark = benchmark.pct_change().dropna()
    benchmark = benchmark.reindex(perf.index)
    benchmark = benchmark.interpolate(method="polynomial", order=2)
    benchmark = benchmark.astype(np.float32)

    if volatility_target:
        benchmark = (
            benchmark * volatility_target / (benchmark.std() * np.sqrt(252))
        )

    perf = pd.concat([perf, benchmark], 1)

    strats = list(perf.keys())
    if sspw_tto:
        cols = [
            "Return",
            "Volatility",
            "Skewness",
            "Excess kurtosis",
            "VaR-5%",
            "ES-5%",
            "SR",
            "PSR",
            "minTRL",
            "MDD",
            "CR",
            "CEQ",
            "SSPW",
            "TTO",
        ]
    else:
        cols = [
            "Return",
            "Volatility",
            "Skewness",
            "Excess kurtosis",
            "VaR-5%",
            "ES-5%",
            "SR",
            "PSR",
            "minTRL",
            "MDD",
            "CR",
            "CEQ",
        ]

    stats = pd.DataFrame(index=strats, columns=cols, dtype=np.float32)
    ports = list(weights.keys())
    assets = weights[ports[0]].columns
    dates = weights[ports[0]].index
    n_assets = weights[ports[0]].shape[-1]
    for strat in strats:
        if strat == "equal":
            weights["equal"] = np.repeat(np.array([[1 / n_assets] * n_assets]),
                                         len(dates), axis=0)
            weights["equal"] = pd.DataFrame(weights["equal"], columns=assets,
                                            index=dates)
        elif strat == "equal_class":
            market_budget = kwargs.get("market_budget")
            market_budget = market_budget.loc[assets, :]
            assert market_budget is not None
            weights["equal_class"] = np.repeat(equal_class_weights(
                market_budget).values.reshape(1, -1), len(dates), axis=0)
            weights["equal_class"] = pd.DataFrame(weights["equal_class"],
                                                  columns=assets, index=dates)

        if sspw_tto:
            if strat not in bench_names:
                if "hedge" in strat:
                    sspw_value = np.mean(
                        [np.mean(
                            weights[strat][k].apply(herfindahl_index, axis=1)
                        )
                    for k in weights[strat]]
                    )
                    raise NotImplementedError("You must verify logic for tto!")
                    tto_value = np.mean(
                        [
                            total_average_turnover(weights[strat],
                                                   prices=kwargs.get("prices"))
                            for k in weights[strat]
                        ]
                    )
                else:
                    sspw_value = np.mean(
                        weights[strat].apply(herfindahl_index, axis=1)
                    )
                    tto_value = total_average_turnover(
                        weights[strat], prices=kwargs.get("prices"))
            else:
                sspw_value = np.nan
                tto_value = 0.0

            stats.loc[strat] = [
                perf[strat].mean() * period,
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
                sspw_value,
                tto_value,
            ]
        else:
            stats.loc[strat] = [
                perf[strat].mean() * period,
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
            ]

    if format:
        LOGGER.debug("Formatting table")
        stats["Return"] = stats["Return"] * 100
        stats["VaR-5%"] = stats["VaR-5%"] * 100
        stats["ES-5%"] = stats["ES-5%"] * 100
        stats["MDD"] = stats["MDD"] * 100
        stats["CEQ"] = stats["CEQ"] * 100
        if sspw_tto:
            stats["TTO"] = stats["TTO"] * 100
    if kwargs.get("round", False):
        stats = np.round(stats, 2)

    return stats


def get_target_vol_other_weights(portfolio: str, window_size=250):
    # Load pyrobustm results
    if portfolio == "GMV_robust_dataset1":
        dataset = "dataset1"
        weights = pd.read_csv(
            "./final_models/run_11_dataset1_20230408_145352/weights_GMV_robust.csv",
            index_col=0,
        )
        data_specs = DATA_SPECS_AE_DATASET1
    elif portfolio == "MeanVar_dataset1":
        dataset = "dataset1"
        weights = pd.read_csv(
            "./final_models/run_11_dataset1_20230408_145352/weights_MeanVar_long.csv",
            index_col=0,
        )
        data_specs = DATA_SPECS_AE_DATASET1
    elif portfolio == "GMV_robust_dataset2":
        dataset = "dataset2"
        data_specs = DATA_SPECS_AE_DATASET2
        weights = pd.read_csv(
            "./final_models/run_12_dataset2_20230408_145946/weights_GMV_robust.csv",
            index_col=0,
        )
    elif portfolio == "MeanVar_dataset2":
        dataset = "dataset2"
        data_specs = DATA_SPECS_AE_DATASET2
        weights = pd.read_csv(
            "./final_models/run_12_dataset2_20230408_145946/weights_MeanVar_long.csv",
            index_col=0,
        )
    else:
        raise NotImplementedError()

    weights.index = pd.to_datetime(weights.index)
    # Load data
    data, assets = load_data(dataset=dataset)

    returns = data.pct_change(1).dropna()

    port_perf = pd.DataFrame()

    leverage = []
    for cv in data_specs:
        test_start = data_specs[cv]["test_start"]
        test_end = data_specs[cv]["end"]
        w = {"other": weights.loc[test_start:test_end]}
        if cv == 0:
            prev_w = {
                "other": pd.DataFrame(np.ones_like(w["other"]), columns=assets)
            }
        else:
            prev_w = {
                "other": weights.loc[
                    data_specs[cv - 1]["test_start"] : data_specs[cv - 1][
                        "end"
                    ]
                ]
            }

        train_returns = returns.loc[:test_start].iloc[-window_size - 1 : -1]
        test_returns = returns.loc[test_start:test_end]

        risk_free = load_risk_free()
        risk_free = risk_free.reindex(test_returns.index)
        risk_free = impute_missing_risk_free(risk_free)
        if risk_free.isna().any().any():
            assert not risk_free.isna().any()

        one_cv_perf, l = get_portfolio_perf_wrapper(
            train_returns,
            test_returns,
            w,
            portfolios=["other"],
            prev_weights=prev_w,
            train_weights={"other": w["other"].iloc[0, :]},
            risk_free=risk_free["risk_free"],
        )
        leverage.append(l["other"])
        prev_w = w.copy()
        port_perf = pd.concat([port_perf, one_cv_perf["other"]])
    leverage = pd.DataFrame(leverage, columns=["other"])
    return port_perf, leverage


def get_ts_weights_from_cv_results(cv_results, port) -> pd.DataFrame:
    dates = [cv_results[0][cv]["returns"].index[0] for cv in cv_results[0]]
    weights = pd.DataFrame()
    for cv in cv_results[0]:
        if "ae" in port:
            avg_weights_cv = pd.DataFrame()
            for i in cv_results:
                w = pd.DataFrame(cv_results[i][cv]["port"][port]).T
                avg_weights_cv = pd.concat([avg_weights_cv, w])
            avg_weights_cv = avg_weights_cv.mean()
            avg_weights_cv = pd.DataFrame(avg_weights_cv).T
            weights = pd.concat([weights, avg_weights_cv])
        else:
            w = pd.DataFrame(cv_results[0][cv]["port"][port]).T
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
    portfolios = cv_results[0][0]["port"].keys()
    for cv in cv_results[0]:
        date = cv_results[0][cv]["returns"].index[0]
        for port in portfolios:
            if "ae" in port:
                weights = pd.DataFrame()
                for i in cv_results:
                    w = pd.DataFrame(cv_results[i][cv]["port"][port]).T
                    weights = pd.concat([weights, w])
                weights = weights.mean()
                weights = pd.DataFrame(weights).T
                weights.index = [date]
                port_weights[cv][port] = weights

    return port_weights


def plot_perf(
    perf,
    strategies=["aerp"],
    save_path=None,
    show=False,
    legend=True,
    figsize=(20, 10),
):
    plt.figure(figsize=figsize)
    for s in strategies:
        plt.plot(np.cumprod(perf[s] + 1) - 1, label=s)
    plt.plot(perf[s] - perf[s], "--", c="lightgrey")
    if legend:
        plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", transparent=True)
    if show:
        plt.show()


def bar_plot_weights(weights, show=False, legend=False, save_path=None):
    labels = [str(d.date()) for d in weights.index]
    fig, ax = plt.subplots(figsize=(20, 10))
    NUM_COLORS = len(weights.columns)
    cm = plt.get_cmap("gist_rainbow")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    )

    for i, c in enumerate(list(weights.columns)):
        ax.bar(
            labels,
            weights[c],
            label=c,
            width=1,
            align="edge",
            bottom=weights.iloc[:, :i].sum(1),
        )

    ax.tick_params(axis="x", labelrotation=45)
    if len(weights) > 45:
        _ = ax.set_xticks(list(range(0, len(weights), 6)))

    if legend:
        plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    plt.close()


def annualized_volatility(perf, period: int = 1):
    return perf.std() * np.sqrt(period)


def hist_VaR(perf, level: float = 0.05):
    return -np.quantile(perf, level)


def hist_ES(perf, level: float = 0.05):
    VaR = np.quantile(perf, level)
    return -np.mean(perf[perf <= VaR])


def ceq(returns, gamma: float = 1.0, period: int = 1):
    """
    See Raffinot paper or DeMiguel et al. [2009]
    :param perf: returns
    :param gamma: risk aversion
    :param risk_free: risk free rate
    :return:
    """

    risk_free = load_risk_free()
    risk_free = risk_free.reindex(returns.index)
    risk_free = impute_missing_risk_free(risk_free)

    return (
        (returns - risk_free["risk_free"]).mean() * period
    ) - gamma / 2 * returns.var() * period


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
    :param weights: pd.DataFrame of shape (n_obs, n_assets)
    :return:
    """
    return weights.diff().abs().dropna().mean()


def total_average_turnover(weights, prices=None):
    """
    compute total average turnover per rebalancing
    :param weights: pd.DataFrame of shape (n_obs, n_assets)
    :return:
    """
    n_assets = weights.shape[-1]
    if prices is None:
        return np.mean(np.sum(np.abs(np.diff(weights)), axis=1))
    else:
        # prev_weights = pd.DataFrame()
        tto = []
        for i, end_date in enumerate(weights.index):
                if i == 0:
                    pw = np.zeros((n_assets))
                else:
                    prev_weights = weights.iloc[i - 1]
                    start_date = weights.index[i - 1]
                    shares = prev_weights / prices.loc[start_date]
                    pw = shares * prices.loc[end_date] / (
                            shares * prices.loc[end_date]).sum()

                tto.append(np.sum(np.abs(weights.loc[end_date] - pw)))
        return np.mean(tto)


def adjusted_sharpe_ratio(perf, period: int = 1):
    # check (Pezier and White (2008))
    # SR x [1 + (S/6) x SR - ((K-3) / 24) x SR^2]

    # annualized ASR = sqrt(period) * ASR
    # Indeed annualized skew = skew / sqrt(period) and annualized kurtosis = kurtosis / period

    sr = sharpe_ratio(perf)
    skew = scipy_stats.skew(perf, axis=0)
    kurtosis = scipy_stats.kurtosis(perf, axis=0)

    return sr * (
        1 + (skew / 6) * sr - ((kurtosis - 3) / 24) * (sr ** 2)
    )  # * np.sqrt(period)


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
    portfolios = cv_results[0]["port"].keys()
    for p in portfolios:
        pweights = pd.DataFrame()
        dates = []
        for cv in cv_results.keys():
            dates.append(cv_results[cv]["returns"].index[0])
            w = cv_results[cv]["port"][p]
            if w is None:
                assets = cv_results[cv]["returns"].columns
                w = pd.Series([None] * len(assets), index=assets)
            pweights = pd.concat([pweights, w], 1)
        pweights = pweights.T
        pweights.index = dates
        weights[p] = pweights
    return weights


def compute_balance(
    price, weights, prev_weights, prev_K, fee=2e-4, leverage=1
):
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


def get_portfolio_perf_wrapper(
    train_returns: pd.DataFrame,
    returns: pd.DataFrame,
    weights: Dict,
    portfolios: List,
    train_weights: Optional[Dict] = None,
    prev_weights: Optional[Dict] = None,
    fee: float = 2e-4,
    volatility_target: Optional[float] = 0.05,
    risk_free: Optional[pd.Series] = None,
    **kwargs,
):
    """

    Logic:
    - Train weights are the portfolio weights on train set.
    If train weights is not given (it corresponds to the usual case where weights are constant on test set),
    then use same weights as on test set
    - prev weights is previous cv weights or vector of 1s for the first cv
    - weights is current weights for the test period

    :param portfolio: one of  ['equal', 'markowitz', 'shrink_markowitz',
    'ivp', 'aerp', 'erc']
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
        if portfolio == "equal":
            port_perf = portfolio_return(returns, weights=1 / N)
        elif portfolio == "equal_class":
            market_budget = kwargs.get("market_budget")
            assert market_budget is not None
            market_budget = market_budget.loc[returns.columns, :]
            w = equal_class_weights(market_budget)
            port_perf = portfolio_return(returns, weights=w)
        else:
            if weights[portfolio] is not None:
                port_perf = portfolio_return(
                    returns, weights=weights[portfolio]
                )
            else:
                LOGGER.info(
                    f"Warning: No weight for {portfolio} portfolio... Setting to NaN"
                )
                port_perf = returns * np.nan
        # Volatility target weights
        if volatility_target is not None:
            if portfolio == "equal":
                train_port_perf = portfolio_return(
                    train_returns, weights=1 / N
                )
                cost = 0
            elif portfolio == "equal_class":
                market_budget = kwargs.get("market_budget")
                assert market_budget is not None
                market_budget = market_budget.loc[returns.columns, :]
                w = equal_class_weights(market_budget)
                train_port_perf = portfolio_return(train_returns, weights=w)
                cost = 0
            else:
                assert train_weights is not None
                train_port_perf = portfolio_return(
                    train_returns, weights=train_weights[portfolio]
                )

                if weights[portfolio].shape[0] > 1:
                    assert isinstance(weights[portfolio], pd.DataFrame)
                    assert isinstance(prev_weights[portfolio], pd.DataFrame)
                    cost = pd.concat(
                        [
                            prev_weights[portfolio].iloc[-1:, :],
                            weights[portfolio],
                        ], ignore_index=True
                    )
                    cost = fee * np.abs(cost.diff().dropna()).sum(1)
                    cost = pd.Series(cost.values, index=returns.index)
                    assert cost.isna().sum() == 0
                else:
                    mu = np.abs(prev_weights[portfolio] - weights[portfolio])
                    cost = fee * np.sum(mu)
                    assert not np.isnan(cost)

            # Check Jaeger et al 2021
            base_vol = np.max(
                (np.std(train_port_perf[-20:]), np.std(train_port_perf[-60:]))
            ) * np.sqrt(252)
            assert not np.isinf(base_vol)
            assert not np.isnan(base_vol)

            leverage = volatility_target / base_vol
            fee_cost = cost * leverage

            port_perf = leverage * port_perf
            if portfolio not in ["equal", "equal_class"]:
                if weights[portfolio].shape[0] > 1:
                    assert isinstance(fee_cost, pd.Series)
                    date = port_perf.index
                    if risk_free is not None:
                        leverage_cost = risk_free.loc[date] * (leverage - 1)
                        total_cost = fee_cost + leverage_cost
                    else:
                        total_cost = fee_cost
                    assert not total_cost.isna().any()
                    port_perf -= total_cost
                else:
                    assert isinstance(fee_cost, float)
                    date = port_perf.iloc[0].index[0]
                    if risk_free is not None:
                        leverage_cost = risk_free.loc[date] * (leverage - 1)
                        total_cost = fee_cost + leverage_cost
                    else:
                        total_cost = fee_cost
                    assert not np.isnan(total_cost)[0]
                    port_perf.loc[date] = port_perf.loc[date] - total_cost
        else:
            leverage = None

        port_perfs[portfolio] = port_perf
        leverages[portfolio] = leverage

    return port_perfs, leverages


def cv_portfolio_perf_df(
    cv_portfolio: Dict,
    dataset,
    portfolios: List[str] = ["ae_rp_c", "aeaa", "aeerc"],
    train_weights: Optional[Dict] = None,
    **kwargs,
):
    """
    Logic:
    - Train weights are the portfolio weights on train set.
    If train weights is not given (it corresponds to the usual case where weights are constant on test set),
    then use same weights as on test set
    - prev weights is previous cv weights or vector of 1s for the first cv

    :param cv_portfolio: Dictionary with keys:
     - first key is cv fold
     - for each cv:
        - "port" for portfolio weights with each strategy as key: [cv][
        "port"], [cv]["port"]["erc"], etc.
        - "train_returns"
        - "returns"
    :poram portfolios: List of portfolio on which to compute weights
    :param train_weights: Dictionary with keys (cv: 0, 1, 2, ...), each containing a pd.DataFrame of weights
    :return:
    """
    assert all([p in PORTFOLIOS for p in portfolios])
    prices, assets = load_data(dataset)
    port_perf = {}
    leverage = {}
    for p in portfolios:
        port_perf[p] = {}
        leverage[p] = []
        port_perf[p]["total"] = pd.DataFrame()

    assets = cv_portfolio[0]["train_returns"].columns
    for cv in cv_portfolio:
        weights = cv_portfolio[cv]["port"].copy()
        if cv == 0:
            prev_weights = {
                p: pd.DataFrame(np.zeros((1, len(assets))), columns=assets) for p
                in portfolios
            }
        else:
            prev_weights = cv_portfolio[cv-1]["port"].copy()
            start_date = cv_portfolio[cv-1]["returns"].index[0]
            end_date = cv_portfolio[cv-1]["returns"].index[-1]
            for p in portfolios:
                if p == "equal":
                    pw = np.ones(len(assets)) * 1/len(assets)
                elif p == "equal_class":
                    market_budget = kwargs.get("market_budget")
                    assert market_budget is not None
                    market_budget = market_budget.loc[assets, :]
                    pw = equal_class_weights(market_budget)
                else:
                    pw = prev_weights[p].loc[start_date]

                shares = pw / prices.loc[start_date]
                prev_weights[p] = shares * prices.loc[end_date] / (
                        shares * prices.loc[end_date]).sum()
                prev_weights[p] = pd.DataFrame(prev_weights[p],
                                               columns=[end_date]).T
        if train_weights is not None:
            train_w = train_weights[cv]
        else:
            train_w = {p: weights[p].iloc[0, :] for p in weights}

        risk_free = load_risk_free()
        risk_free = impute_missing_risk_free(risk_free)
        last_value = risk_free.values[-1, 0]

        if risk_free.isna().any().any():
            risk_free.fillna(method="ffill", inplace=True)
            assert not risk_free.isna().any().any()
        risk_free = risk_free.reindex(cv_portfolio[cv]["returns"].index)
        if risk_free.isna().any().any():
            if np.isnan(risk_free.values[0, 0]):
                risk_free.iloc[0, 0] = last_value
            risk_free.fillna(method="ffill", inplace=True)

        assert not risk_free.isna().any().any()
        one_cv_perf, one_cv_leverage = get_portfolio_perf_wrapper(
            cv_portfolio[cv]["train_returns"],
            cv_portfolio[cv]["returns"],
            weights,
            portfolios,
            train_weights=train_w,
            prev_weights=prev_weights,
            risk_free=risk_free["risk_free"],
            **kwargs,
        )

        for p in portfolios:
            port_perf[p]["total"] = pd.concat(
                [port_perf[p]["total"], one_cv_perf[p]]
            )
            leverage[p].append(one_cv_leverage[p])
    leverage = pd.DataFrame(leverage)

    return port_perf, leverage


def portfolio_return(
    returns, weights: Optional[Union[float, np.ndarray]] = None
):
    if weights is None:
        port_perf = returns.mean(1) * np.nan
    else:
        port_perf = (returns * weights).sum(1)
    return port_perf


def herfindahl_index(p, normalize=True):
    if normalize:
        d = len(p)
        return (d * np.sum(p ** 2) - 1) / (d - 1)
    else:
        return np.sum(p ** 2)


def shannon_entropy(p):
    return np.exp(- np.sum(p * np.log(p), axis=0))


def get_number_of_nmf_bets(rc: Dict, metric: str ="shannon_entropy"):
    """

    :param rc: Dictionary with key corresponding to the
    portfolio name and values a list of pd.Series with risk contribution for
    various experiments, rc["rb_factor"][0], rc["rb_factor"][1], etc.
    :param metric:
    :return:
    """
    assert metric in ["shannon_entropy", "herfindahl"]

    n_exp = len(list(rc.values())[0])
    # Get proba measure from risk contribution
    p_rc = {
        p: [(rc[p][i].abs().T / np.sum(
            rc[p][i].abs().T, axis=0)).T for i in range(n_exp)]
        for p in rc
    }

    if metric == "herfindahl":
        def func(x):
            return 1 / herfindahl_index(x, normalize=False)
    elif metric == "shannon_entropy":
        func = shannon_entropy
    else:
        raise NotImplementedError(metric)

    number_bets = {
        p: pd.concat(
            [
                p_rc[p][i].apply(func, axis=1) for i in range(n_exp)
            ],
            axis=1
        ).mean(1)
        for p in p_rc
    }

    return pd.DataFrame(number_bets)


def get_number_of_pc_bets(cv_results, market_budget: pd.DataFrame,
                                 test_set: str = "test"):
    assets = market_budget.index.tolist()
    d = len(assets)
    portfolios = list(cv_results[0][0]["port"].keys()) + ["equal",
                                                          "equal_class"]
    n_bets = {p: [] for p in portfolios}
    for p in portfolios:
        for i in cv_results.keys():
            p_n_bets = []
            for cv in cv_results[i].keys():
                if test_set == "test":
                    Sigma = np.cov(cv_results[i][cv]["returns"].T)
                elif test_set == "train":
                    Sigma = np.cov(cv_results[i][cv]["train_returns"].T)
                else:
                    raise NotImplementedError()

                if p == "equal":
                    a = np.ones(d) / d
                elif p == "equal_class":
                    assert market_budget is not None
                    a = equal_class_weights(market_budget).loc[assets].values
                else:
                    a = cv_results[i][cv]["port"][p].copy().values
                p_n_bets.append(
                    -get_neg_entropy_from_weights_principal(a.reshape(-1,1), Sigma)
                )
            n_bets[p].append(p_n_bets)
        n_bets[p] = np.mean(pd.DataFrame(p_n_bets), axis=1)
    n_bets = pd.DataFrame(n_bets)

    return n_bets


def get_factors_rc_and_weights(
        cv_results: Dict, market_budget: pd.DataFrame, test_set: str = "test"):
    """

    :param cv_results:
    :param market_budget:
    :param test_set: "test" or "train", if "test" use covariance on test_set,
    otherwise on train_set
    :return:
    """
    assets = market_budget.index.tolist()
    d = len(assets)
    portfolios = list(cv_results[0][0]["port"].keys()) + ["equal",
                                                          "equal_class"]
    risk_contribution = {p: [] for p in portfolios}
    factor_weights = {p: [] for p in portfolios}
    for p in portfolios:
        for i in cv_results.keys():
            p_rc = []
            p_fw = []
            for cv in cv_results[i].keys():
                if test_set == "test":
                    Sigma = np.cov(cv_results[i][cv]["returns"].T)
                elif test_set == "train":
                    Sigma = np.cov(cv_results[i][cv]["train_returns"].T)
                else:
                    raise NotImplementedError()
                W_tilde = np.dot(
                    np.diag(
                        cv_results[i][cv]["scaler"]["attributes"]["scale_"]
                    ),
                    cv_results[i][cv]["loading"]
                )
                if p == "equal":
                    a = np.ones(d) / d
                elif p == "equal_class":
                    assert market_budget is not None
                    a = equal_class_weights(market_budget).loc[assets]
                else:
                    a = cv_results[i][cv]["port"][p].copy()
                rc_z, rc_y = compute_factor_risk_contribution(a,
                                                              W_tilde,
                                                              Sigma)
                w_z, w_y = compute_factor_weight(a, W_tilde)
                p_rc.append(np.concatenate([rc_z, rc_y]))
                p_fw.append(np.concatenate([w_z, w_y]))
            risk_contribution[p].append(pd.DataFrame(p_rc))
            factor_weights[p].append(pd.DataFrame(p_fw))
    return risk_contribution, factor_weights


def one_cv(
    data,
    assets,
    base_dir,
    cv,
    test_set,
    portfolios,
    market_budget=None,
    compute_weights=True,
    window: Optional[int] = 250,
    excess_ret=True,
    reorder_features=True,
    **kwargs,
):
    ae_config = kwargs.get("ae_config")
    res = {}

    (
        model,
        scaler,
        dates,
        test_data,
        test_features,
        pred,
        embedding,
        decoding,
        _,
        decoder_bias,
    ) = load_result(
        ae_config,
        test_set,
        data,
        assets,
        base_dir,
        cv,
        reorder_features=reorder_features,
    )

    data = data.pct_change(1).dropna()
    data = data[assets]
    assert np.sum(data.isna().sum()) == 0
    train_returns = data.loc[dates["train"]]
    returns = data.loc[dates[test_set]]

    if excess_ret:
        risk_free_rate = load_risk_free()
        train_rf = risk_free_rate.reindex(train_returns.index)
        train_rf = impute_missing_risk_free(train_rf)
        excess_train_returns = train_returns - train_rf.values

        rf = risk_free_rate.reindex(returns.index)
        rf = impute_missing_risk_free(rf)
        excess_returns = returns - rf.values
        residuals = excess_returns - pred
    else:
        residuals = returns - pred

    if window is not None:
        assert isinstance(window, int)
        train_returns = train_returns.iloc[-window:]

    if scaler:
        std = scaler["attributes"]["scale_"]
        if std is None:
            std = 1.0
        scaled_residuals = residuals * std
    else:
        scaled_residuals = residuals * 1.0

    res["embedding"] = embedding
    res["loading"] = decoding
    res["scaler"] = scaler
    res["test_features"] = test_features
    res["test_pred"] = pred
    res["Su"] = scaled_residuals.cov()
    res["w"] = decoding
    res["train_returns"] = train_returns
    res["returns"] = returns
    res["decoder_bias"] = decoder_bias
    if excess_ret:
        res["excess_train_returns"] = excess_train_returns
        res["excess_returns"] = excess_returns
    if compute_weights:
        assert market_budget is not None
        res["port"] = portfolio_weights(
            train_returns,
            # shrink_cov=res['H'],
            budget=market_budget.loc[assets],
            embedding=embedding,
            loading=decoding,
            portfolio=portfolios,
            **kwargs,
        )
    else:
        res["port"] = None

    # clusters, _ = get_cluster_labels(embedding)
    res["mean_mse"] = np.mean((residuals ** 2).mean(1))
    res["mse"] = np.sum((residuals ** 2).mean(1))

    return cv, res


def get_cv_results(
    base_dir,
    test_set,
    n_folds,
    portfolios=None,
    market_budget=None,
    compute_weights=True,
    window: Optional[int] = None,
    n_jobs: int = 1,
    dataset="global",
    reorder_features=True,
    **kwargs,
):
    assert test_set in ["val", "test"]

    ae_config = kwargs.get("ae_config")

    data, assets = load_data(dataset=dataset)

    if n_jobs > 1:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            cv_results = _parallel_pool(
                delayed(one_cv)(
                    data,
                    assets,
                    base_dir,
                    cv,
                    test_set,
                    portfolios,
                    market_budget=market_budget,
                    compute_weights=compute_weights,
                    window=window,
                    reorder_features=reorder_features,
                    **kwargs,
                )
                for cv in range(n_folds)
            )
        # Build dictionary
        cv_results = {
            cv_results[i][0]: cv_results[i][1] for i in range(len(cv_results))
        }
        # Reorder dictionary
        cv_results = {cv: cv_results[cv] for cv in range(n_folds)}
    else:
        cv_results = {}
        for cv in range(n_folds):
            _, cv_results[cv] = one_cv(
                data,
                assets,
                base_dir,
                cv,
                test_set,
                portfolios,
                market_budget=market_budget,
                compute_weights=compute_weights,
                window=window,
                reorder_features=reorder_features,
                **kwargs,
            )

    return cv_results

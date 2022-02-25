import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional

from dl_portfolio.logger import LOGGER
from dl_portfolio.constant import AVAILABLE_METHODS


def hedged_portfolio_weights_wrapper(cv: int, returns: pd.DataFrame, cluster: pd.Series, cv_garch_dir: str,
                                     or_port_weights: Dict, strats: List[str] = ['ae_rp_c', 'aeaa', 'aerp', 'aeerc'],
                                     window: Optional[int] = None, target: Optional[pd.DataFrame] = None,
                                     method: Optional[str] = "hedged_strat_cum_excess_return_cluster"):
    LOGGER.info(f"CV: {cv}")
    assets = list(returns.columns)

    train_probas = pd.read_csv(f"{cv_garch_dir}/train_activation_probas.csv", index_col=0)
    train_probas.index = pd.to_datetime(train_probas.index)
    probas = pd.read_csv(f"{cv_garch_dir}/activation_probas.csv", index_col=0)
    probas.index = pd.to_datetime(probas.index)
    # Handle stupid renaming of columns from R
    probas = probas[train_probas.columns]  # Just to be sure
    columns = list(train_probas.columns)
    columns = [c.replace(".", "-") for c in columns]
    train_probas.columns = columns
    probas.columns = columns

    train_returns = returns.loc[train_probas.index]
    train_target = target.loc[train_probas.index]
    test_returns = returns.loc[probas.index]

    if window is not None:
        assert isinstance(window, int)
        train_returns = train_returns.iloc[-window:]

    res = {"port": {}, "signal": {}}
    for strat in strats:
        original_weights = or_port_weights[strat].iloc[cv][assets]
        signals, hedged_weights = hedged_portfolio_weights(train_returns, train_probas, probas, cluster, assets,
                                                           original_weights, target=train_target, method=method)
        res["port"][strat] = hedged_weights
        res["signal"][strat] = signals
    res["train_returns"] = train_returns
    res["returns"] = test_returns

    return cv, res


def hedged_portfolio_weights(train_returns, train_probas, probas, cluster, assets, original_weights,
                             target: Optional[pd.DataFrame] = None,
                             method: Optional[str] = "hedged_strat_cum_excess_return_cluster") -> Union[
    pd.DataFrame, pd.DataFrame]:
    """
    Get the best threshold based on method evaluated on train_returns with train_probas. Then apply threshold on probas
    to get the hedged weights from the original weights
    :param target: True target dataframe
    :param train_returns: asset returns on train set
    :param train_probas: factor exceedance probability on train set
    :param probas: factor exceedance probability on test set
    :param cluster: asset cluster assignemnt: projection on factor
    :param assets: assets
    :param original_weights: original weights
    :param method: method for selecting optimal threshold, must be in AVAILABLE_METHODS
    :return:
    """
    assert method in AVAILABLE_METHODS, method

    cluster_names = np.unique(cluster.dropna()).tolist()
    unnassigned = cluster.index[cluster.isna()]
    weights = pd.DataFrame()
    signals = pd.DataFrame()
    for cluster_name in cluster_names:
        train_w = pd.DataFrame(np.repeat(original_weights.loc[assets].values.reshape(-1, 1).T,
                                         len(train_probas.index),
                                         axis=0),
                               columns=assets,
                               index=train_probas.index)
        test_w = pd.DataFrame(np.repeat(original_weights.loc[assets].values.reshape(-1, 1).T,
                                        len(probas.index),
                                        axis=0),
                              columns=assets,
                              index=probas.index)
        optimal_t = get_best_threshold(train_returns, train_w, train_probas, cluster, cluster_name, target=target,
                                       method=method)
        signal_c, temp_w_c = get_hedged_weight_cluster(test_w, probas, cluster, cluster_name, optimal_t)
        weights = pd.concat([weights, temp_w_c], 1)
        signals = pd.concat([signals, signal_c], 1)

    weights[unnassigned] = 0.
    weights = weights[assets]

    signals[unnassigned] = np.nan
    signals = signals[assets]

    return signals, weights


def get_signals(train_returns, train_probas, probas, cluster, assets, original_weights,
                target: Optional[pd.DataFrame] = None,
                method: Optional[str] = "hedged_strat_cum_excess_return_cluster") -> pd.DataFrame:
    cluster_names = np.unique(cluster.dropna()).tolist()
    unnassigned = cluster.index[cluster.isna()]
    signals = pd.DataFrame()
    for cluster_name in cluster_names:
        train_w = pd.DataFrame(np.repeat(original_weights.loc[assets].values.reshape(-1, 1).T,
                                         len(train_probas.index),
                                         axis=0),
                               columns=assets,
                               index=train_probas.index)
        optimal_t = get_best_threshold(train_returns, train_w, train_probas, cluster, cluster_name, target=target,
                                       method=method)
        temp_signal = get_signal_cluster(probas, cluster, cluster_name, optimal_t, method)
        signals = pd.concat([signals, temp_signal], 1)

    signals[unnassigned] = np.nan
    signals = signals[assets]

    return signals


def get_signal_cluster(probas: pd.DataFrame, cluster, cluster_name, threshold):
    cluster_assets = cluster.index[cluster == cluster_name]
    signal = pd.DataFrame(0, index=probas.index, columns=cluster_assets, dtype=int)
    signal[cluster_assets] = np.repeat((probas[[cluster_name]] < threshold).astype(int).values,
                                       np.sum(cluster == cluster_name), axis=1)
    signal.fillna(0, inplace=True)  # Assets which are not assigned to any cluster have NaN
    return signal


def get_hedged_weight_cluster(weights: pd.DataFrame, probas: pd.DataFrame,
                              cluster: pd.Series, cluster_name: List[str], threshold: float) -> Union[
    pd.DataFrame, pd.DataFrame]:
    """

    :param weights:
    :param probas:
    :param cluster:
    :param cluster_name:
    :param threshold:
    :return:
    """
    signal = get_signal_cluster(probas, cluster, cluster_name, threshold)
    cluster_assets = cluster.index[cluster == cluster_name]

    return signal, weights[cluster_assets] * signal


def get_hedged_return_cluster(returns: pd.DataFrame, probas: pd.DataFrame, cluster: pd.Series, cluster_name: List[str],
                              threshold: float, weights: Optional[pd.DataFrame] = None):
    """

    :param returns:
    :param weights:
    :param probas:
    :param cluster:
    :param cluster_name:
    :param threshold:
    :return:
    """
    signal = get_signal_cluster(probas, cluster, cluster_name, threshold)
    cluster_assets = cluster.index[cluster == cluster_name]
    if weights is not None:
        cluster_return = (returns[cluster_assets] * weights[cluster_assets] * signal).sum(1)
    else:
        cluster_return = (returns[cluster_assets] * signal).mean(1)

    return cluster_return


def hedged_strat_cum_excess_return_cluster(returns: pd.DataFrame, weights: pd.DataFrame, probas: pd.DataFrame,
                                           cluster: pd.Series, cluster_name: List[str], threshold: float) -> float:
    """
    Compute cumulative excess return based on weighted portfolio and hedging probability

    :param returns:
    :param weights:
    :param probas:
    :param cluster:
    :param cluster_name:
    :param threshold:
    :return:
    """
    hedged_cluster_return = get_hedged_return_cluster(returns, probas, cluster, cluster_name, threshold,
                                                      weights=weights)
    cluster_assets = cluster.index[cluster == cluster_name]
    cluster_return = (returns[cluster_assets] * weights[cluster_assets]).sum(1)
    cum_excess_return = np.cumsum(hedged_cluster_return - cluster_return)[-1]

    return cum_excess_return


def hedged_equal_cum_excess_return_cluster(returns: pd.DataFrame, probas: pd.DataFrame, cluster: pd.Series,
                                           cluster_name: List[str], threshold: float) -> float:
    """
    Compute cumulative excess return of equally weighted portfolio and hedging probability

    :param returns:
    :param weights:
    :param probas:
    :param cluster:
    :param cluster_name:
    :param threshold:
    :return:
    """
    hedged_cluster_return = get_hedged_return_cluster(returns, probas, cluster, cluster_name, threshold)
    cluster_assets = cluster.index[cluster == cluster_name]
    cluster_return = (returns[cluster_assets]).mean(1)
    cum_excess_return = np.cumsum(hedged_cluster_return - cluster_return)[-1]

    return cum_excess_return


def get_best_threshold(returns: pd.DataFrame, weights: pd.DataFrame, probas: pd.DataFrame, cluster: pd.Series,
                       cluster_name: List[str], target: Optional[pd.DataFrame] = None,
                       method: Optional[str] = "hedged_strat_cum_excess_return_cluster") -> float:
    # TODO: this should be improved: maybe get optimal threshold based on ROC_CURVE instead of grid search. Must pass target as parameter

    thresholds = np.linspace(0, np.max(probas[cluster_name]) + 1e-6, 50)
    if method == "hedged_strat_cum_excess_return_cluster":
        metric = [[hedged_strat_cum_excess_return_cluster(returns, weights, probas, cluster, cluster_name, t),
                   t] for t in thresholds]
        metric.sort(key=lambda x: x[0])
        optimal_t = metric[-1][1]
    elif method == "hedged_equal_cum_excess_return_cluster":
        metric = [[hedged_equal_cum_excess_return_cluster(returns, probas, cluster, cluster_name, t),
                   t] for t in thresholds]
        metric.sort(key=lambda x: x[0])
        optimal_t = metric[-1][1]
    elif method == "calibrated_exceedance":
        assert target is not None
        optimal_t = calibrated_exceedance_threshold(target[cluster_name], probas[cluster_name], thresholds)
    else:
        raise NotImplementedError(method)

    return optimal_t


def get_exceedance(pred: pd.Series) -> float:
    exceendance = np.sum(pred) / len(pred)
    return exceendance


def calibrated_exceedance_threshold(target: pd.Series, probas: pd.Series,
                                    thresholds: Union[List[float], np.ndarray]):
    true_exceedance = np.sum(target == 1) / len(target)
    calibration = [[t, np.abs(true_exceedance - get_exceedance((probas.dropna() >= t).astype(int)))] for t in
                   thresholds]
    calibration.sort(key=lambda x: x[1])
    optimal_t = calibration[0][0]

    return optimal_t

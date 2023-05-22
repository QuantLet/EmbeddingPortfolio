import json
import os

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional

from sklearn import metrics

from dl_portfolio.logger import LOGGER
from dl_portfolio.constant import AVAILABLE_METHODS, DATASET1_REF_CLUSTER, \
    BASE_FACTOR_ORDER_DATASET1_4, DATASET2_REF_CLUSTER, \
    BASE_FACTOR_ORDER_DATASET2_5
from dl_portfolio.utils import get_features_order


def load_activation(garch_data_dir, ae_dir, garch_dir, perf_ae_dir, dataset):
    cvs = sorted(
        [int(cv) for cv in os.listdir(garch_data_dir) if cv.isdigit()]
    )
    # Load tail events data
    train_activation = pd.DataFrame()
    test_activation = pd.DataFrame()
    train_probas = pd.DataFrame()
    test_probas = pd.DataFrame()
    cv_predictions = pd.read_pickle(f"{perf_ae_dir}/predictions.p")
    predictions = pd.DataFrame()

    if dataset == "dataset1":
        ref_cluster = DATASET1_REF_CLUSTER
        columns = BASE_FACTOR_ORDER_DATASET1_4
    elif dataset == "dataset2":
        ref_cluster = DATASET2_REF_CLUSTER
        columns = BASE_FACTOR_ORDER_DATASET2_5
    else:
        raise NotImplementedError()

    for cv in cvs:
        t = pd.read_csv(f"{garch_data_dir}/{cv}/train_linear_activation.csv",
                        index_col=0)
        t.index = pd.to_datetime(t.index)
        # reorder features
        loading = pd.read_pickle(f"{ae_dir}/{cv}/decoder_weights.p")
        new_order = get_features_order(loading, ref_cluster)
        t = t.iloc[:, new_order]
        if cv > 0:
            last_date = train_activation.index[-1]
            t = t[t.index > last_date]
        train_activation = pd.concat([train_activation, t])

        t = pd.read_csv(f"{garch_data_dir}/{cv}/test_linear_activation.csv",
                        index_col=0)
        t = t.iloc[:, new_order]
        test_activation = pd.concat([test_activation, t])

        t = pd.read_csv(f"{garch_dir}/{cv}/train_activation_probas.csv",
                        index_col=0, parse_dates=True)
        t = t.iloc[:, new_order]
        if cv > 0:
            t = t[t.index > last_date]
        train_probas = pd.concat([train_probas, t])

        t = pd.read_csv(f"{garch_dir}/{cv}/activation_probas.csv",
                        index_col=0, parse_dates=True)
        t = t.iloc[:, new_order]
        test_probas = pd.concat([test_probas, t])

        t = cv_predictions[cv].iloc[:, new_order].copy()
        predictions = pd.concat([predictions, t])

    train_activation.columns = columns
    test_activation.columns = columns
    train_probas.columns = columns
    test_probas.columns = columns
    predictions.columns = columns

    test_activation.index = pd.to_datetime(test_activation.index)

    return (train_activation, test_activation, train_probas, test_probas,
            predictions)


def hedged_portfolio_weights_wrapper(
    cv: int,
    returns: pd.DataFrame,
    garch_dir: str,
    data_dir: str,
    or_port_weights: Dict,
    strats: List[str] = ["ae_rp_c", "aeaa", "aerp", "aeerc"],
    window: Optional[int] = None,
    method: Optional[str] = "calibrated_exceedance",
    use_test_probas: bool = False,
):
    LOGGER.info(f"CV: {cv}")
    assets = list(returns.columns)
    # Load target and predicted probas
    if use_test_probas:
        # Calibrate the threshold on previous test sets
        if cv == 0:
            # Load target
            prev_target = pd.read_csv(
                f"{data_dir}/{cv}/train_linear_activation.csv", index_col=0,
                parse_dates=True
            )
            # Load prediction
            prev_probas = pd.read_csv(
                f"{garch_dir}/{cv}/train_activation_probas.csv", index_col=0,
                parse_dates=True
            )
        else:
            # load first target
            prev_target = [
                pd.read_csv(
                    f"{data_dir}/0/train_linear_activation.csv", index_col=0,
                    parse_dates=True
                )
            ]
            # Load first prediction
            prev_probas = [
                pd.read_csv(
                    f"{garch_dir}/0/train_activation_probas.csv", index_col=0,
                    parse_dates=True
                )
            ]
            window_size = len(prev_target)

            # Include previous target and preds from previous test sets
            for i in range(cv):
                prev_target.append(
                    pd.read_csv(
                        f"{data_dir}/{i}/test_linear_activation.csv", index_col=0,
                        parse_dates=True
                    )
                )
                prev_probas.append(
                    pd.read_csv(
                        f"{garch_dir}/{i}/activation_probas.csv", index_col=0,
                        parse_dates=True
                    )
                )
            prev_probas = pd.concat(prev_probas)
            prev_target = pd.concat(prev_target)
            # remove duplicates if any
            assert isinstance(prev_probas.index,
                              pd.pandas.core.indexes.datetimes.DatetimeIndex)
            assert isinstance(prev_target.index,
                              pd.pandas.core.indexes.datetimes.DatetimeIndex)
            assert not prev_probas.index.duplicated().any()
            assert not prev_target.index.duplicated().any()

            # now take only the window
            # sort just in case
            prev_probas = prev_probas.sort_index().iloc[-window_size:]
            prev_target = prev_target.sort_index().iloc[-window_size:]
    else:
        # Calibrate the threshold on train set
        # Load target
        prev_target = pd.read_csv(
            f"{data_dir}/{cv}/train_linear_activation.csv", index_col=0,
            parse_dates=True
        )
        # Load prediction
        prev_probas = pd.read_csv(
            f"{garch_dir}/{cv}/train_activation_probas.csv", index_col=0,
            parse_dates=True
        )

    prev_target = (prev_target <= 0).astype(int)
    prev_probas.index = pd.to_datetime(
        [pd.to_datetime(d).date() for d in prev_probas.index]
    )

    probas = pd.read_csv(f"{garch_dir}/{cv}/activation_probas.csv", index_col=0)
    probas.index = pd.to_datetime(
        [pd.to_datetime(d).date() for d in probas.index]
    )
    n_cluster = probas.shape[-1]
    # Load clusters
    cluster_assignment = json.load(open(
        f"{data_dir}/{cv}/cluster_assignment.json", "r"))
    cluster = pd.Series(index=assets)
    for k in cluster_assignment:
        e = cluster_assignment[k]
        cluster.loc[e] = int(k)
    cluster = cluster.astype(int)
    # Remove unassigned assets
    cluster[cluster >= n_cluster] = np.nan
    # cluster = cluster.apply(lambda x: train_probas.columns[x])

    # Handle renaming of columns from R
    probas = probas[prev_probas.columns]  # Just to be sure
    columns = list(prev_probas.columns)
    columns = [c.replace(".", "-") for c in columns]
    columns = [c.replace("X", "") for c in columns]

    if all([c.isdigit() for c in columns]):
        columns = [int(c) for c in columns]
    prev_target.columns = columns
    prev_probas.columns = columns
    probas.columns = columns

    train_returns = returns.loc[prev_probas.index]
    test_returns = returns.loc[probas.index]

    if window is not None:
        assert isinstance(window, int)
        train_returns = train_returns.iloc[-window:]

    res = {"port": {}, "signal": {}}
    for strat in strats:
        original_weights = or_port_weights[strat].iloc[cv][assets]
        signals, hedged_weights, pred = hedged_portfolio_weights(
            train_returns,
            prev_probas,
            probas,
            cluster,
            assets,
            original_weights,
            target=prev_target,
            method=method,
        )
        res["port"][strat] = hedged_weights
        res["signal"][strat] = signals
    res["train_returns"] = train_returns
    res["returns"] = test_returns
    res["pred"] = pred

    return cv, res


def hedged_portfolio_weights(
    train_returns,
    train_probas,
    probas,
    cluster,
    assets,
    original_weights,
    target: Optional[pd.DataFrame] = None,
    method: Optional[str] = "calibrated_exceedance",
) -> Union[pd.DataFrame, pd.DataFrame]:
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
    pred = pd.DataFrame()
    for cluster_name in cluster_names:
        train_w = pd.DataFrame(
            np.repeat(
                original_weights.loc[assets].values.reshape(-1, 1).T,
                len(train_probas.index),
                axis=0,
            ),
            columns=assets,
            index=train_probas.index,
        )
        test_w = pd.DataFrame(
            np.repeat(
                original_weights.loc[assets].values.reshape(-1, 1).T,
                len(probas.index),
                axis=0,
            ),
            columns=assets,
            index=probas.index,
        )
        optimal_t = get_best_threshold(
            train_returns,
            train_w,
            train_probas,
            cluster,
            cluster_name,
            target=target,
            method=method,
        )
        pred = pd.concat([pred, (probas.loc[:, cluster_name] >=
                                 optimal_t).astype(int)], axis=1)
        signal_c, temp_w_c = get_hedged_weight_cluster(
            test_w, probas, cluster, cluster_name, optimal_t
        )
        weights = pd.concat([weights, temp_w_c], 1)
        signals = pd.concat([signals, signal_c], 1)

    weights[unnassigned] = 0.0
    weights = weights[assets]

    signals[unnassigned] = np.nan
    signals = signals[assets]

    return signals, weights, pred


def get_signals(
    train_returns,
    train_probas,
    probas,
    cluster,
    assets,
    original_weights,
    target: Optional[pd.DataFrame] = None,
    method: Optional[str] = "hedged_strat_cum_excess_return_cluster",
) -> pd.DataFrame:
    cluster_names = np.unique(cluster.dropna()).tolist()
    unnassigned = cluster.index[cluster.isna()]
    signals = pd.DataFrame()
    for cluster_name in cluster_names:
        train_w = pd.DataFrame(
            np.repeat(
                original_weights.loc[assets].values.reshape(-1, 1).T,
                len(train_probas.index),
                axis=0,
            ),
            columns=assets,
            index=train_probas.index,
        )
        optimal_t = get_best_threshold(
            train_returns,
            train_w,
            train_probas,
            cluster,
            cluster_name,
            target=target,
            method=method,
        )
        temp_signal = get_signal_cluster(
            probas, cluster, cluster_name, optimal_t, method
        )
        signals = pd.concat([signals, temp_signal], 1)

    signals[unnassigned] = np.nan
    signals = signals[assets]

    return signals


def get_signal_cluster(probas: pd.DataFrame, cluster, cluster_name, threshold):
    cluster_assets = cluster.index[cluster == cluster_name]
    signal = pd.DataFrame(
        0, index=probas.index, columns=cluster_assets, dtype=int
    )
    signal[cluster_assets] = np.repeat(
        (probas[[cluster_name]] < threshold).astype(int).values,
        np.sum(cluster == cluster_name),
        axis=1,
    )
    signal.fillna(
        0, inplace=True
    )  # Assets which are not assigned to any cluster have NaN
    return signal


def get_hedged_weight_cluster(
    weights: pd.DataFrame,
    probas: pd.DataFrame,
    cluster: pd.Series,
    cluster_name: List[str],
    threshold: float,
) -> Union[pd.DataFrame, pd.DataFrame]:
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


def get_hedged_return_cluster(
    returns: pd.DataFrame,
    probas: pd.DataFrame,
    cluster: pd.Series,
    cluster_name: List[str],
    threshold: float,
    weights: Optional[pd.DataFrame] = None,
):
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
        cluster_return = (
            returns[cluster_assets] * weights[cluster_assets] * signal
        ).sum(1)
    else:
        cluster_return = (returns[cluster_assets] * signal).mean(1)

    return cluster_return


def hedged_strat_cum_excess_return_cluster(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    probas: pd.DataFrame,
    cluster: pd.Series,
    cluster_name: List[str],
    threshold: float,
) -> float:
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
    hedged_cluster_return = get_hedged_return_cluster(
        returns, probas, cluster, cluster_name, threshold, weights=weights
    )
    cluster_assets = cluster.index[cluster == cluster_name]
    cluster_return = (returns[cluster_assets] * weights[cluster_assets]).sum(1)
    cum_excess_return = np.cumsum(hedged_cluster_return - cluster_return)[-1]

    return cum_excess_return


def hedged_equal_cum_excess_return_cluster(
    returns: pd.DataFrame,
    probas: pd.DataFrame,
    cluster: pd.Series,
    cluster_name: List[str],
    threshold: float,
) -> float:
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
    hedged_cluster_return = get_hedged_return_cluster(
        returns, probas, cluster, cluster_name, threshold
    )
    cluster_assets = cluster.index[cluster == cluster_name]
    cluster_return = (returns[cluster_assets]).mean(1)
    cum_excess_return = np.cumsum(hedged_cluster_return - cluster_return)[-1]

    return cum_excess_return


def get_best_threshold(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    probas: pd.DataFrame,
    cluster: pd.Series = None,
    cluster_name: List[str] = None,
    target: Optional[pd.DataFrame] = None,
    method: Optional[str] = "hedged_strat_cum_excess_return_cluster",
) -> float:
    thresholds = np.linspace(0, np.max(probas.loc[:, cluster_name]) + 1e-6, 50)
    if method == "hedged_strat_cum_excess_return_cluster":
        assert cluster is not None
        metric = [
            [
                hedged_strat_cum_excess_return_cluster(
                    returns, weights, probas, cluster, cluster_name, t
                ),
                t,
            ]
            for t in thresholds
        ]
        metric.sort(key=lambda x: x[0])
        optimal_t = metric[-1][1]
    elif method == "hedged_equal_cum_excess_return_cluster":
        assert cluster is not None
        metric = [
            [
                hedged_equal_cum_excess_return_cluster(
                    returns, probas, cluster, cluster_name, t
                ),
                t,
            ]
            for t in thresholds
        ]
        metric.sort(key=lambda x: x[0])
        optimal_t = metric[-1][1]
    elif method == "calibrated_exceedance":
        assert target is not None
        assert cluster_name is not None
        optimal_t = calibrated_exceedance_threshold(
            target.loc[:, cluster_name], probas.loc[:, cluster_name],
            thresholds
        )
    elif method == "roc_curve":
        # distance top left
        temp = probas.loc[:, cluster_name].dropna()
        fpr, tpr, t = metrics.roc_curve(target.loc[temp.index, cluster_name],
                                        temp)
        squred_distance = fpr ** 2 + (tpr - 1) ** 2
        optimal_t = t[np.argmin(squred_distance)]
    else:
        raise NotImplementedError(method)

    return optimal_t


def get_exceedance(pred: pd.Series) -> float:
    exceendance = np.sum(pred) / len(pred)
    return exceendance


def calibrated_exceedance_threshold(
    target: pd.Series,
    probas: pd.Series,
    thresholds: Union[List[float], np.ndarray],
):
    true_exceedance = np.sum(target == 1) / len(target)
    calibration = [
        [
            t,
            np.abs(
                true_exceedance
                - get_exceedance((probas.dropna() >= t).astype(int))
            ),
        ]
        for t in thresholds
    ]
    calibration.sort(key=lambda x: x[1])
    optimal_t = calibration[0][0]

    return optimal_t

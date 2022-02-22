import numpy as np
import pandas as pd
from typing import List, Dict


def hedged_portfolio_weights_wrapper(cv: int, returns: pd.DataFrame, cluster: pd.Series, cv_garch_dir: str,
                                     or_port_weights: Dict, strats: List[str] = ['ae_rp_c', 'aeaa', 'aerp', 'aeerc'],
                                     window=None):
    assets = list(returns.columns)
    train_probas = pd.read_csv(f"{cv_garch_dir}/train_activation_probas.csv", index_col=0)
    train_probas.index = pd.to_datetime(train_probas.index)
    probas = pd.read_csv(f"{cv_garch_dir}/activation_probas.csv", index_col=0)

    # Handle stupid renaming of columns from R
    probas = probas[train_probas.columns]  # Just to be sure
    columns = list(train_probas.columns)
    columns = [c.replace(".", "-") for c in columns]
    train_probas.columns = columns
    probas.columns = columns

    probas.index = pd.to_datetime(probas.index)
    train_returns = returns.loc[train_probas.index]
    test_returns = returns.loc[probas.index]

    if window is not None:
        assert isinstance(window, int)
        train_returns = train_returns.iloc[-window:]

    res = {"port": {}}
    for strat in strats:
        original_weights = or_port_weights[strat].iloc[cv][assets]
        hedged_weights = hedged_portfolio_weights(train_returns, train_probas, probas, cluster, assets,
                                                  original_weights)
        res["port"][strat] = hedged_weights
    res["train_returns"] = train_returns
    res["returns"] = test_returns

    return res


def hedged_portfolio_weights(train_returns, train_probas, probas, cluster, assets, original_weights):
    cluster_names = np.unique(cluster.dropna()).tolist()
    unnassigned = cluster.index[cluster.isna()]
    weights = pd.DataFrame()
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
        optimal_t = get_best_threshold(train_returns, train_w, train_probas, cluster, cluster_name)
        temp_w_c = get_hedged_weight_cluster(test_w, probas, cluster, cluster_name, optimal_t)
        weights = pd.concat([weights, temp_w_c], 1)

    weights[unnassigned] = 0.
    weights = weights[assets]

    return weights


def get_signal_cluster(probas, cluster, cluster_name, threshold):
    cluster_assets = cluster.index[cluster == cluster_name]
    signal = pd.DataFrame(0, index=probas.index, columns=cluster_assets, dtype=int)
    signal[cluster_assets] = np.repeat((probas[[cluster_name]] < threshold).astype(int).values,
                                       np.sum(cluster == cluster_name), axis=1)
    signal.fillna(0, inplace=True)  # Assets which are not assigned to any cluster have NaN
    return signal


def get_hedged_weight_cluster(weights, probas, cluster, cluster_name, threshold):
    signal = get_signal_cluster(probas, cluster, cluster_name, threshold)
    cluster_assets = cluster.index[cluster == cluster_name]
    return weights[cluster_assets] * signal


def get_hedged_return_cluster(returns, weights, probas, cluster, cluster_name, threshold):
    signal = get_signal_cluster(probas, cluster, cluster_name, threshold)
    cluster_assets = cluster.index[cluster == cluster_name]
    cluster_return = (returns[cluster_assets] * weights[cluster_assets] * signal).sum(1)

    return cluster_return


def get_hedged_cum_excess_return_cluster(returns, weights, probas, cluster, cluster_name, threshold):
    hedged_cluster_return = get_hedged_return_cluster(returns, weights, probas,
                                                      cluster, cluster_name, threshold)
    cluster_assets = cluster.index[cluster == cluster_name]
    cluster_return = (returns[cluster_assets] * weights[cluster_assets]).sum(1)
    cum_excess_return = np.cumsum(hedged_cluster_return - cluster_return)[-1]

    return cum_excess_return


def get_best_threshold(returns, weights, probas, cluster, cluster_name, method="cum_excess_return"):
    print(probas)
    print(cluster_name)
    thresholds = np.linspace(0, np.max(probas[cluster_name]) + 1e-6, 100)
    if method == "cum_excess_return":
        metric = [[get_hedged_cum_excess_return_cluster(returns, weights, probas, cluster, cluster_name, t),
                   t] for t in thresholds]

    metric.sort(key=lambda x: x[0])
    optimal_t = metric[-1][1]

    return optimal_t

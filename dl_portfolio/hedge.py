import numpy as np
import pandas as pd


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
    thresholds = np.linspace(0, np.max(probas[cluster_name]) + 1e-6, 100)
    if method == "cum_excess_return":
        metric = [[get_hedged_cum_excess_return_cluster(returns, weights, probas, cluster, cluster_name, t),
                   t] for t in thresholds]

    metric.sort(key=lambda x: x[0])
    optimal_t = metric[-1][1]

    return optimal_t

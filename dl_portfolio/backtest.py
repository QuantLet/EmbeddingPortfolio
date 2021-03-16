import numpy as np
import pandas as pd


def cluster_portfolio(test_returns, train_cluster_port, encoding_weights):
    n_clusters = encoding_weights.shape[1]
    cov = np.cov(train_cluster_port, rowvar=False)
    cluster_weights = getIVP(cov)
    norm_encoding_weights = (encoding_weights / encoding_weights.sum())

    port_returns = pd.Series(np.zeros(len(test_returns)), index=test_returns.index)
    for i in range(n_clusters):
        cluster_ret = cluster_weights[i] * (
                test_returns * norm_encoding_weights.iloc[:, i].values.reshape(-1, 1).T).sum(1)
        port_returns = port_returns + cluster_ret.values

    return port_returns


def get_portfolio_perf(train_returns, test_returns, train_cluster_port, encoding_weights, fx_levrage=1.):
    # AE port
    ae_port_returns = cluster_portfolio(test_returns, train_cluster_port, encoding_weights)

    # ivp
    cov = np.cov(train_returns, rowvar=False)
    weights = getIVP(cov)
    ivp_returns = (test_returns * weights).sum(1)

    # equally
    equally_weighted = test_returns.mean(1)

    return equally_weighted, ivp_returns, ae_port_returns


def getIVP(cov):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

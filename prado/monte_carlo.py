# From lopez de prado

# On 20151231 by MLdP <lopezdeprado@lbl.gov>
import os
import pdb
import time

import fastcluster
import scipy.cluster.hierarchy as sch
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from statsmodels.stats.correlation_tools import corr_nearest

import seaborn as sns

import prado.cla.CLA as CLA
from dl_portfolio.logger import LOGGER
from dl_portfolio.nmf.convex_nmf import ConvexNMF
from dl_portfolio.weights import ae_riskparity_weights
from prado.hrp import correlDist, getIVP, getQuasiDiag, getRecBipart

import json

# ------------------------------------------------------------------------------

def dgp_mapper(dgp_name, dgp_params):
    if dgp_name == "hrp_mc":
        return generate_data_hrp(**dgp_params)
    else:
        raise NotImplementedError(dgp_name)

def generate_data_hrp(n_obs, sLength, size0, size1, mu0, sigma0, sigma1F):
    # Time series of correlated variables
    # 1) generate random uncorrelated data: each row is a variable
    x = np.random.normal(mu0, sigma0, size=(n_obs, size0))

    # 2) create correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F,
                                      size=(n_obs, len(cols)))
    x = np.append(x, y, axis=1)
    # 3) add common random shock
    point = np.random.randint(sLength, n_obs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]])

    # 4) add specific random shock
    point = np.random.randint(sLength, n_obs - 1, size=2)
    x[point, cols[-1]] = np.array([-.5, 2])
    cluster_mapper, col_cluster_mapper = get_cluster_mapper(size0, x.shape[-1],
                                                            cols)

    return x, col_cluster_mapper, cluster_mapper


def get_cluster_mapper(size0, nvars, cols):
    cluster_labels = np.unique(cols).tolist()
    col_cluster_mapper = {i: i for i in range(nvars)}
    for i, c in enumerate(range(size0, nvars)):
        col_cluster_mapper[c] = cols[i]
    col_cluster_mapper = pd.Series(col_cluster_mapper)
    cluster_mapper = {
        c: col_cluster_mapper.index[col_cluster_mapper == c].tolist() for c
        in cluster_labels
    }
    return cluster_mapper, col_cluster_mapper


def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()


def getIVP_mc(cov, **kargs):
    return pd.Series(getIVP(cov))


def getCLA(cov, **kargs):
    # Compute CLA's minimum variance portfolio
    mean = np.arange(cov.shape[0]).reshape(-1, 1)  # Not used by C portf
    lB = np.zeros(mean.shape)
    uB = np.ones(mean.shape)
    cla = CLA.CLA(mean, cov, lB, uB)
    cla.solve()
    return cla.w[-1].flatten()


def create_art_market_budget(col_cluster_mapper):
    market_budget = pd.DataFrame(col_cluster_mapper, columns=["market"])
    market_budget = market_budget.reset_index(drop=False)
    market_budget["rc"] = 1.
    market_budget.columns = ["assets", "market", "rc"]
    market_budget["assets"] = market_budget["assets"].astype(str)
    market_budget["market"] = market_budget["market"].astype(str)
    return market_budget


def getNMF(train_data, n_components, market_budget, method="ae_rp_c"):
    model = ConvexNMF(
        n_components=n_components,
        random_state=None,
        verbose=0,
    )
    model.fit(train_data)

    embedding = pd.DataFrame(model.encoding.copy())
    loading = pd.DataFrame(model.components.copy())

    if method == "ae_rp_c":
        weights = ae_riskparity_weights(pd.DataFrame(train_data),
                                        embedding, loading,
                                        market_budget=market_budget,
                                        risk_parity="cluster",
                                        threshold=0.)
    else:
        raise NotImplementedError()

    return weights


def worker(steps, methods_mapper, dgp_name=None, dgp_params=None, sLength=260,
           rebal=22, returns=None, col_cluster_mapper=None,
           cluster_mapper=None):
    if steps % 10 == 0:
        LOGGER.info(f"Steps to go: {steps}")

    stats = {i: pd.Series(dtype=np.float32) for i in methods_mapper}
    # 1) Prepare data for one experiment
    if dgp_name:
        returns, col_cluster_mapper, cluster_mapper = dgp_mapper(
            dgp_name, dgp_params)

    r = {i: pd.Series(dtype=np.float32) for i in methods_mapper}
    weights = {i: pd.DataFrame() for i in methods_mapper}

    pointers = range(sLength, len(returns), rebal)

    n_components = len(cluster_mapper)
    market_budget = create_art_market_budget(col_cluster_mapper)

    # 2) Compute portfolios in-sample
    for pointer in pointers:
        in_x_ = returns[pointer - sLength:pointer]
        cov_, corr_ = np.cov(in_x_, rowvar=0), np.corrcoef(in_x_, rowvar=0)

        # 3) Compute performance out-of-sample
        x_ = returns[pointer:pointer + rebal]
        for func_name in methods_mapper:
            if func_name == "NMF":
                w_ = methods_mapper[func_name](in_x_, n_components,
                                               market_budget=market_budget)
            else:
                assert func_name in ["IVP", "HRP"]
                w_ = methods_mapper[func_name](cov=cov_, corr=corr_)
            r_ = pd.Series(np.dot(x_, w_))
            r[func_name] = r[func_name].append(r_)
            weights[func_name] = pd.concat([weights[func_name], w_], axis=0)

    # 4) Evaluate and store results
    port_returns = {}
    for func_name in methods_mapper:
        r_ = r[func_name].reset_index(drop=True)
        p_ = (1 + r_).cumprod()
        stats[func_name] = p_.iloc[-1] - 1  # terminal return

        port_returns[func_name] = r_

    return port_returns, weights, cluster_mapper  # , stats


def mc_hrp(methods_mapper,  dgp_name,  dgp_params, n_jobs=1,
           num_iters=int(1e4), sLength=260, rebal=22,  save_dir=None):
    assert save_dir is not None
    # Monte Carlo experiment on HRP
    # stats = {i: pd.Series() for i in methods_mapper}
    returns = {k: pd.DataFrame() for k in methods_mapper}
    weights = {k: pd.DataFrame() for k in methods_mapper}

    if n_jobs > 1:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            results = _parallel_pool(
                delayed(worker)(
                    num_iters - numIter, methods_mapper, dgp_name,
                    dgp_params, sLength=sLength, rebal=rebal
                )
                for numIter in range(num_iters)
            )
        # for numIter in range(numIters):
        #     for func_name in methods_mapper:
        #         stats[func_name].loc[numIter] = stats_iter[numIter][
        #         func_name]

        clusters = []
        for numIter in range(num_iters):
            for func_name in methods_mapper:
                returns[func_name] = pd.concat(
                    [
                        returns[func_name],
                        results[numIter][0][func_name]
                    ],
                    axis=1
                )
                weights[func_name] = pd.concat(
                    [
                        weights[func_name],
                        results[numIter][1][func_name]
                    ],
                    axis=1
                )
            clusters.append(results[numIter][2])

    else:
        clusters = []
        for numIter in range(num_iters):
            returns_iter, weights_iter, cluster_iter = worker(
                num_iters - numIter, methods_mapper, dgp_name, dgp_params,
                sLength=sLength, rebal=rebal)
            clusters.append(cluster_iter)
            # for func_name in methods_mapper:
            #     stats[func_name].loc[numIter] = stats_iter[func_name]
            for func_name in methods_mapper:
                returns[func_name] = pd.concat(
                    [
                        returns[func_name],
                        returns_iter[func_name]
                    ],
                    axis=1
                )
                weights[func_name] = pd.concat(
                    [
                        weights[func_name],
                        weights_iter[func_name]
                    ],
                    axis=1
                )

    for func_name in methods_mapper:
        returns[func_name].columns = list(range(num_iters))
        weights[func_name].columns = list(range(num_iters))

    # 5) Report results
    # stats = pd.DataFrame.from_dict(returns, orient='columns')
    # stats.to_csv('stats.csv')
    # df0, df1 = stats.std(), stats.var()
    # print(stats)
    # print(pd.concat([df0, df1, df1 / df1['getHRP'] - 1], axis=1))
    for func_name in methods_mapper:
        returns[func_name].to_csv(f"{save_dir}/returns_{func_name}.csv")
        weights[func_name].to_csv(f"{save_dir}/weights_{func_name}.csv")

    json.dump(clusters, open(f"{save_dir}/clusters.json", "w"))
    return


def mc_from_corr(methods_mapper, batch, n_jobs=8,
                 num_iters=int(1e4), sLength=260, rebal=22,  save_dir=None):
    assert os.path.isdir(save_dir)
    generator = tf.keras.models.load_model(
        "prado/corrgan-models/saved_model/generator_100d")

    # Monte Carlo experiment on HRP
    # stats = {i: pd.Series() for i in methods_mapper}
    returns = {k: pd.DataFrame() for k in methods_mapper}
    weights = {k: pd.DataFrame() for k in methods_mapper}

    counter = 0
    if n_jobs > 1:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            results = _parallel_pool(
                    delayed(worker)(
                        num_iters - counter, methods_mapper, dgp_name=None,
                        dgp_params=None, sLength=sLength, rebal=rebal,
                        returns=batch[numIter][0], col_cluster_mapper=batch[
                            numIter][1],
                        cluster_mapper=batch[numIter][2]
                    )
                    for numIter in range(num_iters)
            )
        # for numIter in range(numIters):
        #     for func_name in methods_mapper:
        #         stats[func_name].loc[numIter] = stats_iter[numIter][
        #         func_name]

        clusters = []
        for numIter in range(num_iters):
            for func_name in methods_mapper:
                returns[func_name] = pd.concat(
                    [
                        returns[func_name],
                        results[numIter][0][func_name]
                    ],
                    axis=1
                )
                weights[func_name] = pd.concat(
                    [
                        weights[func_name],
                        results[numIter][1][func_name]
                    ],
                    axis=1
                )
            clusters.append(results[numIter][2])
    else:
        clusters = []
        for numIter in range(num_iters):
            returns_iter, weights_iter, cluster_iter = worker(
                num_iters - counter, methods_mapper, dgp_name=None,
                dgp_params=None, sLength=sLength, rebal=rebal,
                returns=batch[numIter][0], col_cluster_mapper=batch[
                    numIter][1],
                cluster_mapper=batch[numIter][2]
            )
            clusters.append(cluster_iter)
            # for func_name in methods_mapper:
            #     stats[func_name].loc[numIter] = stats_iter[func_name]
            for func_name in methods_mapper:
                returns[func_name] = pd.concat(
                    [
                        returns[func_name],
                        returns_iter[func_name]
                    ],
                    axis=1
                )
                weights[func_name] = pd.concat(
                    [
                        weights[func_name],
                        weights_iter[func_name]
                    ],
                    axis=1
                )



    for func_name in methods_mapper:
        returns[func_name].columns = list(range(num_iters))
        weights[func_name].columns = list(range(num_iters))

    # 5) Report results
    # stats = pd.DataFrame.from_dict(returns, orient='columns')
    # stats.to_csv('stats.csv')
    # df0, df1 = stats.std(), stats.var()
    # print(stats)
    # print(pd.concat([df0, df1, df1 / df1['getHRP'] - 1], axis=1))
    for func_name in methods_mapper:
        returns[func_name].to_csv(f"{save_dir}/returns_{func_name}.csv")
        weights[func_name].to_csv(f"{save_dir}/weights_{func_name}.csv")

    json.dump(clusters, open(f"{save_dir}/clusters.json", "w"))
    return


METHODS_MAPPER = {
    "IVP": getIVP_mc,
    "HRP": getHRP,
    "NMF": getNMF
}

if __name__ == '__main__':
    import datetime as dt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp",
                        type=str,
                        help="DGP name: hrp_mc or corrgan")
    parser.add_argument("--n_jobs",
                        default=os.cpu_count(),
                        type=int,
                        help="Number of parallel jobs")
    parser.add_argument("--num_iters",
                        default=int(1e4),
                        type=int,
                        help="Number of MC iterations")

    args = parser.parse_args()



    save_dir = f"prado/results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if args.dgp == "hrp_mc":
        dgp_params = {
            "n_obs": 520,
            "size0": 6,
            "size1": 9,
            "mu0": 0,
            "sigma0": 1e-2,
            "sigma1F": 0.25,
            "sLength": 260,
        }
    elif args.dgp == "corrgan":
        dgp_params = {
            "n_assets": 100,
            "n_obs": 520,
            "min_sigma": 0.0025,
            "max_sigma": 0.015,
            "seed": 1234,
            # "sLength": 260,
        }
    elif args.dgp == "from_corr":
        raise NotImplementedError
        # dgp_params = {
        #     "n_assets": 100,
        #     "n_obs": 520,
        #     "min_sigma": 0.0025,
        #     "max_sigma": 0.015,
        #     "seed": 1234,
        #     # "sLength": 260,
        # }
        # generator = tf.keras.models.load_model(
        #     "prado/corrgan-models/saved_model/generator_100d")
        # batch = generate_batch_corr_mat(10,
        #                                 dgp_params["n_assets"],
        #                                 generator=generator,
        #                                 seed=None,
        #                                 nearest=False)
        # batch = [
        #     [
        #         generate_data_from_corr_mat(
        #             dgp_params["n_assets"],
        #             dgp_params["n_obs"],
        #             batch[i][0],
        #             np.random.uniform(dgp_params["min_sigma"],
        #                               dgp_params["max_sigma"],
        #                               dgp_params["n_assets"]).reshape(
        #                 dgp_params["n_assets"], 1)
        #         ),
        #         batch[i][1],
        #         batch[i][2],
        #     ] for i in range(args.num_iters)
        # ]
        #
        # mc_from_corr(methods_mapper, batch, n_jobs=1,
        #              num_iters=args.num_iters, sLength=260, rebal=22,
        #              save_dir=save_dir)
    else:
        raise NotImplementedError(args.dgp_name)

    mc_hrp(METHODS_MAPPER, args.dgp,  dgp_params, n_jobs=args.n_jobs,
           num_iters=args.num_iters, save_dir=save_dir)

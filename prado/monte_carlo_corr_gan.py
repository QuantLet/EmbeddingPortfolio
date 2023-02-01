import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed

from dl_portfolio.logger import LOGGER
from dl_portfolio.nmf.convex_nmf import ConvexNMF
from dl_portfolio.weights import ae_riskparity_weights
from prado.constants import METHODS_MAPPER
from prado.corrgan import generate_returns_from_gan
from prado.hrp import correlDist, getIVP, getQuasiDiag, getRecBipart

import json

from prado.monte_carlo import worker, create_art_market_budget


def generate_data_corrgan(corr_path, n_obs=520, min_sigma=0.0025,
                          max_sigma=0.015):
    gan_mat = np.loadtxt(corr_path, delimiter=",")
    returns, col_cluster_mapper, cluster_mapper = generate_returns_from_gan(
        gan_mat, n_obs, min_sigma=min_sigma,
        max_sigma=max_sigma)

    return returns, col_cluster_mapper, cluster_mapper


def worker(steps, methods_mapper, dgp_params=None, sLength=260,
           rebal=22):

    if steps % 10 == 0:
        LOGGER.info(f"Steps to go: {steps}")

    stats = {i: pd.Series(dtype=np.float32) for i in methods_mapper}
    # 1) Prepare data for one experiment
    returns, col_cluster_mapper, cluster_mapper = generate_data_corrgan(
        **dgp_params)

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
        # p_ = (1 + r_).cumprod()
        # stats[func_name] = p_.iloc[-1] - 1  # terminal return
        port_returns[func_name] = r_
        print(func_name, (1 + r_).cumprod().values[-1])

    return port_returns, weights, cluster_mapper  # , stats


def mc_gan(methods_mapper,  dgp_params, n_jobs=1,
           sLength=260, rebal=22,  save_dir=None):
    assert os.path.isdir(save_dir)
    # Monte Carlo experiment on HRP
    # stats = {i: pd.Series() for i in methods_mapper}
    returns = {k: pd.DataFrame() for k in methods_mapper}
    weights = {k: pd.DataFrame() for k in methods_mapper}

    num_iters = len(dgp_params)
    if n_jobs > 1:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            results = _parallel_pool(
                delayed(worker)(
                    num_iters - numIter, methods_mapper,
                    dgp_params[numIter], sLength=sLength, rebal=rebal
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
                    num_iters - numIter, methods_mapper,
                    dgp_params[numIter], sLength=sLength, rebal=rebal
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




if __name__ == '__main__':
    import datetime as dt
    import argparse

    dgp_name = "corrgan"
    corr_paths = (f"prado/corr_matrix/{corr_path}" for corr_path in filter(
        lambda x: "corr" in x, os.listdir("prado/corr_matrix")))
    dgp_params = [
        {
            "n_obs": 520,
            "min_sigma": 0.0025,
            "max_sigma": 0.015,
            "corr_path": corr_path
        } for corr_path in corr_paths
    ]

    batch_size = 100
    num_iters = len(dgp_params)
    n_batch = num_iters // batch_size + 1
    for b in range(n_batch):
        save_dir = f"prado/results_{dgp_name}" \
                   f"_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_batch_{b}"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        mc_gan(METHODS_MAPPER, dgp_params[b*batch_size:(b+1)*batch_size],
               n_jobs=1, sLength=260, rebal=22,  save_dir=save_dir)

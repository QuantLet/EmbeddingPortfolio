# From lopez de prado

# On 20151231 by MLdP <lopezdeprado@lbl.gov>
import os
import scipy.cluster.hierarchy as sch
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import prado.cla.CLA as CLA
from dl_portfolio.cluster import get_cluster_labels
from dl_portfolio.logger import LOGGER
from dl_portfolio.nmf.convex_nmf import ConvexNMF
from dl_portfolio.utils import optimal_target_vol_test
from dl_portfolio.weights import (
    ae_riskparity_weights,
    get_inner_cluster_weights,
)
from prado.hrp import correlDist, getIVP, getQuasiDiag, getRecBipart
from arch import arch_model

import json


def dgp_mapper(dgp_name, dgp_params):
    if dgp_name == "hrp_mc":
        return generate_data_hrp(**dgp_params)
    if dgp_name == "cluster_mc":
        return generate_data_cluster(**dgp_params)
    else:
        raise NotImplementedError(dgp_name)


def generate_data_hrp(n_obs, sLength, size0, size1, mu0, sigma0, sigma1F):
    # Time series of correlated variables
    # 1) generate random uncorrelated data: each row is a variable
    x = np.random.normal(mu0, sigma0, size=(n_obs, size0))

    # 2) create correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(
        0, sigma0 * sigma1F, size=(n_obs, len(cols))
    )
    x = np.append(x, y, axis=1)
    # 3) add common random shock
    point = np.random.randint(sLength, n_obs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-0.5, -0.5], [2, 2]])

    # 4) add specific random shock
    point = np.random.randint(sLength, n_obs - 1, size=2)
    x[point, cols[-1]] = np.array([-0.5, 2])
    cluster_mapper, col_cluster_mapper = get_cluster_mapper(
        size0, x.shape[-1], cols
    )

    return x, col_cluster_mapper, cluster_mapper


def generate_garch(
    assets=["BTC", "US_B", "SPX_X", "EUR_FX", "GOLDS_C"],
    sigmas: list = [0.08, 0.003, 0.01, 0.005, 0.007],
):
    """
    Fit a GARCH process to the selected assets and generate data. Rescaled
    the data to match the volatilities.

    :param assets:
    :param sigmas:
    :return:
    """
    data = pd.read_csv(
        "data/dataset1/dataset1.csv", index_col=0, parse_dates=True
    )
    generated = pd.DataFrame()
    for i, c in enumerate(assets):
        # define model
        model = arch_model(
            data[c], mean="Zero", vol="GARCH", p=1, q=1, rescale=True
        )
        model_fit = model.fit()
        sim = model.simulate(params=model_fit.params, nobs=520) / 10
        generated = pd.concat([generated, sim["data"]], axis=1)

    generated.columns = ["BTC", "US_B", "EUR_FX", "GOLDS_C", "SPX_X"]
    for i, c in enumerate(generated.columns):
        ostd = np.std(generated[c])
        generated[c] *= sigmas[i] / ostd

    return generated.values


def generate_data_cluster(
    n_obs,
    sLength,
    size1: int = 15,
    mu0: list = [0] * 5,
    sigma0: list = [0.08, 0.003, 0.01, 0.005, 0.007],
    sigma1F: float = 0.25,
    process="garch",
):
    # Time series of correlated variables
    # 1) generate random uncorrelated data
    size0 = len(sigma0)
    if process == "garch":
        assert len(sigma0) == 5
        x = generate_garch(sigmas=sigma0)
    elif process == "norm":
        assert len(mu0) == len(sigma0)
        x = np.random.normal(mu0, sigma0, size=(n_obs, size0))
    else:
        raise NotImplementedError(process)

    # 2) create new correlated variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    while len(np.unique(cols)) != size0:
        cols = [random.randint(0, size0 - 1) for i in range(size1)]

    y = np.zeros((n_obs, size1))
    for i, c in enumerate(cols):
        y[:, i] = x[:, c] + np.random.normal(
            0, sigma0[c] * sigma1F, size=n_obs
        )
    x = np.append(x, y, axis=1)
    # 3) add common random shock
    point = np.random.randint(sLength, n_obs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-0.5, -0.5], [2, 2]])

    # 4) add specific random shock
    point = np.random.randint(sLength, n_obs - 1, size=2)
    x[point, cols[-1]] = np.array([-0.5, 2])
    cluster_mapper, col_cluster_mapper = get_cluster_mapper(
        size0, x.shape[-1], cols
    )

    return x, col_cluster_mapper, cluster_mapper


def get_cluster_mapper(size0, nvars, cols):
    cluster_labels = np.unique(cols).tolist()
    col_cluster_mapper = {i: i for i in range(nvars)}
    for i, c in enumerate(range(size0, nvars)):
        col_cluster_mapper[c] = cols[i]
    col_cluster_mapper = pd.Series(col_cluster_mapper)
    cluster_mapper = {
        c: col_cluster_mapper.index[col_cluster_mapper == c].tolist()
        for c in cluster_labels
    }
    return cluster_mapper, col_cluster_mapper


def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(dist, "single")
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
    market_budget["rc"] = 1.0
    market_budget.columns = ["assets", "market", "rc"]
    market_budget["assets"] = market_budget["assets"].astype(str)
    market_budget["market"] = market_budget["market"].astype(str)
    return market_budget


def getNMF(
    train_data, n_components, market_budget, threshold, method="ae_rp_c"
):
    model = ConvexNMF(
        n_components=n_components,
        random_state=None,
        verbose=0,
    )
    model.fit(train_data)

    embedding = pd.DataFrame(model.encoding.copy())
    loading = pd.DataFrame(model.components.copy())

    if method == "ae_rp_c":
        returns = pd.DataFrame(train_data)
        weights = ae_riskparity_weights(
            returns,
            embedding,
            loading,
            market_budget=market_budget,
            risk_parity="cluster",
            threshold=threshold,
        )

        max_cluster = embedding.shape[-1] - 1
        # First get cluster allocation to forget about small contribution
        clusters, _ = get_cluster_labels(embedding, threshold=threshold)
        clusters = {c: clusters[c] for c in clusters if c <= max_cluster}
        inner_weights = get_inner_cluster_weights(
            returns.cov(), loading, clusters, market_budget
        )
        inner_weights = pd.DataFrame(inner_weights)
    else:
        raise NotImplementedError()

    return weights, inner_weights


def worker(
    steps,
    methods_mapper,
    dgp_name=None,
    dgp_params=None,
    sLength=260,
    rebal=22,
    returns=None,
    col_cluster_mapper=None,
    cluster_mapper=None,
    vol_target: float = 0.05,
    max_leverage: float = 3,
):
    if steps % 10 == 0:
        LOGGER.info(f"Steps to go: {steps}")

    stats = {i: pd.Series(dtype=np.float32) for i in methods_mapper}
    # 1) Prepare data for one experiment
    if dgp_name:
        returns, col_cluster_mapper, cluster_mapper = dgp_mapper(
            dgp_name, dgp_params
        )

    r = {i: pd.Series(dtype=np.float32) for i in methods_mapper}
    leverage = {i: [] for i in methods_mapper}
    test_target_vol = {i: [] for i in methods_mapper}
    weights = {i: pd.DataFrame() for i in methods_mapper}
    inner_weights = pd.DataFrame()

    pointers = range(sLength, len(returns), rebal)

    n_components = len(cluster_mapper)
    market_budget = create_art_market_budget(col_cluster_mapper)

    # 2) Compute portfolios in-sample
    for pointer in pointers:
        in_x_ = returns[pointer - sLength : pointer]
        cov_, corr_ = np.cov(in_x_, rowvar=0), np.corrcoef(in_x_, rowvar=0)

        # 3) Compute performance out-of-sample
        x_ = returns[pointer: pointer + rebal]
        in_x_ = returns[:pointer]
        for func_name in methods_mapper:
            try:
                if func_name == "NMF":
                    w_, in_w_ = methods_mapper[func_name](
                        in_x_,
                        n_components,
                        market_budget=market_budget,
                        threshold=1e-2,
                    )
                    inner_weights = pd.concat([inner_weights, in_w_])
                else:
                    assert func_name in ["IVP", "HRP"]
                    w_ = methods_mapper[func_name](cov=cov_, corr=corr_)
                r_ = pd.Series(np.dot(x_, w_))

                # Vol target leverage (Jaeger et al 2021):
                if vol_target is not None:
                    in_r_ = np.dot(in_x_, w_)
                    tvs = optimal_target_vol_test(pd.Series(in_r_))
                    base_vol = np.max(
                        (np.std(in_r_[-20:]), np.std(in_r_[-60:]))
                    ) * np.sqrt(252)
                    assert not np.isinf(base_vol)
                    assert not np.isnan(base_vol)
                    lev = vol_target / base_vol
                    if lev > max_leverage:
                        LOGGER.warning(f"leverage={lev}>{max_leverage}. "
                                       f"Setting to {max_leverage}")
                        lev = max_leverage
                    r_ = lev * r_
                else:
                    lev = None
            except Exception as _exc:
                LOGGER.exception(f"Error with {func_name}...{_exc}")
                r_ = pd.Series([np.nan] * len(x_))
                w_ = pd.Series([np.nan] * x_.shape[-1])

            leverage[func_name].append(lev)
            test_target_vol[func_name].append(tvs)
            r[func_name] = r[func_name].append(r_, ignore_index=True)
            weights[func_name] = pd.concat([weights[func_name], w_])

    # 4) Evaluate and store results
    port_returns = {}
    for func_name in methods_mapper:
        r_ = r[func_name].reset_index(drop=True)
        p_ = (1 + r_).cumprod()
        stats[func_name] = p_.iloc[-1] - 1  # terminal return

        port_returns[func_name] = r_

    return port_returns, weights, cluster_mapper, inner_weights, leverage, test_target_vol


def mc_hrp(
    methods_mapper,
    dgp_name,
    dgp_params,
    n_jobs=1,
    num_iters=int(1e4),
    sLength=260,
    rebal=22,
    save_dir=None,
):
    assert save_dir is not None
    # Monte Carlo experiment on HRP
    # stats = {i: pd.Series() for i in methods_mapper}
    returns = {k: pd.DataFrame() for k in methods_mapper}
    weights = {k: pd.DataFrame() for k in methods_mapper}
    leverage = {k: [] for k in methods_mapper}
    test_target_vol = {k: [] for k in methods_mapper}

    if n_jobs > 1:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            results = _parallel_pool(
                delayed(worker)(
                    num_iters - numIter,
                    methods_mapper,
                    dgp_name,
                    dgp_params,
                    sLength=sLength,
                    rebal=rebal,
                )
                for numIter in range(num_iters)
            )
        # for numIter in range(numIters):
        #     for func_name in methods_mapper:
        #         stats[func_name].loc[numIter] = stats_iter[numIter][
        #         func_name]
        clusters = []
        inner_weights = []
        for numIter in range(num_iters):
            for func_name in methods_mapper:
                returns[func_name] = pd.concat(
                    [returns[func_name], results[numIter][0][func_name]],
                    axis=1,
                )
                weights[func_name] = pd.concat(
                    [weights[func_name], results[numIter][1][func_name]],
                    axis=1,
                )
                leverage[func_name].append(results[numIter][4][func_name])
                test_target_vol[func_name].append(results[numIter][5][
                                                      func_name])

            clusters.append(results[numIter][2])
            inner_weights.append(
                results[numIter][3].reset_index().to_json(orient="index")
            )

    else:
        clusters = []
        inner_weights = []
        for numIter in range(num_iters):
            (
                returns_iter,
                weights_iter,
                cluster_iter,
                inner_weights_iter,
                _,
                _,
            ) = worker(
                num_iters - numIter,
                methods_mapper,
                dgp_name,
                dgp_params,
                sLength=sLength,
                rebal=rebal,
            )
            clusters.append(cluster_iter)
            # for func_name in methods_mapper:
            #     stats[func_name].loc[numIter] = stats_iter[func_name]
            for func_name in methods_mapper:
                returns[func_name] = pd.concat(
                    [returns[func_name], returns_iter[func_name]], axis=1
                )
                weights[func_name] = pd.concat(
                    [weights[func_name], weights_iter[func_name]], axis=1
                )
            inner_weights.append(
                inner_weights_iter.reset_index().to_json(orient="index")
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

    json.dump(inner_weights, open(f"{save_dir}/inner_weights_NMF.json", "w"))
    json.dump(clusters, open(f"{save_dir}/clusters.json", "w"))
    json.dump(leverage, open(f"{save_dir}/leverage.json", "w"))
    json.dump(test_target_vol, open(f"{save_dir}/test_target_vol.json", "w"))
    return


METHODS_MAPPER = {"IVP": getIVP_mc, "HRP": getHRP, "NMF": getNMF}

if __name__ == "__main__":
    import datetime as dt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dgp", default="cluster_mc", type=str, help="Data generating process"
    )
    parser.add_argument(
        "--n_jobs",
        default=os.cpu_count(),
        type=int,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--num_iters",
        default=int(1e4),
        type=int,
        help="Number of MC iterations",
    )

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
    elif args.dgp == "cluster_mc":
        dgp_params = {
            "n_obs": 520,
            "size1": 15,
            "mu0": [0] * 5,
            "sigma0": [
                0.08,
                0.003,
                0.01,
                0.005,
                0.007,
            ],  # (crypto, bond, stock, forex, commodities)
            "sigma1F": 0.25,
            "sLength": 260,
            "process": "garch",
        }
    else:
        raise NotImplementedError(args.dgp)

    mc_hrp(
        METHODS_MAPPER,
        args.dgp,
        dgp_params,
        n_jobs=args.n_jobs,
        num_iters=args.num_iters,
        save_dir=save_dir,
    )

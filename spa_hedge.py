import logging
import os
import pickle
import time

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from dl_portfolio.data import load_data
from dl_portfolio.backtest import cv_portfolio_perf_df
from dl_portfolio.logger import LOGGER

PERF_DIR_1 = "performance/test_final_models/ae_paper/ae" \
             "/dataset1_20230528_123124"

PERF_DIR_2 = "performance/test_final_models/ae_paper/ae" \
             "/dataset2_20230528_135044"

STRATS = ["rb_factor", "rb_factor_es_0.01", "rb_factor_cdar_0.01"]


def worker(seed, ts_signals, strats, cv_dates, port_weights, cv_results,
           train_weights, market_budget, dataset):
    index = ts_signals[strats[0]].index
    cv_folds = range(len(cv_results[0]))

    LOGGER.info(f"seed={seed}")
    ts_permut_signals = {
        strat: ts_signals[strat].sample(
            frac=1, random_state=seed
        ).set_index(index) for strat in strats
    }
    permut_signals = {
        strat: {
            cv: ts_permut_signals[strat].loc[cv_dates[cv]] for cv in cv_dates
        } for strat in strats
    }
    cv_port_weights = {}
    for port in strats:
        port_weights[port] = port_weights[port].reindex(index).fillna(
            method="ffill"
        )
        # Create permutation
        cv_port_weights[port] = {
            cv: port_weights[port].loc[cv_dates[cv]] * permut_signals[port][cv]
            for cv in cv_dates
        }

    cv_portfolio = {
        cv: {
            "returns": cv_results[0][cv]["returns"],
            "train_returns": cv_results[0][cv]["train_returns"],
            "port": {
                port: cv_port_weights[port][cv] for port in strats
            },
        }
        for cv in cv_folds
    }

    # Get portfolio returns
    port_perf, _ = cv_portfolio_perf_df(
        cv_portfolio,
        portfolios=strats,
        volatility_target=None,
        train_weights=train_weights,
        market_budget=market_budget,
        dataset=dataset,
    )
    port_perf = {strat: port_perf[strat]["total"] for strat in strats}

    return port_perf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Dataset name: dataset1 or dataset2",
    )
    parser.add_argument(
        "--n",
        default=10,
        type=int,
        help="Number of permutations"
    )
    parser.add_argument(
        "--n_jobs",
        default=os.cpu_count(),
        type=int,
        help="Number of parallel jobs"
    )
    parser.add_argument("--save", action="store_true", help="Save results")
    args = parser.parse_args()
    LOGGER.setLevel(logging.INFO)
    # ------------------------------------------------ input ------------------------------------------------
    # dataset1
    dataset = args.dataset
    # -------------------------------------------------------------------------------------------------------
    # Get necessary data
    if dataset == "dataset1":
        LOGGER.info("Run for dataset1")
        # Define paths
        perf_dir = PERF_DIR_1

        # Load data
        data, assets = load_data(dataset="dataset1")
        market_budget = pd.read_csv(
            "data/dataset1/market_budget_dataset1.csv", index_col=0
        )
        cryptos = ["BTC", "DASH", "ETH", "LTC", "XRP"]
        market_budget = pd.concat(
            [
                market_budget,
                pd.DataFrame(
                    np.array([["crypto", 1]] * len(cryptos)),
                    index=cryptos,
                    columns=market_budget.columns,
                ),
            ]
        )
        market_budget["rc"] = market_budget["rc"].astype(int)
    elif dataset == "dataset2":
        LOGGER.info("Run for dataset2")
        # Define paths
        perf_dir = PERF_DIR_2

        # Load data
        data, assets = load_data(dataset="dataset2")
        market_budget = pd.read_csv(
            "data/dataset2/market_budget_dataset2.csv", index_col=0
        )
        market_budget["rc"] = market_budget["rc"].astype(int)
    else:
        raise NotImplementedError(dataset)

    port_weights = pickle.load(open(f"{perf_dir}/portfolios_weights.p", "rb"))
    cv_results = pickle.load(open(f"{perf_dir}/cv_results.p", "rb"))
    cv_folds = range(len(cv_results[0]))
    train_weights = {
        cv: {
            strat: port_weights[strat].iloc[cv]
            for strat in list(port_weights.keys())
        }
        for cv in cv_folds
    }
    signals = pd.read_pickle(f"{perf_dir}/hedging_signals_0.p")
    cv_dates = {
        k: list(signals[STRATS[0]][k].index) for k in signals[STRATS[0]]
    }
    ts_signals = {
        strat: pd.concat(list(signals[strat].values())) for strat in STRATS
    }
    index = ts_signals[STRATS[0]].index

    t1 = time.time()
    if args.n_jobs == 1:
        results = {strat: [] for strat in STRATS}
        for i in range(args.n):
            LOGGER.info(f"Steps to go: {args.n - i}")
            port_perf = worker(
                i, ts_signals, STRATS, cv_dates, port_weights, cv_results,
                train_weights, market_budget, dataset
            )
            for strat in STRATS:
                results[strat].append(port_perf[strat])
        for strat in STRATS:
            results[strat] = pd.concat(
                results[strat], axis=1
            ).T.reset_index(drop=True).T
    else:
        with Parallel(
                n_jobs=args.n_jobs,
                backend="threading"
        ) as _parallel_pool:
            port_perf = _parallel_pool(
                delayed(worker)(
                    i, ts_signals, STRATS, cv_dates, port_weights, cv_results,
                    train_weights, market_budget, dataset
                ) for i in range(args.n)
            )
        results = {
            strat: pd.concat(
                [port_perf[i][strat] for i in range(len(port_perf))],
                axis=1
            ).T.reset_index(drop=True).T for strat in STRATS
        }
    t2 = time.time()
    LOGGER.info(f"Done in {round((t2-t1)/60, 2)} min")
    pickle.dump(results, open(f"{perf_dir}/spa_hedge_returns_{args.n}.p",
                              "wb"))

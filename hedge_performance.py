import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from dl_portfolio.data import load_data
from dl_portfolio.hedge import hedged_portfolio_weights_wrapper
from dl_portfolio.backtest import cv_portfolio_perf_df
from dl_portfolio.logger import LOGGER
from dl_portfolio.constant import METHODS_MAPPING, AVAILABLE_METHODS

DATA_BASE_DIR_1 = "activationProba/data/dataset1"
# GARCH_BASE_DIR_1 = "activationProba/output/dataset1/final"
# PERF_DIR_1 = "./performance/test_final_models/ae/dataset1_20230220_110629"

GARCH_BASE_DIR_1 = "activationProba/output/dataset1/20230407134753"
PERF_DIR_1 = "./performance/test_final_models/ae/dataset1_20230418_203337"

DATA_BASE_DIR_2 = "./activationProba/data/dataset2"
# GARCH_BASE_DIR_2 = "activationProba/output/dataset2/final"
# PERF_DIR_2 = "./performance/test_final_models/ae/dataset2_20230220_111557"
PERF_DIR_2 = "./performance/test_final_models/ae/dataset2_20230417_234323"
GARCH_BASE_DIR_2 = "activationProba/output/dataset2/20230407170225"

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
        "--method",
        default="calibrated_exceedance",
        type=str,
        help="Method to compute optimal threshold",
    )
    parser.add_argument(
        "--n_jobs",
        default=os.cpu_count(),
        type=int,
        help="Number of parallel jobs",
    )
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--show", action="store_true", help="Show performance")
    parser.add_argument(
        "-vol",
        "--volatility_target",
        help="Volatility target",
        dest="volatility_target",
        default=0.05,
    )
    args = parser.parse_args()

    if args.volatility_target == "":
        args.volatility_target = None

    assert args.method in AVAILABLE_METHODS, args.method

    # ------------------------------------------------ input ------------------------------------------------
    # dataset1
    dataset = args.dataset
    # -------------------------------------------------------------------------------------------------------
    # Get necessary data
    if dataset == "dataset1":
        LOGGER.info("Run for dataset1")
        # Define paths
        data_base_dir = DATA_BASE_DIR_1
        garch_base_dir = GARCH_BASE_DIR_1
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
        data_base_dir = DATA_BASE_DIR_2
        garch_base_dir = GARCH_BASE_DIR_2
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
    returns = data.pct_change(1).dropna()

    strats = [
        s for s in list(port_weights.keys()) if s in ["aerp", "rb_factor"]
    ]
    cv_folds = range(sum([cv.isdigit() for cv in os.listdir(data_base_dir)]))
    LOGGER.info(f"Method for optimal threshold is: {args.method}")
    if args.n_jobs > 1:
        LOGGER.info(f"Compute weights with {args.n_jobs} jobs...")
        with Parallel(n_jobs=args.n_jobs) as _parallel_pool:
            cv_results = _parallel_pool(
                delayed(hedged_portfolio_weights_wrapper)(
                    cv,
                    returns,
                    f"{garch_base_dir}/{cv}",
                    f"{data_base_dir}/{cv}",
                    port_weights,
                    strats=strats,
                    method=args.method,
                )
                for cv in cv_folds
            )
        cv_results = {cv: res for (cv, res) in cv_results}
    else:
        LOGGER.info(f"n_jobs = 1: compute weights sequentially...")
        cv_results = {}
        for cv in cv_folds:
            _, cv_results[cv] = hedged_portfolio_weights_wrapper(
                cv,
                returns,
                f"{garch_base_dir}/{cv}",
                f"{data_base_dir}/{cv}",
                port_weights,
                strats=strats,
                method=args.method,
            )
    LOGGER.info("Done.")

    pickle.dump(cv_results, open(f"{perf_dir}/hedged_results.p", "wb"))
    # Now parse cv portfolio weights and train weights
    LOGGER.info("Portfolio returns...")
    cv_portfolio = {
        cv: {
            "returns": cv_results[cv]["returns"],
            "train_returns": cv_results[cv]["train_returns"],
            "port": {
                port: cv_results[cv]["port"][port]
                for port in strats
                # if port not in ["equal", "equal_classs"]
            },
        }
        for cv in cv_folds
    }
    train_weights = {
        cv: {
            strat: port_weights[strat].iloc[cv]
            for strat in list(port_weights.keys())
        }
        for cv in cv_folds
    }
    # Get portfolio returns
    port_perf, leverage = cv_portfolio_perf_df(
        cv_portfolio,
        portfolios=strats,
        volatility_target=args.volatility_target,
        train_weights=train_weights,
        market_budget=market_budget,
        dataset=args.dataset,
    )
    LOGGER.info("Done.")

    # Format final results
    port_returns = pd.DataFrame()
    for p in strats:
        port_returns[p] = port_perf[p]["total"].iloc[:, 0]
    new_port_weights = {
        strat: {cv: cv_results[cv]["port"][strat] for cv in cv_results}
        for strat in strats
    }
    signals = {
        strat: {cv: cv_results[cv]["signal"][strat] for cv in cv_results}
        for strat in strats
    }

    if args.save:
        LOGGER.info("Saving results... ")
        port_returns.to_csv(
            f"{perf_dir}/portfolios_returns_hedged_{METHODS_MAPPING[args.method]}.csv"
        )
        leverage.to_csv(
            f"{perf_dir}/leverage_hedged_{METHODS_MAPPING[args.method]}.csv"
        )
        pickle.dump(
            signals,
            open(
                f"{perf_dir}/hedging_signals_{METHODS_MAPPING[args.method]}.p",
                "wb",
            ),
        )
        pickle.dump(
            new_port_weights,
            open(
                f"{perf_dir}/portfolios_weights_hedged_{METHODS_MAPPING[args.method]}.p",
                "wb",
            ),
        )

    if args.show:
        LOGGER.info("Show performance... ")
        or_port_perf = pd.read_csv(
            f"{perf_dir}/portfolios_returns.csv", index_col=0
        )
        or_port_perf.index = pd.to_datetime(or_port_perf.index)
        for strat in strats:
            plt.figure(figsize=(20, 10))
            plt.plot(np.cumsum(or_port_perf[strat]), label="or")
            plt.plot(np.cumsum(port_returns[strat]), label="hedge")
            plt.title(strat)
            plt.legend()
            plt.show()
    LOGGER.info("Done.")

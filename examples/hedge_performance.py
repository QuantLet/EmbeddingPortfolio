import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from dl_portfolio.ae_data import load_data
from dl_portfolio.hedge import hedged_portfolio_weights_wrapper
from dl_portfolio.backtest import cv_portfolio_perf_df
from dl_portfolio.logger import LOGGER
from dl_portfolio.constant import METHODS_MAPPING, AVAILABLE_METHODS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        help="Dataset name: dataset1 or dataset2")
    parser.add_argument("--method",
                        default="hedged_strat_cum_excess_return_cluster",
                        type=str,
                        help="Method to compute optimal threshold")
    parser.add_argument("--n_jobs",
                        default=os.cpu_count() - 1,
                        type=int,
                        help="Number of parallel jobs")
    parser.add_argument("--save",
                        action='store_true',
                        help="Save results")
    parser.add_argument("--show",
                        action='store_true',
                        help="Show performance")
    args = parser.parse_args()
    assert args.method in AVAILABLE_METHODS, args.method

    # ------------------------------------------------ input ------------------------------------------------
    # dataset1
    dataset = args.dataset
    # -------------------------------------------------------------------------------------------------------
    # Get necessary data
    if dataset == "dataset1":
        LOGGER.info("Run for dataset1")
        # Define paths
        ae_base_dir = "final_models/ae/dataset/m_0_bond_nbb_resample_bl_60_seed_4_1640003029645042"
        garch_base_dir = "./activationProba/output/dataset1/test/20220222142525_ec2_run_2"
        perf_dir = "performance/test_final_models/ae/dataset1_20220222_155538"
        # Load data
        data, assets = load_data(dataset="bond", crix=False, crypto_assets=["BTC", "DASH", "ETH", "LTC", "XRP"])
        market_budget = pd.read_csv("data/market_budget_bond.csv", index_col=0)
        cryptos = ["BTC", "DASH", "ETH", "LTC", "XRP"]
        market_budget = pd.concat([market_budget, pd.DataFrame(np.array([["crypto", 1]] * len(cryptos)),
                                                               index=cryptos,
                                                               columns=market_budget.columns)])
        # market_budget = market_budget.drop("CRIX")
        market_budget["rc"] = market_budget["rc"].astype(int)
    elif dataset == "dataset2":
        LOGGER.info("Run for dataset2")
        # Define paths
        ae_base_dir = "final_models/ae/dataset/m_0_raffinot_bloomberg_comb_update_2021_nbb_resample_bl_60_seed_1_1645050812225231"
        garch_base_dir = "./activationProba/output/dataset2/test/20220222100230_ec2_run_1"
        perf_dir = "performance/test_final_models/ae/dataset2_20220222_162628"
        # Load data
        data, assets = load_data(dataset="raffinot_bloomberg_comb_update_2021")
        market_budget = pd.read_csv('data/market_budget_raffinot_multiasset.csv', index_col=0)
        market_budget['rc'] = market_budget['rc'].astype(int)
    else:
        raise NotImplementedError(dataset)

    port_weights = pickle.load(open(f"{perf_dir}/portfolios_weights.p", "rb"))

    linear_activation = pd.read_csv(f"./activationProba/data/{dataset}/linear_activation.csv", index_col=0)
    linear_activation.index = pd.to_datetime(linear_activation.index)
    target = (linear_activation <= 0).astype(int)
    returns = data.pct_change(1).dropna()
    cluster_assignment = pickle.load(open(f"{perf_dir}/cluster_assignment.p", "rb"))
    # Reorder series
    for cv in cluster_assignment:
        cluster_assignment[cv] = cluster_assignment[cv].loc[assets]

    strats = [s for s in list(port_weights.keys()) if "ae" in s]
    cv_folds = list(cluster_assignment.keys())

    if args.n_jobs > 1:
        LOGGER.info(f"Compute weights with {args.n_jobs} jobs...")
        with Parallel(n_jobs=args.n_jobs) as _parallel_pool:
            cv_results = _parallel_pool(
                delayed(hedged_portfolio_weights_wrapper)(cv, returns, cluster_assignment[cv],
                                                          f"{garch_base_dir}/{cv}",
                                                          port_weights, strats=strats, method=args.method)
                for cv in cv_folds
            )
        # cv_results = {cv_results[i][0]: cv_results[i][1] for i in range(len(cv_results))}
        cv_results = {cv: res for (cv, res) in cv_results}
        LOGGER.info("Done.")
    else:
        LOGGER.info(f"n_jobs = 1: compute weights sequentially...")
        cv_results = {}
        for cv in cv_folds:
            cv_results[cv] = hedged_portfolio_weights_wrapper(cv, returns, cluster_assignment[cv],
                                                              f"{garch_base_dir}/{cv}",
                                                              port_weights, strats=strats, method=args.method)
        LOGGER.info("Done.")

    LOGGER.info("Portfolio returns...")
    # Now parse cv portfolio weights and train weights
    cv_portfolio = {
        cv: {
            "returns": cv_results[cv]["returns"],
            "train_returns": cv_results[cv]["train_returns"],
            "port": {port: cv_results[cv]["port"][port] for port in strats
                     # if port not in ["equal", "equal_classs"]
                     }
        } for cv in cv_folds
    }
    train_weights = {
        cv: {
            strat: port_weights[strat].iloc[cv] for strat in list(port_weights.keys())
        } for cv in cv_folds
    }
    # Get portfolio returns
    port_perf, leverage = cv_portfolio_perf_df(cv_portfolio, portfolios=strats, train_weights=train_weights)
    LOGGER.info("Done.")

    # Format final results
    port_returns = pd.DataFrame()
    for p in strats:
        port_returns[p] = port_perf[p]['total'].iloc[:, 0]
    new_port_weights = {
        strat: {cv: cv_results[cv]["port"][strat] for cv in cv_results} for strat in strats
    }
    signals = {
        strat: {cv: cv_results[cv]["signal"][strat] for cv in cv_results} for strat in strats
    }

    if args.save:
        LOGGER.info('Saving results... ')
        port_returns.to_csv(f"{perf_dir}/portfolios_returns_hedged_{METHODS_MAPPING[args.method]}.csv")
        leverage.to_csv(f"{perf_dir}/leverage_hedged_{METHODS_MAPPING[args.method]}.csv")
        pickle.dump(signals, open(f"{perf_dir}/hedging_signals_{METHODS_MAPPING[args.method]}.p", "wb"))
        pickle.dump(new_port_weights,
                    open(f"{perf_dir}/portfolios_weights_hedged_{METHODS_MAPPING[args.method]}.p", "wb"))

    if args.show:
        LOGGER.info('Show performance... ')
        or_port_perf = pd.read_csv(f"{perf_dir}/portfolios_returns.csv",
                                   index_col=0)
        or_port_perf.index = pd.to_datetime(or_port_perf.index)
        for strat in strats:
            plt.figure(figsize=(20, 10))
            plt.plot(np.cumsum(or_port_perf[strat]), label="or")
            plt.plot(np.cumsum(port_returns[strat]), label="hedge")
            plt.title(strat)
            plt.legend()
            plt.show()
    LOGGER.info("Done.")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os
from dl_portfolio.ae_data import load_data
import sys, pickle
from dl_portfolio.utils import load_result
from dl_portfolio.weights import portfolio_weights
from dl_portfolio.cluster import get_cluster_assignment
from dl_portfolio.hedge import hedged_portfolio_weights, hedged_portfolio_weights_wrapper
from dl_portfolio.constant import BASE_FACTOR_ORDER_BOND, BASE_FACTOR_ORDER_RAFFINOT
from dl_portfolio.logger import LOGGER

if __name__ == "__main__":
    # ------------------------------------------------ input ------------------------------------------------
    # dataset1
    dataset = "dataset2"
    # -------------------------------------------------------------------------------------------------------
    # Get necessary data
    if dataset == "dataset1":
        # Define paths
        ae_base_dir = "final_models/ae/dataset/m_0_bond_nbb_resample_bl_60_seed_4_1640003029645042"
        garch_base_dir = "./activationProba/output/dataset1/test/20220221132420_ec2_run_1"
        perf_dir = "performance/test_final_models/ae/dataset1_20220221_232911"
        # Load data
        data, assets = load_data(dataset="bond", crix=False, crypto_assets=["BTC", "DASH", "ETH", "LTC", "XRP"])
        market_budget = pd.read_csv('data/market_budget_bond.csv', index_col=0)
        cryptos = ['BTC', 'DASH', 'ETH', 'LTC', 'XRP']
        market_budget = pd.concat([market_budget, pd.DataFrame(np.array([['crypto', 1]] * len(cryptos)),
                                                               index=cryptos,
                                                               columns=market_budget.columns)])
        # market_budget = market_budget.drop('CRIX')
        market_budget['rc'] = market_budget['rc'].astype(int)
    elif dataset == "dataset2":
        # Define paths
        ae_base_dir = "final_models/ae/dataset/m_0_raffinot_bloomberg_comb_update_2021_nbb_resample_bl_60_seed_1_1645050812225231"
        garch_base_dir = "./activationProba/output/dataset2/test/20220222100230_ec2_run_1"
        perf_dir = "performance/test_final_models/ae/dataset2_20220222_135813"
        # Load data
        data, assets = load_data(dataset="raffinot_bloomberg_comb_update_2021")
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
    cv_results = {}
    for cv in cv_folds:
        LOGGER.info(f"CV: {cv}")
        cv_results[cv] = hedged_portfolio_weights_wrapper(cv, returns, cluster_assignment[cv], f"{garch_base_dir}/{cv}",
                                                          port_weights, strats=strats, window=None)

import datetime as dt
import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, preprocessing

from dl_portfolio.backtest import cv_portfolio_perf, bar_plot_weights, backtest_stats, plot_perf, \
    get_average_perf, get_ts_weights, get_cv_results, get_dl_average_weights
from dl_portfolio.cluster import get_cluster_labels, consensus_matrix, rand_score_permutation, \
    assign_cluster_from_consmat
from dl_portfolio.evaluate import pred_vs_true_plot, average_prediction, average_prediction_cv
from dl_portfolio.logger import LOGGER
from dl_portfolio.constant import BASE_FACTOR_ORDER_DATASET2, BASE_FACTOR_ORDER_DATASET1

# PORTFOLIOS = ['equal', 'markowitz', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']
# STRAT = ['equal', 'markowitz', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']

# AE bond
# PORTFOLIOS = ['equal', 'equal_class', 'markowitz', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']
# STRAT = ['equal', 'equal_class', 'markowitz', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']

# AE raffinot
PORTFOLIOS = ['equal', 'equal_class', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']
STRAT = ['equal', 'equal_class', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']

# PORTFOLIOS = ['equal', 'equal_class', 'aerp', 'aeerc', 'ae_rp_c', 'aeaa']
# STRAT = ['equal', 'equal_class', 'aerp', 'aeerc', 'ae_rp_c', 'aeaa']

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",
                        type=str,
                        help="Experiments dir")
    parser.add_argument("--model_type",
                        default='ae',
                        type=str,
                        help="ae or nmf")
    parser.add_argument("--test_set",
                        default='val',
                        type=str,
                        help="val or test")
    parser.add_argument("--n_jobs",
                        default=2 * os.cpu_count(),
                        type=int,
                        help="Number of parallel jobs")
    parser.add_argument("--window",
                        default=252,
                        type=int,
                        help="Window size for portfolio optimisation")
    parser.add_argument("--show",
                        action='store_true',
                        help="Show plots")
    parser.add_argument("--save",
                        action='store_true',
                        help="Save results")
    parser.add_argument("--legend",
                        action='store_true',
                        help="Add legend to plots")
    parser.add_argument("-v",
                        "--verbose",
                        help="Be verbose",
                        action="store_const",
                        dest="loglevel",
                        const=logging.INFO,
                        default=logging.WARNING)
    parser.add_argument('-d',
                        '--debug',
                        help="Debugging statements",
                        action="store_const",
                        dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.WARNING)
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    LOGGER.setLevel(args.loglevel)
    meta = vars(args)
    if args.save:
        save_dir = f"backtest/{args.test_set}_{args.base_dir}" + '_' + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        LOGGER.info(f"Saving result to {save_dir}")
        # os.makedirs(f"{save_dir}/cv_plots/")

        meta['save_dir'] = save_dir
        json.dump(meta, open(f"{save_dir}/meta.json", "w"))

    EVALUATION = {'model': {}, 'cluster': {}}

    # Load paths
    models = os.listdir(args.base_dir)
    paths = [f"{args.base_dir}/{d}" for d in models if os.path.isdir(f"{args.base_dir}/{d}") and d[0] != "."]
    n_folds = os.listdir(paths[0])
    n_folds = sum([d.isdigit() for d in n_folds])
    sys.path.append(paths[0])

    if args.model_type == "ae":
        import ae_config as config

        assert "ae" in config.model_type
    elif args.model_type == "nmf":
        import nmf_config as config

        assert "nmf" in config.model_type
    else:
        raise ValueError(f"run '{args.run}' is not implemented. Shoule be 'ae' or 'kmeans' or 'nmf'")

    # Load Market budget
    if config.dataset == 'bond':
        market_budget = pd.read_csv('data/market_budget_dataset1.csv', index_col=0)
        cryptos = ['BTC', 'DASH', 'ETH', 'LTC', 'XRP']
        market_budget = pd.concat([market_budget, pd.DataFrame(np.array([['crypto', 1]] * len(cryptos)),
                                                               index=cryptos,
                                                               columns=market_budget.columns)])
        # market_budget = market_budget.drop('CRIX')
        market_budget['rc'] = market_budget['rc'].astype(int)
    elif config.dataset in ["raffinot_multi_asset", "raffinot_bloomberg_comb_update_2021"]:
        market_budget = pd.read_csv('data/market_budget_dataset2.csv', index_col=0)
        market_budget['rc'] = market_budget['rc'].astype(int)
    elif config.dataset == 'cac':
        market_budget = pd.read_csv('data/market_budget_cac.csv', index_col=0)
    else:
        raise NotImplementedError()

    if config.dataset == "dataset1":
        CLUSTER_NAMES = BASE_FACTOR_ORDER_DATASET1
    elif config.dataset == "dataset2":
        CLUSTER_NAMES = BASE_FACTOR_ORDER_DATASET2
    else:
        raise NotImplementedError()

    # Main loop to get results
    cv_results = {}
    train_cov = {}
    test_cov = {}
    port_perf = {}

    LOGGER.info("Starting main loop...")
    for i, path in enumerate(paths):
        LOGGER.info(len(paths) - i)
        if i == 0:
            portfolios = PORTFOLIOS
        else:
            portfolios = [p for p in PORTFOLIOS if 'ae' in p]  # ['aerp', 'aeerc', 'ae_rp_c']
        cv_results[i] = get_cv_results(path,
                                       args.test_set,
                                       n_folds,
                                       dataset=config.dataset,
                                       portfolios=portfolios,
                                       market_budget=market_budget,
                                       window=args.window,
                                       n_jobs=args.n_jobs,
                                       ae_config=config)
    LOGGER.info("Done.")

    # Get average weights for AE portfolio across runs
    port_weights = get_dl_average_weights(cv_results)
    # Build dictionary for cv_portfolio_perf
    cv_returns = {}
    for cv in cv_results[0]:
        cv_returns[cv] = cv_results[0][cv]['returns'].copy()
        date = cv_results[0][cv]['returns'].index[0]
        for port in PORTFOLIOS:
            if port not in ['equal', 'equal_class'] and 'ae' not in port:
                weights = pd.DataFrame(cv_results[0][cv]['port'][port]).T
                weights.index = [date]
                port_weights[cv][port] = weights
    cv_portfolio = {
        cv: {
            'returns': cv_returns[cv],
            'train_returns': cv_results[0][cv]['train_returns'],
            'port': {port: port_weights[cv][port].values for port in port_weights[cv]
                     # if port not in ['equal', 'equal_classs']
                     }
        } for cv in cv_returns
    }
    port_perf, leverage = cv_portfolio_perf(cv_portfolio, portfolios=PORTFOLIOS, volatility_target=0.05,
                                            market_budget=market_budget)

    K = cv_results[i][0]['loading'].shape[-1]
    CV_DATES = [str(cv_results[0][cv]['returns'].index[0].date()) for cv in range(n_folds)]
    ASSETS = list(cv_results[i][0]['returns'].columns)

    # Get portfolio weights time series
    # port_weights = {}
    # for p in PORTFOLIOS:
    #     if p not in ['equal', 'equal_class']:
    #         port_weights[p] = get_ts_weights(cv_results, port=p)
    port_weights = get_ts_weights(port_weights)
    # Get average perf across runs
    ann_perf = pd.DataFrame()
    for p in PORTFOLIOS:
        ann_perf[p] = port_perf[p]['total'].iloc[:, 0]

    # Some renaming
    LOGGER.info("Get cluster assignment...")
    cv_labels = {}
    for cv in range(n_folds):
        cv_labels[cv] = {}
        for i in cv_results:
            c, cv_labels[cv][i] = get_cluster_labels(cv_results[i][cv]['embedding'])

    cluster_assignment = {}
    for cv in cv_labels:
        cons_mat = consensus_matrix(cv_labels[cv], reorder=True, method="single")
        cluster_assignment[cv] = assign_cluster_from_consmat(cons_mat, CLUSTER_NAMES, t=0)
    LOGGER.info("Done.")
    if args.save:
        LOGGER.info('Saving performance... ')
        ann_perf.to_csv(f"{save_dir}/portfolios_returns.csv")
        leverage.to_csv(f"{save_dir}/leverage.csv")
        pickle.dump(port_weights, open(f"{save_dir}/portfolios_weights.p", "wb"))
        pickle.dump(cluster_assignment, open(f"{save_dir}/cluster_assignment.p", "wb"))

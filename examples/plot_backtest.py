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
    get_average_perf, get_ts_weights, get_cv_results
from dl_portfolio.cluster import get_cluster_labels, consensus_matrix, rand_score_permutation
from dl_portfolio.evaluate import pred_vs_true_plot, average_prediction, average_prediction_cv
from dl_portfolio.logger import LOGGER

# PORTFOLIOS = ['equal', 'markowitz', 'ae_ivp', 'hrp', 'hcaa', 'ae_rp', 'ae_rp_c', 'aeaa', 'kmaa']
# STRAT = ['equal', 'markowitz', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']
PORTFOLIOS = ['equal', 'equal_class', 'ae_ivp', 'hrp', 'hcaa', 'ae_rp', 'ae_rp_c', 'aeaa', 'kmaa']
STRAT = ['equal', 'equal_class', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']

# PORTFOLIOS = ['ae_rp_c']
# STRAT = ['ae_rp_c']


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",
                        type=str,
                        help="Backtest result dir")
    parser.add_argument("--save_dir",
                        type=str,
                        nargs="?",
                        const=None,
                        default=None,
                        help="Destination dir")
    parser.add_argument("--save",
                        action='store_true',
                        help="Save results")
    parser.add_argument("--show",
                        action='store_true',
                        help="Show plots")
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
        if args.save_dir:
            save_dir = f"performance/{args.save_dir}"
        else:
            save_dir = f"performance/{args.log_dir}"

    # load result
    meta = json.load(open(f"performance/{args.log_dir}/meta.json", "r"))
    ann_perf = pd.read_csv(f"performance/{args.log_dir}/portfolios_returns.csv")
    port_weights = pickle.load(open(f"performance/{args.log_dir}/portfolios_weights.p", "rb"))
    ASSETS = port_weights['hrp'].columns
    stats = pd.read_csv(f"performance/{args.log_dir}/backtest_stats.csv", index_col=0)

    ##########################
    # Backtest performance

    if args.save:
        LOGGER.info('Saving performance... ')
        legend = False
        plot_perf(ann_perf, strategies=STRAT,
                  save_path=f"{save_dir}/performance_all.png",
                  show=args.show, legend=legend)
        plot_perf(ann_perf, strategies=[p for p in STRAT if p not in ['aerp', 'aeerc']],
                  save_path=f"{save_dir}/performance_ae_rp_c_vs_all.png",
                  show=args.show, legend=legend)
        plot_perf(ann_perf, strategies=['hrp', 'aerp'], save_path=f"{save_dir}/performance_hrp_aerp.png",
                  show=args.show, legend=legend)
        plot_perf(ann_perf, strategies=['hcaa', 'aeerc'], save_path=f"{save_dir}/performance_hcaa_aeerc.png",
                  show=args.show, legend=legend)
        plot_perf(ann_perf, strategies=['hrp', 'ae_rp_c'], save_path=f"{save_dir}/performance_hrp_aeerc_cluster.png",
                  show=args.show, legend=legend)
        plot_perf(ann_perf, strategies=['hcaa', 'ae_rp_c'], save_path=f"{save_dir}/performance_hcaa_aeerc_cluster.png",
                  show=args.show, legend=legend)
        if 'markowitz' in PORTFOLIOS:
            plot_perf(ann_perf, strategies=['markowitz', 'ae_rp_c'],
                      save_path=f"{save_dir}/performance_markowitz_aeerc_cluster.png",
                      show=args.show, legend=legend)

        if 'shrink_markowitz' in PORTFOLIOS:
            bar_plot_weights(port_weights['shrink_markowitz'], save_path=f"{save_dir}/weights_shrink_markowitz.png",
                             show=args.show)
        if 'markowitz' in PORTFOLIOS:
            bar_plot_weights(port_weights['markowitz'], save_path=f"{save_dir}/weights_markowitz.png", show=args.show)
        bar_plot_weights(port_weights['hrp'], save_path=f"{save_dir}/weights_hrp.png", show=args.show)
        bar_plot_weights(port_weights['aerp'], save_path=f"{save_dir}/weights_aerp.png", show=args.show)
        bar_plot_weights(port_weights['hcaa'], save_path=f"{save_dir}/weights_hcaa.png", show=args.show)
        bar_plot_weights(port_weights['aeerc'], save_path=f"{save_dir}/weights_aeerc.png", show=args.show)
        bar_plot_weights(port_weights['hrp'], save_path=f"{save_dir}/weights_hrp.png", show=args.show)
        bar_plot_weights(port_weights['ae_rp_c'], save_path=f"{save_dir}/weights_aeerc_cluster.png", show=args.show)
        bar_plot_weights(port_weights['aeaa'], save_path=f"{save_dir}/weights_aeaa.png", show=args.show)
    else:
        plot_perf(ann_perf, strategies=[p for p in STRAT if p not in ['aerp', 'aeerc']],show=args.show, legend=True)
        bar_plot_weights(port_weights['hrp'], show=args.show)
        bar_plot_weights(port_weights['hcaa'], show=args.show)
        bar_plot_weights(port_weights['ae_rp_c'],  show=args.show)

    # Plot excess return
    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['aerp'] + 1) - np.cumprod(ann_perf['hrp'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_hrp_aerp.png", bbox_inches='tight', transparent=True)

    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['aeerc'] + 1) - np.cumprod(ann_perf['hcaa'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_hcaa_aeerc.png", bbox_inches='tight', transparent=True)

    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['ae_rp_c'] + 1) - np.cumprod(ann_perf['hcaa'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_hcaa_aeerc_cluster.png", bbox_inches='tight', transparent=True)

    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['ae_rp_c'] + 1) - np.cumprod(ann_perf['hrp'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_hrp_aeerc_cluster.png", bbox_inches='tight', transparent=True)

    if 'markowitz' in PORTFOLIOS:
        plt.figure(figsize=(20, 10))
        plt.plot(np.cumprod(ann_perf['ae_rp_c'] + 1) - np.cumprod(ann_perf['markowitz'] + 1))
        if args.save:
            plt.savefig(f"{save_dir}/excess_performance_markowitz_aeerc_cluster.png", bbox_inches='tight',
                        transparent=True)

    # Plot one cv weight
    cv = 0
    date = str(port_weights['hrp'].index[cv].date())

    plt.figure(figsize=(14, 7))
    plt.bar(ASSETS, port_weights['hrp'].iloc[cv].values, label='hrp')
    plt.bar(ASSETS, port_weights['aerp'].iloc[cv].values, label='aerp')
    plt.ylim([0, 0.9])
    x = plt.xticks(rotation=45)
    if args.save:
        plt.savefig(f"{save_dir}/weights_hrp_aerp.png", bbox_inches='tight', transparent=True)

    plt.figure(figsize=(14, 7))
    plt.bar(ASSETS, port_weights['hcaa'].iloc[cv].values, label='hcaa')
    plt.bar(ASSETS, port_weights['aeerc'].iloc[cv].values, label='aeerc')
    plt.ylim([0, 0.9])
    x = plt.xticks(rotation=45)
    if args.save:
        plt.savefig(f"{save_dir}/weights_hcaa_aeerc.png", bbox_inches='tight', transparent=True)

    plt.figure(figsize=(14, 7))
    plt.bar(ASSETS, port_weights['hrp'].iloc[cv].values, label='hrp')
    plt.bar(ASSETS, port_weights['ae_rp_c'].iloc[cv].values, label='ae_rp_c')
    plt.ylim([0, 0.9])
    x = plt.xticks(rotation=45)
    if args.save:
        plt.savefig(f"{save_dir}/weights_hrp_aeerc_cluster.png", bbox_inches='tight', transparent=True)

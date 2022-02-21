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
from dl_portfolio.constant import BASE_FACTOR_ORDER_RAFFINOT, BASE_FACTOR_ORDER_BOND

# PORTFOLIOS = ['equal', 'markowitz', 'ae_ivp', 'hrp', 'hcaa', 'ae_rp', 'ae_rp_c', 'aeaa', 'kmaa']
# STRAT = ['equal', 'markowitz', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']

# AE bond
# PORTFOLIOS = ['equal', 'equal_class', 'markowitz', 'ae_ivp', 'hrp', 'hcaa', 'ae_rp', 'ae_rp_c', 'aeaa', 'kmaa']
# STRAT = ['equal', 'equal_class', 'markowitz', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']

# AE raffinot
PORTFOLIOS = ['equal', 'equal_class', 'ae_ivp', 'hrp', 'hcaa', 'ae_rp', 'ae_rp_c', 'aeaa', 'kmaa']
STRAT = ['equal', 'equal_class', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']

# PORTFOLIOS = ['equal', 'equal_class', 'ae_ivp', 'ae_rp', 'ae_rp_c', 'aeaa']
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
                        default=250,
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
        save_dir = f"performance/{args.test_set}_{args.base_dir}" + '_' + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        market_budget = pd.read_csv('data/market_budget_bond.csv', index_col=0)
        cryptos = ['BTC', 'DASH', 'ETH', 'LTC', 'XRP']
        market_budget = pd.concat([market_budget, pd.DataFrame(np.array([['crypto', 1]] * len(cryptos)),
                                                               index=cryptos,
                                                               columns=market_budget.columns)])
        # market_budget = market_budget.drop('CRIX')
        market_budget['rc'] = market_budget['rc'].astype(int)
    elif config.dataset in ["raffinot_multi_asset", "raffinot_bloomberg_comb_update_2021"]:
        market_budget = pd.read_csv('data/market_budget_raffinot_multiasset.csv', index_col=0)
        market_budget['rc'] = market_budget['rc'].astype(int)
    elif config.dataset == 'cac':
        market_budget = pd.read_csv('data/market_budget_cac.csv', index_col=0)
    else:
        raise NotImplementedError()

    if config.dataset == "bond":
        CLUSTER_NAMES = BASE_FACTOR_ORDER_BOND
    elif config.dataset == "raffinot_bloomberg_comb_update_2021":
        CLUSTER_NAMES = BASE_FACTOR_ORDER_RAFFINOT
    else:
        raise NotImplementedError()

    # Main loop to get results
    cv_results = {}
    train_cov = {}
    test_cov = {}
    port_perf = {}

    for i, path in enumerate(paths):
        LOGGER.info(len(paths) - i)
        if i == 0:
            portfolios = PORTFOLIOS
        else:
            portfolios = [p for p in PORTFOLIOS if 'ae' in p]  # ['ae_ivp', 'ae_rp', 'ae_rp_c']
        cv_results[i] = get_cv_results(path,
                                       args.test_set,
                                       n_folds,
                                       dataset=config.dataset,
                                       portfolios=portfolios,
                                       market_budget=market_budget,
                                       window=args.window,
                                       n_jobs=args.n_jobs,
                                       ae_config=config)

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
    port_weights['aerp'] = port_weights['ae_ivp'].copy()
    port_weights['aeerc'] = port_weights['ae_rp'].copy()
    ann_perf['aerp'] = ann_perf.loc[:, 'ae_ivp'].values
    ann_perf['aeerc'] = ann_perf.loc[:, 'ae_rp'].values

    port_weights.pop('ae_ivp')
    port_weights.pop('ae_rp')
    ann_perf.drop(['ae_ivp', 'ae_rp'], axis=1, inplace=True)

    ##########################
    # Backtest performance
    if args.save:
        LOGGER.info('Saving performance... ')
        ann_perf.to_csv(f"{save_dir}/portfolios_returns.csv")
        leverage.to_csv(f"{save_dir}/leverage.csv")
        pickle.dump(port_weights, open(f"{save_dir}/portfolios_weights.p", "wb"))
        plot_perf(ann_perf, strategies=STRAT,
                  save_path=f"{save_dir}/performance_all.png",
                  show=args.show, legend=args.legend)
        plot_perf(ann_perf, strategies=[p for p in STRAT if p not in ['aerp', 'aeerc']],
                  save_path=f"{save_dir}/performance_ae_rp_c_vs_all.png",
                  show=args.show, legend=args.legend)
        if 'hrp' in PORTFOLIOS:
            plot_perf(ann_perf, strategies=['hrp', 'aerp'], save_path=f"{save_dir}/performance_hrp_aerp.png",
                      show=args.show, legend=args.legend)
            plot_perf(ann_perf, strategies=['hrp', 'ae_rp_c'],
                      save_path=f"{save_dir}/performance_hrp_aeerc_cluster.png",
                      show=args.show, legend=args.legend)
        if 'hcaa' in PORTFOLIOS:
            plot_perf(ann_perf, strategies=['hcaa', 'aeerc'], save_path=f"{save_dir}/performance_hcaa_aeerc.png",
                      show=args.show, legend=args.legend)
            plot_perf(ann_perf, strategies=['hcaa', 'ae_rp_c'],
                      save_path=f"{save_dir}/performance_hcaa_aeerc_cluster.png",
                      show=args.show, legend=args.legend)
        if 'markowitz' in PORTFOLIOS:
            plot_perf(ann_perf, strategies=['markowitz', 'ae_rp_c'],
                      save_path=f"{save_dir}/performance_markowitz_aeerc_cluster.png",
                      show=args.show, legend=args.legend)
        if 'shrink_markowitz' in PORTFOLIOS:
            bar_plot_weights(port_weights['shrink_markowitz'], save_path=f"{save_dir}/weights_shrink_markowitz.png",
                             show=args.show)
        if 'markowitz' in PORTFOLIOS:
            bar_plot_weights(port_weights['markowitz'], save_path=f"{save_dir}/weights_markowitz.png", show=args.show)
        if 'hcaa' in PORTFOLIOS:
            bar_plot_weights(port_weights['hcaa'], save_path=f"{save_dir}/weights_hcaa.png", show=args.show,
                             legend=args.legend)
        if 'hrp' in PORTFOLIOS:
            bar_plot_weights(port_weights['hrp'], save_path=f"{save_dir}/weights_hrp.png", show=args.show,
                             legend=args.legend)
        bar_plot_weights(port_weights['aerp'], save_path=f"{save_dir}/weights_aerp.png", show=args.show,
                         legend=args.legend)
        bar_plot_weights(port_weights['aeerc'], save_path=f"{save_dir}/weights_aeerc.png", show=args.show,
                         legend=args.legend)
        bar_plot_weights(port_weights['ae_rp_c'], save_path=f"{save_dir}/weights_aeerc_cluster.png", show=args.show,
                         legend=args.legend)
        bar_plot_weights(port_weights['aeaa'], save_path=f"{save_dir}/weights_aeaa.png", show=args.show,
                         legend=args.legend)
    else:
        plot_perf(ann_perf, strategies=STRAT, show=args.show, legend=args.legend)
        if 'hrp' in PORTFOLIOS:
            plot_perf(ann_perf, strategies=['hrp', 'aerp'], show=args.show, legend=args.legend)
            bar_plot_weights(port_weights['hrp'], show=args.show)
        bar_plot_weights(port_weights['aerp'], show=args.show)
        if 'hcaa' in PORTFOLIOS:
            plot_perf(ann_perf, strategies=['hcaa', 'aeerc'], show=args.show, legend=args.legend)
            bar_plot_weights(port_weights['hcaa'], show=args.show)
        bar_plot_weights(port_weights['aeerc'], show=args.show)
        bar_plot_weights(port_weights['ae_rp_c'], show=args.show)
        bar_plot_weights(port_weights['aeaa'], show=args.show)

    # Plot excess return
    if 'hrp' in PORTFOLIOS:
        plt.figure(figsize=(20, 10))
        plt.plot(np.cumprod(ann_perf['aerp'] + 1) - np.cumprod(ann_perf['hrp'] + 1))
        if args.save:
            plt.savefig(f"{save_dir}/excess_performance_hrp_aerp.png", bbox_inches='tight', transparent=True)
        plt.figure(figsize=(20, 10))
        plt.plot(np.cumprod(ann_perf['ae_rp_c'] + 1) - np.cumprod(ann_perf['hrp'] + 1))
        if args.save:
            plt.savefig(f"{save_dir}/excess_performance_hrp_aeerc_cluster.png", bbox_inches='tight', transparent=True)

    if 'hcaa' in PORTFOLIOS:
        plt.figure(figsize=(20, 10))
        plt.plot(np.cumprod(ann_perf['aeerc'] + 1) - np.cumprod(ann_perf['hcaa'] + 1))
        if args.save:
            plt.savefig(f"{save_dir}/excess_performance_hcaa_aeerc.png", bbox_inches='tight', transparent=True)

        plt.figure(figsize=(20, 10))
        plt.plot(np.cumprod(ann_perf['ae_rp_c'] + 1) - np.cumprod(ann_perf['hcaa'] + 1))
        if args.save:
            plt.savefig(f"{save_dir}/excess_performance_hcaa_aeerc_cluster.png", bbox_inches='tight', transparent=True)

    if 'markowitz' in PORTFOLIOS:
        plt.figure(figsize=(20, 10))
        plt.plot(np.cumprod(ann_perf['ae_rp_c'] + 1) - np.cumprod(ann_perf['markowitz'] + 1))
        if args.save:
            plt.savefig(f"{save_dir}/excess_performance_markowitz_aeerc_cluster.png", bbox_inches='tight',
                        transparent=True)

    # # Plot one cv weight
    # CV = 0
    # plt.figure(figsize=(14, 7))
    # plt.bar(ASSETS, port_weights['hrp'].iloc[CV].values, label='hrp')
    # plt.bar(ASSETS, port_weights['aerp'].iloc[CV].values, label='aerp')
    # plt.legend()
    # plt.ylim([0, 0.9])
    # x = plt.xticks(rotation=45)
    # if args.save:
    #     plt.savefig(f"{save_dir}/weights_hrp_aerp.png", bbox_inches='tight', transparent=True)
    #
    # plt.figure(figsize=(14, 7))
    # plt.bar(ASSETS, port_weights['hrp'].iloc[CV].values, label='hrp')
    # plt.bar(ASSETS, port_weights['ae_rp_c'].iloc[CV].values, label='ae_rp_c')
    # plt.legend()
    # plt.ylim([0, 0.9])
    # x = plt.xticks(rotation=45)
    # if args.save:
    #     plt.savefig(f"{save_dir}/weights_hrp_aeerc_cluster.png", bbox_inches='tight', transparent=True)
    #
    # plt.figure(figsize=(14, 7))
    # plt.bar(ASSETS, port_weights['hcaa'].iloc[CV].values, label='hcaa')
    # plt.bar(ASSETS, port_weights['aeerc'].iloc[CV].values, label='aeerc')
    # plt.legend()
    # plt.ylim([0, 0.9])
    # x = plt.xticks(rotation=45)
    # if args.save:
    #     plt.savefig(f"{save_dir}/weights_hcaa_aeerc.png", bbox_inches='tight', transparent=True)

    # Get statistics
    stats = backtest_stats(ann_perf, port_weights, period=250, format=True, market_budget=market_budget)
    if args.save:
        stats.to_csv(f"{save_dir}/backtest_stats.csv")
    print(stats.to_string())

    ##########################
    # Model evaluation
    # Average prediction across runs for each cv
    returns, scaled_returns, pred, scaled_pred = average_prediction_cv(cv_results)

    # Compute pred metric
    total_rmse = []
    total_r2 = []
    for cv in returns.keys():
        # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        # scaler.fit(returns[cv])
        # same_ret = pd.DataFrame(scaler.transform(returns[cv]), index=returns.index, columns=returns.columns)
        # same_pred = pd.DataFrame(scaler.transform(pred[cv]), index=returns.index, columns=returns.columns)
        total_rmse.append(float(np.sqrt(np.mean(np.mean((returns[cv] - pred[cv]).values ** 2, axis=-1)))))
        total_r2.append(metrics.r2_score(returns[cv], pred[cv], multioutput='uniform_average'))
    EVALUATION['model']['cv_total_rmse'] = total_rmse
    EVALUATION['model']['cv_total_r2'] = total_r2

    # Average prediction across runs
    returns, scaled_returns, pred, scaled_pred = average_prediction(cv_results)

    # Compute pred metric
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    same_ret = pd.DataFrame(scaler.fit_transform(returns), index=returns.index, columns=returns.columns)
    same_pred = pd.DataFrame(scaler.transform(pred), index=returns.index, columns=returns.columns)
    EVALUATION['model']['scaled_rmse'] = np.sqrt(np.mean((same_ret - same_pred) ** 2)).to_dict()
    EVALUATION['model']['rmse'] = np.sqrt(np.mean((returns - pred) ** 2)).to_dict()
    EVALUATION['model']['total_rmse'] = float(np.sqrt(np.mean(np.mean((returns - pred).values ** 2, axis=-1))))
    EVALUATION['model']['r2'] = {a: metrics.r2_score(returns[a], pred[a]) for a in
                                 returns.columns}
    EVALUATION['model']['total_r2'] = metrics.r2_score(returns, pred, multioutput='uniform_average')

    if args.save:
        pred_vs_true_plot(scaled_returns, scaled_pred, save_path=f"{save_dir}/pred_vs_true.png", show=args.show)
    else:
        pred_vs_true_plot(scaled_returns, scaled_pred, show=args.show)

    # loading analysis
    # loading over cv folds
    p = 0
    n_cv = len(cv_results[p])
    n_cols = 6
    n_rows = n_cv // n_cols + 1
    figsize = (15, int(n_rows * 6))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    row = -1
    col = 0
    for cv in cv_results[p]:
        loading = cv_results[p][cv]['loading'].copy()
        if cv % n_cols == 0:
            col = 0
            row += 1
        sns.heatmap(loading,
                    ax=axs[row, col],
                    vmin=0,
                    vmax=1,
                    cbar=cv == 0,
                    cbar_ax=None if cv else cbar_ax, cmap='Reds')
        date = str(cv_results[p][cv]['returns'].index[0].date())
        axs[row, col].set_title(date)
        col += 1

    fig.tight_layout(rect=[0, 0, .9, 1])
    if args.save:
        plt.savefig(f"{save_dir}/cv_loading_weights.png", bbox_inches='tight', transparent=True)
    if args.show:
        plt.show()
    plt.close()

    # Correlation
    avg_cv_corr = []
    for cv in range(n_folds):
        cv_corr = []
        for i in cv_results.keys():
            corr = cv_results[i][cv]['test_features'].corr().values
            corr = corr[np.triu_indices(len(corr), k=1)]
            cv_corr.append(corr)
        cv_corr = np.array(cv_corr)
        cv_corr = cv_corr.mean(0)
        avg_cv_corr.append(cv_corr)
    avg_cv_corr = np.array(avg_cv_corr)
    avg_cv_corr = np.mean(avg_cv_corr, axis=1).tolist()
    EVALUATION['cluster']['corr'] = {}
    EVALUATION['cluster']['corr']['cv'] = avg_cv_corr
    EVALUATION['cluster']['corr']['avg_corr'] = np.mean(avg_cv_corr)

    # Ex factor correlation cv = 0
    corr_0 = cv_results[i][0]['test_features'].corr()
    sns.heatmap(corr_0,
                cmap='bwr',
                square=True,
                vmax=1,
                vmin=-1,
                cbar=True)
    if args.save:
        plt.savefig(f"{save_dir}/corr_factors_heatmap_0.png", bbox_inches='tight', transparent=True)
    if args.show:
        plt.show()
    plt.close()

    # Ex pred correlation cv = 0
    corr_0 = cv_results[i][0]['test_pred'].corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_0,
                cmap='bwr',
                square=True,
                vmax=1,
                vmin=-1,
                cbar=True)
    if args.save:
        plt.savefig(f"{save_dir}/corr_pred_heatmap_0.png", bbox_inches='tight', transparent=True)
    if args.show:
        plt.show()
    plt.close()

    my_cmap = plt.get_cmap("bwr")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(avg_cv_corr)), avg_cv_corr, color=my_cmap(rescale(avg_cv_corr)), width=0.5)
    _ = plt.xticks(range(len(avg_cv_corr)), range(len(avg_cv_corr)), rotation=90)
    _ = plt.ylim([-1, 1])

    if args.save:
        plt.savefig(f"{save_dir}/avg_corr.png", bbox_inches='tight', transparent=True)
    if args.show:
        plt.show()
    plt.close()

    # Cluster analysis
    cv_labels = {}
    for cv in range(n_folds):
        cv_labels[cv] = {}
        for i in cv_results:
            c, cv_labels[cv][i] = get_cluster_labels(cv_results[i][cv]['embedding'])

    # Compute Rand index
    EVALUATION['cluster']['rand_index'] = {}
    n_runs = len(cv_results)
    cv_rand = {}
    for cv in range(n_folds):
        cv_rand[cv] = rand_score_permutation(cv_labels[cv])

    # Plot heatmap
    trii = np.triu_indices(n_runs, k=1)
    EVALUATION['cluster']['rand_index']['cv'] = [np.mean(cv_rand[cv][trii]) for cv in cv_rand]

    # for cv in cv_rand:
    #     mean = np.mean(cv_rand[cv][trii])
    #     std = np.std(cv_rand[cv][trii])
    #     triu = np.triu(cv_rand[cv], k=1)
    #     sns.heatmap(triu, vmin=0, vmax=1)
    #     plt.title(f"{CV_DATES[cv]}\nMean: {mean.round(2)}, Std: {std.round(2)}")
    #     if args.save:
    #         plt.savefig(f"{save_dir}/cv_plots/rand_cv_{cv}.png", bbox_inches='tight', transparent=True)
    #     if args.show:
    #         plt.show()
    #     plt.close()

    # Plot heatmap of average rand
    avg_rand = np.zeros_like(cv_rand[0])
    trii = np.triu_indices(n_runs, k=1)
    for cv in cv_rand:
        triu = np.triu(cv_rand[cv], k=1)
        avg_rand = avg_rand + triu
    avg_rand = avg_rand / len(cv_rand)

    mean = np.mean(avg_rand[trii])
    std = np.std(avg_rand[trii])
    EVALUATION['cluster']['rand_index']['mean'] = mean

    sns.heatmap(avg_rand, vmin=0, vmax=1)
    plt.title(f"Rand index\nMean: {mean.round(2)}, Std: {std.round(2)}")
    if args.save:
        plt.savefig(f"{save_dir}/rand_avg.png", bbox_inches='tight', transparent=True)
    if args.show:
        plt.show()
    plt.close()

    # Consensus matrix
    assets = cv_labels[cv][0]['label'].index
    avg_cons_mat = pd.DataFrame(0, columns=assets, index=assets)
    cluster_assignment = {}
    for cv in cv_labels:
        cons_mat = consensus_matrix(cv_labels[cv], reorder=True, method="single")
        cluster_assignment[cv] = assign_cluster_from_consmat(cons_mat, CLUSTER_NAMES, t=0)

        if cv == 0:
            order0 = cons_mat.index
            avg_cons_mat = avg_cons_mat.loc[order0, :]
            avg_cons_mat = avg_cons_mat.loc[:, order0]
        else:
            cons_mat = cons_mat.loc[order0, :]
            cons_mat = cons_mat.loc[:, order0]

        # plt.figure(figsize=(10, 10))
        # sns.heatmap(cons_mat, square=True)
        # if args.save:
        #     plt.savefig(f"{save_dir}/cv_plots/cons_mat_cv_{cv}.png", bbox_inches='tight', transparent=True)
        # if args.show:
        #     plt.show()
        # plt.close()

        avg_cons_mat += cons_mat

    avg_cons_mat = avg_cons_mat / len(cv_labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(avg_cons_mat, square=True)
    if args.save:
        plt.savefig(f"{save_dir}/avg_cons_mat.png", bbox_inches='tight', transparent=True)
    if args.show:
        plt.show()
    plt.close()

    # Save final result
    if args.save:
        pickle.dump(cluster_assignment, open(f"{save_dir}/cluster_assignment.p", "wb"))
        json.dump(EVALUATION, open(f"{save_dir}/evaluation.json", "w"))

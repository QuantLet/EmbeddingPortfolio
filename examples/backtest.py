import pandas as pd
import matplotlib.pyplot as plt
import os, pickle
import numpy as np
from dl_portfolio.logger import LOGGER
from dl_portfolio.backtest import cv_portfolio_perf, get_cv_results, bar_plot_weights, backtest_stats, plot_perf, \
    get_average_perf, get_ts_weights
import datetime as dt

PORTFOLIOS = ['equal', 'markowitz', 'shrink_markowitz', 'ae_ivp', 'hrp', 'hcaa', 'ae_rp', 'ae_rp_c', 'aeaa', 'kmaa']
# PORTFOLIOS = ['equal', 'ae_ivp', 'aeaa']
# STRAT = ['equal', 'markowitz', 'shrink_markowitz', 'aerp', 'hrp', 'herc', 'aeerc', 'ae_rp_c']
STRAT = ['equal', 'markowitz', 'shrink_markowitz', 'aerp', 'hrp', 'hcaa', 'aeerc', 'ae_rp_c', 'aeaa', 'kmaa']

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",
                        type=str,
                        help="Experiments dir")
    parser.add_argument("--dataset",
                        type=str,
                        help="Dataset name")
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
    args = parser.parse_args()

    if args.dataset == 'bond':
        market_budget = pd.read_csv('data/market_budget_bond.csv', index_col=0)
        cryptos = ['BTC', 'DASH', 'ETH', 'LTC', 'XRP']
        market_budget = pd.concat([market_budget, pd.DataFrame(np.array([['crypto', 1]] * len(cryptos)),
                                                               index=cryptos,
                                                               columns=market_budget.columns)])
        # market_budget = market_budget.drop('CRIX')
        market_budget['rc'] = market_budget['rc'].astype(int)
    else:
        raise NotImplementedError()

    meta = vars(args)
    if args.save:
        save_dir = f"eval_backtest/{args.test_set}_{args.base_dir}" + '_' + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        LOGGER.info(f"Saving result to {save_dir}")
        # os.makedirs(f"{save_dir}/cv_plots/")
        meta['save_dir'] = save_dir
        json.dump(meta, open(f"{save_dir}/meta.json", "w"))

    models = os.listdir(args.base_dir)
    paths = [f"{args.base_dir}/{d}" for d in models if models[0] != '.']

    n_folds = os.listdir(paths[0])
    n_folds = sum([d.isdigit() for d in n_folds])

    #####
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
                                       dataset=args.dataset,
                                       portfolios=portfolios,
                                       market_budget=market_budget,
                                       window=args.window,
                                       n_jobs=args.n_jobs)
        port_perf[i] = cv_portfolio_perf(cv_results[i], portfolios=portfolios, annualized=True)

    dates = [cv_results[0][cv]['test_features'].index[0] for cv in cv_results[0]]
    ASSETS = list(cv_results[i][0]['returns'].columns)

    # Weights
    port_weights = {}
    for p in PORTFOLIOS:
        if p != 'equal':
            port_weights[p] = get_ts_weights(cv_results, port=p)

    # Get average perf across runs
    ann_perf = pd.DataFrame()
    for p in PORTFOLIOS:
        if 'ae' in p:
            ann_perf[p] = get_average_perf(port_perf, port=p)
        else:
            ann_perf[p] = port_perf[0][p]['total'][0]

    # Some renaming
    port_weights['aerp'] = port_weights['ae_ivp'].copy()
    port_weights['aeerc'] = port_weights['ae_rp'].copy()
    ann_perf['aerp'] = ann_perf.loc[:, 'ae_ivp'].values
    ann_perf['aeerc'] = ann_perf.loc[:, 'ae_rp'].values

    port_weights.pop('ae_ivp')
    port_weights.pop('ae_rp')
    ann_perf.drop(['ae_ivp', 'ae_rp'], axis=1, inplace=True)

    # Plot perf
    if args.save:
        LOGGER.info('Saving performance... ')
        ann_perf.to_csv(f"{save_dir}/portfolios_returns.csv")
        pickle.dump(port_weights, open(f"{save_dir}/portfolios_weights.p", "wb"))

        plot_perf(ann_perf, strategies=STRAT,
                  save_path=f"{save_dir}/performance_all.png",
                  show=args.show, legend=True)
        plot_perf(ann_perf, strategies=[p for p in STRAT if p not in ['aerp', 'aeerc']],
                  save_path=f"{save_dir}/performance_ae_rp_c_vs_all.png",
                  show=args.show, legend=True)
        plot_perf(ann_perf, strategies=['hrp', 'aerp'], save_path=f"{save_dir}/performance_hrp_aerp.png",
                  show=args.show, legend=True)
        plot_perf(ann_perf, strategies=['hcaa', 'aeerc'], save_path=f"{save_dir}/performance_hcaa_aeerc.png",
                  show=args.show, legend=True)
        plot_perf(ann_perf, strategies=['hrp', 'ae_rp_c'], save_path=f"{save_dir}/performance_hrp_aeerc_cluster.png",
                  show=args.show, legend=True)
        plot_perf(ann_perf, strategies=['hcaa', 'ae_rp_c'], save_path=f"{save_dir}/performance_hcaa_aeerc_cluster.png",
                  show=args.show, legend=True)
        plot_perf(ann_perf, strategies=['markowitz', 'ae_rp_c'],
                  save_path=f"{save_dir}/performance_markowitz_aeerc_cluster.png",
                  show=args.show, legend=True)

        if 'shrink_markowitz' in PORTFOLIOS:
            bar_plot_weights(port_weights['shrink_markowitz'], save_path=f"{save_dir}/weights_shrink_markowitz.png",
                             show=args.show)
        bar_plot_weights(port_weights['markowitz'], save_path=f"{save_dir}/weights_markowitz.png", show=args.show)
        bar_plot_weights(port_weights['hrp'], save_path=f"{save_dir}/weights_hrp.png", show=args.show)
        bar_plot_weights(port_weights['aerp'], save_path=f"{save_dir}/weights_aerp.png", show=args.show)
        bar_plot_weights(port_weights['hcaa'], save_path=f"{save_dir}/weights_hcaa.png", show=args.show)
        bar_plot_weights(port_weights['aeerc'], save_path=f"{save_dir}/weights_aeerc.png", show=args.show)
        bar_plot_weights(port_weights['hrp'], save_path=f"{save_dir}/weights_hrp.png", show=args.show)
        bar_plot_weights(port_weights['ae_rp_c'], save_path=f"{save_dir}/weights_aeerc_cluster.png", show=args.show)
        bar_plot_weights(port_weights['aeaa'], save_path=f"{save_dir}/weights_aeaa.png", show=args.show)
    else:
        plot_perf(ann_perf, strategies=STRAT, show=args.show, legend=True)
        plot_perf(ann_perf, strategies=['hrp', 'aerp'], show=args.show, legend=True)
        bar_plot_weights(port_weights['hrp'], show=args.show)
        bar_plot_weights(port_weights['aerp'], show=args.show)

        plot_perf(ann_perf, strategies=['hcaa', 'aeerc'], show=args.show, legend=True)
        bar_plot_weights(port_weights['hcaa'], show=args.show)
        bar_plot_weights(port_weights['aeerc'], show=args.show)

        plot_perf(ann_perf, strategies=['hrp', 'ae_rp_c'], show=args.show, legend=True)
        bar_plot_weights(port_weights['hrp'], show=args.show)
        bar_plot_weights(port_weights['ae_rp_c'], show=args.show)
        bar_plot_weights(port_weights['aeaa'], show=args.show)

    # Plot excess return
    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['aerp'] + 1) - np.cumprod(ann_perf['hrp'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_hrp_aerp.png", bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['aeerc'] + 1) - np.cumprod(ann_perf['hcaa'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_hcaa_aeerc.png", bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['ae_rp_c'] + 1) - np.cumprod(ann_perf['hcaa'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_hcaa_aeerc_cluster.png", bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['ae_rp_c'] + 1) - np.cumprod(ann_perf['hrp'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_hrp_aeerc_cluster.png", bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['ae_rp_c'] + 1) - np.cumprod(ann_perf['markowitz'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_markowitz_aeerc_cluster.png", bbox_inches='tight')

    # Plot one cv weight
    cv = 0
    date = str(port_weights['hrp'].index[cv].date())
    plt.figure(figsize=(14, 7))
    plt.bar(ASSETS, port_weights['hrp'].iloc[cv].values, label='hrp')
    plt.bar(ASSETS, port_weights['aerp'].iloc[cv].values, label='aerp')
    plt.legend()
    plt.ylim([0, 0.9])
    x = plt.xticks(rotation=45)
    if args.save:
        plt.savefig(f"{save_dir}/weights_hrp_aerp.png", bbox_inches='tight')

    plt.figure(figsize=(14, 7))
    plt.bar(ASSETS, port_weights['hcaa'].iloc[cv].values, label='hcaa')
    plt.bar(ASSETS, port_weights['aeerc'].iloc[cv].values, label='aeerc')
    plt.legend()
    plt.ylim([0, 0.9])
    x = plt.xticks(rotation=45)
    if args.save:
        plt.savefig(f"{save_dir}/weights_hcaa_aeerc.png", bbox_inches='tight')

    plt.figure(figsize=(14, 7))
    plt.bar(ASSETS, port_weights['hrp'].iloc[cv].values, label='hrp')
    plt.bar(ASSETS, port_weights['ae_rp_c'].iloc[cv].values, label='ae_rp_c')
    plt.legend()
    plt.ylim([0, 0.9])
    x = plt.xticks(rotation=45)
    if args.save:
        plt.savefig(f"{save_dir}/weights_hrp_aeerc_cluster.png", bbox_inches='tight')

    # Get statistics
    stats = backtest_stats(ann_perf, port_weights, period=252, format=True)
    if args.save:
        stats.to_csv(f"{save_dir}/backtest_stats.csv")
    print(stats.to_string())

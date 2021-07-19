import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from dl_portfolio.logger import LOGGER
from dl_portfolio.backtest import portfolio_weights, cv_portfolio_perf, get_cv_results, bar_plot_weights, get_mdd, \
    calmar_ratio, sharpe_ratio
from dl_portfolio.evaluate import plot_perf
import datetime as dt

PORTFOLIOS = ['ae_ivp', 'hrp', 'herc', 'ae_rp']

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
                        default=365,
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
        save_dir = f"eval_backtest/{args.base_dir}" + '_' + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        LOGGER.info(f"Saving result to {save_dir}")
        os.makedirs(f"{save_dir}/cv_plots/")
        meta['save_dir'] = save_dir
        json.dump(meta, open(f"{save_dir}/meta.json", "w"))

    models = os.listdir(args.base_dir)
    paths = [f"{args.base_dir}/{d}" for d in models]

    n_folds = os.listdir(paths[0])
    n_folds = sum([d.isdigit() for d in n_folds])

    #####
    cv_results = {}
    train_cov = {}
    test_cov = {}
    port_perf = {}
    mse = []
    paths = paths[:2]
    for i, path in enumerate(paths):
        LOGGER.info(len(paths) - i)
        # try:
        cv_results[i] = get_cv_results(path,
                                       args.test_set,
                                       n_folds,
                                       dataset=args.dataset,
                                       portfolios=PORTFOLIOS,
                                       market_budget=market_budget,
                                       window=args.window,
                                       n_jobs=args.n_jobs)
        port_perf[i] = cv_portfolio_perf(cv_results[i],
                                         portfolios=PORTFOLIOS)
        # except Exception as _exc:
        #     LOGGER.info(f"Error with {path}... :")
        #     LOGGER.info(_exc)

    dates = [cv_results[0][cv]['test_features'].index[0] for cv in cv_results[0]]
    ASSETS = list(cv_results[i][0]['returns'].columns)

    # Weights
    port_weights = {}
    port_weights['hrp'] = pd.DataFrame()
    for cv in cv_results[0]:
        w = pd.DataFrame(cv_results[0][cv]['port']['hrp']).T
        port_weights['hrp'] = pd.concat([port_weights['hrp'], w])
    port_weights['hrp'].index = dates

    port_weights['herc'] = pd.DataFrame()
    for cv in cv_results[0]:
        w = pd.DataFrame(cv_results[0][cv]['port']['herc']).T
        port_weights['herc'] = pd.concat([port_weights['herc'], w])
    port_weights['herc'].index = dates

    port_weights['aerp'] = pd.DataFrame()
    for cv in cv_results[0]:
        avg_weights_cv = pd.DataFrame()
        for i in cv_results:
            w = pd.DataFrame(cv_results[i][cv]['port']['ae_ivp']).T
            avg_weights_cv = pd.concat([avg_weights_cv, w])
        avg_weights_cv = avg_weights_cv.mean()
        avg_weights_cv = pd.DataFrame(avg_weights_cv).T

        port_weights['aerp'] = pd.concat([port_weights['aerp'], avg_weights_cv])
    port_weights['aerp'].index = dates

    port_weights['aeerc'] = pd.DataFrame()
    for cv in cv_results[0]:
        avg_weights_cv = pd.DataFrame()
        for i in cv_results:
            w = pd.DataFrame(cv_results[i][cv]['port']['ae_rp']).T
            avg_weights_cv = pd.concat([avg_weights_cv, w])
        avg_weights_cv = avg_weights_cv.mean()
        avg_weights_cv = pd.DataFrame(avg_weights_cv).T
        port_weights['aeerc'] = pd.concat([port_weights['aeerc'], avg_weights_cv])
    port_weights['aeerc'].index = dates

    # Get average perf across runs
    ann_perf = {}
    ann_perf['aerp'] = pd.DataFrame()
    for i in port_perf:
        ann_perf['aerp'] = pd.concat([ann_perf['aerp'], port_perf[i]['ae_ivp']['total']], 1)
    ann_perf['aerp'] = ann_perf['aerp'].mean(1)
    ann_perf['aerp'] = 0.05 / (ann_perf['aerp'].std() * np.sqrt(252)) * ann_perf['aerp']

    ann_perf['aeerc'] = pd.DataFrame()
    for i in port_perf:
        ann_perf['aeerc'] = pd.concat([ann_perf['aeerc'], port_perf[i]['ae_rp']['total']], 1)
    ann_perf['aeerc'] = ann_perf['aeerc'].mean(1)
    ann_perf['aeerc'] = 0.05 / (ann_perf['aeerc'].std() * np.sqrt(252)) * ann_perf['aeerc']

    ann_perf['herc'] = port_perf[i]['herc']['total'][0]
    ann_perf['herc'] = 0.05 / (ann_perf['herc'].std() * np.sqrt(252)) * ann_perf['herc']

    ann_perf['hrp'] = port_perf[i]['hrp']['total'][0]
    ann_perf['hrp'] = 0.05 / (ann_perf['hrp'].std() * np.sqrt(252)) * ann_perf['hrp']

    # Plot perf
    if args.save:
        plot_perf(ann_perf, strategies=['hrp', 'aerp'], save_path=f"{save_dir}/performance_hrp_aerp.png",
                  show=args.show, legend=True)
        bar_plot_weights(port_weights['hrp'], save_path=f"{save_dir}/weights_hrp.png", show=args.show)
        bar_plot_weights(port_weights['aerp'], save_path=f"{save_dir}/weights_aerp.png", show=args.show)

        plot_perf(ann_perf, strategies=['herc', 'aeerc'], save_path=f"{save_dir}/performance_herc_aeerc.png",
                  show=args.show, legend=True)
        bar_plot_weights(port_weights['herc'], save_path=f"{save_dir}/weights_herc.png", show=args.show)
        bar_plot_weights(port_weights['aeerc'], save_path=f"{save_dir}/weights_aeerc.png", show=args.show)

    else:
        plot_perf(ann_perf, strategies=['hrp', 'aerp'], show=args.show, legend=True)
        bar_plot_weights(port_weights['hrp'], show=args.show)
        bar_plot_weights(port_weights['aerp'], show=args.show)

        plot_perf(ann_perf, strategies=['herc', 'aeerc'], show=args.show, legend=True)
        bar_plot_weights(port_weights['herc'], show=args.show)
        bar_plot_weights(port_weights['aeerc'], show=args.show)

    # Plot excess return
    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['aerp'] + 1) - np.cumprod(ann_perf['hrp'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_hrp_aerp.png", bbox_inches='tight')
    if args.show:
        plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(np.cumprod(ann_perf['aeerc'] + 1) - np.cumprod(ann_perf['herc'] + 1))
    if args.save:
        plt.savefig(f"{save_dir}/excess_performance_herc_aeerc.png", bbox_inches='tight')
    if args.show:
        plt.show()

    # Plot one cv weight
    cv = 0
    date = str(port_weights['hrp'].index[cv].date())
    plt.figure(figsize=(14, 7))
    plt.bar(ASSETS, port_weights['hrp'].iloc[cv].values, label='hrp')
    plt.bar(ASSETS, port_weights['aerp'].iloc[cv].values, label='aerp')
    plt.legend()
    x = plt.xticks(rotation=45)
    if args.save:
        plt.savefig(f"{save_dir}/weights_hrp_aerp.png", bbox_inches='tight')
    if args.show:
        plt.show()

    plt.figure(figsize=(14, 7))
    plt.bar(ASSETS, port_weights['herc'].iloc[cv].values, label='herc')
    plt.bar(ASSETS, port_weights['aeerc'].iloc[cv].values, label='aeerc')
    plt.legend()
    x = plt.xticks(rotation=45)
    if args.save:
        plt.savefig(f"{save_dir}/weights_herc_aeerc.png", bbox_inches='tight')
    if args.show:
        plt.show()

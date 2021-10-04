import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, pickle
import seaborn as sns
import sys, pickle
import numpy as np
from dl_portfolio.logger import LOGGER
from dl_portfolio.evaluate import pred_vs_true_plot, average_prediction, average_prediction_cv
from dl_portfolio.backtest import portfolio_weights, cv_portfolio_perf, get_cv_results, bar_plot_weights, get_mdd, \
    calmar_ratio, sharpe_ratio
from sklearn import metrics, preprocessing
from dl_portfolio.cluster import get_cluster_labels, consensus_matrix, rand_score_permutation, compute_serial_matrix
import datetime as dt

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
    parser.add_argument("--show",
                        action='store_true',
                        help="Show plots")
    parser.add_argument("--save",
                        action='store_true',
                        help="Save results")
    args = parser.parse_args()

    EVALUATION = {'model': {}, 'cluster': {}}

    # ORDER0 = ['UKX_X', 'SX5E_X', 'SPX_X', 'EPRA_X', 'MXWD_X', 'SHCOMP_X', 'JPY_FX',
    #           'NKY_X', 'GOLDS_C', 'GE_B', 'JP_B', 'UK_B', 'US_B', 'CRIX', 'GBP_FX',
    #           'CNY_FX', 'EUR_FX']

    meta = vars(args)
    if args.save:
        save_dir = f"evaluation/{args.test_set}_{args.base_dir}" + '_' + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        LOGGER.info(f"Saving result to {save_dir}")
        os.makedirs(f"{save_dir}/cv_plots/")
        meta['save_dir'] = save_dir
        json.dump(meta, open(f"{save_dir}/meta.json", "w"))

    models = os.listdir(args.base_dir)
    models = [m for m in models if m[0] != '.']
    paths = [f"{args.base_dir}/{d}" for d in models]

    # Get results for all runs
    cv_results = {}
    n_folds = os.listdir(paths[0])
    n_folds = sum([d.isdigit() for d in n_folds])
    for i, path in enumerate(paths):
        print(len(paths) - i)
        cv_results[i] = get_cv_results(path, args.test_set, n_folds, dataset=args.dataset, compute_weights=False,
                                       n_jobs=args.n_jobs)
    K = cv_results[i][0]['embedding'].shape[-1]

    cv_dates = [str(cv_results[0][cv]['returns'].index[0].date()) for cv in range(n_folds)]

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

    # Embedding analysis
    # Embedding over cv folds
    p = 0
    n_cv = len(cv_results[p])
    n_cols = 6
    n_rows = n_cv // n_cols + 1
    if n_rows == 3:
        figsize = (15, 18)
    else:
        figsize = (15, 12)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    row = -1
    col = 0
    for cv in cv_results[p]:
        embedding = cv_results[p][cv]['embedding'].copy()
        if cv % n_cols == 0:
            col = 0
            row += 1
        sns.heatmap(embedding,
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
        plt.savefig(f"{save_dir}/cv_embedding_weights.png", bbox_inches='tight')
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
        plt.savefig(f"{save_dir}/corr_factors_heatmap_0.png", bbox_inches='tight')
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
        plt.savefig(f"{save_dir}/corr_pred_heatmap_0.png", bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()

    my_cmap = plt.get_cmap("bwr")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(avg_cv_corr)), avg_cv_corr, color=my_cmap(rescale(avg_cv_corr)), width=0.5)
    _ = plt.xticks(range(len(avg_cv_corr)), range(len(avg_cv_corr)))
    _ = plt.ylim([-1, 1])

    if args.save:
        plt.savefig(f"{save_dir}/avg_corr.png", bbox_inches='tight')
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
    for cv in cv_rand:
        mean = np.mean(cv_rand[cv][trii])
        std = np.std(cv_rand[cv][trii])
        triu = np.triu(cv_rand[cv], k=1)
        sns.heatmap(triu, vmin=0, vmax=1)
        plt.title(f"{cv_dates[cv]}\nMean: {mean.round(2)}, Std: {std.round(2)}")
        if args.save:
            plt.savefig(f"{save_dir}/cv_plots/rand_cv_{cv}.png", bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()

    EVALUATION['cluster']['rand_index']['cv'] = [np.mean(cv_rand[cv][trii]) for cv in cv_rand]

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
        plt.savefig(f"{save_dir}/rand_avg.png", bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()

    # Specific ordered consensus matrix
    # avg_cons_mat = pd.DataFrame(0, columns=ORDER0, index=ORDER0)
    # for cv in cv_labels:
    #     cons_mat = consensus_matrix(cv_labels[cv], reorder=True, method="single")
    #     cons_mat = cons_mat.loc[ORDER0, :]
    #     cons_mat = cons_mat.loc[:, ORDER0]
    #     avg_cons_mat += cons_mat
    #
    # avg_cons_mat = avg_cons_mat / len(cv_labels)
    #
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(avg_cons_mat, square=True)
    # if args.save:
    #     plt.savefig(f"{save_dir}/avg_cons_mat_ordered.png", bbox_inches='tight')
    # if args.show:
    #     plt.show()
    # plt.close()

    # Consensus matrix
    assets = cv_labels[cv][0]['label'].index
    avg_cons_mat = pd.DataFrame(0, columns=assets, index=assets)
    for cv in cv_labels:

        cons_mat = consensus_matrix(cv_labels[cv], reorder=True, method="single")
        if cv == 0:
            order0 = cons_mat.index
            avg_cons_mat = avg_cons_mat.loc[order0, :]
            avg_cons_mat = avg_cons_mat.loc[:, order0]
        else:
            cons_mat = cons_mat.loc[order0, :]
            cons_mat = cons_mat.loc[:, order0]

        plt.figure(figsize=(10, 10))
        sns.heatmap(cons_mat, square=True)
        if args.save:
            plt.savefig(f"{save_dir}/cv_plots/cons_mat_cv_{cv}.png", bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()

        avg_cons_mat += cons_mat

    avg_cons_mat = avg_cons_mat / len(cv_labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(avg_cons_mat, square=True)
    if args.save:
        plt.savefig(f"{save_dir}/avg_cons_mat.png", bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close()

    # Save final result
    if args.save:
        json.dump(EVALUATION, open(f"{save_dir}/evaluation.json", "w"))

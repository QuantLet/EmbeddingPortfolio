import datetime as dt
import logging
import os
import pickle
import sys

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, preprocessing

from dl_portfolio.backtest import (
    bar_plot_weights,
    backtest_stats,
    plot_perf,
    get_ts_weights,
    get_cv_results,
    get_dl_average_weights,
    cv_portfolio_perf_df,
    get_number_of_nmf_bets,
    get_factors_rc_and_weights,
    get_cv_portfolio_weights,
    get_cv_loadings,
)

from dl_portfolio.cluster import (
    get_cluster_labels,
    consensus_matrix,
    rand_score_permutation,
    assign_cluster_from_consmat,
)
from dl_portfolio.data import load_data
from dl_portfolio.evaluate import (
    average_prediction_cv,
    load_prediction_cv,
    total_r2
)
from dl_portfolio.logger import LOGGER
from dl_portfolio.constant import (
    BASE_FACTOR_ORDER_DATASET1_4,
    BASE_FACTOR_ORDER_DATASET2_5,
)

PORTFOLIOS = [
    "equal",
    "equal_class",
    "hrp",
    "hcaa",
    "aerp",
    "erc",
    "rb_factor"
]

np.random.seed(0) # there is some variance with HCAA...

if __name__ == "__main__":
    if "hcaa" in PORTFOLIOS:
        try:
            from portfoliolab.clustering.herc import \
                HierarchicalEqualRiskContribution
        except ModuleNotFoundError as _exc:
            LOGGER.exception("You must install portfoliolab or remove 'hcaa' "
                             "from the portfolios list or implement "
                             "'HierarchicalEqualRiskContribution' yourself!")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, help="Experiments dir")
    parser.add_argument(
        "--model_type", default="ae", type=str, help="ae or nmf"
    )
    parser.add_argument(
        "--test_set", default="test", type=str, help="val or test"
    )
    parser.add_argument(
        "--n_jobs",
        default=os.cpu_count(),
        type=int,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--window",
        default=250,
        type=int,
        help="Window size for portfolio allocation",
    )
    parser.add_argument("--eval_only", action="store_true", help="Perform "
                                                                 "model"
                                                                 "evaluation only")
    parser.add_argument("--backtest_only", action="store_true", help="Perform "
                                                                 "model"
                                                                 "backtest "
                                                                     "only")
    parser.add_argument("--show", action="store_true", help="Show plots")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument(
        "--legend", action="store_true", help="Add legend to plots"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-vol",
        "--volatility_target",
        help="Volatility target",
        dest="volatility_target",
        default=0.05,
    )
    args = parser.parse_args()

    try:
        args.volatility_target = float(args.volatility_target)
    except ValueError:
        if args.volatility_target == "":
            args.volatility_target = None
        else:
            raise ValueError(args.volatility_target)
    logging.basicConfig(level=args.loglevel)
    LOGGER.setLevel(args.loglevel)
    meta = vars(args)
    if args.save:
        save_dir = (
            f"performance/{args.test_set}_{args.base_dir}"
            + "_"
            + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        LOGGER.info(f"Saving result to {save_dir}")
        # os.makedirs(f"{save_dir}/cv_plots/")

        meta["save_dir"] = save_dir
        json.dump(meta, open(f"{save_dir}/meta.json", "w"))

    EVALUATION = {"model": {}, "cluster": {}}

    LOGGER.info("Loading data...")
    # Load paths
    models = os.listdir(args.base_dir)
    paths = sorted([
        f"{args.base_dir}/{d}"
        for d in models
        if os.path.isdir(f"{args.base_dir}/{d}") and d[0] != "."
    ])
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
        raise ValueError(
            f"model_type '{args.model_type}' is not implemented. Shoule be 'ae' or 'kmeans' or 'nmf'"
        )

    # Load Market budget
    if config.dataset == "dataset1":
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
        # market_budget = market_budget.drop('CRIX')
        market_budget["rc"] = market_budget["rc"].astype(int)
    elif config.dataset == "dataset2":
        market_budget = pd.read_csv(
            "data/dataset2/market_budget_dataset2.csv", index_col=0
        )
        market_budget["rc"] = market_budget["rc"].astype(int)
    else:
        raise NotImplementedError(config.dataset)

    CLUSTER_NAMES = None
    if config.dataset == "dataset1":
        if config.encoding_dim == 4:
            CLUSTER_NAMES = BASE_FACTOR_ORDER_DATASET1_4
        else:
            CLUSTER_NAMES = None
    elif config.dataset == "dataset2":
        if config.encoding_dim == 5:
            CLUSTER_NAMES = BASE_FACTOR_ORDER_DATASET2_5
        else:
            CLUSTER_NAMES = None
    else:
        raise NotImplementedError()
    if CLUSTER_NAMES is not None:
        assert config.encoding_dim == len(CLUSTER_NAMES)

    LOGGER.info("Main loop to get results and portfolio weights...")
    # Main loop to get results
    cv_results = {}
    train_cov = {}
    test_cov = {}
    port_perf = {}

    N_EXP = len(paths)

    cv_loading = get_cv_loadings(args.base_dir)
    cv_port_weights = get_cv_portfolio_weights(
        args.base_dir,
        config,
        args.test_set,
        PORTFOLIOS,
        market_budget,
        window=args.window,
        n_jobs=args.n_jobs,
        dataset=config.dataset,
    )

    for i, path in enumerate(paths):
        LOGGER.info(len(paths) - i)
        cv_results[i] = get_cv_results(
            path,
            args.test_set,
            n_folds,
            dataset=config.dataset,
            window=args.window,
            n_jobs=args.n_jobs,
            ae_config=config,
            excess_ret=config.excess_ret,
            reorder_features=CLUSTER_NAMES is not None,
        )

    LOGGER.info("Done.")
    if args.save:
        pd.to_pickle(cv_results, f"{save_dir}/cv_results.p")

    CV_DATES = [
        str(cv_results[0][cv]["returns"].index[0].date())
        for cv in range(n_folds)
    ]
    ASSETS = list(cv_results[i][0]["returns"].columns)

    if not args.backtest_only:
        ##########################
        # Model evaluation
        # Average prediction across runs for each cv
        LOGGER.info("Starting with evaluation...")
        if N_EXP > 1:
            returns, scaled_returns, pred, scaled_pred = average_prediction_cv(
                cv_results, excess_ret=config.excess_ret
            )
        else:
            returns, scaled_returns, pred, scaled_pred = load_prediction_cv(
                cv_results, excess_ret=config.excess_ret
            )

        LOGGER.info("Prediction metric")
        EVALUATION["model"]["total_r2"] = {
            a: [
                np.mean(
                    [
                        total_r2(
                            cv_results[i][cv]["returns"][[a]].values,
                            cv_results[i][cv]["test_pred"][[a]].values
                        )
                        for i in cv_results
                    ]
                ) for cv in cv_results[0]
            ] for a in ASSETS
        }

        # Now make one df
        returns = pd.concat(returns.values())
        scaled_returns = pd.concat(scaled_returns.values())
        pred = pd.concat(pred.values())
        scaled_pred = pd.concat(scaled_pred.values())

        # Compute pred metric
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        same_ret = pd.DataFrame(
            scaler.fit_transform(returns),
            index=returns.index,
            columns=returns.columns,
        )
        same_pred = pd.DataFrame(
            scaler.transform(pred), index=returns.index, columns=returns.columns
        )
        EVALUATION["model"]["scaled_rmse"] = np.sqrt(
            np.mean((same_ret - same_pred) ** 2)
        ).to_dict()
        EVALUATION["model"]["rmse"] = np.sqrt(
            np.mean((returns - pred) ** 2)
        ).to_dict()
        EVALUATION["model"]["total_rmse"] = float(
            np.sqrt(np.mean(np.mean((returns - pred).values ** 2, axis=-1)))
        )
        LOGGER.info("Done.")

        if False:
            # loading analysis
            # loading over cv folds
            LOGGER.info("CV loadings plots")
            p = 0
            n_cv = len(cv_results[p])
            n_cols = 6
            n_rows = n_cv // n_cols + 1
            figsize = (15, int(n_rows * 6))
            fig, axs = plt.subplots(
                n_rows, n_cols, figsize=figsize, sharex=True, sharey=True
            )
            cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
            row = -1
            col = 0
            for cv in cv_results[p]:
                loading = cv_results[p][cv]["loading"].copy()
                if cv % n_cols == 0:
                    col = 0
                    row += 1
                sns.heatmap(
                    loading,
                    ax=axs[row, col],
                    vmin=0,
                    vmax=1,
                    cbar=cv == 0,
                    cbar_ax=None if cv else cbar_ax,
                    cmap="Reds",
                )
                date = str(cv_results[p][cv]["returns"].index[0].date())
                axs[row, col].set_title(date)
                col += 1

            fig.tight_layout(rect=[0, 0, 0.9, 1])
            if args.save:
                plt.savefig(
                    f"{save_dir}/cv_loading_weights.png",
                    bbox_inches="tight",
                    transparent=True,
                )
            if args.show:
                plt.show()
            plt.close()
            LOGGER.info("Done.")

        # Correlation
        LOGGER.info("Correlation...")
        avg_cv_corr = []
        for cv in range(n_folds):
            cv_corr = []
            for i in cv_results.keys():
                corr = cv_results[i][cv]["test_features"].corr().values
                corr = corr[np.triu_indices(len(corr), k=1)]
                cv_corr.append(corr)
            cv_corr = np.mean(cv_corr, axis=0).tolist()
            avg_cv_corr.append(cv_corr)
        if config.encoding_dim is not None:
            avg_cv_factor_corr = np.array(avg_cv_corr)
            avg_cv_factor_corr = np.mean(avg_cv_factor_corr, axis=1).tolist()
        else:
            avg_cv_factor_corr = None
        avg_cv_corr = np.array([np.mean(cv) for cv in avg_cv_corr]).tolist()
        EVALUATION["cluster"]["corr"] = {}
        EVALUATION["cluster"]["corr"]["cv"] = avg_cv_factor_corr
        EVALUATION["cluster"]["corr"]["avg_corr"] = avg_cv_corr

        # Ex factor correlation cv = 0
        corr_0 = cv_results[i][10]["test_features"].corr()
        sns.heatmap(corr_0, cmap="bwr", square=True, vmax=1, vmin=-1, cbar=True)
        if args.save:
            plt.savefig(
                f"{save_dir}/corr_factors_heatmap_0.png",
                bbox_inches="tight",
                transparent=True,
            )
        if args.show:
            plt.show()
        plt.close()

        # Ex pred correlation cv = 0
        corr_0 = cv_results[i][10]["test_pred"].corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr_0, cmap="bwr", square=True, vmax=1, vmin=-1, cbar=True)
        if args.save:
            plt.savefig(
                f"{save_dir}/corr_pred_heatmap_0.png",
                bbox_inches="tight",
                transparent=True,
            )
        if args.show:
            plt.show()
        plt.close()

        my_cmap = plt.get_cmap("bwr")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(len(avg_cv_corr)),
            avg_cv_corr,
            color=my_cmap(rescale(avg_cv_corr)),
            width=0.5,
        )
        if len(avg_cv_corr) > 22:
            xticks = range(0, len(avg_cv_corr), 6)
        else:
            xticks = range(len(avg_cv_corr))
        xticks_labels = np.array(CV_DATES)[xticks].tolist()
        _ = plt.xticks(xticks, xticks_labels, rotation=45)
        _ = plt.ylim([-1, 1])

        if args.save:
            plt.savefig(
                f"{save_dir}/avg_corr.png", bbox_inches="tight", transparent=True
            )
        if args.show:
            plt.show()
        plt.close()
        LOGGER.info("Done.")

        if N_EXP > 1:
            LOGGER.info("Cluster analysis...")
            # Cluster analysis
            LOGGER.info("Get cluster labels...")
            cv_labels = {}
            for cv in range(n_folds):
                cv_labels[cv] = {}
                for i in cv_results:
                    c, cv_labels[cv][i] = get_cluster_labels(
                        cv_results[i][cv]["loading"]
                    )
            LOGGER.info("Done.")

            LOGGER.info("Compute Rand Index...")
            EVALUATION["cluster"]["rand_index"] = {}
            n_runs = len(cv_results)
            cv_rand = {}
            for cv in range(n_folds):
                cv_rand[cv] = rand_score_permutation(cv_labels[cv])
            LOGGER.info("Done.")

            LOGGER.info("Rand Index heatmap...")
            # Plot heatmap
            trii = np.triu_indices(n_runs, k=1)
            EVALUATION["cluster"]["rand_index"]["cv"] = [
                np.mean(cv_rand[cv][trii]) for cv in cv_rand
            ]
            # Plot heatmap of average rand
            avg_rand = np.zeros_like(cv_rand[0])
            trii = np.triu_indices(n_runs, k=1)
            for cv in cv_rand:
                triu = np.triu(cv_rand[cv], k=1)
                avg_rand = avg_rand + triu
            avg_rand = avg_rand / len(cv_rand)

            mean = np.mean(avg_rand[trii])
            std = np.std(avg_rand[trii])
            EVALUATION["cluster"]["rand_index"]["mean"] = mean

            sns.heatmap(avg_rand, vmin=0, vmax=1)
            plt.title(f"Rand index\nMean: {mean.round(2)}, Std: {std.round(2)}")
            if args.save:
                plt.savefig(
                    f"{save_dir}/rand_avg.png", bbox_inches="tight", transparent=True
                )
            if args.show:
                plt.show()
            plt.close()
            LOGGER.info("Done.")

            LOGGER.info("Consensus matrix...")
            # Consensus matrix
            ASSETS = cv_labels[cv][0]["label"].index
            avg_cons_mat = pd.DataFrame(0, columns=ASSETS, index=ASSETS)
            cluster_assignment = {}
            if args.save:
                if not os.path.isdir(f"{save_dir}/cv_cons_mat/"):
                    os.makedirs(f"{save_dir}/cv_cons_mat/")
            for cv in cv_labels:
                cons_mat = consensus_matrix(
                    cv_labels[cv], reorder=True, method="single"
                )
                plt.figure(figsize=(10, 10))
                sns.heatmap(cons_mat, square=True)
                if args.save:
                    plt.savefig(
                        f"{save_dir}/cv_cons_mat/cons_mat_{cv}.png",
                        bbox_inches="tight",
                        transparent=True,
                    )
                if args.show:
                    plt.show()
                plt.close()

                if cv == 0:
                    order0 = cons_mat.index
                    avg_cons_mat = avg_cons_mat.loc[order0, :]
                    avg_cons_mat = avg_cons_mat.loc[:, order0]
                else:
                    cons_mat = cons_mat.loc[order0, :]
                avg_cons_mat += cons_mat

            avg_cons_mat = avg_cons_mat / len(cv_labels)
            plt.figure(figsize=(10, 10))
            sns.heatmap(avg_cons_mat, square=True)
            if args.save:
                plt.savefig(
                    f"{save_dir}/avg_cons_mat.png",
                    bbox_inches="tight",
                    transparent=True,
                )
            if args.show:
                plt.show()
            plt.close()
            LOGGER.info("Done.")

        cluster_assignment = {}
        for cv in range(n_folds):
            cluster_assignment[cv], _ = get_cluster_labels(
                cv_loading[cv]
            )

        LOGGER.info("Saving final results...")
        # Save final result
        if args.save:
            pickle.dump(
                cluster_assignment, open(f"{save_dir}/cluster_assignment.p", "wb")
            )
            json.dump(EVALUATION, open(f"{save_dir}/evaluation.json", "w"))
        LOGGER.info("Done.")

    if not args.eval_only:
        ##########################
        # Portfolio performance
        LOGGER.info("Backtest weights...")
        # Get average weights for AE portfolio across runs
        # port_weights = get_dl_average_weights(cv_results)
        # Build dictionary for cv_portfolio_perf
        cv_returns = {}
        port_weights = {cv: {} for cv in cv_results[0]}
        for cv in cv_results[0]:
            cv_returns[cv] = cv_results[0][cv]["returns"].copy()
            date = cv_results[0][cv]["returns"].index[0]
            for port in cv_port_weights[0]:
                weights = pd.DataFrame(cv_port_weights[cv][port]).T
                weights.index = [date]
                port_weights[cv][port] = weights

        port_weights_df = {}
        for port in port_weights[0]:
            port_weights_df[port] = {}
            for cv in cv_results[0]:
                dates = cv_returns[cv].index
                cv_weights = port_weights[cv][port]
                cv_weights = pd.DataFrame(
                    np.repeat(cv_weights.values, len(dates), axis=0),
                    index=dates,
                    columns=cv_weights.columns,
                )
                cv_weights = cv_weights[cv_returns[cv].columns]
                port_weights_df[port][cv] = cv_weights

        cv_portfolio_df = {
            cv: {
                "returns": cv_returns[cv],
                "train_returns": cv_results[0][cv]["train_returns"],
                "port": {
                    port: port_weights_df[port][cv]
                    for port in port_weights_df
                },
            }
            for cv in cv_returns
        }

        port_perf, leverage = cv_portfolio_perf_df(
            cv_portfolio_df,
            portfolios=PORTFOLIOS,
            volatility_target=args.volatility_target,
            market_budget=market_budget,
            dataset=config.dataset,
        )
        LOGGER.info("Done.")

        # Get portfolio weights time series
        # port_weights = {}
        # for p in PORTFOLIOS:
        #     if p not in ['equal', 'equal_class']:
        #         port_weights[p] = get_ts_weights(cv_results, port=p)
        port_weights = get_ts_weights(port_weights)
        # Get average perf across runs
        ann_perf = pd.DataFrame()
        for p in PORTFOLIOS:
            ann_perf[p] = port_perf[p]["total"].iloc[:, 0]

        LOGGER.info("Saving backtest performance and plots...")
        if args.save:
            LOGGER.info("Saving performance... ")
            ann_perf.to_csv(f"{save_dir}/portfolios_returns.csv")
            leverage.to_csv(f"{save_dir}/leverage.csv")
            pickle.dump(
                port_weights, open(f"{save_dir}/portfolios_weights.p", "wb")
            )
            plot_perf(
                ann_perf,
                strategies=PORTFOLIOS,
                save_path=f"{save_dir}/performance_all.png",
                show=args.show,
                legend=args.legend,
            )

        # Get statistics
        risk_contribution, factor_weights = get_factors_rc_and_weights(
            cv_results,
            cv_port_weights,
            cv_loading,
            market_budget.drop("CRIX") if config.dataset == "dataset1"  else
            market_budget
        )
        number_bets = get_number_of_nmf_bets(risk_contribution)
        if args.save:
            number_bets.to_csv(f"{save_dir}/number_bets.csv")
            plt.figure()
            plt.boxplot(number_bets.T, showmeans=True)
            plt.xticks(range(1, len(number_bets.columns) + 1),
                       number_bets.columns, rotation=90)
            plt.savefig(f"{save_dir}/number_bets.png", bbox_inches="tight",
                        transparent=True)
        if args.show:
            plt.show()
        plt.close()

        stats = backtest_stats(
            ann_perf,
            port_weights,
            period=250,
            format=True,
            market_budget=market_budget,
            volatility_target=args.volatility_target,
            prices=load_data(config.dataset)[0],
        )
        if args.save:
            stats.to_csv(f"{save_dir}/backtest_stats.csv")
        LOGGER.info(stats.to_string())
        LOGGER.info("Done with backtest.")

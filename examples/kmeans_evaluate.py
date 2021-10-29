import pandas as pd
import matplotlib.pyplot as plt
import os, pickle, json
import seaborn as sns
import numpy as np
from dl_portfolio.logger import LOGGER
from dl_portfolio.cluster import consensus_matrix, rand_score_permutation
import datetime as dt


def get_cv_result(dir_: str):
    n_folds = len([c for c in os.listdir(dir_) if c.isdigit()])
    result = {}
    for cv in range(n_folds):
        result[cv] = {'clusters': pickle.load(open(f"{dir_}/{cv}/clusters.p", "rb")),
                      'model': pickle.load(open(f"{dir_}/{cv}/model.p", "rb")),
                      'labels': pd.read_pickle(f"{dir_}/{cv}/labels.p")}
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",
                        type=str,
                        default='log_kmeans',
                        help="Experiments dir")
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

    if args.save:
        save_dir = f"evaluation/{args.base_dir}" + '_' + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        LOGGER.info(f"Saving result to {save_dir}")
        os.makedirs(f"{save_dir}/cv_plots/")


    # Get results for all runs
    LOGGER.info(f"Loading results")
    models = os.listdir(args.base_dir)
    models = [m for m in models if m[0] != '.']
    paths = [f"{args.base_dir}/{d}" for d in models]

    cv_results = {}
    n_folds = os.listdir(paths[0])
    n_folds = sum([d.isdigit() for d in n_folds])
    for i, path in enumerate(paths):
        cv_results[i] = get_cv_result(path)

    cv_labels = {
        cv: {i: cv_results[i][cv]['labels'] for i in cv_results.keys()} for cv in range(n_folds)
    }
    # Compute Rand index
    LOGGER.info(f"Compute Rand Index")
    EVALUATION['cluster']['rand_index'] = {}
    n_runs = len(cv_results)
    cv_rand = {}
    for cv in range(n_folds):
        cv_rand[cv] = rand_score_permutation(cv_labels[cv])

    # Plot heatmap
    LOGGER.info(f"CV Rand Index heatmap")
    trii = np.triu_indices(n_runs, k=1)
    for cv in cv_rand:
        mean = np.mean(cv_rand[cv][trii])
        std = np.std(cv_rand[cv][trii])
        triu = np.triu(cv_rand[cv], k=1)
        sns.heatmap(triu, vmin=0, vmax=1)
        plt.title(f"cv {cv}\nMean: {mean.round(2)}, Std: {std.round(2)}")
        if args.save:
            plt.savefig(f"{save_dir}/cv_plots/rand_cv_{cv}.png", bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()

    EVALUATION['cluster']['rand_index']['cv'] = [np.mean(cv_rand[cv][trii]) for cv in cv_rand]

    # Plot heatmap of average rand
    LOGGER.info(f"Average Rand Index heatmap")
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

    # Consensus matrix
    LOGGER.info(f"Consensus matrix heatmap")
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
    LOGGER.info(f"Saving result evaluation")
    if args.save:
        json.dump(EVALUATION, open(f"{save_dir}/evaluation.json", "w"))
    LOGGER.info(f"Done")
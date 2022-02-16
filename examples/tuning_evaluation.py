import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from sklearn.model_selection._search import ParameterGrid

from dl_portfolio.cluster import rand_score_permutation, get_cluster_labels
from dl_portfolio.cluster import consensus_matrix


def worker(params):
    cv_labels = {}
    metric = {"rand": {}, "mse": {}}
    param_dir = f"{args.base_dir}/{str(params)}"
    model_paths = os.listdir(param_dir)
    model_paths = [f"{args.base_dir}/{str(params)}/{m}" for m in model_paths if
                   os.path.isdir(f"{args.base_dir}/{str(params)}/{m}")]
    n_runs = len(model_paths)
    trii = np.triu_indices(n_runs, k=1)
    cvs = len([cv for cv in os.listdir(f"{model_paths[0]}") if os.path.isdir(f"{model_paths[0]}/{cv}")])
    for cv in range(cvs):
        cv_labels[cv] = {}
        for i, model_path in enumerate(model_paths):
            loadings = pd.read_pickle(f'{model_path}/{cv}/encoder_weights.p')
            c, cv_labels[cv][i] = get_cluster_labels(loadings)

        metric["rand"][cv] = rand_score_permutation(cv_labels[cv])
        metric["rand"][cv] = np.mean(metric["rand"][cv][trii])

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
        avg_cons_mat += cons_mat

    avg_cons_mat = avg_cons_mat / len(cv_labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(avg_cons_mat, square=True)
    plt.savefig(f"{args.save_dir}/avg_cons_mat_{str(params)}.png", bbox_inches='tight', transparent=True)
    plt.close()

    for cv in range(cvs):
        metric["mse"][cv] = np.mean(
            [json.load(open(f'{model_path}/evaluation.json', 'r'))[str(cv)]["test"] for model_path in model_paths])

    for m in metric:
        metric[m]["mean"] = np.mean(list(metric[m].values()))

    return metric


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",
                        type=str,
                        help="Experiments dir")
    parser.add_argument("--save_dir",
                        type=str,
                        help="Save dir")
    parser.add_argument("--n_jobs",
                        default=2 * os.cpu_count(),
                        type=int,
                        help="Number of parallel jobs")
    parser.add_argument("--save",
                        action='store_true',
                        help="Save results")
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    tuning_params = json.load(open(f'{args.base_dir}/tuning_config.json', 'r'))
    param_grid = list(ParameterGrid(tuning_params))
    nb_params = len(param_grid)

    with Parallel(n_jobs=2 * os.cpu_count() - 1) as _parallel_pool:
        metric = _parallel_pool(
            delayed(worker)(params) for params in param_grid
        )
    metric = {str(params): metric[i] for i, params in enumerate(param_grid)}
    print(metric)

    results = pd.DataFrame(index=[str(p) for p in param_grid], columns=["rand", "mse"])
    for params in results.index:
        results.loc[params, :] = [metric[params]["rand"]["mean"], metric[params]["mse"]["mean"]]

    print(results.T.astype(np.float32).round(2))
    results.T.astype(np.float32).round(2).to_csv(f"{args.save_dir}/p_tuning_result.csv")

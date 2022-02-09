import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection._search import ParameterGrid
from dl_portfolio.cluster import rand_score_permutation, get_cluster_labels


def worker(params):
    cv_labels = {}
    metric = {"rand": {}, "mse": {}}
    param_dir = f"{args.base_dir}/{str(params)}"
    model_paths = os.listdir(param_dir)
    model_paths = [f"{args.base_dir}/{str(params)}/{m}" for m in model_paths if os.path.isdir(f"{args.base_dir}/{str(params)}/{m}")]
    n_runs = len(model_paths)
    cvs = len([cv for cv in os.listdir(f"{model_paths[0]}") if os.path.isdir(f"{model_paths[0]}/{cv}")])
    for cv in range(cvs):
        cv_labels[cv] = {}
        for i, model_path in enumerate(model_paths):
            loadings = pd.read_pickle(f'{model_path}/{cv}/encoder_weights.p')
            c, cv_labels[cv][i] = get_cluster_labels(loadings)

        metric["rand"][cv] = rand_score_permutation(cv_labels[cv])

        trii = np.triu_indices(n_runs, k=1)
        for cv in range(cvs):
            metric["rand"][cv] = np.mean(metric["rand"][cv][trii])

    for cv in range(cvs):
        metric["mse"][cv] = np.mean([json.load(open(f'{model_path}/evaluation.json', 'r'))[str(cv)]["test"] for model_path in model_paths])

    for m in metric:
        metric[m]["mean"] = np.mean(list(metric[m].values()))

    return metric


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",
                        type=str,
                        help="Experiments dir")
    parser.add_argument("--n_jobs",
                        default=2 * os.cpu_count(),
                        type=int,
                        help="Number of parallel jobs")
    parser.add_argument("--save",
                        action='store_true',
                        help="Save results")
    args = parser.parse_args()

    tuning_params = json.load(open(f'{args.base_dir}/tuning_config.json', 'r'))
    param_grid = list(ParameterGrid(tuning_params))
    nb_params = len(param_grid)


    metric = {}
    for i, params in enumerate(param_grid):
        metric[str(params)] = worker(params)
    print(metric)

    results = pd.DataFrame(index = [str(p) for p in param_grid], columns = ["rand", "mse"])
    for params in results.index:
        results.loc[params,:] = [metric[params]["rand"]["mean"], metric[params]["mse"]["mean"]]

    print(results)


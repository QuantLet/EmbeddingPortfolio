import os
import json
import logging

from joblib import Parallel, delayed
from typing import Dict, Optional

from sklearn.model_selection import ParameterGrid

from dl_portfolio.utils import config_setter
from dl_portfolio.ae_data import load_data
from dl_portfolio.run import run_ae, run_kmeans, run_nmf
from dl_portfolio.logger import LOGGER

BASE_DIR = 'tuning_log'


def worker(config, params: Dict, log_dir: str, seed: Optional[int] = None):
    config = config_setter(run_type, config, params)
    run(config, data, assets, log_dir=log_dir, seed=seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n",
                        default=15,
                        type=int,
                        help="Number of experience to run")
    parser.add_argument("--n_jobs",
                        default=2 * os.cpu_count() - 1,
                        type=int,
                        help="Number of parallel jobs")
    parser.add_argument("--run",
                        type=str,
                        default='ae',
                        help="Type of run: 'ae' or 'kmeans' or 'convex_nmf'")
    parser.add_argument("--backend",
                        type=str,
                        default="loky",
                        help="Joblib backend")
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

    run_type = args.run
    if run_type == "ae":
        from dl_portfolio.config import ae_config as config

        run = run_ae
    elif run_type == "kmeans":
        run = run_kmeans
    elif run_type == "nmf":
        from dl_portfolio.config import nmf_config as config

        run = run_nmf
    else:
        raise ValueError(f"run '{args.run}' is not implemented. Shoule be 'ae' or 'kmeans' or 'convex_nmf'")

    tuning = json.load(open('dl_portfolio/config/tuning_config.json', 'r'))
    param_grid = list(ParameterGrid(tuning))
    nb_params = len(param_grid)

    if os.path.isdir(BASE_DIR):
        raise ValueError(f"{BASE_DIR} already exists, rename it or move it")
    else:
        os.mkdir(BASE_DIR)
        json.dump(tuning, open(f'{BASE_DIR}/tuning_config.json', 'w'))

    if config.dataset == 'bond':
        data, assets = load_data(dataset=config.dataset, assets=config.assets, dropnan=config.dropnan,
                                 freq=config.freq, crix=config.crix, crypto_assets=config.crypto_assets)
    else:
        data, assets = load_data(dataset=config.dataset, assets=config.assets, dropnan=config.dropnan,
                                 freq=config.freq)

    for i, params in enumerate(param_grid):
        LOGGER.warning(f"Params left to test: {nb_params - i}")
        LOGGER.warning(f"Testing params:\n{params}")
        log_dir = f"{BASE_DIR}/{str(params)}"
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
            json.dump(params, open(f"{log_dir}/params.json", "w"))
        config = config_setter(run_type, config, params)

        Parallel(n_jobs=args.n_jobs, backend=args.backend)(
            delayed(worker)(config, params, log_dir, seed=seed) for seed in range(args.n)
        )

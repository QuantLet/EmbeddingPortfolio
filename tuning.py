from dl_portfolio.run import run
from dl_portfolio.logger import LOGGER
from joblib import Parallel, delayed
import os
from dl_portfolio.utils import config_setter
import json
from sklearn.model_selection import ParameterGrid
from typing import Dict, Optional
import logging


def worker(ae_config, params: Dict, log_dir: str, seed: Optional[int] = None):
    ae_config = config_setter(ae_config, params)
    run(ae_config, log_dir=log_dir, seed=seed)


if __name__ == "__main__":
    import argparse
    from dl_portfolio.config import ae_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--n",
                        default=15,
                        type=int,
                        help="Number of experience to run")
    parser.add_argument("--n_jobs",
                        default=2 * os.cpu_count() - 1,
                        type=int,
                        help="Number of parallel jobs")
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
    tuning = json.load(open('dl_portfolio/config/tuning_config.json', 'rb'))
    param_grid = list(ParameterGrid(tuning))
    nb_params = len(param_grid)

    BASE_DIR = 'log_tuning'
    if os.path.isdir(BASE_DIR):
        raise ValueError(f"{BASE_DIR} already exists, rename it or move it")
    else:
        os.mkdir(BASE_DIR)

    for i, params in enumerate(param_grid):
        LOGGER.warning(f"Params left to test: {nb_params - i}")
        LOGGER.warning(f"Testing params:\n{params}")
        log_dir = f"{BASE_DIR}/{str(params)}"
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        json.dump(params, open(f"{log_dir}/params.json", "w"))

        Parallel(n_jobs=args.n_jobs, backend=args.backend)(
            delayed(worker)(ae_config, params, log_dir, seed=seed) for seed in range(args.n)
        )
        # for seed in range(args.n):
        #     worker(ae_config, params, log_dir, seed=seed)

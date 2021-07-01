import matplotlib.pyplot as plt
from typing import Dict, Optional
from dl_portfolio.logger import LOGGER
import pickle, os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def plot_train_history(train_history: Dict, test_history: Dict, save_dir: Optional[str] = None, show: bool = False):
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    axs[0].plot(train_history['loss'])
    axs[0].plot(test_history['loss'])
    axs[0].set_title('Loss')
    axs[1].plot(train_history['avg_ret'])
    axs[1].plot(test_history['avg_ret'])
    axs[1].set_title('Average return')
    axs[2].plot(train_history['cum_ret'])
    axs[2].plot(test_history['cum_ret'])
    axs[2].set_title('Cum return')
    if save_dir:
        plt.savefig(save_dir)
    if show:
        plt.show()


def cv_evaluation(base_dir: str, test_set: str, n_folds: int, metrics: list = ['mse']):
    assert test_set in ['val', 'test']

    def run(cv):
        LOGGER.info(f'CV {cv}')
        res = {}
        returns = pd.read_pickle(f'{base_dir}/{cv}/{test_set}_returns.p')
        pred = pd.read_pickle(f'{base_dir}/{cv}/{test_set}_prediction.p')

        residuals = returns - pred
        if 'mse' in metrics:
            res['mean_mse'] = np.mean((residuals ** 2).mean(1))
            res['mse'] = np.sum((residuals ** 2).mean(1))
        else:
            raise NotImplementedError("Available metrics are: 'mse'")

        return cv, res

    with Parallel(n_jobs=2 * os.cpu_count() - 1) as _parallel_pool:
        cv_results = _parallel_pool(
            delayed(run)(cv) for cv in range(n_folds)
        )

    # Build dictionary
    cv_results = {cv_results[i][0]: cv_results[i][1] for i in range(len(cv_results))}
    # Reorder dictionary
    cv_results = {cv: cv_results[cv] for cv in range(n_folds)}

    return cv_results

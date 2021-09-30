import matplotlib.pyplot as plt
from typing import Dict, Optional
from dl_portfolio.logger import LOGGER
import pickle, os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def average_prediction(cv_results: Dict):
    """

    :param cv_results: Dict with shape {run_1: {cv_0: {}, cv_1: {}}, run_2: {...} ...}
    :return:
    """
    assert len(cv_results) >= 2
    returns = pd.DataFrame()
    scaled_returns = pd.DataFrame()
    n_cv = len(cv_results[0])
    for cv in range(n_cv):
        ret = cv_results[0][cv]['returns'].copy()

        scaler = cv_results[0][cv]['scaler']
        scaled_ret = (ret - scaler['attributes']['mean_']) / np.sqrt(scaler['attributes']['var_'])

        returns = pd.concat([returns, ret])
        scaled_returns = pd.concat([scaled_returns, scaled_ret])

    temp_sp = pd.DataFrame()
    temp_p = pd.DataFrame()
    assets = cv_results[0][0]['returns'].columns
    for p in cv_results:
        ppred = pd.DataFrame()
        sspred = pd.DataFrame()
        for cv in range(n_cv):
            test_pred = cv_results[p][cv]['test_pred']

            scaler = cv_results[p][cv]['scaler']
            scaled_test_pred = (test_pred - scaler['attributes']['mean_']) / np.sqrt(scaler['attributes']['var_'])

            ppred = pd.concat([ppred, test_pred])
            sspred = pd.concat([sspred, scaled_test_pred])

        temp_p = pd.concat([temp_p, ppred], 1)
        temp_sp = pd.concat([temp_sp, sspred], 1)

    pred = pd.DataFrame()
    scaled_pred = pd.DataFrame()
    for a in assets:
        pred = pd.concat([pred, pd.DataFrame(temp_p[a].mean(1), columns=[a])], 1)
        scaled_pred = pd.concat([scaled_pred, pd.DataFrame(temp_sp[a].mean(1), columns=[a])], 1)

    return returns, scaled_returns, pred, scaled_pred


def average_prediction_cv(cv_results: Dict):
    """

    :param cv_results: Dict with shape {run_1: {cv_0: {}, cv_1: {}}, run_2: {...} ...}
    :return:
    """
    assert len(cv_results) >= 2
    returns = {}
    scaled_returns = {}
    predictions = {}
    scaled_predictions = {}
    n_cv = len(cv_results[0])
    for cv in range(n_cv):

        ret = cv_results[0][cv]['returns'].copy()
        scaler = cv_results[0][cv]['scaler']
        scaled_ret = (ret - scaler['attributes']['mean_']) / np.sqrt(scaler['attributes']['var_'])
        returns[cv] = ret
        scaled_returns[cv] = scaled_ret

        temp_pred = pd.DataFrame()
        temp_scaled_pred = pd.DataFrame()
        assets = cv_results[0][0]['returns'].columns
        for p in cv_results:
            p_pred = cv_results[p][cv]['test_pred']
            temp_pred = pd.concat([temp_pred, p_pred], 1)

            scaler = cv_results[p][cv]['scaler']
            temp_p_pred = (p_pred - scaler['attributes']['mean_']) / np.sqrt(scaler['attributes']['var_'])
            temp_scaled_pred = pd.concat([temp_scaled_pred, temp_p_pred], 1)

        pred = pd.DataFrame()
        scaled_pred = pd.DataFrame()
        for a in assets:
            pred = pd.concat([pred, pd.DataFrame(temp_pred[a].mean(1), columns=[a])], 1)
            scaled_pred = pd.concat([scaled_pred, pd.DataFrame(temp_scaled_pred[a].mean(1), columns=[a])], 1)

        predictions[cv] = pred
        scaled_predictions[cv] = scaled_pred

    return returns, scaled_returns, predictions, scaled_predictions


def qqplot(true: pd.DataFrame, pred: pd.DataFrame, save_path: Optional[str] = None, show: bool = False):
    n_rows = true.shape[-1] // 6 + 1
    fig, axs = plt.subplots(n_rows, 6, figsize=(20, 12))
    row = -1
    for i, a in enumerate(list(true.columns)):
        if i % 6 == 0:
            row += 1
        col = i % 6
        percs = np.linspace(0, 100, 41)
        qn_a = np.percentile(true[a].values, percs)
        qn_b = np.percentile(pred[a].values, percs)

        xlim = (min(min(qn_a), min(qn_b)) - 5e-1, max(max(qn_a), max(qn_b)) + 5e-1)
        axs[row, col].plot(qn_a, qn_b, ls="", marker="o")
        axs[row, col].set_ylim(xlim)
        axs[row, col].set_xlim(xlim)

        x = np.linspace(np.min((qn_a.min(), qn_b.min())) - 5e-1, np.max((qn_a.max(), qn_b.max())) + 5e-1)
        axs[row, col].plot(x, x, color="k", ls="--")
        axs[row, col].set_title(a)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()


def pred_vs_true_plot(true: pd.DataFrame, pred: pd.DataFrame, save_path: Optional[str] = None, show: bool = False):
    n_rows = true.shape[-1] // 6 + 1
    fig, axs = plt.subplots(n_rows, 6, figsize=(20, 14))
    row = -1
    for i, a in enumerate(list(true.columns)):
        if i % 6 == 0:
            row += 1
        col = i % 6
        axs[row, col].scatter(true[a].values, pred[a].values, s=5)
        x = np.linspace(np.min((true[a].min(), pred[a].min())) - 5e-1, np.max((true[a].max(), pred[a].max())) + 5e-1)
        axs[row, col].plot(x, x, color="k", ls="--")
        axs[row, col].set_title(a)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()


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

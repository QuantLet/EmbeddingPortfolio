import os

import numpy as np
import pandas as pd
import multiprocessing
import time

from joblib import Parallel, delayed
from sklearn import linear_model

from dl_portfolio.logger import LOGGER


def get_reg_coef_d(data, p):
    """

    :param data:
    :param p:
    :return:
    """
    n = len(data)
    varn = np.abs(data['var_norm'])
    varf = np.abs(data['var_evt'])
    spread = varf - varn

    dates = list(data.index)
    resreg = pd.DataFrame(index=data.index, columns=['spread', 'alpha', 'beta'])
    X = np.array(range(1, p + 1)).reshape(-1, 1)
    for date in dates[p:]:
        ind = dates.index(date)
        reg = linear_model.LinearRegression()
        reg.fit(X, spread[dates[ind - p + 1]:date])
        resreg.loc[date, :] = [spread.loc[date], reg.intercept_, reg.coef_[0]]

    return resreg


def get_reg_coef(data, p):
    """

    :param data:
    :param p:
    :return:
    """
    n = len(data)
    varn = np.abs(data['var'])
    varf = np.abs(data['evt_var'])
    spread = varf - varn

    resreg = pd.DataFrame(index=range(p, n), columns=['dates', 'spread', 'alpha', 'beta'])
    X = np.array(range(1, p + 1)).reshape(-1, 1)
    for i in range(p, n):
        reg = linear_model.LinearRegression()
        reg.fit(X, spread[i - p:i])
        resreg.loc[i, :] = [spread.index[i], spread[i], reg.intercept_, reg.coef_[0]]
        """
        if i % 500 == 0:
            print('Steps to go: ', n - i)
            # The coefficients
            print('Beta: ', reg.coef_[0])
            print('Alpha: ', reg.intercept_)
        """

    resreg['index'] = resreg.index
    resreg.index = resreg['dates']

    return resreg


def get_var_spread_signal(resreg, q):
    """

    :param resreg:
    :param q:
    :return:
    """

    posBeta = resreg['beta'].copy()
    posBeta = posBeta.apply(lambda x: max(x, 0))
    label = (resreg['beta'] >= q * np.cumsum(posBeta) / np.array(
        range(1, len(resreg) + 1))).astype(int)
    label = pd.DataFrame(label.values, index=label.index, columns=['label'])

    return label


def get_var_spread_perf(data, p, q):
    """

    :param data:
    :param p:
    :param q:
    :return:
    """
    result = {}
    signal = get_var_spread_signal(data, p, q)
    perf = (
            (1 - signal['label']) * data.loc[signal.index, 'returns'] -
            data.loc[signal.index, 'returns']
    ).cumsum()
    result[p, q] = perf.values[-1]
    return result


def get_reg_coef_parallel(data, lags=range(3, 25)):
    """

    :param data:
    :param lags:
    :return:
    """

    def get_one(p):
        return p, get_reg_coef(data, p)

    with Parallel(n_jobs=2 * os.cpu_count() - 1) as _parallel_pool:
        res = _parallel_pool(
            delayed(get_one)(p) for p in lags
        )

    result = {}
    for r in res:
        result[r[0]] = r[1]

    return result


def get_q_parallel(k, returns, resreg, qs=np.arange(0, 15.25, 0.25)):
    """

    :param returns:
    :param resreg:
    :param qs:
    :return:
    """
    def get_one(q):
        signal = get_var_spread_signal(resreg, q)
        perf = ((1 - signal['label']) * returns.loc[signal.index] -
                returns.loc[signal.index]).cumsum()
        return perf.values[-1]

    result = {}
    for q in qs:
        result[q] = get_one(q)
    return k, result


def get_var_spread_param(
        data, returns, lags=range(3, 6), qs=np.arange(0, 15.25, 0.25)):
    """

    :param data:
    :param lags:
    :param qs:
    :return:
    """
    LOGGER.info("Get reg coef parallel")
    t1 = time.time()
    resregs = get_reg_coef_parallel(data, lags)
    t2 = time.time()
    LOGGER.info(f"Done in {round(t2 - t1, 2)} sec.")

    result = {}
    LOGGER.info("Get q parallel")
    t1 = time.time()
    with Parallel(n_jobs=2 * os.cpu_count() - 1) as _parallel_pool:
        res = _parallel_pool(
            delayed(get_q_parallel)(k, returns, resregs[k], qs) for k in
            resregs.keys()
        )
    for r in res:
        result[r[0]] = r[1]
    # for k in resregs.keys():
    #     result[k] = get_q_parallel(returns, resregs[k], qs)
    t2 = time.time()
    LOGGER.info(f"Done in {round(t2 - t1, 2)} sec.")
    result = pd.DataFrame(result)

    max_ = np.where(result.values == np.amax(result.values))
    listOfCordinates = list(zip(max_[0], max_[1]))
    if len(listOfCordinates) > 1:  # multiple maximum, get the simple one with min p
        maxind = np.argmin([c[1] for c in listOfCordinates])
        cord = listOfCordinates[maxind]
    else:
        cord = listOfCordinates[0]

    q, p = list(result.index)[cord[0]], list(result.columns)[cord[1]]

    return p, q

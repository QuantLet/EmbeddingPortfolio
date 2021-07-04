from dl_portfolio.backtest import get_cv_results, cv_portfolio_perf
from typing import List, Optional
import numpy as np
import pandas as pd
from dl_portfolio.logger import LOGGER


def sharpe_ratio(perf, period: int = 1):
    return perf.mean() / perf.std() * np.sqrt(period)


def tuning(paths, portfolios: List[str], test_set: str, n_folds: int, market_budget=None, window: Optional[int] = None,
           n_jobs: int = None, metric: str = 'sharpe_ratio'):
    if metric == 'sharpe_ratio':
        metric_func = sharpe_ratio
    else:
        raise NotImplementedError()

    cv_results = {}
    port_perf = {}
    for i, path in enumerate(paths):
        try:

            LOGGER.info(f'Steps to go: {len(paths) - i}')
            cv_results[path] = get_cv_results(path, test_set, n_folds, portfolios, market_budget=market_budget, window=window,
                                              n_jobs=n_jobs)
            port_perf[path] = cv_portfolio_perf(cv_results[path], portfolios=portfolios)
        except Exception as _exc:
            LOGGER.info(f'Error with {path}')
            raise _exc

    best_models = {}
    best_perf = {}

    for p in portfolios:
        best_perf_port = {}
        best_perf_port['total'] = pd.DataFrame()
        for cv in range(n_folds):
            metric_value = []
            for path in port_perf:
                sr = metric_func(port_perf[path][p][cv])
                metric_value.append([path, sr])
            metric_value.sort(key=lambda x: x[1], reverse=True)
            best = metric_value[0][0]
            best_models[cv] = best

            if cv < n_folds - 1:
                best_perf_port[cv] = port_perf[best][p][cv + 1]
                best_perf_port['total'] = pd.concat([best_perf_port['total'], best_perf_port[cv]])
        best_perf_port['model'] = best_models
        best_perf_port[metric] = metric_func(best_perf_port['total'])

        best_perf[p] = best_perf_port

    return best_perf

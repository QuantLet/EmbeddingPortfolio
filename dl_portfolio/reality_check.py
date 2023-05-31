import numpy as np
import pandas as pd

from arch.bootstrap import StationaryBootstrap, optimal_block_length
from typing import List, Optional

from dl_portfolio.backtest import sharpe_ratio, get_mdd


def reality_check(returns: pd.DataFrame, metric: str, col_test: List[str],
                  col_bench: str, block_size: Optional[float] = None,
                  n_b: int = 1000,  seed=None):

    if metric == "sharpe_ratio":
        f = sharpe_ratio(returns[col_test]) - sharpe_ratio(returns[col_bench])
    elif metric == "mdd":
        f = - get_mdd(np.cumprod(returns[col_test] + 1)) + get_mdd(
            np.cumprod(returns[col_bench] + 1)
        )
    else:
        raise NotImplementedError(metric)

    n = len(returns)
    T = np.max(np.sqrt(n) * f)
    if block_size is None:
        block_size = optimal_block_length(returns)
        block_size = np.mean(block_size["stationary"])
    bs = StationaryBootstrap(block_size, returns, seed=seed)
    bs_f = []
    for data in bs.bootstrap(n_b):
        if metric == "sharpe_ratio":
            m = sharpe_ratio(data[0][0])
        elif metric == "mdd":
            m = - get_mdd(np.cumprod(data[0][0] + 1))
        else:
            raise NotImplementedError(metric)
        f = m[col_test] - m[col_bench]
        bs_f.append(f)

    bs_f = pd.concat(bs_f, axis=1).T

    T_bs = np.max(np.sqrt(n) * (bs_f.T - f.values.reshape(-1, 1)).T, axis=1)
    p_val = sum(T_bs > T) / n_b

    return p_val

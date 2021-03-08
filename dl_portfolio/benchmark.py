from dl_portfolio.constant import BASE_FREQ
from dl_portfolio.metrics import np_portfolio_returns
import pandas as pd
import numpy as np


def market_cap_weights(freq):
    market_cap = pd.read_pickle('crypto_data/marketcap/clean_marketcap_daily.p')
    market_cap = market_cap.astype(np.float32)
    if freq != BASE_FREQ * 48:
        if freq == BASE_FREQ:
            freq = '30min'
        elif freq == BASE_FREQ * 2:
            freq = '1H'
        elif freq == BASE_FREQ * 4:
            freq = '2H'
        elif freq == BASE_FREQ * 8:
            freq = '4H'
        elif freq == BASE_FREQ * 24:
            freq = '12H'
        else:
            raise NotImplementedError()
        market_cap = market_cap.resample(freq).first().fillna(method='ffill')
    weights = market_cap.divide(market_cap.sum(1), 0)

    return weights


def market_cap_returns(asset_return: pd.DataFrame, freq: int, trading_fee: float = 0.):
    weights = market_cap_weights(freq)
    weights = weights.loc[asset_return.index]

    strat_perf_no_fee, strat_perf = np_portfolio_returns(weights, asset_return,
                                                         initial_position=weights.values[:1,:],
                                                         trading_fee=trading_fee,
                                                         cash_bias=False)
    return strat_perf, np.exp(np.cumsum(strat_perf))


def equally_weighted_returns(asset_return: pd.DataFrame, **kwargs):
    # raise NotImplementedError()
    # strat_perf_no_fee, strat_perf = np_portfolio_returns(weights, asset_return,
    #                                                      initial_position=weights.values[:1, :],
    #                                                      trading_fee=trading_fee,
    #                                                      cash_bias=False)

    port_returns = asset_return.mean(1)
    port_value = np.exp(np.cumsum(port_returns))

    return port_returns, port_value

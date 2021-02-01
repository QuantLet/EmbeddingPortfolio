from dl_portfolio.constant import BASE_FREQ
import pandas as pd
import numpy as np


def market_cap_weights(freq):
    if freq != BASE_FREQ * 48:
        if freq == BASE_FREQ:
            freq = '30min'
        elif freq == BASE_FREQ * 2:
            freq = '1H'
        elif freq == BASE_FREQ * 4:
            freq = '2H'
        elif freq == BASE_FREQ * 8:
            freq = '4H'

    market_cap = pd.read_pickle('crypto_data/marketcap/clean_marketcap_daily.p')
    market_cap = market_cap.astype(np.float32)
    market_cap = market_cap.resample(freq).first().fillna(method='ffill')
    weights = market_cap.divide(market_cap.sum(1), 0)

    return weights


def market_cap_returns(asset_return: pd.DataFrame, freq: int):
    weights = market_cap_weights(freq)
    weights = weights.loc[asset_return.index]
    port_returns = (asset_return * weights).sum(1)
    port_value = (port_returns + 1).cumprod()

    return port_returns, port_value

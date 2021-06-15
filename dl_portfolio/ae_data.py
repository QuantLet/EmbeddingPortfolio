import pandas as pd
from dl_portfolio.logger import LOGGER
from typing import List, Optional
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import datetime as dt


def hour_in_week(dates: List[dt.datetime]) -> np.ndarray:
    hinw = np.array([date.weekday() * 24 + date.hour for date in dates], dtype=np.float32)
    hinw = np.round(np.sin(2 * np.pi * hinw / 168), 4)
    return hinw


def get_features(data, start: str, end: str, assets: List, val_size=30 * 6, rescale=None, scaler='StandardScaler',
                 features_config: Optional[List] = None, **kwargs):
    data = data.loc[start:end, assets]

    # Train/val/test split
    train_data = data.iloc[:-val_size * 2, :]
    val_data = data.loc[train_data.index[-1]:, :].iloc[1:val_size]

    test_data = data.loc[val_data.index[-1]:, :].iloc[1:]

    LOGGER.info(f"Train from {train_data.index[0]} to {train_data.index[-1]}")
    LOGGER.info(f"Validation from {val_data.index[0]} to {val_data.index[-1]}")
    LOGGER.info(f"Test from {test_data.index[0]} to {test_data.index[-1]}")

    # featurization
    train_data = train_data.pct_change(1).dropna()
    train_dates = train_data.index
    train_data = train_data.values
    val_data = val_data.pct_change(1).dropna()
    val_dates = val_data.index
    val_data = val_data.values
    test_data = test_data.pct_change(1).dropna()
    test_dates = test_data.index
    test_data = test_data.values

    # standardization
    if scaler == 'StandardScaler':
        kwargs['with_std'] = kwargs.get('with_std', True)
        kwargs['with_mean'] = kwargs.get('with_mean', True)
        scaler = preprocessing.StandardScaler(**kwargs)
    elif scaler == 'MinMaxScaler':
        assert 'feature_range' in kwargs
        scaler = preprocessing.MinMaxScaler(**kwargs)
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    if rescale is not None:
        train_data = train_data * rescale
        val_data = val_data * rescale
        test_data = test_data * rescale

    dates = {
        'train': train_dates,
        'val': val_dates,
        'test': test_dates
    }

    if features_config:
        n_features = len(features_config)
        features = {
            'train': [],
            'val': [],
            'test': [],
        }
        for feature_config in features_config:
            if feature_config['name'] == 'hour_in_week':
                f = {
                    'train': hour_in_week(dates['train']),
                    'val': hour_in_week(dates['val']),
                    'test': hour_in_week(dates['test'])

                }
            features['train'].append(f['train'])
            features['val'].append(f['val'])
            features['test'].append(f['test'])
        if n_features == 1:
            features['train'] = features['train'][0].reshape(-1, 1)
            features['val'] = features['val'][0].reshape(-1, 1)
            features['test'] = features['test'][0].reshape(-1, 1)
        else:
            raise NotImplementedError()
    else:
        features = None

    return train_data, val_data, test_data, scaler, dates, features


def load_data(type=['indices', 'forex', 'forex_metals', 'crypto', 'commodities'], dropnan=False, fillnan=True):
    data = pd.DataFrame()
    assets = []
    end = '2021-01-30 12:30:00'
    for asset_class in type:
        if asset_class == 'crypto':
            LOGGER.info('Loading crypto data')
            crypto_assets = ['BTC', 'DASH', 'DOGE', 'ETH', 'LTC', 'XEM', 'XMR', 'XRP']
            # Load data
            crypto_data = pd.read_pickle('./data/crypto_data/price/clean_data_1800.p')
            crypto_data = crypto_data.loc[:, pd.IndexSlice[crypto_assets, 'close']].droplevel(1, 1)
            crypto_data.index = crypto_data.index.tz_localize('UTC')
            crypto_data = crypto_data.resample('1H',
                                               closed='right',
                                               label='right').agg('last')
            data = pd.concat([data, crypto_data], 1)
            assets = assets + crypto_assets
            del crypto_data
        elif asset_class == 'forex':
            LOGGER.info('Loading forex data')
            fx_assets = ['CADUSD', 'CHFUSD', 'EURUSD', 'GBPUSD', 'JPYUSD', 'AUDUSD', 'HKDUSD']
            fxdata = pd.read_pickle('./data/histdatacom/forex_f_3600_2014_2021_close_index.p')
            fxdata = fxdata.loc[:, pd.IndexSlice[fx_assets, 'close']].droplevel(1, 1)
            data = pd.concat([data, fxdata], 1)
            del fxdata
            assets = assets + fx_assets

        elif asset_class == 'forex_metals':
            LOGGER.info('Loading forex metals data')
            fx_metals_assets = ['XAUUSD', 'XAGUSD']
            fx_m_data = pd.read_pickle('./data/histdatacom/forex_metals_f_3600_2014_2021_close_index.p')
            fx_m_data = fx_m_data.loc[:, pd.IndexSlice[fx_metals_assets, 'close']].droplevel(1, 1)
            data = pd.concat([data, fx_m_data], 1)
            del fx_m_data
            assets = assets + fx_metals_assets
        elif asset_class == 'indices':
            LOGGER.info('Loading indices data')
            indices = ['UKXUSD', 'FRXUSD', 'JPXUSD', 'SPXUSD', 'NSXUSD', 'HKXUSD', 'AUXUSD']
            indices_data = pd.read_pickle('./data/histdatacom/indices_f_3600_2014_2021_close_index.p')
            indices_data = indices_data.loc[:, pd.IndexSlice[indices, 'close']].droplevel(1, 1)
            data = pd.concat([data, indices_data], 1)
            del indices_data
            assets = assets + indices
        elif asset_class == 'commodities':
            LOGGER.info('Loading commodities data')
            com_assets = ['WTIUSD', 'BCOUSD']
            com_data = pd.read_pickle('./data/histdatacom/commodities_f_3600_2014_2021_close_index.p')
            com_data = com_data.loc[:, pd.IndexSlice[com_assets, 'close']].droplevel(1, 1)
            data = pd.concat([data, com_data], 1)
            del com_data
            assets = assets + com_assets
        else:
            raise ValueError(asset_class)

    # assets = np.random.choice(assets, len(assets), replace=False).tolist()
    # data = data.loc[:, pd.IndexSlice[assets, 'price']]
    # data = pd.DataFrame(data.values, columns=pd.MultiIndex.from_product([assets, ['price']]), index=data.index)

    data = data.loc[:end]
    if 'crypto' in type:
        if dropnan:
            data = data.dropna()
        else:
            if fillnan:
                data = data.fillna(method='ffill')
                data = data.dropna()
    else:
        if dropnan:
            data = data.dropna()
    data = data.loc[:, assets]

    return data, assets


def labelQuantile(close,
                  lq=0.1,
                  uq=0.9,
                  window=30,
                  log=True,
                  binary=False):
    """
    # label_t = 1 if we hit the upper band in the next lookfront steps: r_t+l >= upper_band where 1 <= l <= lookfront
    # label_t = 2 if we hit the lower band in the next lookfront steps: r_t+l >= upper_band where 1 <= l <= lookfront
    # else label = 0
    :param close: numpy, close price
    :param lq: float, lower quantile
    :param uq: float, upper quantile
    :param lookfront: int, horizon forecast
    :param window: int, rolling window size for computing the quantile
    :param log: boolean, log scale or simple
    :param fee: float, fee
    :param binary: boolean, output is two classes or three classes
    :return:
    """

    hist_returns = np.zeros(len(close), dtype=float)

    if log:
        hist_returns[1:] = np.log(close[1:] / close[0:-1])
    else:
        hist_returns[1:] = close[1:] / close[0:-1] - 1

    labels = np.zeros(len(close), dtype=int)
    returns = np.zeros(len(close), dtype=float)
    lower_q = np.zeros(len(close), dtype=float)
    upper_q = np.zeros(len(close), dtype=float)

    for t in range(window, len(close)):
        data_w = hist_returns[t - window: t + 1]  # select past window
        lower_q_t = np.quantile(data_w, lq)
        upper_q_t = np.quantile(data_w, uq)

        r_t = hist_returns[t].copy()

        if r_t <= lower_q_t:
            if binary:
                labels[t] = 1
            else:
                labels[t] = 2
        elif r_t >= upper_q_t:
            labels[t] = 1

        returns[t] = hist_returns[t]
        lower_q[t] = lower_q_t
        upper_q[t] = upper_q_t

    quantiles = np.concatenate([lower_q.reshape(-1, 1),
                                upper_q.reshape(-1, 1)],
                               axis=1)

    quantiles[:window, 0] = lower_q[window]
    quantiles[:window, 1] = upper_q[window]
    returns[:window] = hist_returns[:window]
    labels[:window] = np.nan

    return labels, returns, quantiles


def get_sample_weights(close, label_func, **kwargs):
    window = kwargs.get('window')
    binary = kwargs.get('binary', False)
    labels, returns, quantiles = label_func(close, **kwargs)
    if binary:
        classes = [0, 1]
    else:
        classes = [0, 1, 2]

    class_weights = compute_class_weight('balanced', classes=classes, y=labels if window is None else labels[window:])
    class_weights = {
        c: class_weights[c] for c in classes
    }
    LOGGER.info(f"Class weights:\n{class_weights}")
    sample_weights = np.zeros_like(labels, dtype=np.float32)
    for c in class_weights:
        sample_weights[labels == c] = class_weights[c]

    return sample_weights, labels, returns, quantiles


def get_sample_weights_from_df(data: pd.DataFrame, label_func, **kwargs):
    sample_weights = pd.DataFrame(columns=data.columns, index=data.index)
    labels = pd.DataFrame(columns=data.columns, index=data.index)
    for c in data.columns:
        sample_weights[c], labels[c], _, _ = get_sample_weights(data[c].values, label_func, **kwargs)
    labels = labels.astype(int)
    return sample_weights, labels

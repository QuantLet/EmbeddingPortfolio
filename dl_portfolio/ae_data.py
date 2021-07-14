import pandas as pd
from dl_portfolio.logger import LOGGER
from typing import List, Optional
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import datetime as dt
from dl_portfolio.sample import id_nb_bootstrap


def hour_in_week(dates: List[dt.datetime]) -> np.ndarray:
    hinw = np.array([date.weekday() * 24 + date.hour for date in dates], dtype=np.float32)
    hinw = np.round(np.sin(2 * np.pi * hinw / 168), 4)
    return hinw


def get_features(data, start: str, end: str, assets: List, val_start: str = None, test_start: str = None,
                 rescale=None, scaler='StandardScaler', resample=None, features_config: Optional[List] = None,
                 **kwargs):
    data = data[assets]
    # Train/val/test split
    assert dt.datetime.strptime(start, '%Y-%m-%d') < dt.datetime.strptime(end, '%Y-%m-%d')
    if val_start is not None:
        assert dt.datetime.strptime(start, '%Y-%m-%d') < dt.datetime.strptime(val_start, '%Y-%m-%d')
        assert dt.datetime.strptime(val_start, '%Y-%m-%d') < dt.datetime.strptime(end, '%Y-%m-%d')
    if test_start is not None:
        assert dt.datetime.strptime(start, '%Y-%m-%d') < dt.datetime.strptime(test_start, '%Y-%m-%d')
        assert dt.datetime.strptime(val_start, '%Y-%m-%d') < dt.datetime.strptime(test_start, '%Y-%m-%d')
        assert dt.datetime.strptime(test_start, '%Y-%m-%d') < dt.datetime.strptime(end, '%Y-%m-%d')

    if val_start is not None:
        train_data = data.loc[start:val_start].iloc[:-1]
        if test_start is not None:
            val_data = data.loc[val_start:test_start].iloc[:-1]
            test_data = data.loc[test_start:end]
        else:
            val_data = data.loc[val_start:end]
            test_data = None
    else:
        raise NotImplementedError()

    LOGGER.info(f"Train from {train_data.index[0]} to {train_data.index[-1]}")
    LOGGER.info(f"Validation from {val_data.index[0]} to {val_data.index[-1]}")
    if test_data is not None:
        LOGGER.info(f"Test from {test_data.index[0]} to {test_data.index[-1]}")

    # featurization
    train_data = train_data.pct_change(1).dropna()
    train_dates = train_data.index
    train_data = train_data.values
    val_data = val_data.pct_change(1).dropna()
    val_dates = val_data.index
    val_data = val_data.values

    if test_data is not None:
        test_data = test_data.pct_change(1).dropna()
        test_dates = test_data.index
        test_data = test_data.values
    else:
        test_dates = None

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
    if test_data is not None:
        test_data = scaler.transform(test_data)

    if rescale is not None:
        train_data = train_data * rescale
        val_data = val_data * rescale
        if test_data is not None:
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
                    'test': hour_in_week(dates['test']) if test_data is not None else None

                }
            features['train'].append(f['train'])
            features['val'].append(f['val'])
            features['test'].append(f['test'])
        if n_features == 1:
            features['train'] = features['train'][0].reshape(-1, 1)
            features['val'] = features['val'][0].reshape(-1, 1)
            features['test'] = features['test'][0].reshape(-1, 1) if test_data is not None else None
        else:
            raise NotImplementedError()
    else:
        features = None

    if resample is not None:
        if resample['method'] == 'nbb':
            where = resample.get('where', ['train'])
            block_length = resample.get('block_length', 44)
            if 'train' in where:
                LOGGER.info(f"Resampling training data with 'nbb' method with block length {block_length}")
                nbb_id = id_nb_bootstrap(len(train_data), block_length=block_length)
                train_data = train_data[nbb_id]
                dates['train'] = dates['train'][nbb_id]
            if 'val' in where:
                LOGGER.info(f"Resampling val data with 'nbb' method with block length {block_length}")
                nbb_id = id_nb_bootstrap(len(train_data), block_length=block_length)
                val_data = val_data[nbb_id]
                dates['val'] = dates['val'][nbb_id]
            if 'test' in where:
                LOGGER.info(f"Resampling test data with 'nbb' method with block length {block_length}")
                nbb_id = id_nb_bootstrap(len(train_data), block_length=block_length)
                test_data = test_data[nbb_id]
                dates['test'] = dates['test'][nbb_id]

        else:
            raise NotImplementedError(resample)

    return train_data, val_data, test_data, scaler, dates, features


def load_data_old(type: List = ['indices', 'forex', 'forex_metals', 'crypto', 'commodities'], dropnan: bool = False,
                  fillnan: bool = True, freq: str = '1H'):
    assert isinstance(freq, str)
    data = pd.DataFrame()
    assets = []
    end = '2021-01-30 12:30:00'
    start_date = []
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
            start_date.append(crypto_data.dropna().index[0])
            data = pd.concat([data, crypto_data], 1)
            assets = assets + crypto_assets
            del crypto_data
        elif asset_class == 'forex':
            LOGGER.info('Loading forex data')
            fx_assets = ['CADUSD', 'CHFUSD', 'EURUSD', 'GBPUSD', 'JPYUSD', 'AUDUSD']
            fxdata = pd.read_pickle('./data/histdatacom/forex_f_3600_2014_2021_close_index.p')
            fxdata = fxdata.loc[:, pd.IndexSlice[fx_assets, 'close']].droplevel(1, 1)
            start_date.append(fxdata.dropna().index[0])
            data = pd.concat([data, fxdata], 1)
            del fxdata
            assets = assets + fx_assets

        elif asset_class == 'forex_metals':
            LOGGER.info('Loading forex metals data')
            fx_metals_assets = ['XAUUSD', 'XAGUSD']
            fx_m_data = pd.read_pickle('./data/histdatacom/forex_metals_f_3600_2014_2021_close_index.p')
            fx_m_data = fx_m_data.loc[:, pd.IndexSlice[fx_metals_assets, 'close']].droplevel(1, 1)
            start_date.append(fx_m_data.dropna().index[0])
            data = pd.concat([data, fx_m_data], 1)
            del fx_m_data
            assets = assets + fx_metals_assets
        elif asset_class == 'indices':
            LOGGER.info('Loading indices data')
            indices = ['UKXUSD', 'FRXUSD', 'JPXUSD', 'SPXUSD', 'NSXUSD', 'HKXUSD', 'AUXUSD']
            indices_data = pd.read_pickle('./data/histdatacom/indices_f_3600_2014_2021_close_index.p')

            indices_data = indices_data.loc[:, pd.IndexSlice[indices, 'close']].droplevel(1, 1)
            us_market_hours = indices_data['SPXUSD'].dropna().index
            start_date.append(indices_data.dropna().index[0])
            data = pd.concat([data, indices_data], 1)
            del indices_data
            assets = assets + indices
        elif asset_class == 'commodities':
            LOGGER.info('Loading commodities data')
            com_assets = ['WTIUSD', 'BCOUSD']
            com_data = pd.read_pickle('./data/histdatacom/commodities_f_3600_2014_2021_close_index.p')
            com_data = com_data.loc[:, pd.IndexSlice[com_assets, 'close']].droplevel(1, 1)
            start_date.append(com_data.dropna().index[0])
            data = pd.concat([data, com_data], 1)
            del com_data
            assets = assets + com_assets
        else:
            raise ValueError(asset_class)

    start_date = max(start_date)
    data = data.loc[start_date:, :]
    # assets = np.random.choice(assets, len(assets), replace=False).tolist()
    # data = data.loc[:, pd.IndexSlice[assets, 'price']]
    # data = pd.DataFrame(data.values, columns=pd.MultiIndex.from_product([assets, ['price']]), index=data.index)

    if freq not in ["1H", "H"]:
        # TODO
        raise NotImplementedError('You must verify logic')
        LOGGER.info(f"Resampling to {freq} frequency")
        data = data.resample(freq,
                             closed='right',
                             label='right').agg('last')

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


def load_data(dataset='global', **kwargs):
    if dataset == 'bond':
        data, assets = load_global_bond_data(crix=kwargs.get('crix', False), crypto_assets=kwargs.get('crypto_assets', None))
    elif dataset == 'global':
        assets = kwargs.get('assets', None)
        dropnan = kwargs.get('dropnan', False)
        fillnan = kwargs.get('fillnan', True)
        freq = kwargs.get('freq', '1H')
        base = kwargs.get('base', 'SPXUSD')
        data, assets = load_global_data(assets=assets, dropnan=dropnan, fillnan=fillnan, freq=freq, base=base)
    elif dataset == 'global_crypto':
        data, assets = load_global_crypto_data()
    elif dataset == 'raffinot_multi_asset':
        data, assets = load_raffinot_multi_asset()
    elif dataset == 'sp500':
        data, assets = load_sp500_assets(kwargs.get('start_date', '1989-01-01'))
    else:
        raise NotImplementedError(f"dataset must be one of ['global', 'bond', 'global_crypto']: {dataset}")

    return data, assets


def load_sp500_assets(start_date='1989-01-01'):
    data = pd.read_csv('data/sp500_data.csv', index_col=0, header=[0, 1])
    data.index = pd.to_datetime(data.index)
    data = data.loc[:, pd.IndexSlice[:, 'close']].droplevel(1, 1)
    data = data.dropna(how='all')
    data = data.astype(np.float32)

    # Get assets that have prices after start date
    all_starts = pd.DataFrame(columns=['start'], index=data.columns)
    for a in data.columns:
        all_starts.loc[a, 'start'] = data[a].dropna().index[0]
    assets = list(all_starts.index[[d <= pd.to_datetime([start_date])[0] for d in all_starts['start']]])
    max_start_date = max(all_starts.loc[assets, 'start'])
    data = data.loc[max_start_date:, assets]
    data = data.interpolate(method='polynomial', order=2)
    data.dropna(inplace=True)

    return data, assets


def load_raffinot_multi_asset():
    data = pd.read_csv('data/raffinot/multiassets.csv')
    data = data.set_index('Dates')
    data.index = pd.to_datetime(data.index)
    data = data.astype(np.float32)
    data.dropna(inplace=True)
    assets = list(data.columns)

    return data, assets


def load_global_bond_data(crix=False, crypto_assets=None):
    data = pd.read_csv('./data/ALLA/alla_data.csv')
    data = data.interpolate(method='polynomial', order=2)
    data = data.set_index('Date')
    data.index = pd.to_datetime(data.index)
    data = data.dropna()

    if crix:
        assert crypto_assets is None
        crix = pd.read_pickle('./data/crypto_data/crix_1H_20160630_20210614.p')
        crix = crix.resample('1D', closed='right', label='left').agg('last')
        crix.index = pd.to_datetime([d.date() for d in crix.index])

        data = pd.concat([data, crix], 1)
        data = data.dropna()
    elif crypto_assets is not None:
        crypto_data = pd.read_pickle('./data/crypto_data/price/clean_data_1800_20150808_20210624.p')
        crypto_data = crypto_data.loc[:, pd.IndexSlice[crypto_assets, 'close']].droplevel(1, 1)
        crypto_data = crypto_data.resample('1H',
                                           closed='right',
                                           label='right').agg('last')
        crypto_data = crypto_data.resample('1D',
                                           closed='right',
                                           offset="23h",
                                           label='right').agg('last')
        crypto_data.index = pd.to_datetime([str(d.date()) for d in crypto_data.index])
        data = pd.concat([data, crypto_data], 1).dropna()

    data = data.astype(np.float32)

    assets = list(data.columns)
    return data, assets


def load_global_crypto_data():
    data = pd.read_csv('./data/crypto_data/crix/table_top_100_20180101_20201231.csv', header=[0, 1], index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.astype(np.float32)
    data = data['price']
    assets = list(data.columns)

    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    return data, assets


def load_global_data(assets: Optional[List] = None, dropnan: bool = False, fillnan: bool = True, freq: str = '1H',
                     base='SPXUSD'):
    assert freq in ['1H', '1D']
    assert isinstance(freq, str)
    data = pd.DataFrame()
    end = '2021-06-11 19:00:00'

    crypto_assets = ['BTC', 'DASH', 'DOGE', 'ETH', 'LTC', 'XEM', 'XMR', 'XRP']
    fx_assets = ['CADUSD', 'CHFUSD', 'EURUSD', 'GBPUSD', 'JPYUSD', 'AUDUSD']
    fx_metals_assets = ['XAUUSD', 'XAGUSD']
    indices = ['UKXUSD', 'FRXUSD', 'JPXUSD', 'SPXUSD', 'NSXUSD', 'HKXUSD', 'AUXUSD']
    com_assets = ['WTIUSD', 'BCOUSD']
    available_assets = indices + fx_assets + fx_metals_assets + com_assets + crypto_assets

    if assets:
        assert any([a in available_assets for a in assets])

    start_date = []
    # Load crypto data
    if assets is None or any([a in crypto_assets for a in assets]):
        crypto_data = pd.read_pickle('./data/crypto_data/price/clean_data_1800_20150808_20210624.p')
        crypto_data = crypto_data.loc[:, pd.IndexSlice[crypto_assets, 'close']].droplevel(1, 1)
        crypto_data = crypto_data.resample('1H',
                                           closed='right',
                                           label='right').agg('last')
        start_date.append(crypto_data.dropna().index[0])
        data = pd.concat([data, crypto_data], 1)

    if assets is None or any([a in indices + fx_assets + fx_metals_assets + com_assets for a in assets]):
        trad_data = pd.read_pickle('./data/histdatacom/data_close_1H_20140102_20210611.p')
        start_date.append(trad_data.dropna().index[0])
        data = pd.concat([data, trad_data], 1)

    if 'CRIX' in assets:
        crix = pd.read_pickle('./data/crypto_data/crix_1H_20160630_20210614.p')
        start_date.append(crix.dropna().index[0])
        data = pd.concat([data, crix], 1)

    if assets is None:
        assets = available_assets
        data = data[assets]
    else:
        data = data[assets]

    start_date = max(start_date)
    data = data.loc[start_date:, :]

    if freq == '1D':
        if base != 'SPXUSD':
            raise NotImplementedError()
        if base in assets:
            assert base == 'SPXUSD'
            # Convert data to EST time and get latest available price for SP500 for each day
            data = data.tz_convert('EST')
            sp = data[['SPXUSD']].dropna()
            sp = sp.groupby(sp.index.date).apply(lambda x: x.iloc[[-1]])
            sp_dates = list(sp.index.get_level_values(1))

            # Now we might have missing data at exact hour from sp500 for the other assets
            # so we just take the latest available price
            # we fill nan with latest value "ffill"
            data = data.fillna(method='ffill')
            data = data.reindex(sp_dates)

            # Convert index to days
            data.index = pd.to_datetime([d.date() for d in data.index])

            assert sum(data.isna().sum()) == 0

    # assets = np.random.choice(assets, len(assets), replace=False).tolist()
    # data = data.loc[:, pd.IndexSlice[assets, 'price']]
    # data = pd.DataFrame(data.values, columns=pd.MultiIndex.from_product([assets, ['price']]), index=data.index)

    data = data.loc[:end]
    if any([a in crypto_assets for a in data.columns]):
        # We have cryptos in our table
        if dropnan:
            data = data.dropna()
        else:
            if fillnan:
                data = data.fillna(method='ffill')
                data = data.dropna()
    else:
        if dropnan:
            data = data.dropna()

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

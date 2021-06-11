import pandas as pd
from dl_portfolio.logger import LOGGER
from typing import List
from sklearn import preprocessing


def get_features(data, start: str, end: str, assets: List, val_size=30 * 6, rescale=None, randomize_columns=False):
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
    scaler = preprocessing.StandardScaler(with_std=True, with_mean=True)
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
    return train_data, val_data, test_data, scaler, dates


def load_data(type=['indices', 'forex', 'forex_metals', 'crypto', 'commodities'], drop_weekends=False):
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
        if drop_weekends:
            data = data.dropna()
        else:
            data = data.fillna(method='ffill')
            data = data.dropna()
    else:
        data = data.dropna()
    data = data.loc[:, assets]

    return data, assets

import pandas as pd
from dl_portfolio.logger import LOGGER
from typing import List, Optional, Dict, Union
from sklearn import preprocessing
import numpy as np
import datetime as dt
from dl_portfolio.sample import id_nb_bootstrap

DATASETS = ['dataset1', 'dataset2']


def drop_remainder(indices, batch_size, last=False):
    drop = np.remainder(len(indices), batch_size)
    if len(indices) - drop < batch_size:
        raise ValueError('After dropping data is too small to generate one batch')
    if drop > 0:
        if last:
            indices = indices[:-drop]
        else:
            indices = indices[drop:]
    return indices


def load_data(dataset):
    assert dataset in DATASETS, dataset
    if dataset == 'dataset1':
        data, assets = load_dataset1()
    elif dataset == 'dataset2':
        data, assets = load_dataset2()
    else:
        raise NotImplementedError(f"dataset must be one of ['dataset1', 'dataset2']: {dataset}")

    return data, assets


def load_dataset1():
    data = pd.read_csv("data/dataset1/dataset1.csv", index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.astype(np.float32)
    return data, list(data.columns)


def load_dataset2():
    data = pd.read_csv('data/dataset2/dataset2.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    data = data.interpolate(method='polynomial', order=2)
    data = data.astype(np.float32)
    assets = list(data.columns)

    return data, assets


def bb_resample_sample(data: np.ndarray, dates: List, block_length: int = 44):
    nbb_id = id_nb_bootstrap(len(data), block_length=block_length)
    data = data[nbb_id]
    dates = dates[nbb_id]
    return data, dates


def hour_in_week(dates: List[dt.datetime]) -> np.ndarray:
    hinw = np.array([date.weekday() * 24 + date.hour for date in dates], dtype=np.float32)
    hinw = np.round(np.sin(2 * np.pi * hinw / 168), 4)
    return hinw


def get_features(data, start: str, end: str, assets: List, val_start: str = None, test_start: str = None,
                 rescale=None, scaler: Union[str, Dict] = 'StandardScaler', resample=None,
                 features_config: Optional[List] = None,
                 **kwargs):
    """

    :param data:
    :param start:
    :param end:
    :param assets:
    :param val_start:
    :param test_start:
    :param rescale:
    :param scaler: if str, then must be name of scaler and we fit the scaler, if Dict, then we use the parameter defined in scaler to transform (used for inference)
    :param resample:
    :param features_config:
    :param kwargs:
    :return:
    """
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
        train_data = data.loc[start:end]
        val_data = None
        test_data = None

    LOGGER.debug(f"Train from {train_data.index[0]} to {train_data.index[-1]}")
    if val_data is not None:
        LOGGER.debug(f"Validation from {val_data.index[0]} to {val_data.index[-1]}")
    if test_data is not None:
        LOGGER.debug(f"Test from {test_data.index[0]} to {test_data.index[-1]}")

    # featurization
    train_data = train_data.pct_change(1).dropna()
    train_dates = train_data.index
    train_data = train_data.values

    if val_data is not None:
        val_data = val_data.pct_change(1).dropna()
        val_dates = val_data.index
        val_data = val_data.values
    else:
        val_dates = None

    if test_data is not None:
        test_data = test_data.pct_change(1).dropna()
        test_dates = test_data.index
        test_data = test_data.values
    else:
        test_dates = None

    # standardization
    if scaler is not None:
        if isinstance(scaler, str):
            if scaler == 'StandardScaler':
                kwargs['with_std'] = kwargs.get('with_std', True)
                kwargs['with_mean'] = kwargs.get('with_mean', True)
                scaler = preprocessing.StandardScaler(**kwargs)
            elif scaler == 'MinMaxScaler':
                assert 'feature_range' in kwargs
                scaler = preprocessing.MinMaxScaler(**kwargs)
            else:
                raise NotImplementedError(scaler)

            scaler.fit(train_data)
            train_data = scaler.transform(train_data)
            if val_data is not None:
                val_data = scaler.transform(val_data)

            if test_data is not None:
                test_data = scaler.transform(test_data)

        elif isinstance(scaler, dict):
            mean_ = scaler['attributes']['mean_']
            std = scaler['attributes']['scale_']  # same as np.sqrt(scaler['attributes']['var_'])
            train_data = (train_data - mean_) / std
            if val_data is not None:
                val_data = (val_data - mean_) / std
            if test_data is not None:
                test_data = (test_data - mean_) / std

        else:
            raise NotImplementedError(scaler)

    if rescale is not None:
        train_data = train_data * rescale
        if val_data is not None:
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
                LOGGER.debug(f"Resampling training data with 'nbb' method with block length {block_length}")
                # nbb_id = id_nb_bootstrap(len(train_data), block_length=block_length)
                # train_data = train_data[nbb_id]
                # dates['train'] = dates['train'][nbb_id]
                train_data, dates['train'] = bb_resample_sample(train_data, dates['train'], block_length=block_length)
            if 'val' in where:
                LOGGER.debug(f"Resampling val data with 'nbb' method with block length {block_length}")
                # nbb_id = id_nb_bootstrap(len(val_data), block_length=block_length)
                # val_data = val_data[nbb_id]
                # dates['val'] = dates['val'][nbb_id]
                val_data, dates['val'] = bb_resample_sample(val_data, dates['val'], block_length=block_length)
            if 'test' in where:
                LOGGER.debug(f"Resampling test data with 'nbb' method with block length {block_length}")
                # nbb_id = id_nb_bootstrap(len(train_data), block_length=block_length)
                # test_data = test_data[nbb_id]
                # dates['test'] = dates['test'][nbb_id]
                test_data, dates['test'] = bb_resample_sample(test_data, dates['test'], block_length=block_length)
        else:
            raise NotImplementedError(resample)

    return train_data, val_data, test_data, scaler, dates, features

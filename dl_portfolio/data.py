import numpy as np
from typing import List, Dict
import pandas as pd
import datetime as dt
from dl_portfolio.logger import LOGGER
from dl_portfolio.constant import BASE_FREQ, BASE_COLUMNS, RESAMPLE_DICT


def one_month_from_freq(freq):
    hour = BASE_FREQ * 2
    if freq == hour:
        month = 24 * 30
    elif freq == hour * 2:
        month = 12 * 30
    elif freq == hour * 4:
        month = 6 * 30
    elif freq == hour * 24:
        month = 30
    return month


def data_to_freq(data, freq):
    assert freq in [BASE_FREQ, BASE_FREQ * 2, BASE_FREQ * 4, BASE_FREQ * 8,
                    BASE_FREQ * 48], 'Specified freq must be higher than 30sec'
    assert data.index.freq == '30T', 'Data must have BASE_FREQ'
    if freq != BASE_FREQ:
        if freq == BASE_FREQ * 2:
            freq = '1H'
        elif freq == BASE_FREQ * 4:
            freq = '2H'
        elif freq == BASE_FREQ * 8:
            freq = '4H'
        elif freq == BASE_FREQ * 48:
            freq = '1D'

        re_data = pd.DataFrame()
        assets = np.unique(list(data.columns.get_level_values(0))).tolist()
        for asset in assets:
            re_data = pd.concat([re_data, data[asset].resample(freq,
                                                               closed='right',
                                                               label='right').agg(RESAMPLE_DICT)], 1)
        re_data.columns = pd.MultiIndex.from_product([assets, BASE_COLUMNS])

        return re_data


def get_feature(feature_name: str, data: pd.DataFrame, **kwargs):
    if feature_name in BASE_COLUMNS:
        feature = data[feature_name]
    elif feature_name == 'returns':
        time_period = kwargs.get('time_period', 1)
        feature = data['close'].pct_change(time_period)
    elif feature_name == 'log_returns':
        time_period = kwargs.get('time_period', 1)
        feature = np.log(data['close'].pct_change(time_period) + 1)

    return feature


def build_delayed_window(data: np.ndarray, seq_len: int, return_3d: bool = False):
    """

    :param data: data
    :param seq_len: length of window
    :param return_3d: if True then return  (n, seq_len, n_features)
    :return:
    """
    n = len(data)
    n_features = data.shape[-1]
    # sequence data: (n, seq_len, n_features)
    seq_data = np.zeros((n, seq_len, n_features))
    seq_data[:] = np.nan
    seq_data[seq_len - 1:, :] = np.array([data[i - seq_len:i] for i in range(seq_len, n + 1)], dtype=np.float32)

    if return_3d:
        data = seq_data
    else:
        # concatenate columns: (n, seq_len * n_features)
        data = np.zeros((n, seq_len * n_features))
        data[:] = np.nan
        for i in range(n_features):
            data[:, i * seq_len:seq_len * (i + 1)] = seq_data[:, :, i]
    return data


def reshape_to_2d_data(data: np.ndarray, n_features: int, seq_len: int):
    """

    :param data: array with shape (n, seq_len * n_features). Columns are organised as (x^1_t-k-1, ..., x^1_t, x^2_t-k-1,
     ..., x^2_t, ..., x^f_t-k-1, ..., x^f_t)
    :param n_features:
    :param seq_len:
    :return:
    """
    seq_data = np.zeros((len(data), seq_len, n_features))
    seq_data[:] = np.nan
    for i in range(n_features):
        seq_data[:, :, i] = data[:, i * seq_len:seq_len * (i + 1)]
    assert np.isnan(seq_data).sum() == 0
    return seq_data


def min_max_scaler(X: np.ndarray, feature_range: tuple, minX: float = None, maxX: float = None):
    """

    :param X: data
    :param feature_range: (min, max)
    :param minX: min from train set to apply transformation on unseen data
    :param maxX: max from train set to apply transformation on unseen data
    :return:
    """
    if minX is None and maxX is None:
        minX = X.min(axis=0)
        maxX = X.max(axis=0)

    X_std = (X - minX) / (maxX - minX)
    X_scaled = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]

    return X_scaled, minX, maxX


def normalize_2d(data):
    """

    :param data: (n, seq_len, n_features)
    :return:
    """

    raise NotImplementedError()

def drop_remainder(indices, batch_size, last=False):
    drop = np.remainder(len(indices), batch_size)
    if last:
        indices = indices[:drop]
    else:
        indices = indices[drop:]
    return indices

class DataLoader(object):
    def __init__(self, model_type: str, features: List, freq: int = 3600,
                 path: str = 'crypto_data/price/train_data_1800.p',
                 pairs: List[Dict] = ['BTC', 'DASH', 'DOGE', 'ETH', 'LTC', 'XEM', 'XMR', 'XRP'],
                 nb_folds: int = 1, val_size: int = 6, no_cash: bool = False, window: int = 1, batch_size: int = 32):
        self._freq = freq
        self._window = window
        self._features = features
        self._n_features = len(features)
        self._features_name = [f['name'] for f in self._features]
        self._pairs = pairs
        self._nb_folds = nb_folds
        self._val_size = val_size
        self._model_type = model_type
        self._batch_size = batch_size

        if not no_cash:
            self._assets = self._pairs + ['cash']
        else:
            self._assets = self._pairs
        self._n_assets = len(self._assets)
        LOGGER.info(f'Creating data_loader for {self._n_assets} assets : {self._assets}')

        # load data
        self.df_data = pd.read_pickle(path)
        self.df_data = self.df_data.astype(np.float32)
        # resample
        self.df_data = data_to_freq(self.df_data, freq)
        # Get returns
        self.df_returns = self.df_data.loc[:, pd.IndexSlice[:, 'close']].pct_change().droplevel(1, 1)
        if not no_cash:
            # Add cash column
            self.df_returns['cash'] = 0.
        self.df_returns = self.df_returns.astype(np.float32)
        # daily_risk_free_rate = (1 + US_10Y_BOND) ** (1 / 3650) - 1
        # returns[:, -1] = daily_risk_free_rate

        # dropna
        before_drop = len(self.df_data)
        has_nan = np.sum(self.df_data.isna().sum()) > 0
        if has_nan:
            LOGGER.info('They are NaNs in original dataframe, dropping...')
        self.df_data = self.df_data.dropna()
        after_drop = len(self.df_data)
        if has_nan:
            LOGGER.info(f'Dropped {before_drop - after_drop} NaNs')

        # get last feature index
        last_ind = self.df_returns.index[-1] - dt.timedelta(seconds=freq)
        self.df_data = self.df_data.loc[:last_ind]


        # TODO: move this into cv fold generation
        # Build features, returns and corresponding base index
        LOGGER.info(f'Building {len(self._features_name)} features: {self._features_name}')
        if self._model_type == 'EIIE_model':
            self._input_data, self.df_returns, self._indices, self._dates = self.build_features_EIIE()
        else:
            self._input_data, self.df_returns, self._indices, self._dates = self.build_1d_features_and_returns()

        # Train / Test split
        LOGGER.info('Train / test split')
        self._cv_indices = self.cv_folds(self._nb_folds, self._val_size, type='incremental')

    def build_1d_pair_features(self, pair):
        df_features = pd.DataFrame()
        for feature_spec in self._features:
            params = feature_spec.get('params')
            if params is not None:
                feature = get_feature(feature_spec['name'], self.df_data[pair], **params)
            else:
                feature = get_feature(feature_spec['name'], self.df_data[pair])
            df_features = pd.concat([df_features, feature], 1)

        df_features.columns = self._features_name

        if self._window > 1:
            df_features = pd.DataFrame(build_delayed_window(df_features.values, self._window),
                                       index=df_features.index)
        self._lookback = np.max(df_features.isna().sum())
        max_feature_lookback = np.max([f['params'].get('time_period') for f in self._features if 'params' in f])
        assert self._lookback == max_feature_lookback + self._window - 1
        LOGGER.info(f'Lookback is {self._lookback}')

        before_drop = len(df_features)
        # drop na
        df_features.dropna(inplace=True)
        after_drop = len(df_features)
        if self._lookback != before_drop - after_drop:
            raise ValueError(f'Problem with NaNs count:\n{df_features.isna().sum()}')

        return df_features

    def build_1d_features_and_returns(self):
        df_features = pd.DataFrame()
        for pair in self._pairs:
            pair_feature = self.build_1d_pair_features(pair)
            df_features = pd.concat([df_features, pair_feature], 1)
        df_features.columns = pd.MultiIndex.from_product([self._pairs, pair_feature.columns])
        assert not any(df_features.isna().sum(1)), 'Problem in df_features: there are NaNs'

        # Get corresponding returns
        dates = df_features.index
        return_dates = dates + dt.timedelta(seconds=self._freq)
        df_returns = self.df_returns.reindex(return_dates)
        if np.sum(df_returns.isna().sum()) != 0:
            raise NotImplementedError(
                'If returns does not exist for one date, then we need to delete corresponding raw in df_feature')
        assert len(df_features) == len(df_returns)
        assert np.sum(df_features.index != df_returns.index - dt.timedelta(seconds=self._freq)) == 0

        # Base index
        n_samples = len(df_features)
        indices = list(range(n_samples))
        dates = df_features.index
        # Convert to float32
        df_features, df_returns = df_features.astype(np.float32), df_returns.astype(np.float32)

        return df_features, df_returns, indices, dates

    def build_features_EIIE(self):
        pairs_features = {}
        df_features, df_returns, indices, dates = self.build_1d_features_and_returns()

        """for pair in self._pairs:
            pairs_features[pair] = reshape_to_2d_data(df_features[pair].values, n_features=self._n_features,
                                                      seq_len=self._window)
        assert np.sum(
            np.unique([pairs_features[k].shape for k in pairs_features.keys()]) != [self._n_features, self._window,
                                                                                    len(dates)]) == 0"""
        for pair in self._pairs:
            pairs_features[pair] = df_features[pair]

        return pairs_features, df_returns, indices, dates

    def cv_folds(self, nb_folds: int = 1, n_months: int = 6, type='incremental'):
        """

        :param nb_folds:
        :param val_size:
        :param type:
        :return:
        """
        if type != 'incremental':
            raise NotImplementedError()

        month = one_month_from_freq(self._freq)
        val_size = n_months * month
        cv_indices = {}
        for i in range(nb_folds, 0, -1):
            if i > 1:
                cv_indices[nb_folds - i] = {
                    'train': drop_remainder(self._indices[:-val_size * i], self._batch_size, last=False),
                    'test': drop_remainder(self._indices[- val_size * i:- val_size * (i - 1)], self._batch_size, last=True)
                }
            else:
                cv_indices[nb_folds - i] = {
                    'train': drop_remainder(self._indices[:-val_size * i], self._batch_size, last=False),
                    'test': drop_remainder(self._indices[- val_size:], self._batch_size, last=True)
                }

        return cv_indices

    def cv_split(self, cv):
        raise ValueError('No need for that, drop_remainder has been implemented')
        LOGGER.info('Train / test split')
        train_indices = self._cv_indices[cv]['train']
        train_nb_batch = len(train_indices) // self._batch_size
        LOGGER.info(f'nb_batch in training: {train_nb_batch}')
        drop_first = np.remainder(len(train_indices), self._batch_size)
        LOGGER.info(f'Drop first {drop_first} in train set')
        self._train_indices = train_indices[drop_first:]
        self._train_dates = self._dates[train_indices]

        test_indices = self._cv_indices[cv]['test']
        test_nb_batch = len(test_indices) // self._batch_size
        LOGGER.info(f'nb_batch in test: {test_nb_batch}')
        drop_first = np.remainder(len(test_indices), self._batch_size)
        LOGGER.info(f'Drop first {drop_first} in train set')
        self._test_indices = test_indices[drop_first:]
        self._test_dates = self._dates[test_indices]

    @property
    def cv_indices(self):
        return self._cv_indices

    @property
    def assets(self):
        return self._assets

    @property
    def n_assets(self):
        return self._n_assets

    @property
    def pairs(self):
        return self._pairs

    @property
    def n_pairs(self):
        return len(self._pairs)

    @property
    def n_features(self):
        return self._n_features

    @property
    def input_data(self):
        return self._input_data

    @property
    def returns(self):
        return self.df_returns.values

    @property
    def dates(self):
        return self._dates

    @property
    def window(self):
        return self._window

    @property
    def train_indices(self):
        return self._train_indices

    @property
    def train_dates(self):
        return self._train_dates

    @property
    def test_indices(self):
        return self._test_indices

    @property
    def test_dates(self):
        return self._test_dates


def features_generator(dataset):
    for features, _ in dataset:
        yield features


def returns_generator(dataset):
    for _, next_returns in dataset:
        yield next_returns

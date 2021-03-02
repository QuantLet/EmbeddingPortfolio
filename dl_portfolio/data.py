import numpy as np
from typing import List, Dict
import pandas as pd
import datetime as dt
from dl_portfolio.logger import LOGGER
from dl_portfolio.constant import BASE_FREQ, BASE_COLUMNS, RESAMPLE_DICT
from sklearn import preprocessing
import tensorflow as tf


def one_month_from_freq(freq, base_freq=BASE_FREQ):
    if base_freq == BASE_FREQ:
        hour = base_freq * 2
        if freq == hour:
            month = 24 * 30
        elif freq == hour * 2:
            month = 12 * 30
        elif freq == hour * 4:
            month = 6 * 30
        elif freq == hour * 12:
            month = 2 * 30
        elif freq == hour * 24:
            month = 30
    elif base_freq == 'D':
        month = 22

    return month


def build_seq(data, seq_len):
    data = np.array(
        [data[i - seq_len:i] for i in range(seq_len, len(data) + 1)])
    return data


def data_to_freq(data, freq):
    # raise NotImplementedError('Verify resamplig method to make sure we dont look into the future, in particular for freq < D')
    assert freq in [BASE_FREQ, BASE_FREQ * 2, BASE_FREQ * 4, BASE_FREQ * 8, BASE_FREQ * 24,
                    BASE_FREQ * 48], f'Specified freq must be one of [BASE_FREQ, BASE_FREQ * 2, BASE_FREQ * 4, BASE_FREQ * 8, BASE_FREQ * 48], freq is: {freq}'
    assert data.index.freq == '30T', 'Data must have BASE_FREQ'
    if freq != BASE_FREQ:
        if freq == BASE_FREQ * 2:
            freq = '1H'
        elif freq == BASE_FREQ * 4:
            freq = '2H'
        elif freq == BASE_FREQ * 8:
            freq = '4H'
        elif freq == BASE_FREQ * 24:
            freq = '12H'
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
        raise NotImplementedError('It seems that we are looking into the future with that feature....')
        time_period = kwargs.get('time_period', 1)
        feature = data['close'].pct_change(time_period)
    elif feature_name == 'open_close_returns':
        feature = data['close'] / data['open'].values - 1

    elif feature_name == 'log_returns':
        time_period = kwargs.get('time_period', 1)
        feature = np.log(data['close'].pct_change(time_period) + 1)

    return feature


def build_delayed_window(data: np.ndarray, seq_len: int, return_2d: bool = False):
    """

    :param data: data
    :param seq_len: length of window
    :param return_2d: if True then return  (n, seq_len, n_features)
    :return:
    """
    n = len(data)
    n_features = data.shape[-1]
    # sequence data: (n, seq_len, n_features)
    seq_data = np.zeros((n, seq_len, n_features))
    seq_data[:] = np.nan
    seq_data[seq_len - 1:, :] = np.array([data[i - seq_len:i] for i in range(seq_len, n + 1)], dtype=np.float32)

    if return_2d:
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
    if drop > 0:
        if last:
            indices = indices[:drop]
        else:
            indices = indices[drop:]
    return indices


class DataLoader(object):
    def __init__(self, model_type: str, features: List, freq: int = 3600,
                 path: str = 'crypto_data/price/train_data_1800.p',
                 pairs: List[Dict] = ['BTC', 'DASH', 'DOGE', 'ETH', 'LTC', 'XEM', 'XMR', 'XRP'],
                 nb_folds: int = 5, val_size: int = 6, no_cash: bool = False, window: int = 1, batch_size: int = 32,
                 cv_type: str = 'incremental'):
        self._freq = freq
        if 'crypto_data' in path:
            self._base_freq = 1800
        else:
            self._base_freq = 'D'
        self._window = window
        self._features = features
        self._n_features = len(features)
        self._features_name = [f['name'] for f in self._features]
        self._pairs = pairs
        self._nb_folds = nb_folds
        self._val_size = val_size
        self._model_type = model_type
        self._batch_size = batch_size
        self._cv_type = cv_type

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
        if 'crypto_data' in path:
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
        if self._model_type in ['EIIE', 'asset_independent_model']:
            self._input_data, self.df_returns, self._indices, self._dates = self.build_features_EIIE()
        else:
            self._input_data, self.df_returns, self._indices, self._dates = self.build_1d_features_and_returns()

        # Train / Test split
        LOGGER.info('Train / test split')
        self._cv_indices = self.cv_folds(self._nb_folds, self._val_size, type=self._cv_type)

    def build_1d_pair_features(self, pair, window=None):
        if window is None:
            window = self._window

        df_features = pd.DataFrame()
        for feature_spec in self._features:
            params = feature_spec.get('params')
            if params is not None:
                feature = get_feature(feature_spec['name'], self.df_data[pair], **params)
            else:
                feature = get_feature(feature_spec['name'], self.df_data[pair])
            df_features = pd.concat([df_features, feature], 1)

        df_features.columns = self._features_name

        if window > 1:
            df_features = pd.DataFrame(build_delayed_window(df_features.values, window),
                                       index=df_features.index)
        self._lookback = np.max(df_features.isna().sum())
        max_feature_lookback = np.max([f['params'].get('time_period') for f in self._features if 'params' in f])
        assert self._lookback == max_feature_lookback + window - 1
        LOGGER.info(f'Lookback is {self._lookback}')

        before_drop = len(df_features)
        # drop na
        df_features.dropna(inplace=True)
        after_drop = len(df_features)
        if self._lookback != before_drop - after_drop:
            raise ValueError(f'Problem with NaNs count:\n{df_features.isna().sum()}')

        return df_features

    def build_1d_features_and_returns(self, **kwargs):
        df_features = pd.DataFrame()
        for pair in self._pairs:
            pair_feature = self.build_1d_pair_features(pair, **kwargs)
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
        df_features, df_returns, indices, dates = self.build_1d_features_and_returns(window=1)

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

        month = one_month_from_freq(self._freq, base_freq=self._base_freq)
        val_size = n_months * month
        cv_indices = {}
        for i in range(nb_folds, 0, -1):
            if i > 1:
                cv_indices[nb_folds - i] = {
                    'train': drop_remainder(self._indices[:-val_size * i], self._batch_size, last=False),
                    'test': drop_remainder(self._indices[- val_size * i:- val_size * (i - 1)], self._batch_size,
                                           last=True)
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


class SeqDataLoader(object):
    def __init__(self, model_type: str, features: List, start_date: str, freq: int = 3600,
                 path: str = 'crypto_data/price/train_data_1800.p',
                 pairs: List[Dict] = ['BTC', 'DASH', 'DOGE', 'ETH', 'LTC', 'XEM', 'XMR', 'XRP'],
                 preprocess_param: Dict = None, nb_folds: int = 5, val_size: int = 6, no_cash: bool = False,
                 seq_len: int = 1, batch_size: int = 32, cv_type: str = 'incremental', horizon: int = 1,
                 lookfront: int = 1):
        self._preprocess_param = preprocess_param
        self._freq = freq
        if 'crypto_data' in path:
            self._base_freq = 1800
        else:
            self._base_freq = 'D'
        self._seq_len = seq_len
        self._features = features
        self._n_features = len(features)
        self._features_name = [f['name'] for f in self._features]
        self._pairs = pairs
        self._nb_folds = nb_folds
        self._val_size = val_size
        self._model_type = model_type
        self._batch_size = batch_size
        self._cv_type = cv_type
        self._horizon = horizon
        self.lookfront = lookfront
        if self._horizon > 1:
            raise NotImplementedError()
        if not no_cash:
            self._assets = self._pairs + ['cash']
        else:
            self._assets = self._pairs
        self._n_assets = len(self._assets)
        LOGGER.info(f'Creating data_loader for {self._n_assets} assets : {self._assets}')

        # load data
        self.df_data = pd.read_pickle(path)
        self.df_data = self.df_data.loc[start_date:, self._pairs]
        self.df_data = self.df_data.astype(np.float32)
        # resample
        if 'crypto_data' in path:
            self.df_data = data_to_freq(self.df_data, freq)
        # Get returns
        self.df_returns = (self.df_data.loc[:, pd.IndexSlice[:, 'close']] / self.df_data.loc[:, pd.IndexSlice[:, 'open']].values - 1).droplevel(1, 1)
        # self.df_returns = self.df_data.loc[:, pd.IndexSlice[:, 'close']].pct_change().droplevel(1, 1)
        if not no_cash:
            # Add cash column
            self.df_returns['cash'] = 0.
        self.df_returns = self.df_returns.astype(np.float32)
        self.df_returns = np.log(self.df_returns + 1.)
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

        # Build features, returns and corresponding base index
        LOGGER.info(f'Building {len(self._features_name)} features: {self._features_name}')
        self._input_data, self.df_returns, self._indices, self._dates = self.build_features_EIIE()

        # Train / Test split
        LOGGER.info('Train / test split')
        self._cv_indices = self.cv_folds(self._nb_folds, self._val_size, type=self._cv_type)

    def build_1d_pair_features(self, pair):
        df_features = pd.DataFrame()
        self._feature_index = {}
        for i, feature_spec in enumerate(self._features):
            self._feature_index[feature_spec['name']] = i
            params = feature_spec.get('params')
            if params is not None:
                feature = get_feature(feature_spec['name'], self.df_data[pair], **params)
            else:
                feature = get_feature(feature_spec['name'], self.df_data[pair])
            df_features = pd.concat([df_features, feature], 1)

        df_features.columns = self._features_name
        self._lookback = np.max(df_features.isna().sum())
        max_feature_lookback = np.max([f['params'].get('time_period') if 'params' in f else 0 for f in self._features])
        assert self._lookback == max_feature_lookback
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

        # df_returns = df_features.loc[:, pd.IndexSlice[:,'returns']].droplevel(1,1)
        # df_returns = df_returns.iloc[1:,:]

        # Get corresponding returns
        dates = df_features.index
        if self.lookfront > 0:
            # return_dates = dates + dt.timedelta(seconds=self._freq)
            return_dates = list(dates)[self.lookfront:]
        else:
            return_dates = dates

        df_returns = self.df_returns.reindex(return_dates)
        # df_features['index'] = list(range(len(df_features)))
        # df_returns['index'] = list(range(len(df_returns)))
        # df_returns = df_returns.dropna()

        # features_dates = df_returns.index - dt.timedelta(seconds=self._freq)
        # df_features = df_features.reindex(features_dates)
        # df_features = df_features.dropna()

        # print(df_features['index'])
        # print(df_returns['index'])
        # exit()

        if np.sum(df_features.isna().sum()) != 0:
            raise NotImplementedError()
        if np.sum(df_returns.isna().sum()) != 0:
            raise NotImplementedError(
                'If returns does not exist for one date, then we need to delete corresponding raw in df_feature')
        if self.lookfront > 0:
            df_features = df_features.iloc[:-self.lookfront]
        assert len(df_features) == len(df_returns)
        # assert np.sum(df_features.index != df_returns.index - dt.timedelta(seconds=self._freq)) == 0

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

        month = one_month_from_freq(self._freq, base_freq=self._base_freq)
        val_size = n_months * month
        assert val_size * nb_folds < len(
            self._indices), f'val_size * nb_folds is too big: {val_size * nb_folds}\n val_size: {val_size}, ' \
                            f'nb_folds: {nb_folds} for {len(self._indices)} samples'
        cv_indices = {}
        if type == 'incremental':
            for i in range(nb_folds, 0, -1):
                if i > 1:
                    cv_indices[nb_folds - i] = {
                        'train': self._indices[:-val_size * i - self._horizon + 1],
                        'test': self._indices[- val_size * i:- val_size * (i - 1)]
                    }
                else:
                    cv_indices[nb_folds - i] = {
                        'train': self._indices[:-val_size * i - self._horizon + 1],
                        'test': self._indices[- val_size:]
                    }
        elif type == 'fold':
            for i in range(nb_folds, 0, -1):
                if i == nb_folds:
                    cv_indices[nb_folds - i] = {
                        'train': self._indices[:-val_size * i - self._horizon + 1],
                        'test': self._indices[- val_size * i:- val_size * (i - 1)]
                    }
                elif 1 < i < nb_folds:
                    cv_indices[nb_folds - i] = {
                        'train': self._indices[-val_size * (i + 1):-val_size * i - self._horizon + 1],
                        'test': self._indices[- val_size * i:- val_size * (i - 1)]
                    }
                else:
                    cv_indices[nb_folds - i] = {
                        'train': self._indices[-val_size * (i + 1):-val_size * i - self._horizon + 1],
                        'test': self._indices[- val_size:]
                    }
        else:
            raise NotImplementedError()

        return cv_indices

    def get_cv_data(self, cv):
        seq_normalization = False
        train_indices = self.cv_indices[cv]['train']
        test_indices = self.cv_indices[cv]['test']

        train_data = {pair: self._input_data[pair].iloc[train_indices] for pair in self._input_data}
        test_data = {pair: self._input_data[pair].iloc[test_indices] for pair in self._input_data}

        if self._preprocess_param is not None:
            LOGGER.info('Preprocessing ...')
            for pair in self._input_data:
                features_seq_normalize = []
                for feature_name in self._preprocess_param:
                    print(feature_name, self._preprocess_param[feature_name])
                    if self._preprocess_param[feature_name]['method'] == 'minmax':
                        scaler = preprocessing.MinMaxScaler(
                            self._preprocess_param[feature_name]['params']['feature_range'])
                        LOGGER.info('Fit to train set and transform')
                        scaler.fit(train_data[pair][[feature_name]])
                        train_data[pair].loc[:, feature_name] = scaler.transform(
                            train_data[pair][[feature_name]].values)
                        LOGGER.info('Transform test set')
                        test_data[pair].loc[:, feature_name] = scaler.transform(
                            test_data[pair][[feature_name]].values)
                    elif self._preprocess_param[feature_name]['method'] == 'mean_std':
                        scaler = preprocessing.StandardScaler(**self._preprocess_param[feature_name]['params'])
                        LOGGER.info('Fit to train set and transform')
                        scaler.fit(train_data[pair][[feature_name]])
                        train_data[pair].loc[:, feature_name] = scaler.transform(
                            train_data[pair][[feature_name]].values)
                        LOGGER.info('Transform test set')
                        test_data[pair].loc[:, feature_name] = scaler.transform(
                            test_data[pair][[feature_name]].values)
                    elif self._preprocess_param[feature_name]['method'] == 'seq_normalization':
                        seq_normalization = True
                        features_seq_normalize.append(feature_name)
                        pass

                    else:
                        raise NotImplementedError()

        LOGGER.info('Reshape to sequence data ...')
        self._train_indices = build_seq(train_indices, self._seq_len)
        self._train_indices = self._train_indices[:, -1]
        self._test_indices = build_seq(test_indices, self._seq_len)
        self._test_indices = self._test_indices[:, -1]

        train_data = np.array([build_seq(train_data[pair], self._seq_len) for pair in self._input_data])
        test_data = np.array([build_seq(test_data[pair], self._seq_len) for pair in self._input_data])

        if seq_normalization:
            LOGGER.info('Sequence normalization')
            for feature_name in features_seq_normalize:
                feature_index = self._feature_index[feature_name]
                base_norm = self._preprocess_param[feature_name]['params']['base']
                LOGGER.info(
                    f'Normalization sequence with base {base_norm} for feature: {feature_name} at index: {feature_index}')
                train_data[:, :, :, feature_index] = train_data[:, :, :, feature_index] / np.expand_dims(
                    train_data[:, :, base_norm, feature_index], -1)
                test_data[:, :, :, feature_index] = test_data[:, :, :, feature_index] / np.expand_dims(
                    test_data[:, :, base_norm, feature_index], -1)

        return train_data, test_data

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
        return self._seqlen

    @property
    def train_indices(self):
        return self._train_indices

    @property
    def test_indices(self):
        return self._test_indices

    @property
    def train_dates(self):
        return self._dates[self._train_indices]

    @property
    def test_dates(self):
        return self._dates[self._test_indices]

    @property
    def df_train_returns(self):
        return self.df_returns.iloc[self.train_indices]

    @property
    def df_test_returns(self):
        return self.df_returns.iloc[self.test_indices]

    @property
    def train_returns(self):
        # shape: (n, horizon, n_pairs) if horizon > 1
        if self._horizon > 1:
            returns = np.array([self.df_returns.values[i: i + self._horizon] for i in range(len(self.train_indices))])
        else:
            returns = self.df_returns.iloc[self.train_indices]

        return returns

    @property
    def test_returns(self):
        # shape: (n, horizon, n_pairs) if horizon > 1
        if self._horizon > 1:
            returns = np.array([self.df_returns.values[i: i + self._horizon] for i in range(len(self.test_indices))])
        else:
            returns = self.df_returns.iloc[self.test_indices]
        return returns


def features_generator(dataset, model_type: str = None):
    for ind, features, _ in dataset:
        if model_type == "EIIE":
            features = tf.transpose(features, [0, 3, 1, 2])
        elif model_type == 'asset_independent_model':
            features = [features[:, :, :, i] for i in range(features.shape[-1])]

        yield features


def returns_generator(dataset):
    for _, next_returns in dataset:
        yield next_returns

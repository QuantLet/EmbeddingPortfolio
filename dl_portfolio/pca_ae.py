import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import backend as K
from dl_portfolio.custom_layer import DenseTied, TransposeDense
from dl_portfolio.constraints import WeightsOrthogonalityConstraint, PositiveSkewnessConstraint, NonNegAndUnitNorm, \
    UncorrelatedFeaturesConstraint
from typing import List
from dl_portfolio.logger import LOGGER
import tensorflow as tf
import datetime as dt
import os
import seaborn as sns
from superkeras.permutational_layer import *
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.callbacks import Callback
import tensorflow_probability as tfp

LOG_DIR = 'dl_portfolio/log_AE'


def covariance_penalty(x, x_hat):
    weightage = 1.
    norm = '1/2'
    encoding_dim = x.shape[-1]

    def get_covariance(x):
        x_centered_list = []
        for i in range(encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        x_centered = K.transpose(x_centered)

        covariance = tfp.stats.covariance(
            x_centered
        )
        return covariance

    def uncorrelated_feature(x):
        m = get_covariance(x)
        if encoding_dim <= 1:
            return 0.0
        else:
            output = K.sum(K.square(m - tf.math.multiply(m, tf.eye(encoding_dim)))) / 2
            if norm == '1':
                return output
            elif norm == '1/2':
                # avoid overweighting fat tails
                return K.sqrt(output)
            else:
                raise NotImplementedError("norm must be '1' or '1/2' ")

    pen = weightage * uncorrelated_feature(x)
    return pen


def mse_with_covariance_penalty(x, x_hat: List):
    loss = K.mean(K.square(x - x_hat[0])) + covariance_penalty(x_hat[1], '')

    return loss


def _get_activity_regularizers(model):
    activity_regularizers = []
    for layer in model.layers:
        a_reg = getattr(layer, 'activity_regularizer', None)
        if a_reg is not None:
            activity_regularizers.append(a_reg)
    return activity_regularizers


class ActivityRegularizer(Callback):
    """ 'on_batch_end' gets automatically called by .fit when finishing
    iterating over a batch. The model, and its attributes, are inherited by
    'Callback' (except at __init__) and can be accessed via, e.g., self.model """

    def __init__(self, model):
        self.activity_regularizers = _get_activity_regularizers(model)

    def on_epoch_end(self, epoch, logs=None):
        LOGGER.info(f'epoch {epoch}')
        # 'activity_regularizer' references model layer's activity_regularizer (in this
        # case 'MyActivityRegularizer'), so its attributes ('a') can be set directly
        for i, activity_regularizer in enumerate(self.activity_regularizers):
            LOGGER.info(f'pen: {K.eval(activity_regularizer.pen)}')
            LOGGER.info(f'm:\n {K.eval(activity_regularizer.m)}')


def heat_map(encoder_weights, show=False, save=False, save_dir=None, **kwargs):
    n_clusters = len(encoder_weights.columns)
    yticks = list(encoder_weights.index)

    fig, axs = plt.subplots(1, n_clusters, figsize=(10, 10), sharey=True)

    for j, c in enumerate(list(encoder_weights.columns)):
        ax = sns.heatmap(encoder_weights[c].values.reshape(-1, 1), xticklabels=[c], yticklabels=yticks,
                         ax=axs[j], cbar=j == n_clusters - 1, **kwargs)
    if save:
        plt.savefig(f'{save_dir}/clusters_heatmap.png', bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()


def heat_map_cluster(load_dir, show=False, save=False, filename='encoder_weights.p', **kwargs):
    sets = [sets for sets in os.listdir(load_dir) if sets.isdigit()]
    sets.sort(key=lambda x: int(x))
    encoder_weights = {}
    for set_ in sets:
        encoder_weights[int(set_)] = pd.read_pickle(f"{load_dir}/{set_}/{filename}")
    n_clusters = len(encoder_weights[int(set_)].columns)
    last_set = max(list(encoder_weights.keys()))
    yticks = list(encoder_weights[last_set].index)
    n_sets = len(encoder_weights.keys())

    fig, axs = plt.subplots(n_sets, n_clusters, figsize=(10, 10 * n_sets), sharey=True)
    if n_sets > 1:
        for i, s in enumerate(encoder_weights.keys()):
            print(i)
            for j, c in enumerate(list(encoder_weights[s].columns)):
                ax = sns.heatmap(encoder_weights[s][c].values.reshape(-1, 1), xticklabels=[c], yticklabels=yticks,
                                 ax=axs[i, j], cbar=j == n_clusters - 1, **kwargs)
    else:
        for j, c in enumerate(list(encoder_weights[last_set].columns)):
            ax = sns.heatmap(encoder_weights[last_set][c].values.reshape(-1, 1), xticklabels=[c], yticklabels=yticks,
                             ax=axs[j], cbar=j == n_clusters - 1, **kwargs)
    if save:
        plt.savefig(f'{load_dir}/clusters_heatmap.png', bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()


def get_layer_by_name(name, model):
    return [l for l in model.layers if l.name == name][0]


class NonNegAndUnitNormInit(tf.keras.initializers.Initializer):

    def __init__(self, initializer: str, **kwargs):
        if initializer == 'glorot_uniform':
            self.initializer = tf.keras.initializers.GlorotUniform()
        elif initializer == 'random_normal':
            self.initializer = tf.keras.initializers.RandomNormal(mean=kwargs.get('mean', 0.0),
                                                                  stddev=kwargs.get('stddev', 0.05))
        else:
            raise NotImplementedError()

    def __call__(self, shape, dtype=None, **kwargs):
        values = self.initializer(shape=shape, dtype=dtype, **kwargs)
        values = NonNegAndUnitNorm(axis=0)(values)
        return values

    def get_config(self):  # To support serialization
        return {"initializer": self.initializer}


def ae_model(input_dim: int,
             encoding_dim: int,
             n_features: int = None,
             extra_features_dim: int = 1,
             activation: str = 'linear',
             kernel_initializer: str = 'glorot_uniform',
             activity_regularizer=None,
             kernel_constraint=None,
             kernel_regularizer=None,
             **kwargs
             ):
    batch_size = kwargs.get('batch_size', None)

    with CustomObjectScope({'MyActivityRegularizer': activity_regularizer}):  # required for Keras to recognize
        asset_input = tf.keras.layers.Input(input_dim, batch_size=batch_size, dtype=tf.float32, name='asset_input')
        encoder_layer = tf.keras.layers.Dense(encoding_dim,
                                              activation=activation,
                                              kernel_initializer=kernel_initializer,
                                              kernel_regularizer=kernel_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint,
                                              use_bias=True,
                                              name='encoder',
                                              dtype=tf.float32)
        decoder_layer = tf.keras.layers.Dense(input_dim,
                                              activation='linear',
                                              # kernel_initializer=kernel_initializer,
                                              # kernel_regularizer=kernel_regularizer,
                                              # activity_regularizer=activity_regularizer,
                                              # kernel_constraint=kernel_constraint,
                                              use_bias=True,
                                              name='decoder',
                                              dtype=tf.float32)
        encoding = encoder_layer(asset_input)

        # Extra input
        if n_features is not None:
            extra_input = tf.keras.layers.Input(n_features, batch_size=batch_size, dtype=tf.float32, name='extra_input')
            extra_features_layer = tf.keras.layers.Dense(extra_features_dim,
                                                         activation='linear',
                                                         use_bias=True,
                                                         name='extra_features',
                                                         dtype=tf.float32)
            extra_features = extra_features_layer(extra_input)
            hidden_layer = tf.keras.layers.concatenate([encoding, extra_features])
            output = decoder_layer(hidden_layer)
            autoencoder = tf.keras.models.Model([asset_input, extra_input], output)
        else:
            output = decoder_layer(encoding)
            autoencoder = tf.keras.models.Model(asset_input, output)
            extra_features = None

        encoder = tf.keras.models.Model(asset_input, encoding)

        return autoencoder, encoder, extra_features


def pca_ae_model(input_dim: int,
                 encoding_dim: int,
                 n_features: int = None,
                 extra_features_dim: int = 1,
                 activation: str = 'linear',
                 kernel_initializer: str = 'glorot_uniform',
                 activity_regularizer=None,
                 kernel_constraint=None,
                 kernel_regularizer=None,
                 **kwargs
                 ):
    batch_size = kwargs.get('batch_size', None)
    loss = kwargs.get('loss', None)

    with CustomObjectScope({'MyActivityRegularizer': activity_regularizer}):  # required for Keras to recognize
        asset_input = tf.keras.layers.Input(input_dim, batch_size=batch_size, dtype=tf.float32, name='asset_input')
        encoder_layer = tf.keras.layers.Dense(encoding_dim,
                                              activation=activation,
                                              kernel_initializer=kernel_initializer,
                                              kernel_regularizer=kernel_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint,
                                              use_bias=True,
                                              name='encoder',
                                              dtype=tf.float32)
        decoder_layer = DenseTied(input_dim,
                                  tied_to=encoder_layer,
                                  n_features=n_features,
                                  activation='linear',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  use_bias=True,
                                  dtype=tf.float32,
                                  name='decoder')
        encoding = encoder_layer(asset_input)
        encoder = tf.keras.models.Model(asset_input, encoding)

        if n_features:
            # Extra input
            extra_input = tf.keras.layers.Input(n_features, batch_size=batch_size, dtype=tf.float32, name='extra_input')
            extra_features_layer = tf.keras.layers.Dense(extra_features_dim,
                                                         activation='linear',
                                                         use_bias=True,
                                                         name='extra_features',
                                                         dtype=tf.float32)
            extra_features = extra_features_layer(extra_input)
            hidden_layer = tf.keras.layers.concatenate([encoding, extra_features])
            output = decoder_layer(hidden_layer)
            autoencoder = tf.keras.models.Model([asset_input, extra_input], output)
            if loss == 'mse_with_covariance_penalty':
                loss = mse_with_covariance_penalty([asset_input, extra_input], [output, encoding])
                autoencoder.add_loss(loss)
        else:
            output = decoder_layer(encoding)
            autoencoder = tf.keras.models.Model(asset_input, output)
            extra_features = None
            if loss == 'mse_with_covariance_penalty':
                loss = mse_with_covariance_penalty(asset_input, [output, encoding])
                autoencoder.add_loss(loss)

        return autoencoder, encoder, extra_features


def pca_permut_ae_model(input_dim: int, encoding_dim: int, activation: str = 'linear',
                        kernel_initializer: str = 'glorot_uniform',
                        ortho_weights: bool = True,
                        non_neg_unit_norm: bool = True,
                        uncorr_features: bool = True,
                        activity_regularizer=None,
                        non_neg=False,
                        pooling=None,
                        **kwargs
                        ):
    inputs = repeat_layers(Input, [(1,) for i in range(input_dim)])  # [a, b, c]
    perm_layer1 = PermutationalLayer(
        PermutationalEncoder(
            PairwiseModel((1,), repeat_layers(Dense, [input_dim], activation="relu", name="hidden")),
            input_dim
        ),
        name="permutational_layer1"
    )

    outputs = perm_layer1(inputs)
    print(len(outputs))
    if pooling == 'maximum':
        perm_output = maximum(outputs)
    elif pooling == 'average':
        perm_output = average(outputs)
    else:
        perm_output = tf.concat(outputs, axis=-1)

    permutation_model = Model(inputs, perm_output, name='permutation')
    print("# Multi-layer model summary")
    print(permutation_model.summary())

    kernel_regularizer = WeightsOrthogonalityConstraint(encoding_dim, axis=0) if ortho_weights else None
    if non_neg_unit_norm:
        assert not non_neg
        kernel_constraint = NonNegAndUnitNorm(axis=0)
    elif non_neg:
        kernel_constraint = tf.keras.constraints.NonNeg()
    else:
        kernel_constraint = None
    if activity_regularizer is None:
        weightage = kwargs.get('weightage', 1.)
        activity_regularizer = UncorrelatedFeaturesConstraint(encoding_dim,
                                                              weightage=weightage) if uncorr_features else None
    else:
        assert not uncorr_features

    encoder_layer = tf.keras.layers.Dense(encoding_dim,
                                          activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer,
                                          activity_regularizer=activity_regularizer,
                                          kernel_constraint=kernel_constraint,
                                          use_bias=True,
                                          name='encoder',
                                          dtype=tf.float32)
    decoder_layer = DenseTied(input_dim,
                              tied_to=encoder_layer,
                              activation='linear',
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              use_bias=True,
                              dtype=tf.float32,
                              name='decoder')
    encoder_output = encoder_layer(perm_output)
    encoder = tf.keras.models.Model(inputs, encoder_output)
    decoder_output = decoder_layer(encoder_output)
    autoencoder = tf.keras.models.Model(inputs, decoder_output)

    return autoencoder, encoder


def get_features(data, start: str, end: str, assets: List, val_size=30 * 6, rescale=None):
    if end == str(data.index[-1]):
        end = '2020-03-08 00:00:00'
    else:
        end = pd.to_datetime(end) + dt.timedelta(days=6 * 30)
    train_data = data.loc[start:end, assets].iloc[:-1, :]
    train_data = train_data.loc[:, pd.IndexSlice[:, 'price']].droplevel(1, 1)

    val_data = data.loc[end:, assets]
    val_data = val_data.loc[:, pd.IndexSlice[:, 'price']].droplevel(1, 1)
    val_data = val_data.iloc[:val_size]

    test_data = data.loc[val_data.index[-1]:, assets].iloc[1:]
    test_data = test_data.loc[:, pd.IndexSlice[:, 'price']].droplevel(1, 1)
    test_data = test_data.iloc[:val_size]

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


if __name__ == "__main__":
    n_features = 1
    model, encoder, extra_features = pca_ae_model(input_dim=5,
                                                  encoding_dim=1,
                                                  n_features=n_features,
                                                  extra_features_dim=1,
                                                  activation='linear',
                                                  kernel_initializer='glorot_uniform',
                                                  activity_regularizer=None,
                                                  kernel_constraint=None,
                                                  kernel_regularizer=None)
    data = np.random.normal(size=1000).reshape((200, 5))
    extra = np.random.normal(size=200).reshape((200, 1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    model.fit([data, extra], data, epochs=1)
    model.save('test_model.h5')

    model, encoder, extra_features = pca_ae_model(input_dim=5,
                                                  encoding_dim=1,
                                                  n_features=n_features,
                                                  extra_features_dim=1,
                                                  activation='linear',
                                                  kernel_initializer='glorot_uniform',
                                                  activity_regularizer=None,
                                                  kernel_constraint=None,
                                                  kernel_regularizer=None)
    model.load_weights('test_model.h5')
    print(model.summary())

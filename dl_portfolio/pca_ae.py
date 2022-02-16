import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import backend as K
from dl_portfolio.custom_layer import DenseTied, TransposeDense, UncorrelatedFeaturesLayer
from dl_portfolio.constraints import PositiveSkewnessConstraint, NonNegAndUnitNorm, \
    UncorrelatedFeaturesConstraint
from dl_portfolio.regularizers import WeightsOrthogonality
from typing import List, Optional
from dl_portfolio.logger import LOGGER
import tensorflow as tf
import datetime as dt
import os
import seaborn as sns
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.callbacks import Callback
import tensorflow_probability as tfp


def create_linear_encoder_with_constraint(input_dim, encoding_dim):
    asset_input = tf.keras.layers.Input(input_dim, dtype=tf.float32, name='asset_input')
    kernel_constraint = NonNegAndUnitNorm(max_value=1., axis=0)  # tf.keras.constraints.NonNeg()#
    kernel_regularizer = WeightsOrthogonality(
        encoding_dim,
        weightage=1,
        axis=0,
        regularizer={'name': 'l2', 'params': {'l2': 1e-3}}
    )
    encoder_layer = tf.keras.layers.Dense(encoding_dim,
                                          activation='linear',
                                          kernel_initializer=tf.keras.initializers.HeNormal(),
                                          kernel_regularizer=kernel_regularizer,
                                          kernel_constraint=kernel_constraint,
                                          use_bias=True,
                                          name='encoder',
                                          dtype=tf.float32)
    encoder = tf.keras.models.Model(asset_input, encoder_layer(asset_input))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    encoder.compile(optimizer, 'mse')

    return encoder


def create_decoder(pca_ae_model, weights: Optional[List[np.ndarray]] = None):
    output_dim = pca_ae_model.layers[0].input_shape[0][-1]
    encoding_dim = pca_ae_model.layers[1].output_shape[-1]

    factors = tf.keras.layers.Input(encoding_dim, dtype=tf.float32, name='relu_factor')
    batch_norm = pca_ae_model.layers[2]
    output = tf.keras.layers.Dense(output_dim,
                                   activation='linear',
                                   dtype=tf.float32)
    if weights is None:
        W = pca_ae_model.layers[1].get_weights()[0].T
        b = pca_ae_model.layers[-1].get_weights()[0]
        weights = [W, b]

    decoder = tf.keras.models.Model(factors, output(batch_norm(factors)))
    decoder.layers[-1].set_weights(weights)

    return decoder


def build_model(model_type, input_dim, encoding_dim, **kwargs):
    if model_type == 'pca_permut_ae_model':
        raise NotImplementedError()
        # model, encoder = pca_permut_ae_model(input_dim,
        #                                      encoding_dim,
        #                                      activation=activation,
        #                                      kernel_initializer=kernel_initializer,
        #                                      kernel_constraint=kernel_constraint,
        #                                      kernel_regularizer=kernel_regularizer,
        #                                      activity_regularizer=activity_regularizer,
        #                                      pooling=pooling
        #                                      )
        # train_input = [train_data[:, i].reshape(-1, 1) for i in range(len(assets))]
        # val_input = [val_data[:, i].reshape(-1, 1) for i in range(len(assets))]
        # test_input = [test_data[:, i].reshape(-1, 1) for i in range(len(assets))]

    elif model_type == 'ae_model':
        model, encoder, extra_features = ae_model(input_dim,
                                                  encoding_dim,
                                                  **kwargs
                                                  )

    elif model_type == 'ae_model2':
        model, encoder, extra_features = ae_model2(input_dim,
                                                   encoding_dim,
                                                   **kwargs
                                                   )
    elif model_type == 'pca_ae_model':
        model, encoder, extra_features = pca_ae_model(input_dim,
                                                      encoding_dim,
                                                      **kwargs
                                                      )
    elif model_type == 'nl_pca_ae_model':
        model, encoder, extra_features = nl_pca_ae_model(input_dim,
                                                         encoding_dim,
                                                         **kwargs
                                                         )

    else:
        raise NotImplementedError()

    return model, encoder, extra_features


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


def heat_map(encoder_weights, show=False, save_dir=None, **kwargs):
    n_clusters = len(encoder_weights.columns)
    yticks = list(encoder_weights.index)

    fig, axs = plt.subplots(1, n_clusters, figsize=(10, 10), sharey=True)

    for j, c in enumerate(list(encoder_weights.columns)):
        ax = sns.heatmap(encoder_weights[c].values.reshape(-1, 1), xticklabels=[c], yticklabels=yticks,
                         ax=axs[j], cbar=j == n_clusters - 1, **kwargs)
    if save_dir:
        plt.savefig(f'{save_dir}/clusters_heatmap.png', bbox_inches='tight', pad_inches=0, transparent=True)
    if show:
        plt.show()


def heat_map_cluster(load_dir, show=False, save=False, filename='encoder_weights.p', **kwargs):
    """
    Heatmap of embedding matrix accros cv folds
    :param load_dir:
    :param show:
    :param save:
    :param filename:
    :param kwargs:
    :return:
    """
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
            for j, c in enumerate(list(encoder_weights[s].columns)):
                ax = sns.heatmap(encoder_weights[s][c].values.reshape(-1, 1), xticklabels=[c], yticklabels=yticks,
                                 ax=axs[i, j], cbar=j == n_clusters - 1, **kwargs)
    else:
        for j, c in enumerate(list(encoder_weights[last_set].columns)):
            ax = sns.heatmap(encoder_weights[last_set][c].values.reshape(-1, 1), xticklabels=[c], yticklabels=yticks,
                             ax=axs[j], cbar=j == n_clusters - 1, **kwargs)
    if save:
        plt.savefig(f'{load_dir}/clusters_heatmap.png', bbox_inches='tight', pad_inches=0, transparent=True)
    if show:
        plt.show()
    plt.close()


def get_layer_by_name(name, model):
    return [l for l in model.layers if l.name == name][0]


class NonNegAndUnitNormInit(tf.keras.initializers.Initializer):

    def __init__(self, initializer: str, **kwargs):
        if initializer == 'glorot_uniform':
            self.initializer = tf.keras.initializers.GlorotUniform(seed=kwargs.get('seed'))
        elif initializer == 'random_normal':
            self.initializer = tf.keras.initializers.RandomNormal(mean=kwargs.get('mean', 0.2),
                                                                  stddev=kwargs.get('stddev', 0.05))
        elif initializer == 'zeros':
            self.initializer = tf.keras.initializers.Zeros()
        elif initializer == 'ones':
            self.initializer = tf.keras.initializers.Ones()

        elif initializer == 'orthogonal':
            self.initializer = tf.keras.initializers.Orthogonal(
                gain=kwargs.get('gain', 1.0), seed=kwargs.get('seed')
            )
        elif initializer == 'truncated_normal':
            self.initializer = tf.keras.initializers.TruncatedNormal(
                mean=kwargs.get('mean', 0.1), stddev=kwargs.get('stddev', 0.05), seed=kwargs.get('seed')
            )
        elif initializer == 'he_normal':
            self.initializer = tf.keras.initializers.HeNormal(seed=kwargs.get('seed'))
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
    uncorrelated_features = kwargs.get('uncorrelated_features', True)
    batch_size = kwargs.get('batch_size', None)
    weightage = kwargs.get('weightage', 1.)
    batch_normalization = kwargs.get('batch_normalization', False)
    dropout = kwargs.get('dropout', None)

    if type(kernel_regularizer).__name__ == "WeightsOrthogonality":
        dkernel_regularizer = WeightsOrthogonality(
            input_dim,
            weightage=kernel_regularizer.weightage,
            axis=0)
        dkernel_regularizer.regularizer = dkernel_regularizer.regularizer
        # kernel_regularizer = WeightsOrthogonality(
        #     encoding_dim,
        #     weightage=kernel_regularizer.weightage,
        #     axis=0,
        #     regularizer={'name': "l2", 'params': {"l2": 1e-3}}
        # )
    if type(kernel_constraint).__name__ == "NonNegAndUnitNorm":
        dkernel_constraint = NonNegAndUnitNorm(max_value=1., axis=1)

    with CustomObjectScope({'MyActivityRegularizer': activity_regularizer}):  # required for Keras to recognize
        asset_input = tf.keras.layers.Input(input_dim, batch_size=batch_size, dtype=tf.float32, name='asset_input')
        encoder_layer = tf.keras.layers.Dense(encoding_dim,
                                              activation=activation,
                                              kernel_initializer=kernel_initializer,
                                              activity_regularizer=activity_regularizer,
                                              kernel_constraint=kernel_constraint,
                                              use_bias=True,
                                              name='encoder',
                                              dtype=tf.float32)
        decoder_layer = tf.keras.layers.Dense(input_dim,
                                              activation='linear',
                                              kernel_initializer=kernel_initializer,
                                              kernel_regularizer=dkernel_regularizer,
                                              kernel_constraint=dkernel_constraint,
                                              use_bias=True,
                                              name='decoder',
                                              dtype=tf.float32)
        encoding = encoder_layer(asset_input)

        if dropout is not None:
            dropout_layer = tf.keras.layers.Dropout(dropout)
            encoding = dropout_layer(encoding)

        if batch_normalization:
            batch_norm_layer = tf.keras.layers.BatchNormalization()
            encoding = batch_norm_layer(encoding)

        if uncorrelated_features:
            activity_regularizer_layer = UncorrelatedFeaturesLayer(encoding_dim, norm='1', use_cov=True,
                                                                   weightage=weightage)

            encoding = activity_regularizer_layer(encoding)
        encoder = tf.keras.models.Model(asset_input, encoding)

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

        return autoencoder, encoder, extra_features


def ae_model2(input_dim: int,
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
    uncorrelated_features = kwargs.get('uncorrelated_features', True)
    batch_size = kwargs.get('batch_size', None)
    weightage = kwargs.get('weightage', 1.)

    if type(kernel_regularizer).__name__ == "WeightsOrthogonality":
        kernel_regularizer1 = WeightsOrthogonality(
            int(input_dim / 2),
            weightage=kernel_regularizer.weightage,
            axis=0,
            # regularizer=kernel_regularizer.regularizer
        )
        kernel_regularizer1.regularizer = kernel_regularizer.regularizer
        kernel_regularizer2 = WeightsOrthogonality(
            encoding_dim,
            weightage=kernel_regularizer.weightage,
            axis=0,
            # regularizer=kernel_regularizer.regularizer
        )
        kernel_regularizer2.regularizer = kernel_regularizer.regularizer

    with CustomObjectScope({'MyActivityRegularizer': activity_regularizer}):  # required for Keras to recognize
        asset_input = tf.keras.layers.Input(input_dim, batch_size=batch_size, dtype=tf.float32, name='asset_input')
        encoder_layer1 = tf.keras.layers.Dense(int(input_dim / 2),
                                               activation=activation,
                                               kernel_initializer=kernel_initializer,
                                               kernel_regularizer=kernel_regularizer1,
                                               activity_regularizer=activity_regularizer,
                                               kernel_constraint=kernel_constraint,
                                               use_bias=True,
                                               name='encoder1',
                                               dtype=tf.float32)
        encoder_layer2 = tf.keras.layers.Dense(encoding_dim,
                                               activation=activation,
                                               kernel_initializer=kernel_initializer,
                                               kernel_regularizer=kernel_regularizer2,
                                               activity_regularizer=activity_regularizer,
                                               kernel_constraint=kernel_constraint,
                                               use_bias=True,
                                               name='encoder2',
                                               dtype=tf.float32)
        decoder_layer1 = tf.keras.layers.Dense(int(input_dim / 2),
                                               activation='linear',
                                               kernel_constraint=kernel_constraint,
                                               use_bias=True,
                                               name='decoder1',
                                               dtype=tf.float32)
        decoder_layer2 = tf.keras.layers.Dense(input_dim,
                                               activation='linear',
                                               kernel_constraint=kernel_constraint,
                                               use_bias=True,
                                               name='decoder2',
                                               dtype=tf.float32)
        encoding = encoder_layer2(encoder_layer1(asset_input))

        if uncorrelated_features:
            activity_regularizer_layer = UncorrelatedFeaturesLayer(encoding_dim, norm='1', use_cov=True,
                                                                   weightage=weightage)

            encoding = activity_regularizer_layer(encoding)
        encoder = tf.keras.models.Model(asset_input, encoding)

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
            output = decoder_layer2(decoder_layer1(hidden_layer))
            autoencoder = tf.keras.models.Model([asset_input, extra_input], output)
        else:
            output = decoder_layer2(decoder_layer1(encoding))
            autoencoder = tf.keras.models.Model(asset_input, output)
            extra_features = None

        encoder = tf.keras.models.Model(asset_input, encoding)

        return autoencoder, encoder, extra_features


def nl_pca_ae_model(input_dim: int,
                    encoding_dim: int,
                    n_features: int = None,
                    activation: str = 'linear',
                    kernel_initializer: str = 'glorot_uniform',
                    kernel_constraint=None,
                    kernel_regularizer=None,
                    **kwargs
                    ):
    uncorrelated_features = kwargs.get('uncorrelated_features', True)
    weightage = kwargs.get('weightage', 1.)
    batch_size = kwargs.get('batch_size', None)
    loss = kwargs.get('loss', None)
    batch_normalization = kwargs.get('batch_normalization', False)
    dropout = kwargs.get('dropout', None)

    asset_input = tf.keras.layers.Input(input_dim, batch_size=batch_size, dtype=tf.float32, name='asset_input')

    kernel_regularizer_1 = WeightsOrthogonality(
        int(input_dim / 2),
        weightage=1e-2,
        axis=0,
        regularizer={'name': 'l2', 'params': {'l2': 1e-3}}
    )
    kernel_regularizer_2 = WeightsOrthogonality(
        encoding_dim,
        weightage=1e-2,
        axis=0,
        regularizer={'name': 'l2', 'params': {'l2': 1e-3}}
    )
    encoder_layer_1 = tf.keras.layers.Dense(int(input_dim / 2),
                                            activation=activation,
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer_1,
                                            kernel_constraint=kernel_constraint,
                                            use_bias=True,
                                            name='encoder1',
                                            dtype=tf.float32)

    encoder_layer = tf.keras.layers.Dense(encoding_dim,
                                          activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer_2,
                                          kernel_constraint=kernel_constraint,
                                          use_bias=True,
                                          name='encoder2',
                                          dtype=tf.float32)

    decoder_layer_1 = DenseTied(int(input_dim / 2),
                                tied_to=encoder_layer,
                                n_features=n_features,
                                activation=activation,
                                use_bias=True,
                                dtype=tf.float32,
                                name='decoder1')

    decoder_layer = DenseTied(input_dim,
                              tied_to=encoder_layer_1,
                              n_features=n_features,
                              activation='linear',
                              use_bias=True,
                              dtype=tf.float32,
                              name='decoder2')

    encoding = encoder_layer(encoder_layer_1(asset_input))

    if dropout is not None:
        dropout_layer = tf.keras.layers.Dropout(dropout)
        encoding = dropout_layer(encoding)

    # if batch_normalization:
    #     batch_norm_layer = tf.keras.layers.BatchNormalization()
    #     encoding = batch_norm_layer(encoding)

    if uncorrelated_features:
        activity_regularizer_layer = UncorrelatedFeaturesLayer(encoding_dim, norm='1', use_cov=True,
                                                               weightage=weightage)

        encoding = activity_regularizer_layer(encoding)
    encoder = tf.keras.models.Model(asset_input, encoding)

    output = decoder_layer(decoder_layer_1(encoding))
    autoencoder = tf.keras.models.Model(asset_input, output)
    extra_features = None
    if loss == 'mse_with_covariance_penalty':
        loss = mse_with_covariance_penalty(asset_input, [output, encoding])
        autoencoder.add_loss(loss)

    return autoencoder, encoder, extra_features


def pca_ae_model(input_dim: int,
                 encoding_dim: int,
                 n_features: int = None,
                 extra_features_dim: Optional[int] = None,
                 activation: str = 'linear',
                 kernel_initializer: str = 'glorot_uniform',
                 kernel_constraint=None,
                 kernel_regularizer=None,
                 **kwargs
                 ):
    uncorrelated_features = kwargs.get('uncorrelated_features', True)
    weightage = kwargs.get('weightage', 1.)
    batch_size = kwargs.get('batch_size', None)
    loss = kwargs.get('loss', None)
    batch_normalization = kwargs.get('batch_normalization', False)
    dropout = kwargs.get('dropout', None)

    asset_input = tf.keras.layers.Input(input_dim, batch_size=batch_size, dtype=tf.float32, name='asset_input')
    encoder_layer = tf.keras.layers.Dense(encoding_dim,
                                          activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer,
                                          kernel_constraint=kernel_constraint,
                                          use_bias=True,
                                          name='encoder',
                                          dtype=tf.float32)

    decoder_layer = DenseTied(input_dim,
                              tied_to=encoder_layer,
                              n_features=n_features,
                              activation='linear',
                              use_bias=True,
                              dtype=tf.float32,
                              name='decoder')

    encoding = encoder_layer(asset_input)

    if dropout is not None:
        dropout_layer = tf.keras.layers.Dropout(dropout)
        encoding = dropout_layer(encoding)

    if batch_normalization:
        batch_norm_layer = tf.keras.layers.BatchNormalization()
        encoding = batch_norm_layer(encoding)
    if uncorrelated_features:
        activity_regularizer_layer = UncorrelatedFeaturesLayer(encoding_dim, norm='1', use_cov=True,
                                                               weightage=weightage)

        encoding = activity_regularizer_layer(encoding)
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


# def pca_permut_ae_model(input_dim: int, encoding_dim: int, activation: str = 'linear',
#                         kernel_initializer: str = 'glorot_uniform',
#                         ortho_weights: bool = True,
#                         non_neg_unit_norm: bool = True,
#                         uncorr_features: bool = True,
#                         activity_regularizer=None,
#                         non_neg=False,
#                         pooling=None,
#                         **kwargs
#                         ):
#     inputs = repeat_layers(Input, [(1,) for i in range(input_dim)])  # [a, b, c]
#     perm_layer1 = PermutationalLayer(
#         PermutationalEncoder(
#             PairwiseModel((1,), repeat_layers(Dense, [input_dim], activation="relu", name="hidden")),
#             input_dim
#         ),
#         name="permutational_layer1"
#     )
#
#     outputs = perm_layer1(inputs)
#     print(len(outputs))
#     if pooling == 'maximum':
#         perm_output = maximum(outputs)
#     elif pooling == 'average':
#         perm_output = average(outputs)
#     else:
#         perm_output = tf.concat(outputs, axis=-1)
#
#     permutation_model = Model(inputs, perm_output, name='permutation')
#     print("# Multi-layer model summary")
#     print(permutation_model.summary())
#
#     kernel_regularizer = WeightsOrthogonality(encoding_dim, axis=0) if ortho_weights else None
#     if non_neg_unit_norm:
#         assert not non_neg
#         kernel_constraint = NonNegAndUnitNorm(axis=0)
#     elif non_neg:
#         kernel_constraint = tf.keras.constraints.NonNeg()
#     else:
#         kernel_constraint = None
#     if activity_regularizer is None:
#         weightage = kwargs.get('weightage', 1.)
#         activity_regularizer = UncorrelatedFeaturesConstraint(encoding_dim,
#                                                               weightage=weightage) if uncorr_features else None
#     else:
#         assert not uncorr_features
#
#     encoder_layer = tf.keras.layers.Dense(encoding_dim,
#                                           activation=activation,
#                                           kernel_initializer=kernel_initializer,
#                                           kernel_regularizer=kernel_regularizer,
#                                           activity_regularizer=activity_regularizer,
#                                           kernel_constraint=kernel_constraint,
#                                           use_bias=True,
#                                           name='encoder',
#                                           dtype=tf.float32)
#     decoder_layer = DenseTied(input_dim,
#                               tied_to=encoder_layer,
#                               activation='linear',
#                               kernel_initializer=kernel_initializer,
#                               kernel_regularizer=kernel_regularizer,
#                               use_bias=True,
#                               dtype=tf.float32,
#                               name='decoder')
#     encoder_output = encoder_layer(perm_output)
#     encoder = tf.keras.models.Model(inputs, encoder_output)
#     decoder_output = decoder_layer(encoder_output)
#     autoencoder = tf.keras.models.Model(inputs, decoder_output)
#
#     return autoencoder, encoder


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

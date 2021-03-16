import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import backend as K
from dl_portfolio.custom_layer import DenseTied, TransposeDense
from dl_portfolio.constraints import WeightsOrthogonalityConstraint, NonNegAndUnitNorm, UncorrelatedFeaturesConstraint
from typing import List, Dict
from dl_portfolio.logger import LOGGER
import tensorflow as tf
import datetime as dt
import os
import seaborn as sns
from dl_portfolio.data import build_seq
from dl_portfolio.pca_ae import NonNegAndUnitNormInit, heat_map_cluster, get_layer_by_name, get_features
from dl_portfolio.model import build_layer
import pickle

LOG_DIR = 'dl_portfolio/log_RAE'


def get_layer_config(encoding_dim, n_assets, seq_len):
    layers = [
        {
            'type': 'lstm',
            'neurons': encoding_dim,
            'params': {
                'activation': 'linear',
                'activity_regularizer': UncorrelatedFeaturesConstraint(encoding_dim, weightage=1.),
                # tf.keras.regularizers.l1(1e-6),
                # 'kernel_initializer': tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                # 'glorot_uniform' # NonNegAndUnitNormInit(initializer='glorot_uniform')
                # 'kernel_constraint': tf.keras.constraints.NonNeg(),
                'kernel_regularizer': WeightsOrthogonalityConstraint(encoding_dim, max_dim=encoding_dim, axis=0),
                'kernel_constraint': NonNegAndUnitNorm(max_dim=encoding_dim),
                #'kernel_constraint': tf.keras.constraints.NonNeg(),
                'return_sequences': False,
                'use_bias': True,
                'name': 'encoder'
            }
        },
        {
            'type': 'RepeatVector',
            'neurons': seq_len
        },
        {
            'type': 'lstm',
            'neurons': n_assets,
            'params': {
                'activation': 'linear',
                'activity_regularizer': tf.keras.regularizers.l1(1e-3), # tf.keras.regularizers.l1(1e-6),
                # 'kernel_initializer': tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                # 'glorot_uniform' # NonNegAndUnitNormInit(initializer='glorot_uniform')
                # 'kernel_constraint': tf.keras.constraints.NonNeg(),
                'kernel_regularizer': WeightsOrthogonalityConstraint(n_assets, max_dim=n_assets, axis=0),
                # 'kernel_constraint': NonNegAndUnitNorm(max_dim=n_assets),
                # 'kernel_constraint': tf.keras.constraints.NonNeg(),
                'return_sequences': True,
                'use_bias': True,
                'name': 'decoder_2'
            }
        },
        {
            'type': 'output',
            'params':
                {
                    'activity_regularizer': tf.keras.regularizers.l1(1e-3)
                }
        }
    ]

    return layers


def lstm_ae_model(input_dim: int,
                  layers: List[Dict],
                  **kwargs
                  ):
    assert len(input_dim) == 2
    input_ = tf.keras.layers.Input(input_dim, dtype=tf.float32, name='input')

    for i, config_layer in enumerate(layers):
        if config_layer['type'] != 'output':
            layer = build_layer(config_layer)
            if i == 0:
                hidden = layer(input_)
            else:
                hidden = layer(hidden)
            if 'params' in config_layer and config_layer['params'].get('name') == 'encoder':
                encoder = hidden
        else:
            output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim[-1],
                                                                           **config_layer.get('params', {}),
                                                                           dtype=tf.float32))(hidden)

    """
    encoder_layer_1 = tf.keras.layers.LSTM(encoding_dim,
                                           **kwargs,
                                           use_bias=True,
                                           name='encoder_1',
                                           return_sequences=False,
                                           dtype=tf.float32)
    repeat = tf.keras.layers.RepeatVector(input_dim[0])
    decoder_layer_1 = tf.keras.layers.LSTM(input_dim[-1],
                                           **kwargs,
                                           use_bias=True,
                                           name='decoder_1',
                                           return_sequences=True,
                                           dtype=tf.float32)
    output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim[-1]))
    output = output_layer(decoder_layer_1(repeat(encoder_layer_1(input_))))
    autoencoder = tf.keras.models.Model(input_, output)
    encoding = encoder_layer_1(input_)
    encoder = tf.keras.models.Model(input_, encoding)
    """
    autoencoder = tf.keras.models.Model(input_, output)
    encoder = tf.keras.models.Model(input_, encoder)

    return autoencoder, encoder


if __name__ == "__main__":
    # seed = np.random.randint(100)
    seed = 10
    np.random.seed(seed)
    tf.random.set_seed(seed)
    LOGGER.info(f"Set seed: {seed}")
    save = True
    model_name = f'linear_simple_{seed}_uncorr_lr_e-3_l1_reg'
    seq_len = 5
    learning_rate = 1e-3
    epochs = 600
    batch_size = 128
    activation = 'linear'
    encoding_dim = 2
    val_size = 30 * 6
    loss = 'mae'
    rescale = None

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_rmse', min_delta=0, patience=300, verbose=1,
            mode='min', baseline=None, restore_best_weights=True
        )
    ]

    if save:
        subdir = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')
        if model_name is not None and model_name != '':
            subdir = subdir + '_' + model_name
        save_dir = f"{LOG_DIR}/{subdir}"
        os.mkdir(save_dir)

    # Load data
    data_specs = json.load(open('./data/crypto_data/coingecko/data_spec_coins_selected.json', 'r'))
    data = pd.read_csv('./data/crypto_data/coingecko/coins_selected.csv', index_col=0, header=[0, 1])
    data = data.astype(np.float32)
    data.index = pd.to_datetime(data.index)
    print(data.head())

    fxdata = pd.read_csv('./data/forex/daily_price.csv', index_col=0, header=[0, 1])
    fxdata = fxdata.astype(np.float32)
    fxdata.index = pd.to_datetime(fxdata.index)
    fxdata = fxdata.loc[:,pd.IndexSlice[:,'close']]
    fx_assets = np.unique(list(fxdata.columns.get_level_values(0))).tolist()
    fxdata.columns = pd.MultiIndex.from_product(([fx_assets, ['price']]))
    print(fx_assets)
    print(fxdata.head())

    data = pd.concat([data, fxdata], 1)
    asset_to_train_on = data_specs['0']['assets'] + ['monero', 'stellar', 'nem', 'ethereum'] + fx_assets
    spec = {str(len(data_specs) - 1): data_specs[str(len(data_specs) - 1)]}
    spec[str(len(data_specs) - 1)]['start'] = data_specs['0']['start']
    spec[str(len(data_specs) - 1)]['assets'] = asset_to_train_on
    data_specs = spec

    asset_to_train_on = {}
    for set_ in data_specs:
        LOGGER.info(f'Starting with set: {set_}')
        if save:
            os.mkdir(f"{save_dir}/{set_}")
        data_spec = data_specs[set_]
        # if int(set_) > 0:
        #     prev_set = str(int(set_) - 1)
        #     assets = asset_to_train_on[prev_set].copy()
        #     assets = assets + data_spec['assets']
        # else:
        #     assets = data_spec['assets'].copy()
        # asset_to_train_on[set_] = assets
        asset_to_train_on[set_] = data_spec['assets'].copy()
        assets = data_spec['assets'].copy()

        train_data, val_data, test_data, scaler, dates = get_features(data, data_spec['start'], data_spec['end'],
                                                                      assets,
                                                                      val_size=val_size, rescale=rescale)
        LOGGER.info(f'Train shape: {train_data.shape}')
        LOGGER.info(f'Validation shape: {val_data.shape}')

        # Build model
        train_data = build_seq(train_data, seq_len)
        val_data = build_seq(val_data, seq_len)
        test_data = build_seq(test_data, seq_len)

        input_dim = (seq_len, len(assets))
        layers = get_layer_config(encoding_dim, len(assets), seq_len)
        model, encoder = lstm_ae_model(input_dim, layers)

        print(model.summary())

        # print(model.layers[1].get_weights())
        # weights = model.layers[1].get_weights()[0]
        # rec_weights = model.layers[1].get_weights()[1]
        # print(weights.shape)
        # print(rec_weights.shape)
        #
        # weights = model.layers[3].get_weights()[0]
        # rec_weights = model.layers[3].get_weights()[1]
        # print(weights.shape)
        # print(rec_weights.shape)

        # if int(set_) > 0:
        #     LOGGER.info('Set weights')
        #     # Set encoder layer weights
        #     weights = model.layers[1].get_weights()[0]
        #     weights[:-1] = prev_encoder_weights[0]
        #     bias = prev_encoder_weights[1]
        #     model.layers[1].set_weights([weights, bias])
        #     encoder.layers[1].set_weights([weights, bias])
        #
        #     # Set decoder layer weights
        #     weights = model.layers[3].get_weights()[0]
        #     weights[:-1] = prev_decoder_weights[0]
        #     bias = model.layers[3].get_weights()[1]
        #     bias[:-1] = prev_decoder_weights[1]
        #     model.layers[2].set_weights([weights, bias])

        # Train
        LOGGER.info('Start training')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Very low learning rate
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                               tf.keras.metrics.RootMeanSquaredError(name='rmse')]
                      )
        history = model.fit(train_data, train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(val_data, val_data),
                            validation_batch_size=batch_size,
                            callbacks=callbacks,
                            shuffle=False,
                            verbose=1)
        if save:
            model.save(f"{save_dir}/{set_}/model.h5")

        prev_encoder_weights = model.layers[1].get_weights()
        prev_decoder_weights = model.layers[3].get_weights()

        fix, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(history.history['loss'], label='loss')
        axs[0].plot(history.history['val_loss'], label='val loss')
        axs[0].legend()
        axs[1].plot(history.history['mae'], label='mae')
        axs[1].plot(history.history['val_mae'], label='val_mae')
        axs[1].legend()
        axs[2].plot(history.history['rmse'], label='rmse')
        axs[2].plot(history.history['val_rmse'], label='val_rmse')
        axs[2].legend()
        if save:
            plt.savefig(f"{save_dir}/{set_}/history.png")
        plt.show()

        # Evaluate
        model.evaluate(train_data, train_data)
        model.evaluate(val_data, val_data)

        val_prediction = model.predict(val_data)
        # val_prediction = pd.DataFrame(val_prediction, columns=assets, index=dates['val'])
        encoder_weights = pd.DataFrame(prev_encoder_weights[0][:, :encoding_dim], index=assets)
        print(encoder_weights)
        print(encoder_weights.sum())
        test_prediction = model.predict(test_data)
        # test_prediction = pd.DataFrame(test_prediction, columns=assets, index=dates['test'])

        # if save:
        #     val_prediction.to_pickle(f"{save_dir}/{set_}/val_prediction.p")
        #     test_prediction.to_pickle(f"{save_dir}/{set_}/test_prediction.p")
        #     encoder_weights.to_pickle(f"{save_dir}/{set_}/encoder_weights.p")
        if save:
            pickle.dump(val_prediction, open(f"{save_dir}/{set_}/val_prediction.p", 'wb'))
            pickle.dump(test_prediction, open(f"{save_dir}/{set_}/test_prediction.p", "wb"))
            encoder_weights.to_pickle(f"{save_dir}/{set_}/encoder_weights.p")

        # indices = np.random.choice(list(range(len(val_data))), 5).tolist()
        # xticks = assets
        # for i in indices:
        #     plt.figure()
        #     plt.scatter(xticks, val_data[i], label='truth')
        #     plt.scatter(xticks, val_prediction.values[i], label='prediction')
        #     plt.legend()
        #     plt.show()

        # for i in range(input_dim):
        #     rmse = np.sqrt(np.mean((val_prediction.values[:, i] - val_data[:, i]) ** 2))
        #     print(assets[i], rmse)

        val_features = encoder.predict(val_data)
        print(np.corrcoef(val_features.T))

    if save:
        heat_map_cluster(save_dir, show=True, save=save)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from dl_portfolio.logger import LOGGER
import tensorflow as tf
import datetime as dt
import os
from dl_portfolio.pca_ae import NonNegAndUnitNormInit, heat_map_cluster, pca_ae_model, get_features
from typing import List
from sklearn import preprocessing

LOG_DIR = 'dl_portfolio/log_fx_AE'


def get_features(data, start: str, end: str, assets: List, val_size=30 * 6, rescale=None):
    data = data.loc[start:end, assets]
    data = data.loc[:, pd.IndexSlice[:, 'price']].droplevel(1, 1)

    # Train/val/test split
    train_data = data.iloc[:-val_size * 2, :]
    val_data = data.loc[train_data.index[-1]:, :].iloc[1:val_size]
    test_data = data.loc[val_data.index[-1]:, assets].iloc[1:]

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


if __name__ == "__main__":
    seed = np.random.randint(100)
    seed = 69
    np.random.seed(seed)
    tf.random.set_seed(seed)
    LOGGER.info(f"Set seed: {seed}")
    fx = True
    save = True
    model_name = f'linear_simple_{seed}_uncorr_lr_e-3_fx_cv_encoding_3'
    learning_rate = 1e-3
    epochs = 600
    batch_size = 128
    activation = 'linear'
    encoding_dim = 3
    val_size = 30 * 3
    uncorr_features = True
    weightage = 1e-3  # 1e-2
    activity_regularizer = None  # tf.keras.regularizers.l1(1e-3)
    loss = 'mse'
    rescale = None
    kernel_initializer = NonNegAndUnitNormInit(initializer='glorot_uniform')
    ortho_weights = True
    non_neg_unit_norm = True
    non_neg = False

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_rmse', min_delta=1e-4, patience=300, verbose=1,
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

    fxdata = pd.read_csv('./data/forex/daily_price.csv', index_col=0, header=[0, 1])
    fxdata = fxdata.astype(np.float32)
    fxdata.index = pd.to_datetime(fxdata.index)
    fxdata = fxdata.loc[:, pd.IndexSlice[:, 'close']]
    fx_assets = np.unique(list(fxdata.columns.get_level_values(0))).tolist()
    fxdata.columns = pd.MultiIndex.from_product(([fx_assets, ['price']]))
    data = pd.concat([data, fxdata], 1)
    assets = ["bitcoin", "dogecoin", "litecoin", "ripple", 'monero', 'stellar', 'nem',
              'ethereum'] + fx_assets

    data_specs = {
        0: {
            'start': '2015-08-07',
            'end': '2019-12-08'
        },
        1: {
            'start': '2015-08-07',
            'end': '2020-03-08'
        },
        2: {
            'start': '2015-08-07',
            'end': '2020-06-08'
        },
        3: {
            'start': '2015-08-07',
            'end': '2020-09-08'
        },
        4: {
            'start': '2015-08-07',
            'end': '2020-12-08'
        },
        5: {
            'start': '2015-08-07',
            'end': '2021-03-08'
        }
    }

    for cv in data_specs:
        LOGGER.info(f'Starting with cv: {cv}')
        if save:
            os.mkdir(f"{save_dir}/{cv}")
        data_spec = data_specs[cv]

        train_data, val_data, test_data, scaler, dates = get_features(data, data_spec['start'], data_spec['end'],
                                                                      assets, val_size=val_size, rescale=rescale)
        LOGGER.info(f'Train shape: {train_data.shape}')
        LOGGER.info(f'Validation shape: {val_data.shape}')
        # Build model
        input_dim = len(assets)
        model, encoder = pca_ae_model(input_dim, encoding_dim, activation=activation,
                                      kernel_initializer=kernel_initializer,
                                      ortho_weights=ortho_weights,
                                      non_neg_unit_norm=non_neg_unit_norm,
                                      uncorr_features=uncorr_features,
                                      activity_regularizer=activity_regularizer,
                                      non_neg=non_neg,
                                      weightage=weightage
                                      )

        print(model.summary())
        # Train
        LOGGER.info('Start training')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Very low learning rate
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanSquaredError(name='mse'),
                               tf.keras.metrics.RootMeanSquaredError(name='rmse')]
                      )
        if save:
            callbacks.append(
                [
                    tf.keras.callbacks.ModelCheckpoint(
                        f"{save_dir}/{cv}/best_model.h5",
                        verbose=1,
                        save_best_only=True,
                        monitor="val_rmse")
                ]
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
            model.save(f"{save_dir}/{cv}/model.h5")

        fix, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(history.history['loss'], label='loss')
        axs[0].plot(history.history['val_loss'], label='val loss')
        axs[0].legend()
        axs[1].plot(history.history['mse'], label='mse')
        axs[1].plot(history.history['val_mse'], label='val_mse')
        axs[1].legend()
        axs[2].plot(history.history['rmse'], label='rmse')
        axs[2].plot(history.history['val_rmse'], label='val_rmse')
        axs[2].legend()
        if save:
            plt.savefig(f"{save_dir}/{cv}/history.png")
        plt.show()

        if save:
            LOGGER.info(f"Loading weights from {save_dir}/{cv}/best_model.h5")
            model.load_weights(f"{save_dir}/{cv}/best_model.h5")

        # Evaluate
        model.evaluate(train_data, train_data)
        model.evaluate(val_data, val_data)

        # Results
        val_prediction = model.predict(val_data)
        val_prediction = pd.DataFrame(val_prediction, columns=assets, index=dates['val'])
        # indices = np.random.choice(list(range(len(val_data))), 5).tolist()
        # xticks = assets
        # for i in indices:
        #     plt.figure()
        #     plt.scatter(xticks, val_data[i], label='truth')
        #     plt.scatter(xticks, val_prediction.values[i], label='prediction')
        #     plt.legend()
        #     plt.show()

        encoder_weights = model.layers[1].get_weights()
        # decoder_weights = model.layers[2].get_weights()
        encoder_weights = pd.DataFrame(encoder_weights[0], index=assets)
        LOGGER.info(f"Encoder weights:\n{encoder_weights}")

        test_prediction = model.predict(test_data)
        test_prediction = scaler.inverse_transform(test_prediction)
        test_prediction = pd.DataFrame(test_prediction, columns=assets, index=dates['test'])
        val_data = scaler.inverse_transform(val_data)
        val_data = pd.DataFrame(val_data, index=dates['val'], columns=assets)
        test_data = scaler.inverse_transform(test_data)
        test_data = pd.DataFrame(test_data, index=dates['test'], columns=assets)

        val_cluster_portfolio = encoder.predict(val_data)
        val_cluster_portfolio = pd.DataFrame(val_cluster_portfolio, index=dates['val'])

        LOGGER.info(f"Encoder feature correlation:\n{np.corrcoef(val_cluster_portfolio.T)}")
        LOGGER.info(f"Unit norm constraint:\n{encoder.layers[-1].kernel.numpy().sum(0)}")

        if save:
            val_data.to_pickle(f"{save_dir}/{cv}/val_returns.p")
            test_data.to_pickle(f"{save_dir}/{cv}/test_returns.p")
            val_prediction.to_pickle(f"{save_dir}/{cv}/val_prediction.p")
            test_prediction.to_pickle(f"{save_dir}/{cv}/test_prediction.p")
            encoder_weights.to_pickle(f"{save_dir}/{cv}/encoder_weights.p")
            val_cluster_portfolio.to_pickle(f"{save_dir}/{cv}/val_cluster_portfolio.p")

    if save:
        heat_map_cluster(save_dir, show=True, save=save, vmax=1., vmin=0.)
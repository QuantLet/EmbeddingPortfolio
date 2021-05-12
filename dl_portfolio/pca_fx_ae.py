import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os, pickle
from dl_portfolio.logger import LOGGER
import tensorflow as tf
import datetime as dt

from dl_portfolio.pca_ae import NonNegAndUnitNormInit, heat_map_cluster, pca_ae_model, get_layer_by_name, heat_map, pca_permut_ae_model
from typing import List
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import backend as K
from shutil import copyfile

LOG_DIR = 'dl_portfolio/log_fx_AE'


# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


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


def load_data(type=['indices', 'forex', 'forex_metals', 'crypto']):
    data = pd.DataFrame()
    assets = []
    END = '2021-01-30 12:30:00'
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
            fx_assets = ['CADUSD', 'CHFUSD', 'EURUSD', 'GBPUSD', 'JPYUSD']
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
            indices = ['UKXUSD', 'FRXUSD', 'JPXUSD', 'SPXUSD']
            indices_data = pd.read_pickle('./data/histdatacom/indices_f_3600_2014_2021_close_index.p')
            indices_data = indices_data.loc[:, pd.IndexSlice[indices, 'close']].droplevel(1, 1)
            data = pd.concat([data, indices_data], 1)
            del indices_data
            assets = assets + indices
        else:
            raise ValueError(asset_class)

    # assets = np.random.choice(assets, len(assets), replace=False).tolist()
    # data = data.loc[:, pd.IndexSlice[assets, 'price']]
    # data = pd.DataFrame(data.values, columns=pd.MultiIndex.from_product([assets, ['price']]), index=data.index)

    data = data.loc[:END]
    if 'crypto' in type:
        data = data.fillna(method='ffill')
        data = data.dropna()
    else:
        data = data.dropna()
    data = data.loc[:, assets]

    return data, assets


if __name__ == "__main__":
    from dl_portfolio.config.ae_config import *
    random_seed = np.random.randint (0, 100)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    LOGGER.info(f"Set seed: {seed}")
    if save:
        subdir = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_s%H%M%S')
        if model_name is not None and model_name != '':
            subdir = subdir + '_' + model_name
        save_dir = f"{LOG_DIR}/{subdir}"
        os.mkdir(save_dir)
        copyfile('./dl_portfolio/config/ae_config.py',
                 os.path.join(save_dir, 'ae_config.py'))

    for cv in data_specs:
        LOGGER.info(f'Starting with cv: {cv}')
        if save:
            os.mkdir(f"{save_dir}/{cv}")
        data_spec = data_specs[cv]

        data, assets = load_data(type=data_type)

        if shuffle_columns:
            LOGGER.info('Shuffle assets order')
            base_asset_order = assets.copy()
            np.random.seed(random_seed)
            np.random.shuffle(assets)
            np.random.seed(seed)
        LOGGER.info(f'Assets order: {assets}')

        train_data, val_data, test_data, scaler, dates = get_features(data, data_spec['start'], data_spec['end'],
                                                                      assets, val_size=val_size, rescale=rescale)

        if shuffle_columns_while_training:
            train_data = np.transpose(train_data)
            np.random.seed(random_seed)
            np.random.shuffle(train_data)
            np.random.seed(seed)
            train_data = np.transpose(train_data)

        LOGGER.info(f'Train shape: {train_data.shape}')
        LOGGER.info(f'Validation shape: {val_data.shape}')
        # Build model
        input_dim = len(assets)
        if model_type == 'pca_permut_ae_model':
            model, encoder = pca_permut_ae_model(input_dim, encoding_dim, activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          ortho_weights=ortho_weights,
                                          non_neg_unit_norm=non_neg_unit_norm,
                                          uncorr_features=uncorr_features,
                                          activity_regularizer=activity_regularizer,
                                          non_neg=non_neg,
                                          weightage=weightage,
                                          pooling=pooling
                                          )
            train_input = [train_data[:,i].reshape(-1,1) for i in range(len(assets))]
            val_input = [val_data[:,i].reshape(-1,1) for i in range(len(assets))]
            test_input = [test_data[:,i].reshape(-1,1) for i in range(len(assets))]

        elif model_type == 'pca_ae_model':
            model, encoder = pca_ae_model(input_dim, encoding_dim, activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          ortho_weights=ortho_weights,
                                          non_neg_unit_norm=non_neg_unit_norm,
                                          uncorr_features=uncorr_features,
                                          activity_regularizer=activity_regularizer,
                                          non_neg=non_neg,
                                          weightage=weightage
                                          )
            train_input = train_data
            val_input = val_data
            test_input = test_data
        else:
            raise NotImplementedError()

        print(model.summary())
        # Train
        LOGGER.info('Start training')
        print(input_dim)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Very low learning rate
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.MeanSquaredError(name='mse'),
                               r_square,
                               tf.keras.metrics.RootMeanSquaredError(name='rmse')]
                      )
        if save:
            callbacks.append(
                [
                    tf.keras.callbacks.ModelCheckpoint(
                        f"{save_dir}/{cv}/best_model.h5",
                        verbose=1,
                        save_best_only=True,
                        mode='max',
                        monitor="val_r_square")
                ]
            )

        history = model.fit(train_input, train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(val_input, val_data),
                            validation_batch_size=batch_size,
                            callbacks=callbacks,
                            shuffle=False,
                            verbose=1)
        if save:
            model.save(f"{save_dir}/{cv}/model.h5")

        fix, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].plot(history.history['loss'], label='loss')
        axs[0].plot(history.history['val_loss'], label='val loss')
        axs[0].legend()
        axs[1].plot(history.history['mse'], label='mse')
        axs[1].plot(history.history['val_mse'], label='val_mse')
        axs[1].legend()
        axs[2].plot(history.history['rmse'], label='rmse')
        axs[2].plot(history.history['val_rmse'], label='val_rmse')
        axs[2].legend()
        axs[3].plot(history.history['r_square'], label='R2')
        axs[3].plot(history.history['val_r_square'], label='val_R2')
        axs[3].legend()
        if save:
            plt.savefig(f"{save_dir}/{cv}/history.png")
        plt.show()

        if save:
            LOGGER.info(f"Loading weights from {save_dir}/{cv}/best_model.h5")
            model.load_weights(f"{save_dir}/{cv}/best_model.h5")

        # Evaluate
        model.evaluate(train_input, train_data)
        model.evaluate(val_input, val_data)

        # PCA baseline
        # pca = PCA(n_components=encoding_dim, random_state=seed)
        # pca.fit(train_data)
        # encoding_pca = pca.components_.T
        # encoding_pca = pd.DataFrame(encoding_pca, index=assets)
        #
        # train_pca_cluster_portfolio = pca.transform(train_data)
        # train_pca_cluster_portfolio = pd.DataFrame(train_pca_cluster_portfolio, index=dates['train'])
        #
        # val_pca_cluster_portfolio = pca.transform(val_data)
        # val_pca_cluster_portfolio = pd.DataFrame(val_pca_cluster_portfolio, index=dates['val'])
        #
        # test_pca_cluster_portfolio = pca.transform(test_data)
        # test_pca_cluster_portfolio = pd.DataFrame(test_pca_cluster_portfolio, index=dates['test'])
        #
        # pca_cluster_portfolio = {
        #     'train': train_pca_cluster_portfolio,
        #     'val': val_pca_cluster_portfolio,
        #     'test': test_pca_cluster_portfolio
        # }

        # Results
        val_prediction = model.predict(val_input)
        val_prediction = pd.DataFrame(val_prediction, columns=assets, index=dates['val'])
        # indices = np.random.choice(list(range(len(val_data))), 5).tolist()
        # xticks = assets
        # for i in indices:
        #     plt.figure()
        #     plt.scatter(xticks, val_data[i], label='truth')
        #     plt.scatter(xticks, val_prediction.values[i], label='prediction')
        #     plt.legend()
        #     plt.show()

        encoder_layer = get_layer_by_name(name='encoder', model=model)
        encoder_weights = encoder_layer.get_weights()
        # decoder_weights = model.layers[2].get_weights()
        encoder_weights = pd.DataFrame(encoder_weights[0], index=assets)
        LOGGER.info(f"Encoder weights:\n{encoder_weights}")

        test_prediction = model.predict(test_input)
        test_prediction = scaler.inverse_transform(test_prediction)
        test_prediction = pd.DataFrame(test_prediction, columns=assets, index=dates['test'])

        train_data = scaler.inverse_transform(train_data)
        train_data = pd.DataFrame(train_data, index=dates['train'], columns=assets)
        val_data = scaler.inverse_transform(val_data)
        val_data = pd.DataFrame(val_data, index=dates['val'], columns=assets)
        test_data = scaler.inverse_transform(test_data)
        test_data = pd.DataFrame(test_data, index=dates['test'], columns=assets)

        # train_cluster_portfolio = encoder.predict(train_data)
        # train_cluster_portfolio = pd.DataFrame(train_cluster_portfolio, index=dates['train'])
        train_cluster_portfolio = pd.DataFrame(np.dot(train_data, encoder_weights / encoder_weights.sum()),
                                               index=dates['train'])

        # val_cluster_portfolio = encoder.predict(val_data)
        # val_cluster_portfolio = pd.DataFrame(val_cluster_portfolio, index=dates['val'])
        val_cluster_portfolio = pd.DataFrame(np.dot(val_data, encoder_weights / encoder_weights.sum()),
                                             index=dates['val'])

        # test_cluster_portfolio = encoder.predict(test_data)
        # test_cluster_portfolio = pd.DataFrame(test_cluster_portfolio, index=dates['test'])
        test_cluster_portfolio = pd.DataFrame(np.dot(test_data, encoder_weights / encoder_weights.sum()),
                                              index=dates['test'])

        if shuffle_columns:
            LOGGER.info('Reorder results with base asset order')
            val_prediction = val_prediction.loc[:, base_asset_order]
            train_data = train_data.loc[:, base_asset_order]
            val_data = val_data.loc[:, base_asset_order]
            test_data = test_data.loc[:, base_asset_order]
            test_prediction = test_prediction.loc[:, base_asset_order]
            encoder_weights = encoder_weights.loc[base_asset_order, :]

        if save:
            heat_map(encoder_weights, show=True, save=save, save_dir=f"{save_dir}/{cv}", vmax=1., vmin=0.)
        else:
            heat_map(encoder_weights, show=True, vmax=1., vmin=0.)

        cluster_portfolio = {
            'train': train_cluster_portfolio,
            'val': val_cluster_portfolio,
            'test': test_cluster_portfolio
        }


        LOGGER.info(f"Encoder feature correlation:\n{np.corrcoef(val_cluster_portfolio.T)}")
        LOGGER.info(f"Unit norm constraint:\n{encoder.layers[-1].kernel.numpy().sum(0)}")

        if save:
            train_data.to_pickle(f"{save_dir}/{cv}/train_returns.p")
            val_data.to_pickle(f"{save_dir}/{cv}/val_returns.p")
            test_data.to_pickle(f"{save_dir}/{cv}/test_returns.p")
            val_prediction.to_pickle(f"{save_dir}/{cv}/val_prediction.p")
            test_prediction.to_pickle(f"{save_dir}/{cv}/test_prediction.p")
            encoder_weights.to_pickle(f"{save_dir}/{cv}/encoder_weights.p")
            # encoding_pca.to_pickle(f"{save_dir}/{cv}/encoding_pca.p")
            pickle.dump(cluster_portfolio, open(f"{save_dir}/{cv}/cluster_portfolio.p", "wb"))
            # pickle.dump(pca_cluster_portfolio, open(f"{save_dir}/{cv}/pca_cluster_portfolio.p", "wb"))

    if save:
        heat_map_cluster(save_dir, show=True, save=save, vmax=1., vmin=0.)

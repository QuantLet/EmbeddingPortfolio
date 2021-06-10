import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os, pickle
from dl_portfolio.logger import LOGGER
import tensorflow as tf
import datetime as dt
from dl_portfolio.pymarkowitz.Moments import MomentGenerator
from dl_portfolio.pca_ae import ActivityRegularizer, NonNegAndUnitNormInit, heat_map_cluster, pca_ae_model, \
    get_layer_by_name, heat_map, pca_permut_ae_model
from typing import List
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import backend as K
from shutil import copyfile
from dl_portfolio.data import drop_remainder
import tensorflow as tf
from tensorboard.plugins import projector
from dl_portfolio.losses import weighted_mae, weighted_mse

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


if __name__ == "__main__":
    from dl_portfolio.config.ae_config import *

    random_seed = np.random.randint(0, 100)
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
    data, assets = load_data(type=data_type, drop_weekends=drop_weekends)
    base_asset_order = assets.copy()
    assets_mapping = {i: base_asset_order[i] for i in range(len(base_asset_order))}

    for cv in data_specs:
        cv_callbacks = [c for c in callbacks]

        LOGGER.info(f'Starting with cv: {cv}')
        if save:
            os.mkdir(f"{save_dir}/{cv}")
        data_spec = data_specs[cv]

        if shuffle_columns:
            LOGGER.info('Shuffle assets order')
            if cv == 0:
                random_assets = assets.copy()
                np.random.seed(random_seed)
                np.random.shuffle(random_assets)
                np.random.seed(seed)
                assets = random_assets
            else:
                np.random.seed(random_seed)
                np.random.shuffle(assets)
                np.random.seed(seed)

        LOGGER.info(f'Assets order: {assets}')
        train_data, val_data, test_data, scaler, dates = get_features(data, data_spec['start'], data_spec['end'],
                                                                      assets, val_size=val_size, rescale=rescale)

        # if shuffle_columns_while_training:
        #     train_data = np.transpose(train_data)
        #     np.random.seed(random_seed)
        #     np.random.shuffle(train_data)
        #     np.random.seed(seed)
        #     train_data = np.transpose(train_data)

        LOGGER.info(f'Train shape: {train_data.shape}')
        LOGGER.info(f'Validation shape: {val_data.shape}')
        # Build model
        input_dim = len(assets)
        if model_type == 'pca_permut_ae_model':
            model, encoder = pca_permut_ae_model(input_dim, encoding_dim, activation=activation,
                                                 kernel_initializer=kernel_initializer,
                                                 kernel_constraint=kernel_constraint,
                                                 kernel_regularizer=kernel_regularizer,
                                                 activity_regularizer=activity_regularizer,
                                                 pooling=pooling
                                                 )
            train_input = [train_data[:, i].reshape(-1, 1) for i in range(len(assets))]
            val_input = [val_data[:, i].reshape(-1, 1) for i in range(len(assets))]
            test_input = [test_data[:, i].reshape(-1, 1) for i in range(len(assets))]

        elif model_type == 'pca_ae_model':
            model, encoder = pca_ae_model(input_dim, encoding_dim, activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          kernel_constraint=kernel_constraint,
                                          kernel_regularizer=kernel_regularizer,
                                          activity_regularizer=activity_regularizer,
                                          batch_size=batch_size if drop_remainder_obs else None,
                                          loss=loss
                                          )
            train_input = train_data
            val_input = val_data
            test_input = test_data
        else:
            raise NotImplementedError()

        if save:
            cv_callbacks.append(
                [
                    tf.keras.callbacks.ModelCheckpoint(
                        f"{save_dir}/{cv}/best_model.h5",
                        verbose=1,
                        save_best_only=True,
                        mode='max',
                        monitor="val_r_square")
                ]

            )

        if callback_activity_regularizer:
            tf.config.run_functions_eagerly(True)
            cv_callbacks.append(ActivityRegularizer(model))

        if drop_remainder_obs:
            indices = list(range(train_input.shape[0]))
            indices = drop_remainder(indices, batch_size, last=False)
            train_input = train_input[indices, :]
            train_data = train_data[indices, :]
            dates['train'] = dates['train'][indices]
            indices = list(range(val_input.shape[0]))
            indices = drop_remainder(indices, batch_size, last=False)
            val_input = val_input[indices, :]
            val_data = val_data[indices, :]
            dates['val'] = dates['val'][indices]

        print(model.summary())
        # Train
        LOGGER.info('Start training')
        print(input_dim)

        if loss == 'mse_with_covariance_penalty':
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Very low learning rate
                          metrics=[tf.keras.metrics.MeanSquaredError(name='mse'),
                                   r_square,
                                   tf.keras.metrics.RootMeanSquaredError(name='rmse')]
                          )
        elif loss == 'mse':
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Very low learning rate
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanSquaredError(name='mse'),
                                   r_square,
                                   tf.keras.metrics.RootMeanSquaredError(name='rmse')]
                          )
        elif loss == 'weighted_mse':
            raise NotImplementedError()
            weights = tf.Variable(
                np.random.normal(size=train_data.shape[0] * train_data.shape[1]).reshape(train_data.shape[0],
                                                                                         train_data.shape[1]),
                dtype=tf.float32
            )
            def custom_loss(y_true, y_pred):
                return weighted_mse(y_true, y_pred, weights)


            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Very low learning rate
                          loss=custom_loss,
                          metrics=[tf.keras.metrics.MeanSquaredError(name='mse'),
                                   r_square,
                                   tf.keras.metrics.RootMeanSquaredError(name='rmse')]
                          )

        history = model.fit(train_input, train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(val_input, val_data),
                            validation_batch_size=batch_size,
                            callbacks=cv_callbacks,
                            shuffle=False,
                            verbose=1)
        if save:
            model.save(f"{save_dir}/{cv}/model.h5")

            # Tensorboard Embedding visualization
            # Set up a logs directory, so Tensorboard knows where to look for files.
            log_dir = f"{save_dir}/{cv}/tensorboard/"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Save Labels separately on a line-by-line manner.
            with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
                for asset in assets:
                    f.write("{}\n".format(asset))

            # Save the weights we want to analyze as a variable.
            encoder_layer = get_layer_by_name(name='encoder', model=model)
            encoder_weights = tf.Variable(encoder_layer.get_weights()[0])
            # Create a checkpoint from embedding, the filename and key are the name of the tensor.
            checkpoint = tf.train.Checkpoint(embedding=encoder_weights)
            checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
            # Set up config.
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
            embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
            embedding.metadata_path = 'metadata.tsv'
            projector.visualize_embeddings(log_dir, config)

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
        # model.evaluate(train_input, train_data)
        # model.evaluate(val_input, val_data)

        # Results
        val_prediction = model.predict(val_input)
        val_prediction = scaler.inverse_transform(val_prediction)
        val_prediction = pd.DataFrame(val_prediction, columns=assets, index=dates['val'])

        encoder_layer = get_layer_by_name(name='encoder', model=model)
        encoder_weights = encoder_layer.get_weights()
        # decoder_weights = model.layers[2].get_weights()
        encoder_weights = pd.DataFrame(encoder_weights[0], index=assets)
        LOGGER.info(f"Encoder weights:\n{encoder_weights}")

        test_prediction = model.predict(test_input)
        test_prediction = scaler.inverse_transform(test_prediction)
        test_prediction = pd.DataFrame(test_prediction, columns=assets, index=dates['test'])

        # train_cluster_portfolio = encoder.predict(train_data)
        # train_cluster_portfolio = pd.DataFrame(train_cluster_portfolio, index=dates['train'])
        train_cluster_portfolio = pd.DataFrame(np.dot(train_data, encoder_weights / encoder_weights.sum()),
                                               index=dates['train'])

        # val_cluster_portfolio = encoder.predict(val_data)
        # val_cluster_portfolio = pd.DataFrame(val_cluster_portfolio, index=dates['val'])
        val_cluster_portfolio = pd.DataFrame(np.dot(val_data, encoder_weights / encoder_weights.sum()),
                                             index=dates['val'])

        coskewness = PositiveSkewnessConstraint(encoding_dim, weightage=1, norm='1', normalize=False)
        LOGGER.info(
            f'Coskewness on validation set: {coskewness(tf.constant(val_cluster_portfolio.values, dtype=tf.float32)).numpy()}')

        # test_cluster_portfolio = encoder.predict(test_data)
        # test_cluster_portfolio = pd.DataFrame(test_cluster_portfolio, index=dates['test'])
        test_cluster_portfolio = pd.DataFrame(np.dot(test_data, encoder_weights / encoder_weights.sum()),
                                              index=dates['test'])

        LOGGER.info(
            f'Coskewness on test set: {coskewness(tf.constant(test_cluster_portfolio.values, dtype=tf.float32)).numpy()}')
        train_data = scaler.inverse_transform(train_data)
        train_data = pd.DataFrame(train_data, index=dates['train'], columns=assets)
        val_data = scaler.inverse_transform(val_data)
        val_data = pd.DataFrame(val_data, index=dates['val'], columns=assets)
        test_data = scaler.inverse_transform(test_data)
        test_data = pd.DataFrame(test_data, index=dates['test'], columns=assets)

        if shuffle_columns:
            LOGGER.info('Reorder results with base asset order')
            val_prediction = val_prediction.loc[:, base_asset_order]
            train_data = train_data.loc[:, base_asset_order]
            val_data = val_data.loc[:, base_asset_order]
            test_data = test_data.loc[:, base_asset_order]
            test_prediction = test_prediction.loc[:, base_asset_order]
            encoder_weights = encoder_weights.loc[base_asset_order, :]

        if kernel_constraint is not None:
            vmax = 1.
            vmin = 0.
        else:
            vmax = None
            vmin = None

        if save:
            heat_map(encoder_weights, show=True, save=save, save_dir=f"{save_dir}/{cv}", vmax=vmax, vmin=vmin)
        else:
            heat_map(encoder_weights, show=True, vmax=vmax, vmin=vmin)

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

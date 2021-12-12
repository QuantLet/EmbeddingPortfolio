import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import datetime as dt

from shutil import copyfile
from typing import Optional
from sklearn.cluster import KMeans

from dl_portfolio.logger import LOGGER
from dl_portfolio.pca_ae import heat_map_cluster, get_layer_by_name, heat_map, build_model
from dl_portfolio.data import drop_remainder
from dl_portfolio.ae_data import get_features, load_data, get_sample_weights_from_df, labelQuantile
from dl_portfolio.train import fit, embedding_visualization, plot_history, create_dataset, build_model_input
from dl_portfolio.constant import LOG_DIR
from dl_portfolio.nmf.convex_nmf import ConvexNMF


def run_ae(config, data, assets, log_dir: Optional[str] = None, seed: Optional[int] = None):
    """

    :param config: config
    :param log_dir: if given save the result in log_dir folder, if not given use LOG_DIR
    :param seed: if given use specific seed
    :return:
    """
    random_seed = np.random.randint(0, 100)
    if config.seed:
        seed = config.seed
    if seed is None:
        seed = np.random.randint(0, 1000)

    np.random.seed(seed)
    tf.random.set_seed(seed)
    LOGGER.debug(f"Set seed: {seed}")

    if config.save:
        if log_dir is None:
            log_dir = LOG_DIR

        iter = len(os.listdir(log_dir))
        if config.model_name is not None and config.model_name != '':
            subdir = f'm_{iter}_' + config.model_name + f'_seed_{seed}'
        else:
            subdir = f'm_{iter}_'
        subdir = subdir + '_' + str(dt.datetime.timestamp(dt.datetime.now())).replace('.', '')
        save_dir = f"{log_dir}/{subdir}"
        os.makedirs(save_dir)
        copyfile('./dl_portfolio/config/config.py',
                 os.path.join(save_dir, 'config.py'))

    base_asset_order = assets.copy()
    assets_mapping = {i: base_asset_order[i] for i in range(len(base_asset_order))}

    if config.loss == 'weighted_mse':
        file_name = f"./data/sample_weights_lq_{config.label_param['lq']}_uq_{config.label_param['uq']}_w_{config.label_param['window']}.p"
        if os.path.isfile(file_name):
            LOGGER.debug(f'Loading sample weights from {file_name}')
            df_sample_weights = pd.read_pickle(file_name)
            df_sample_weights = df_sample_weights[assets]
        else:
            LOGGER.debug('Computing sample weights ...')
            d, _ = load_data(type=['indices', 'forex', 'forex_metals', 'commodities'], dropnan=True)
            t_sample_weights, _ = get_sample_weights_from_df(d, labelQuantile, **config.label_param)
            d, _ = load_data(type=['crypto'], dropnan=False)
            c_sample_weights, _ = get_sample_weights_from_df(d, labelQuantile, **config.label_param)
            df_sample_weights = pd.concat([t_sample_weights, c_sample_weights], 1)
            df_sample_weights = df_sample_weights.fillna(0.0)
            df_sample_weights = df_sample_weights[assets]
            del d
            del c_sample_weights
            del t_sample_weights
            LOGGER.debug(f'Saving sample weights to {file_name}')
            df_sample_weights.to_pickle(
                file_name)
            LOGGER.debug('Done')

    for cv in config.data_specs:
        LOGGER.debug(f'Starting with cv: {cv}')
        if config.save:
            save_path = f"{save_dir}/{cv}"
            os.mkdir(f"{save_dir}/{cv}")
        else:
            save_path = None

        if config.shuffle_columns:
            LOGGER.debug('Shuffle assets order')
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

        LOGGER.debug(f'Assets order: {assets}')
        if config.loss == 'weighted_mse':
            # reorder columns
            df_sample_weights = df_sample_weights[assets]

        # Build model
        input_dim = len(assets)
        n_features = None
        model, encoder, extra_features = build_model(config.model_type,
                                                     input_dim,
                                                     config.encoding_dim,
                                                     n_features=n_features,
                                                     extra_features_dim=1,
                                                     activation=config.activation,
                                                     batch_normalization=config.batch_normalization,
                                                     kernel_initializer=config.kernel_initializer,
                                                     kernel_constraint=config.kernel_constraint,
                                                     kernel_regularizer=config.kernel_regularizer,
                                                     activity_regularizer=config.activity_regularizer,
                                                     batch_size=config.batch_size if config.drop_remainder_obs else None,
                                                     loss=config.loss,
                                                     uncorrelated_features=config.uncorrelated_features,
                                                     weightage=config.weightage)
        # LOGGER.info(model.summary())

        # Create dataset:
        shuffle = False
        if config.resample is not None:
            if config.resample.get('when', None) != 'each_epoch':
                train_dataset, val_dataset = create_dataset(data, assets,
                                                            config.data_specs[cv],
                                                            config.model_type,
                                                            batch_size=config.batch_size,
                                                            rescale=config.rescale,
                                                            features_config=config.features_config,
                                                            scaler_func=config.scaler_func,
                                                            resample=config.resample,
                                                            loss=config.loss,
                                                            drop_remainder_obs=config.drop_remainder_obs,
                                                            df_sample_weights=df_sample_weights if config.loss == 'weighted_mse' else None
                                                            )

            else:
                shuffle = True

        # Set extra loss parameters
        if config.loss_asset_weights is not None:
            loss_asset_weights = {a: 1. for a in assets}
            for a in config.loss_asset_weights:
                loss_asset_weights[a] = config.loss_asset_weights[a]
            LOGGER.debug(f'Loss asset weights is: {loss_asset_weights}')
            loss_asset_weights = np.array(list(loss_asset_weights.values()))
            loss_asset_weights = tf.cast(loss_asset_weights, dtype=tf.float32)
        else:
            loss_asset_weights = None

        if shuffle:
            model, history = fit(model,
                                 None,
                                 config.epochs,
                                 config.learning_rate,
                                 loss=config.loss,
                                 loss_asset_weights=loss_asset_weights,
                                 callbacks=config.callbacks,
                                 val_dataset=None,
                                 extra_features=n_features is not None,
                                 save_path=f"{save_path}" if config.save else None,
                                 shuffle=True,
                                 cv=cv,
                                 data=data,
                                 assets=assets,
                                 config=config,
                                 df_sample_weights=df_sample_weights if config.loss == 'weighted_mse' else None)
        else:
            model, history = fit(model,
                                 train_dataset,
                                 config.epochs,
                                 config.learning_rate,
                                 loss=config.loss,
                                 loss_asset_weights=loss_asset_weights,
                                 callbacks=config.callbacks,
                                 val_dataset=val_dataset,
                                 extra_features=n_features is not None,
                                 save_path=f"{save_path}" if config.save else None,
                                 shuffle=False)

        if config.save:
            # tensorboard viz
            if config.model_type != 'ae_model2':
                embedding_visualization(model, assets, log_dir=f"{save_path}/tensorboard/")
            LOGGER.debug(f"Loading weights from {save_path}/model.h5")
            model.load_weights(f"{save_path}/model.h5")

        plot_history(history, save_path=save_path, show=config.show_plot)

        # Evaluate
        # model.evaluate(train_input, train_data)
        # model.evaluate(val_input, val_data)

        # Get results for later analysis
        data_spec = config.data_specs[cv]
        train_data, val_data, test_data, scaler, dates, features = get_features(data,
                                                                                data_spec['start'],
                                                                                data_spec['end'],
                                                                                assets,
                                                                                val_start=data_spec['val_start'],
                                                                                test_start=data_spec.get('test_start'),
                                                                                rescale=config.rescale,
                                                                                scaler=config.scaler_func['name'],
                                                                                resample=config.resample,
                                                                                features_config=config.features_config,
                                                                                **config.scaler_func.get('params',
                                                                                                            {}))

        if config.drop_remainder_obs:
            indices = list(range(train_data.shape[0]))
            indices = drop_remainder(indices, config.batch_size, last=False)
            train_data = train_data[indices, :]
            features['train'] = features['train'][indices, :]
            dates['train'] = dates['train'][indices]

            indices = list(range(val_data.shape[0]))
            indices = drop_remainder(indices, config.batch_size, last=False)
            val_data = val_data[indices, :]
            features['val'] = features['val'][indices, :]
            dates['val'] = dates['val'][indices]

        # if shuffle_columns_while_training:
        #     train_data = np.transpose(train_data)
        #     np.random.seed(random_seed)
        #     np.random.shuffle(train_data)
        #     np.random.seed(seed)
        #     train_data = np.transpose(train_data)

        LOGGER.debug(f'Train shape: {train_data.shape}')
        LOGGER.debug(f'Validation shape: {val_data.shape}')

        # train_input, val_input, test_input = build_model_input(train_data, val_data, test_data, config.model_type, features=features)
        if features:
            train_input = build_model_input(train_data, config.model_type, features=features['train'], assets=assets)
            val_input = build_model_input(val_data, config.model_type, features=features['val'])
            if test_data is not None:
                test_input = build_model_input(test_data, config.model_type, features=features['test'],
                                               assets=assets)
        else:
            train_input = build_model_input(train_data, config.model_type, features=None, assets=assets)
            val_input = build_model_input(val_data, config.model_type, features=None, assets=assets)
            if test_data is not None:
                test_input = build_model_input(test_data, config.model_type, features=None, assets=assets)

        ## Get prediction
        if n_features:
            train_features = encoder.predict(train_input[0])
            val_features = encoder.predict(val_input[0])
        else:
            train_features = encoder.predict(train_input)
            val_features = encoder.predict(val_input)

        train_features = pd.DataFrame(train_features, index=dates['train'])
        LOGGER.info(f"Train features correlation:\n{train_features.corr()}")
        val_features = pd.DataFrame(val_features, index=dates['val'])
        LOGGER.info(f"Val features correlation:\n{val_features.corr()}")
        val_prediction = model.predict(val_input)
        val_prediction = scaler.inverse_transform(val_prediction)
        val_prediction = pd.DataFrame(val_prediction, columns=assets, index=dates['val'])

        ## Get encoder weights
        if config.model_type in ['ae_model2', 'nl_pca_ae_model']:
            encoder_layer1 = get_layer_by_name(name='encoder1', model=model)
            encoder_weights1 = encoder_layer1.get_weights()[0]
            encoder_layer2 = get_layer_by_name(name='encoder2', model=model)
            encoder_weights2 = encoder_layer2.get_weights()[0]
            encoder_weights = np.dot(encoder_weights1, encoder_weights2)
            print(pd.DataFrame(encoder_weights1))
            print(pd.DataFrame(encoder_weights2))
            encoder_weights = pd.DataFrame(encoder_weights, index=assets)
            heat_map(pd.DataFrame(encoder_weights1), show=True, vmin=0., vmax=1.)
            heat_map(pd.DataFrame(encoder_weights2), show=True, vmin=0., vmax=1.)
            heat_map(encoder_weights, show=True)
        else:
            encoder_layer = get_layer_by_name(name='encoder', model=model)
            encoder_weights = encoder_layer.get_weights()
            # decoder_weights = model.layers[2].get_weights()
            encoder_weights = pd.DataFrame(encoder_weights[0], index=assets)
        LOGGER.debug(f"Encoder weights:\n{encoder_weights}")

        # train_cluster_portfolio = encoder.predict(train_data)
        # train_cluster_portfolio = pd.DataFrame(train_cluster_portfolio, index=dates['train'])
        # train_cluster_portfolio = pd.DataFrame(np.dot(train_data, encoder_weights / encoder_weights.sum()), index=dates['train'])

        # val_cluster_portfolio = encoder.predict(val_data)
        # val_cluster_portfolio = pd.DataFrame(val_cluster_portfolio, index=dates['val'])
        # val_cluster_portfolio = pd.DataFrame(np.dot(val_data, encoder_weights / encoder_weights.sum()), index=dates['val'])

        ## Get prediction on test_data
        if test_data is not None:
            if features:
                test_input = [test_data, features['test']]
            test_prediction = model.predict(test_input)
            test_prediction = scaler.inverse_transform(test_prediction)
            test_prediction = pd.DataFrame(test_prediction, columns=assets, index=dates['test'])

            if n_features:
                test_features = encoder.predict(test_input[0])
            else:
                test_features = encoder.predict(test_input)
            test_features = pd.DataFrame(test_features, index=dates['test'])

            # test_cluster_portfolio = encoder.predict(test_data)
            # test_cluster_portfolio = pd.DataFrame(test_cluster_portfolio, index=dates['test'])
            # test_cluster_portfolio = pd.DataFrame(np.dot(test_data, encoder_weights / encoder_weights.sum()), index=dates['test'])

        # cluster_portfolio = {
        #     'train': train_cluster_portfolio,
        #     'val': val_cluster_portfolio,
        #     'test': test_cluster_portfolio if test_data is not None else None
        # }

        # coskewness = PositiveSkewnessConstraint(encoding_dim, weightage=1, norm='1', normalize=False)
        # LOGGER.debug(
        #     f'Coskewness on validation set: {coskewness(tf.constant(val_cluster_portfolio.values, dtype=tf.float32)).numpy()}')

        # LOGGER.debug(
        #     f'Coskewness on test set: {coskewness(tf.constant(test_cluster_portfolio.values, dtype=tf.float32)).numpy()}')

        # Rescale back input data
        train_data = scaler.inverse_transform(train_data)
        train_data = pd.DataFrame(train_data, index=dates['train'], columns=assets)
        val_data = scaler.inverse_transform(val_data)
        val_data = pd.DataFrame(val_data, index=dates['val'], columns=assets)
        if test_data is not None:
            test_data = scaler.inverse_transform(test_data)
            test_data = pd.DataFrame(test_data, index=dates['test'], columns=assets)

        if config.shuffle_columns:
            LOGGER.debug('Reorder results with base asset order')
            val_prediction = val_prediction.loc[:, base_asset_order]
            train_data = train_data.loc[:, base_asset_order]
            val_data = val_data.loc[:, base_asset_order]
            if test_data is not None:
                test_data = test_data.loc[:, base_asset_order]
                test_prediction = test_prediction.loc[:, base_asset_order]
            encoder_weights = encoder_weights.loc[base_asset_order, :]

        # Sort index in case of random sampling
        train_data.sort_index(inplace=True)
        train_features.sort_index(inplace=True)
        val_data.sort_index(inplace=True)
        val_prediction.sort_index(inplace=True)
        val_features.sort_index(inplace=True)

        if test_data is not None:
            test_data.sort_index(inplace=True)
            test_prediction.sort_index(inplace=True)
            test_features.sort_index(inplace=True)

        # Plot heatmap
        if config.kernel_constraint is not None:
            if type(config.kernel_constraint).__name__ == 'NonNegAndUnitNorm':
                vmax = 1
                vmin = 0.
            elif type(config.kernel_constraint).__name__ == 'UnitNorm':
                vmax = 1
                vmin = -1
            else:
                vmax = None  # np.max(encoder_weights.max())
                vmin = 0.
        else:
            vmax = None
            vmin = None

        if config.show_plot:
            heat_map(encoder_weights, show=config.show_plot, vmax=vmax, vmin=vmin)

        # LOGGER.debug(f"Encoder feature correlation:\n{np.corrcoef(val_cluster_portfolio.T)}")
        LOGGER.debug(f"Unit norm constraint:\n{(encoder_weights ** 2).sum(0)}")
        LOGGER.debug(f"Orthogonality constraint:\n{np.dot(encoder_weights.T, encoder_weights)}")

        if config.show_plot:
            if test_data is not None:
                for c in test_data.columns:
                    plt.plot(test_data[c], label='true')
                    plt.plot(test_prediction[c], label='pred')
                    plt.title(c)
                    plt.show()

        if config.save:
            # train_data.to_pickle(f"{save_path}/train_returns.p")
            # val_data.to_pickle(f"{save_path}/val_returns.p")
            # val_prediction.to_pickle(f"{save_path}/val_prediction.p")
            encoder_weights.to_pickle(f"{save_path}/encoder_weights.p")
            # train_features.to_pickle(f"{save_path}/train_features.p")
            # val_features.to_pickle(f"{save_path}/val_features.p")
            config.scaler_func['attributes'] = scaler.__dict__
            pickle.dump(config.scaler_func, open(f"{save_path}/scaler.p", "wb"))
            # encoding_pca.to_pickle(f"{save_path}/encoding_pca.p")
            # pickle.dump(cluster_portfolio, open(f"{save_path}/cluster_portfolio.p", "wb"))
            # pickle.dump(pca_cluster_portfolio, open(f"{save_path}/pca_cluster_portfolio.p", "wb"))

            if test_data is not None:
                pass
                # test_data.to_pickle(f"{save_path}/test_returns.p")
                # test_prediction.to_pickle(f"{save_path}/test_prediction.p")
                # test_features.to_pickle(f"{save_path}/test_features.p")

    # if config.save:
    #     heat_map_cluster(save_dir, show=True, save=config.save, vmax=1., vmin=0.)


def run_kmeans(config, data, assets, seed=None):
    if config.seed:
        seed = config.seed
    if seed is None:
        seed = np.random.randint(0, 1000)

    np.random.seed(seed)
    LOGGER.info(f"Set seed: {seed}")

    if config.save:
        if not os.path.isdir('log_kmeans'):
            os.mkdir('log_kmeans')
        iter = len(os.listdir('log_kmeans'))
        save_dir = f"log_kmeans/m_{iter}_seed_{seed}_{dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir)
        copyfile('./dl_portfolio/config/config.py',
                 os.path.join(save_dir, 'config.py'))

    for cv in config.data_specs:
        LOGGER.info(f'Starting with cv: {cv}')
        if config.save:
            save_path = f"{save_dir}/{cv}"
            os.mkdir(f"{save_dir}/{cv}")
        else:
            save_path = None

        LOGGER.info(f'Assets order: {assets}')
        data_spec = config.data_specs[cv]
        train_data, val_data, test_data, scaler, dates, features = get_features(data,
                                                                                data_spec['start'],
                                                                                data_spec['end'],
                                                                                assets,
                                                                                val_start=data_spec['val_start'],
                                                                                test_start=data_spec.get('test_start'),
                                                                                scaler='StandardScaler',
                                                                                resample={
                                                                                    'method': 'nbb',
                                                                                    'where': ['train'],
                                                                                    'block_length': 60
                                                                                })
        kmeans = KMeans(n_clusters=config.encoding_dim, random_state=seed)
        kmeans.fit(train_data.T)
        labels = pd.DataFrame(kmeans.labels_.reshape(1, -1), columns=assets).T
        labels.columns = ['label']
        clusters = {i: list(labels[labels['label'] == i].index) for i in range(config.encoding_dim)}

        if config.save:
            pickle.dump(config.scaler_func, open(f"{save_path}/scaler.p", "wb"))
            pickle.dump(kmeans, open(f"{save_path}/model.p", "wb"))
            pickle.dump(clusters, open(f"{save_path}/clusters.p", "wb"))
            labels.to_pickle(f"{save_path}/labels.p")


def run_convex_nmf(config, data, assets, log_dir: Optional[str] = None, seed: Optional[int] = None, verbose=0):
    LOG_DIR = 'log_convex_nmf'

    if config.seed:
        seed = config.seed
    if seed is None:
        seed = np.random.randint(0, 1000)

    np.random.seed(seed)
    LOGGER.info(f"Set seed: {seed}")

    if config.save:
        if log_dir is None:
            log_dir = LOG_DIR

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        iter = len(os.listdir(log_dir))
        save_dir = f"{log_dir}/m_{iter}_seed_{seed}_{dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir)
        copyfile('./dl_portfolio/config/nmf_config.py',
                 os.path.join(save_dir, 'nmf_config.py'))
    mse = {}
    for cv in config.data_specs:
        LOGGER.info(f'Starting with cv: {cv}')
        if config.save:
            save_path = f"{save_dir}/{cv}"
            os.mkdir(f"{save_dir}/{cv}")
        else:
            save_path = None

        LOGGER.info(f'Assets order: {assets}')
        data_spec = config.data_specs[cv]
        train_data, val_data, test_data, scaler, dates, features = get_features(data,
                                                                                data_spec['start'],
                                                                                data_spec['end'],
                                                                                assets,
                                                                                val_start=data_spec['val_start'],
                                                                                test_start=data_spec.get('test_start'),
                                                                                scaler='StandardScaler',
                                                                                resample={
                                                                                    'method': 'nbb',
                                                                                    'where': ['train'],
                                                                                    'block_length': 60
                                                                                })
        nmf = ConvexNMF(n_components=config.encoding_dim, random_state=seed, verbose=verbose)
        nmf.fit(train_data)
        encoder_weights = pd.DataFrame(nmf.components, index=assets)
        mse[cv] = {
            'train': nmf.evaluate(train_data),
            'test': nmf.evaluate(val_data) if test_data is None else nmf.evaluate(test_data)
        }

        if config.save:
            LOGGER.debug(f'Saving result at cv {cv} at {save_path} ...')
            nmf.save(f"{save_path}/model.p")
            encoder_weights.to_pickle(f"{save_path}/encoder_weights.p")
            config.scaler_func['attributes'] = scaler.__dict__
            pickle.dump(config.scaler_func, open(f"{save_path}/scaler.p", "wb"))
            LOGGER.debug('Done')

        if config.show_plot:
            heat_map(encoder_weights, show=config.show_plot)

    if config.save:
        json.dump(mse, open(f"{save_dir}/evaluation.json", "w"))

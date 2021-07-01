import pandas as pd
import os, pickle
from dl_portfolio.logger import LOGGER
import datetime as dt
from dl_portfolio.pca_ae import heat_map_cluster, get_layer_by_name, heat_map, build_model
from shutil import copyfile
from dl_portfolio.data import drop_remainder
from dl_portfolio.ae_data import get_features, load_data, get_sample_weights_from_df, labelQuantile
from dl_portfolio.train import fit, embedding_visualization, plot_history
import tensorflow as tf
import numpy as np
from dl_portfolio.constant import LOG_DIR


def run(ae_config, seed=None):
    random_seed = np.random.randint(0, 100)
    if ae_config.seed:
        seed = ae_config.seed
    if seed is None:
        seed = np.random.randint(0, 1000)

    np.random.seed(seed)
    tf.random.set_seed(seed)
    LOGGER.info(f"Set seed: {seed}")

    if ae_config.save:
        iter = len(os.listdir(LOG_DIR))

        if ae_config.model_name is not None and ae_config.model_name != '':
            subdir = f'm_{iter}_' + ae_config.model_name + f'_seed_{seed}'
        else:
            subdir = f'm_{iter}_'
        subdir = subdir + '_' + dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')
        save_dir = f"{LOG_DIR}/{subdir}"
        os.makedirs(save_dir)
        copyfile('./dl_portfolio/config/ae_config.py',
                 os.path.join(save_dir, 'ae_config.py'))

    data, assets = load_data(dataset=ae_config.dataset, assets=ae_config.assets, dropnan=ae_config.dropnan, freq=ae_config.freq)

    base_asset_order = assets.copy()
    assets_mapping = {i: base_asset_order[i] for i in range(len(base_asset_order))}

    if ae_config.loss == 'weighted_mse':
        file_name = f"./data/sample_weights_lq_{ae_config.label_param['lq']}_uq_{ae_config.label_param['uq']}_w_{ae_config.label_param['window']}.p"
        if os.path.isfile(file_name):
            LOGGER.info(f'Loading sample weights from {file_name}')
            df_sample_weights = pd.read_pickle(file_name)
            df_sample_weights = df_sample_weights[assets]
        else:
            LOGGER.info('Computing sample weights ...')
            d, _ = load_data(type=['indices', 'forex', 'forex_metals', 'commodities'], dropnan=True)
            t_sample_weights, _ = get_sample_weights_from_df(d, labelQuantile, **ae_config.label_param)
            d, _ = load_data(type=['crypto'], dropnan=False)
            c_sample_weights, _ = get_sample_weights_from_df(d, labelQuantile, **ae_config.label_param)
            df_sample_weights = pd.concat([t_sample_weights, c_sample_weights], 1)
            df_sample_weights = df_sample_weights.fillna(0.0)
            df_sample_weights = df_sample_weights[assets]
            del d
            del c_sample_weights
            del t_sample_weights
            LOGGER.info(f'Saving sample weights to {file_name}')
            df_sample_weights.to_pickle(
                file_name)
            LOGGER.info('Done')

    for cv in ae_config.data_specs:
        LOGGER.info(f'Starting with cv: {cv}')
        if ae_config.save:
            os.mkdir(f"{save_dir}/{cv}")
        data_spec = ae_config.data_specs[cv]

        if ae_config.shuffle_columns:
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
        if ae_config.loss == 'weighted_mse':
            # reorder columns
            df_sample_weights = df_sample_weights[assets]

        train_data, val_data, test_data, scaler, dates, features = get_features(data,
                                                                                data_spec['start'],
                                                                                data_spec['end'],
                                                                                assets,
                                                                                val_start=data_spec['val_start'],
                                                                                test_start=data_spec.get('test_start'),
                                                                                rescale=ae_config.rescale,
                                                                                scaler=ae_config.scaler_func['name'],
                                                                                features_config=ae_config.features_config,
                                                                                **ae_config.scaler_func.get('params', {}))

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
        n_features = None
        if ae_config.model_type == 'pca_permut_ae_model':
            raise NotImplementedError()
            train_input = [train_data[:, i].reshape(-1, 1) for i in range(len(assets))]
            val_input = [val_data[:, i].reshape(-1, 1) for i in range(len(assets))]
            if test_data is not None:
                test_input = [test_data[:, i].reshape(-1, 1) for i in range(len(assets))]

        elif ae_config.model_type in ['ae_model', 'pca_ae_model']:
            if features:
                n_features = features['train'].shape[-1]
            else:
                n_features = None
            model, encoder, extra_features = build_model(ae_config.model_type,
                                                         input_dim,
                                                         ae_config.encoding_dim,
                                                         n_features=n_features,
                                                         extra_features_dim=1,
                                                         activation=ae_config.activation,
                                                         batch_normalization=ae_config.batch_normalization,
                                                         kernel_initializer=ae_config.kernel_initializer,
                                                         kernel_constraint=ae_config.kernel_constraint,
                                                         kernel_regularizer=ae_config.kernel_regularizer,
                                                         activity_regularizer=ae_config.activity_regularizer,
                                                         batch_size=ae_config.batch_size if ae_config.drop_remainder_obs else None,
                                                         loss=ae_config.loss,
                                                         uncorrelated_features=ae_config.uncorrelated_features,
                                                         weightage=ae_config.weightage)
            train_input = train_data
            val_input = val_data
            test_input = test_data

        else:
            raise NotImplementedError()
        print(model.summary())

        if ae_config.drop_remainder_obs:
            indices = list(range(train_input.shape[0]))
            indices = drop_remainder(indices, batch_size, last=False)
            train_input = train_input[indices, :]
            train_data = train_data[indices, :]
            dates['train'] = dates['train'][indices]
            if features:
                train_input = [train_data, features['train'][indices, :]]
            indices = list(range(val_input.shape[0]))
            indices = drop_remainder(indices, batch_size, last=False)
            val_input = val_input[indices, :]
            val_data = val_data[indices, :]
            dates['val'] = dates['val'][indices]
            if features:
                val_input = [val_data, features['val'][indices, :]]
        else:
            if features:
                train_input = [train_data, features['train']]
                val_input = [val_data, features['val']]

        if ae_config.loss == 'weighted_mse':
            sample_weights = df_sample_weights.loc[dates['train']]
            sample_weights = tf.Variable(
                sample_weights.values,
                dtype=tf.float32
            )
            if n_features:
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (train_input[0], train_input[1], train_data, sample_weights))
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_data, sample_weights))
        else:
            if n_features:
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (train_input[0], train_input[1], train_data))
            else:
                train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_data))

        if n_features:
            val_dataset = tf.data.Dataset.from_tensor_slices((val_input[0], val_input[1], val_data))
        else:
            val_dataset = tf.data.Dataset.from_tensor_slices((val_input, val_data))

        train_dataset = train_dataset.batch(ae_config.batch_size)
        val_dataset = val_dataset.batch(ae_config.batch_size)

        if ae_config.save:
            save_path = f"{save_dir}/{cv}"
        else:
            save_path = None

        model, history = fit(model,
                             train_dataset,
                             ae_config.epochs,
                             ae_config.learning_rate,
                             loss=ae_config.loss,
                             callbacks=ae_config.callbacks,
                             val_dataset=val_dataset,
                             extra_features=n_features > 0,
                             save_path=f"{save_path}")

        if ae_config.save:
            # tensorboard viz
            embedding_visualization(model, assets, log_dir=f"{save_path}/tensorboard/")
            LOGGER.info(f"Loading weights from {save_path}/best_model.h5")
            model.load_weights(f"{save_path}/best_model.h5")

        plot_history(history, save_path=save_path, show=ae_config.show_plot)

        # Evaluate
        # model.evaluate(train_input, train_data)
        # model.evaluate(val_input, val_data)

        # Get results for later analysis
        ## Get prediction
        if n_features:
            train_features = encoder.predict(train_input[0])
            val_features = encoder.predict(val_input[0])
        else:
            train_features = encoder.predict(train_input)
            val_features = encoder.predict(val_input)
        train_features = pd.DataFrame(train_features, index=dates['train'])
        val_features = pd.DataFrame(val_features, index=dates['val'])
        val_prediction = model.predict(val_input)
        val_prediction = scaler.inverse_transform(val_prediction)
        val_prediction = pd.DataFrame(val_prediction, columns=assets, index=dates['val'])

        ## Get encoder weights
        encoder_layer = get_layer_by_name(name='encoder', model=model)
        encoder_weights = encoder_layer.get_weights()
        # decoder_weights = model.layers[2].get_weights()
        encoder_weights = pd.DataFrame(encoder_weights[0], index=assets)
        LOGGER.info(f"Encoder weights:\n{encoder_weights}")

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
        # LOGGER.info(
        #     f'Coskewness on validation set: {coskewness(tf.constant(val_cluster_portfolio.values, dtype=tf.float32)).numpy()}')

        # LOGGER.info(
        #     f'Coskewness on test set: {coskewness(tf.constant(test_cluster_portfolio.values, dtype=tf.float32)).numpy()}')

        # Rescale back input data
        train_data = scaler.inverse_transform(train_data)
        train_data = pd.DataFrame(train_data, index=dates['train'], columns=assets)
        val_data = scaler.inverse_transform(val_data)
        val_data = pd.DataFrame(val_data, index=dates['val'], columns=assets)
        if test_data is not None:
            test_data = scaler.inverse_transform(test_data)
            test_data = pd.DataFrame(test_data, index=dates['test'], columns=assets)

        if ae_config.shuffle_columns:
            LOGGER.info('Reorder results with base asset order')
            val_prediction = val_prediction.loc[:, base_asset_order]
            train_data = train_data.loc[:, base_asset_order]
            val_data = val_data.loc[:, base_asset_order]
            if test_data is not None:
                test_data = test_data.loc[:, base_asset_order]
                test_prediction = test_prediction.loc[:, base_asset_order]
            encoder_weights = encoder_weights.loc[base_asset_order, :]

        # Plot heatmap
        if ae_config.kernel_constraint is not None:
            vmax = 1.
            vmin = 0.
        else:
            vmax = None
            vmin = None
        if ae_config.save:
            heat_map(encoder_weights, show=ae_config.show_plot, save_dir=f"{save_path}", vmax=vmax, vmin=vmin)
        else:
            heat_map(encoder_weights, show=ae_config.show_plot, vmax=vmax, vmin=vmin)

        # LOGGER.info(f"Encoder feature correlation:\n{np.corrcoef(val_cluster_portfolio.T)}")
        LOGGER.info(f"Unit norm constraint:\n{(encoder_weights ** 2).sum(0)}")

        if ae_config.save:
            train_data.to_pickle(f"{save_path}/train_returns.p")
            val_data.to_pickle(f"{save_path}/val_returns.p")
            val_prediction.to_pickle(f"{save_path}/val_prediction.p")
            encoder_weights.to_pickle(f"{save_path}/encoder_weights.p")
            train_features.to_pickle(f"{save_path}/train_features.p")
            val_features.to_pickle(f"{save_path}/val_features.p")
            ae_config.scaler_func['attributes'] = scaler.__dict__
            pickle.dump(ae_config.scaler_func, open(f"{save_path}/scaler.p", "wb"))
            # encoding_pca.to_pickle(f"{save_path}/encoding_pca.p")
            # pickle.dump(cluster_portfolio, open(f"{save_path}/cluster_portfolio.p", "wb"))
            # pickle.dump(pca_cluster_portfolio, open(f"{save_path}/pca_cluster_portfolio.p", "wb"))

            if test_data is not None:
                test_data.to_pickle(f"{save_path}/test_returns.p")
                test_prediction.to_pickle(f"{save_path}/test_prediction.p")
                test_features.to_pickle(f"{save_path}/test_features.p")

    if ae_config.save:
        heat_map_cluster(save_dir, show=True, save=ae_config.save, vmax=1., vmin=0.)

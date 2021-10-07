import pandas as pd
import os, pickle
from dl_portfolio.logger import LOGGER
import datetime as dt
from dl_portfolio.pca_ae import heat_map_cluster, get_layer_by_name, heat_map, build_model
from shutil import copyfile
from dl_portfolio.data import drop_remainder
from dl_portfolio.ae_data import get_features, load_data, get_sample_weights_from_df, labelQuantile
from dl_portfolio.train import fit, embedding_visualization, plot_history, create_dataset, build_model_input
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

    if ae_config.dataset == 'bond':
        data, assets = load_data(dataset=ae_config.dataset, assets=ae_config.assets, dropnan=ae_config.dropnan,
                                 freq=ae_config.freq, crix=ae_config.crix, crypto_assets=ae_config.crypto_assets)
    else:
        data, assets = load_data(dataset=ae_config.dataset, assets=ae_config.assets, dropnan=ae_config.dropnan,
                                 freq=ae_config.freq)

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

    if ae_config.save:
        save_path = save_dir
    else:
        save_path = None

    if ae_config.shuffle_columns:
        LOGGER.info('Shuffle assets order')
        np.random.seed(random_seed)
        np.random.shuffle(assets)
        np.random.seed(seed)

    LOGGER.info(f'Assets order: {assets}')
    if ae_config.loss == 'weighted_mse':
        # reorder columns
        df_sample_weights = df_sample_weights[assets]

    # Build model
    input_dim = len(assets)
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
    print(model.summary())

    # Create dataset:
    data_spec = ae_config.data_specs
    train_dataset, val_dataset, test_dataset = create_dataset(data, assets,
                                                              data_spec,
                                                              ae_config.model_type,
                                                              batch_size=ae_config.batch_size,
                                                              rescale=ae_config.rescale,
                                                              features_config=ae_config.features_config,
                                                              scaler_func=ae_config.scaler_func,
                                                              resample=ae_config.resample,
                                                              loss=ae_config.loss,
                                                              drop_remainder_obs=ae_config.drop_remainder_obs,
                                                              df_sample_weights=df_sample_weights if ae_config.loss == 'weighted_mse' else None
                                                              )
    # Set extra loss parameters
    if ae_config.loss_asset_weights is not None:
        loss_asset_weights = {a: 1. for a in assets}
        for a in ae_config.loss_asset_weights:
            loss_asset_weights[a] = ae_config.loss_asset_weights[a]
        LOGGER.info(f'Loss asset weights is: {loss_asset_weights}')
        loss_asset_weights = np.array(list(loss_asset_weights.values()))
        loss_asset_weights = tf.cast(loss_asset_weights, dtype=tf.float32)
    else:
        loss_asset_weights = None

    model, history = fit(model,
                         train_dataset,
                         ae_config.epochs,
                         ae_config.learning_rate,
                         loss=ae_config.loss,
                         loss_asset_weights=loss_asset_weights,
                         callbacks=ae_config.callbacks,
                         val_dataset=val_dataset,
                         extra_features=n_features is not None,
                         save_path=f"{save_path}" if ae_config.save else None,
                         shuffle=False)

    if ae_config.save:
        # tensorboard viz
        embedding_visualization(model, assets, log_dir=f"{save_path}/tensorboard/")
        LOGGER.info(f"Loading weights from {save_path}/model.h5")
        model.load_weights(f"{save_path}/model.h5")

    plot_history(history, save_path=save_path, show=ae_config.show_plot)
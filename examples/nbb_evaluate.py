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


def eval_one_cv(ae_config, weights, cv, scaler, seed=None):
    if ae_config.dataset == 'bond':
        data, assets = load_data(dataset=ae_config.dataset, assets=ae_config.assets, dropnan=ae_config.dropnan,
                                 freq=ae_config.freq, crix=ae_config.crix, crypto_assets=ae_config.crypto_assets)
    else:
        data, assets = load_data(dataset=ae_config.dataset, assets=ae_config.assets, dropnan=ae_config.dropnan,
                                 freq=ae_config.freq)

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
    # encoder.set_weights(best_weights)
    model.set_weights(weights)
    print(model.summary())

    # Get results for later analysis
    data_spec = ae_config.data_specs[cv]
    _, data, _, scaler, dates, features = get_features(data,
                                                       data_spec['start'],
                                                       data_spec['end'],
                                                       assets,
                                                       val_start=data_spec['val_start'],
                                                       test_start=data_spec.get('test_start'),
                                                       rescale=ae_config.rescale,
                                                       scaler=ae_config.scaler_func['name'],
                                                       resample=ae_config.resample,
                                                       features_config=ae_config.features_config,
                                                       **ae_config.scaler_func.get('params',
                                                                                   {}))

    if ae_config.drop_remainder_obs:
        indices = list(range(data.shape[0]))
        indices = drop_remainder(indices, ae_config.batch_size, last=False)
        data = data[indices, :]
        features['val'] = features['val'][indices, :]
        dates['val'] = dates['val'][indices]

    LOGGER.info(f'Data shape: {data.shape}')

    if features:
        input = build_model_input(data, ae_config.model_type, features=features['val'])
    else:
        input = build_model_input(data, ae_config.model_type, features=None, assets=assets)

    prediction = model.predict(input)


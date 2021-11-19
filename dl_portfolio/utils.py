import datetime as dt
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

from dl_portfolio.ae_data import get_features
from dl_portfolio.pca_ae import build_model
from dl_portfolio.regularizers import WeightsOrthogonality

LOG_BASE_DIR = './dl_portfolio/log'


def load_result(test_set: str, data: pd.DataFrame, assets: List[str], base_dir: str, cv: str, ae_config):
    """

    :param test_set:
    :param data:
    :param assets:
    :param base_dir:
    :param cv:
    :param ae_config:
    :return:
    """
    scaler = pickle.load(open(f'{base_dir}/{cv}/scaler.p', 'rb'))
    embedding = pd.read_pickle(f'{base_dir}/{cv}/encoder_weights.p')
    input_dim = len(assets)
    model, encoder, extra_features = build_model(ae_config.model_type,
                                                 input_dim,
                                                 ae_config.encoding_dim,
                                                 n_features=None,
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
    model.load_weights(f'{base_dir}/{cv}/model.h5')
    layer_name = list(filter(lambda x: 'uncorrelated_features_layer' in x, [l.name for l in model.layers]))[0]
    encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    data_spec = ae_config.data_specs[cv]
    if test_set:
        _, _, test_data, _, dates, features = get_features(data,
                                                           data_spec['start'],
                                                           data_spec['end'],
                                                           assets,
                                                           val_start=data_spec['val_start'],
                                                           test_start=data_spec.get('test_start'))
    else:
        _, test_data, _, _, dates, features = get_features(data,
                                                           data_spec['start'],
                                                           data_spec['end'],
                                                           assets,
                                                           val_start=data_spec['val_start'],
                                                           test_start=data_spec.get('test_start'))
    std = np.sqrt(scaler['attributes']['var_'])
    test_data = (test_data - scaler['attributes']['mean_']) / std

    # Prediction
    pred = model.predict(test_data)
    pred = pred * std + scaler['attributes']['mean_']
    pred = pd.DataFrame(pred, columns=assets, index=dates[test_set])

    test_features = encoder.predict(test_data)
    test_features = pd.DataFrame(test_features, index=dates[test_set])

    return scaler, dates, test_data, test_features, pred, embedding


def create_log_dir(model_name, model_type):
    subdir = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(LOG_BASE_DIR, model_name, model_type, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    return log_dir


def get_best_model_from_dir(dir_):
    files = os.listdir(dir_)
    files = list(filter(lambda x: 'model' in x, files))
    files = [[f, f.split('e_')[-1].split('.')[0]] for f in files]
    files.sort(key=lambda x: x[1])
    file = files[-1][0]
    return file


def config_setter(config, params: Dict):
    for k in params:
        if k == 'encoding_dim':
            config.encoding_dim = params[k]
        elif k == 'ortho_weightage':
            config.ortho_weightage = params[k]
            config.kernel_regularizer = WeightsOrthogonality(
                config.encoding_dim,
                weightage=config.ortho_weightage,
                axis=0,
                regularizer={
                    'name': config.l_name,
                    'params': {config.l_name: config.l}
                }
            )
        elif k == 'weightage':
            config.weightage = params[k]
        else:
            raise NotImplementedError()

    return config

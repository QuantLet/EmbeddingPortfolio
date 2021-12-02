import datetime as dt
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

from dl_portfolio.ae_data import get_features
from dl_portfolio.pca_ae import build_model, create_decoder
from dl_portfolio.regularizers import WeightsOrthogonality

from sklearn.linear_model import LinearRegression

LOG_BASE_DIR = './dl_portfolio/log'


def get_nnls_analysis(test_set: str, data: pd.DataFrame, assets: List[str], base_dir: str, ae_config):
    """

    :param test_set:
    :param data:
    :param assets:
    :param base_dir:
    :param ae_config:
    :return:
    """

    test_data = pd.DataFrame()
    prediction = pd.DataFrame()
    # pred_nnls_factors = pd.DataFrame()
    pred_nnls_model = pd.DataFrame()
    factors_nnls = pd.DataFrame()
    relu_activation = pd.DataFrame()
    embedding = {}
    mse = {
        'original': [],
        'nnls_factors': [],
        'nnls_model': []
    }

    # cv = 0
    for cv in ae_config.data_specs:
        print(f'CV: {cv}')
        model, scaler, dates, test_data_i, test_features, pred, embed = load_result(test_set,
                                                                                        data,
                                                                                        assets,
                                                                                        base_dir,
                                                                                        cv,
                                                                                        ae_config)
        embedding[cv] = embed
        pred -= scaler['attributes']['mean_']
        pred /= np.sqrt(scaler['attributes']['var_'])
        mse_or = np.mean((test_data_i - pred) ** 2, 0)

        relu_activation_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer('encoder').output)
        relu_activation_i = relu_activation_layer.predict(test_data_i)
        relu_activation = pd.concat([relu_activation, pd.DataFrame(relu_activation_i,
                                                                   index=pred.index)])

        # Fit linear encoder to the factors
        # input_dim = model.layers[0].input_shape[0][-1]
        # encoding_dim = model.layers[1].output_shape[-1]
        # vlin_encoder = create_linear_encoder_with_constraint(input_dim, encoding_dim)
        # lin_encoder.fit(test_data_i, relu_activation_i, batch_size = 1, epochs=500, verbose=2,
        #                 max_queue_size=20, workers=2*os.cpu_count()-1, use_multiprocessing=True)
        # factors_nnls_i = lin_encoder.predict(test_data_i)
        # lin_embedding = pd.DataFrame(encoder.layers[1].weights[0].numpy(), index=embed.index)

        # # Fit non-negative linear least square to the factor
        reg_nnls = LinearRegression(positive=True)
        reg_nnls.fit(test_data_i, relu_activation_i)
        factors_nnls_i = reg_nnls.predict(test_data_i)
        factors_nnls = pd.concat([factors_nnls, pd.DataFrame(factors_nnls_i, index=pred.index)])

        # Get reconstruction error based on nnls embedding
        weights = reg_nnls.coef_.copy()
        # Compute bias
        bias = np.mean(test_data_i, 0) - np.dot(np.mean(factors_nnls_i, 0), weights)
        # Reconstruction
        pred_nnls_model_i = np.dot(factors_nnls_i, weights) + bias

        mse_nnls_model = np.mean((test_data_i - pred_nnls_model_i) ** 2, 0)

        prediction = pd.concat([prediction, pred])
        # pred_nnls_factors = pd.concat([pred_nnls_factors, pd.DataFrame(pred_nnls_factors_i,
        #                                                                columns=pred.columns,
        #                                                                index=pred.index)])
        pred_nnls_model = pd.concat([pred_nnls_model, pd.DataFrame(pred_nnls_model_i,
                                                                   columns=pred.columns,
                                                                   index=pred.index)])

        test_data = pd.concat([test_data, pd.DataFrame(test_data_i,
                                                       columns=pred.columns,
                                                       index=pred.index)])

        mse['original'].append(mse_or)
        mse['nnls_model'].append(mse_nnls_model)

    results = {
        'test_data': test_data,
        'prediction': prediction,
        # 'pred_nnls_factors': pred_nnls_factors,
        'pred_nnls_model': pred_nnls_model,
        'factors_nnls': factors_nnls,
        'relu_activation': relu_activation,
        'mse': mse,
        'embedding': embedding
    }

    return results


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
                                                           test_start=data_spec.get('test_start'),
                                                           scaler=scaler)
    else:
        _, test_data, _, _, dates, features = get_features(data,
                                                           data_spec['start'],
                                                           data_spec['end'],
                                                           assets,
                                                           val_start=data_spec['val_start'],
                                                           test_start=data_spec.get('test_start'),
                                                           scaler=scaler)
    # Prediction
    pred = model.predict(test_data)
    pred *= np.sqrt(scaler['attributes']['var_'])
    pred += scaler['attributes']['mean_']
    pred = pd.DataFrame(pred, columns=assets, index=dates[test_set])

    test_features = encoder.predict(test_data)
    test_features = pd.DataFrame(test_features, index=dates[test_set])

    return model, scaler, dates, test_data, test_features, pred, embedding


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

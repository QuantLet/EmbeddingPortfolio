import datetime as dt
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import activations

from dl_portfolio.logger import LOGGER
from dl_portfolio.ae_data import get_features
from dl_portfolio.pca_ae import build_model, create_decoder
from dl_portfolio.regularizers import WeightsOrthogonality
from dl_portfolio.regressors.nonnegative_linear.ridge import NonnegativeRidge
from dl_portfolio.regressors.nonnegative_linear.base import NonnegativeLinear

from sklearn.linear_model import LinearRegression, Lasso

LOG_BASE_DIR = './dl_portfolio/log'
BASE_FACTOR_ORDER_BOND = ["GE_B", "SPX_X", "EUR_FX", "BTC"]
BASE_FACTOR_ORDER_RAFFINOT = ["SP500", "EuroStox_Small", "Gold", "US-5Y", "French-5Y"]


def build_linear_model(ae_config, reg_type: str, **kwargs):
    if reg_type == 'nn_ridge':
        if ae_config.l_name == 'l2':
            alpha = kwargs.get('alpha', ae_config.l)
            kwargs['alpha'] = alpha
        else:
            alpha = kwargs.get('alpha')
            assert alpha is not None
        model = NonnegativeRidge(**kwargs)
    elif reg_type == 'nn_ls_custom':
        model = NonnegativeLinear()
    elif reg_type == 'nn_ls':
        model = LinearRegression(positive=True, fit_intercept=False, **kwargs)
    elif reg_type == 'nn_lasso':
        if ae_config.l_name == 'l1':
            alpha = kwargs.get('alpha', ae_config.l)
            kwargs['alpha'] = alpha
        else:
            alpha = kwargs.get('alpha')
            assert alpha is not None
        model = Lasso(positive=True, fit_intercept=False, **kwargs)
    else:
        raise NotImplementedError(reg_type)

    return model


def fit_nnls_one_cv(cv: int, test_set: str, data: pd.DataFrame, assets: List[str], base_dir: str,
                    ae_config, reg_type: str = 'nn_ridge', **kwargs):
    model, scaler, dates, test_data, test_features, prediction, embedding, decoding = load_result(ae_config,
                                                                                                  test_set,
                                                                                                  data,
                                                                                                  assets,
                                                                                                  base_dir,
                                                                                                  cv)
    prediction -= scaler['attributes']['mean_']
    prediction /= np.sqrt(scaler['attributes']['var_'])
    mse_or = np.mean((test_data - prediction) ** 2, 0)

    relu_activation_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer('encoder').output)
    relu_activation = relu_activation_layer.predict(test_data)
    relu_activation = pd.DataFrame(relu_activation, index=prediction.index)

    # Fit linear encoder to the factors
    # input_dim = model.layers[0].input_shape[0][-1]
    # encoding_dim = model.layers[1].output_shape[-1]
    # vlin_encoder = create_linear_encoder_with_constraint(input_dim, encoding_dim)
    # lin_encoder.fit(test_data_i, relu_activation_i, batch_size = 1, epochs=500, verbose=2,
    #                 max_queue_size=20, workers=2*os.cpu_count()-1, use_multiprocessing=True)
    # factors_nnls_i = lin_encoder.predict(test_data_i)
    # lin_embedding = pd.DataFrame(encoder.layers[1].weights[0].numpy(), index=embed.index)

    # # Fit non-negative linear least square to the factor
    reg_nnls = build_linear_model(ae_config, reg_type, **kwargs)
    x = test_data.copy()
    mean_ = np.mean(x, 0)
    # Center the data as we do not fit intercept
    x = x - mean_
    reg_nnls.fit(x, relu_activation)
    # Now compute intercept: it is just the mean of the dependent variable
    intercept_ = np.mean(relu_activation).values
    factors_nnls = reg_nnls.predict(x) + intercept_
    factors_nnls = pd.DataFrame(factors_nnls, index=prediction.index)

    # Get reconstruction error based on nnls embedding
    if ae_config.model_type == "pca_ae_model":
        # For PCA AE model encoder and decoder share weights
        weights = reg_nnls.coef_.copy()
        # Compute bias (reconstruction intercept)
        bias = mean_ - np.dot(np.mean(factors_nnls, 0), weights)
    elif ae_config.model_type == "ae_model":
        weights = model.get_layer('decoder').get_weights()[0]
        bias = model.get_layer('decoder').get_weights()[1]
    else:
        raise NotImplementedError(ae_config.model_type)

    # Reconstruction
    pred_nnls_model = np.dot(factors_nnls, weights) + bias
    mse_nnls_model = np.mean((test_data - pred_nnls_model) ** 2, 0)
    # pred_nnls_factors = pd.concat([pred_nnls_factors, pd.DataFrame(pred_nnls_factors_i,
    #                                                                columns=pred.columns,
    #                                                                index=pred.index)])
    pred_nnls_model = pd.DataFrame(pred_nnls_model, columns=prediction.columns, index=prediction.index)
    test_data = pd.DataFrame(test_data, columns=prediction.columns, index=prediction.index)
    reg_coef = pd.DataFrame(weights.T, index=embedding.index)

    return test_data, embedding, decoding, reg_coef, relu_activation, factors_nnls, prediction, pred_nnls_model, mse_or, mse_nnls_model


def get_nnls_analysis(test_set: str, data: pd.DataFrame, assets: List[str], base_dir: str, ae_config,
                      reg_type: str = 'nn_ridge', **kwargs):
    """

    :param test_set:
    :param data:
    :param assets:
    :param base_dir:
    :param ae_config:
    :param reg_type: regression type to fit "nn_ridge" for non negative Ridge or "nn_ls" for non negative LS
    :return:
    """

    test_data = pd.DataFrame()
    prediction = pd.DataFrame()
    # pred_nnls_factors = pd.DataFrame()
    pred_nnls_model = pd.DataFrame()
    factors_nnls = pd.DataFrame()
    relu_activation = pd.DataFrame()
    embedding = {}
    decoding = {}
    reg_coef = {}
    mse = {
        'original': [],
        'nnls_factors': [],
        'nnls_model': []
    }

    # cv = 0
    for cv in ae_config.data_specs:
        LOGGER.info(f'CV: {cv}')
        test_data_i, embedding_i, decoding_i, reg_coef_i, relu_activation_i, factors_nnls_i, pred, pred_nnls_model_i, mse_or, mse_nnls_model = fit_nnls_one_cv(
            cv,
            test_set,
            data,
            assets,
            base_dir,
            ae_config,
            reg_type=reg_type,
            **kwargs)

        embedding[cv] = embedding_i
        decoding[cv] = decoding_i
        reg_coef[cv] = reg_coef_i
        relu_activation = pd.concat([relu_activation, relu_activation_i])
        factors_nnls = pd.concat([factors_nnls, factors_nnls_i])
        prediction = pd.concat([prediction, pred])
        pred_nnls_model = pd.concat([pred_nnls_model, pred_nnls_model_i])
        test_data = pd.concat([test_data, test_data_i])
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
        'embedding': embedding,
        'decoding': decoding,
        'reg_coef': reg_coef
    }

    return results


def reorder_columns(data, new_order):
    return data.iloc[:, new_order]


def load_result_wrapper(config, test_set: str, data: pd.DataFrame, assets: List[str], base_dir: str,
                        reorder_features: bool = True, first_cv=None):
    test_data = pd.DataFrame()
    prediction = pd.DataFrame()
    features = pd.DataFrame()
    relu_activation = pd.DataFrame()
    residuals = pd.DataFrame()
    embedding = {}
    decoding = {}

    cvs = list(config.data_specs.keys())
    if first_cv:
        cvs = [cv for cv in cvs if cv >= first_cv]

    for cv in cvs:
        embedding[cv] = {}
        model, scaler, dates, t_data, f, pred, embed, decod, relu_act = load_result(config,
                                                                                    test_set,
                                                                                    data,
                                                                                    assets,
                                                                                    base_dir,
                                                                                    cv,
                                                                                    reorder_features)
        t_data = pd.DataFrame(t_data, columns=pred.columns, index=pred.index)
        t_data *= scaler["attributes"]["scale_"]
        t_data += scaler["attributes"]["mean_"]

        test_data = pd.concat([test_data, t_data])
        prediction = pd.concat([prediction, pred])

        features = pd.concat([features, f])
        if relu_act is not None:
            relu_activation = pd.concat([relu_activation, relu_act])
        residuals = pd.concat([residuals, t_data - pred])
        embedding[cv] = embed
        decoding[cv] = decod

    return test_data, prediction, features, residuals, embedding, decoding, relu_activation


def get_linear_encoder(config, test_set: str, data: pd.DataFrame, assets: List[str], base_dir: str, cv: str,
                       reorder_features=True):
    """

    :param model_type: 'ae' or 'nmf'
    :param test_set:
    :param data:
    :param assets:
    :param base_dir:
    :param cv:
    :param ae_config:
    :return:
    """
    model_type = config.model_type
    assert model_type in ["pca_ae_model", "ae_model", "convex_nmf", "semi_nmf"]
    assert test_set in ["train", "val", "test"]

    scaler = pickle.load(open(f'{base_dir}/{cv}/scaler.p', 'rb'))
    input_dim = len(assets)

    model, encoder, extra_features = build_model(config.model_type,
                                                 input_dim,
                                                 config.encoding_dim,
                                                 n_features=None,
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
    model.load_weights(f'{base_dir}/{cv}/model.h5')
    layer_name = list(filter(lambda x: 'uncorrelated_features_layer' in x, [l.name for l in model.layers]))[0]
    encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    dense_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer('encoder').output)
    dense_layer.layers[-1].activation = activations.linear

    assert dense_layer.layers[-1].activation == activations.linear
    assert encoder.layers[1].activation == activations.linear

    data_spec = config.data_specs[cv]
    if test_set == 'test':
        _, _, test_data, _, dates, _ = get_features(data,
                                                    data_spec['start'],
                                                    data_spec['end'],
                                                    assets,
                                                    val_start=data_spec['val_start'],
                                                    test_start=data_spec.get('test_start'),
                                                    scaler=scaler)
    elif test_set == 'val':
        _, test_data, _, _, dates, _ = get_features(data,
                                                    data_spec['start'],
                                                    data_spec['end'],
                                                    assets,
                                                    val_start=data_spec['val_start'],
                                                    test_start=data_spec.get('test_start'),
                                                    scaler=scaler)
    elif test_set == 'train':
        # For first cv: predict on train data then for the others used previous validation data for prediction
        if cv == 0:
            test_data, _, _, _, dates, _ = get_features(data,
                                                        data_spec['start'],
                                                        data_spec['end'],
                                                        assets,
                                                        val_start=data_spec['val_start'],
                                                        test_start=data_spec.get('test_start'),
                                                        scaler=scaler)
        else:
            data_spec = config.data_specs[cv - 1]
            _, test_data, _, _, dates, _ = get_features(data,
                                                        data_spec['start'],
                                                        data_spec['end'],
                                                        assets,
                                                        val_start=data_spec['val_start'],
                                                        test_start=data_spec.get('test_start'),
                                                        scaler=scaler)
    else:
        raise NotImplementedError(test_set)

    # Prediction
    test_features = encoder.predict(test_data)
    lin_activation = dense_layer.predict(test_data)

    if test_set == "train" and cv > 0:
        index = dates["val"]
    else:
        index = dates[test_set]

    test_features = pd.DataFrame(test_features, index=index)
    lin_activation = pd.DataFrame(lin_activation, index=index)

    if reorder_features:
        embedding = pd.read_pickle(f'{base_dir}/{cv}/encoder_weights.p')
        if config.dataset == "bond":
            base_order = BASE_FACTOR_ORDER_BOND
        elif config.dataset == "raffinot_bloomberg_comb_update_2021":
            base_order = BASE_FACTOR_ORDER_RAFFINOT
        else:
            raise NotImplementedError()
        new_order = [embedding.loc[c].idxmax() for c in base_order]
        test_features = reorder_columns(test_features, new_order)
        test_features.columns = base_order
        lin_activation = reorder_columns(lin_activation, new_order)
        lin_activation.columns = base_order

    return model, test_features, lin_activation


def load_result(config, test_set: str, data: pd.DataFrame, assets: List[str], base_dir: str, cv: str,
                reorder_features=True):
    """

    :param model_type: 'ae' or 'nmf'
    :param test_set:
    :param data:
    :param assets:
    :param base_dir:
    :param cv:
    :param ae_config:
    :return:
    """
    model_type = config.model_type
    assert model_type in ["pca_ae_model", "ae_model", "convex_nmf", "semi_nmf"]
    assert test_set in ["train", "val", "test"]

    scaler = pickle.load(open(f'{base_dir}/{cv}/scaler.p', 'rb'))
    input_dim = len(assets)

    if "ae" in model_type:
        embedding = pd.read_pickle(f'{base_dir}/{cv}/encoder_weights.p')
        if model_type == "pca_ae_model":
            decoding = embedding.copy()
        elif model_type == "ae_model":
            decoding = pd.read_pickle(f'{base_dir}/{cv}/decoder_weights.p')
        else:
            pass
        model, encoder, extra_features = build_model(config.model_type,
                                                     input_dim,
                                                     config.encoding_dim,
                                                     n_features=None,
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
        model.load_weights(f'{base_dir}/{cv}/model.h5')
        layer_name = list(filter(lambda x: 'uncorrelated_features_layer' in x, [l.name for l in model.layers]))[0]
        encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    elif model_type == "convex_nmf":
        model = pickle.load(open(f'{base_dir}/{cv}/model.p', "rb"))
        embedding = model.encoding.copy()
        embedding = pd.DataFrame(embedding, index=assets)
        decoding = model.components.copy()
        decoding = pd.DataFrame(decoding, index=assets)
    elif model_type == "semi_nmf":
        model = pickle.load(open(f'{base_dir}/{cv}/model.p', "rb"))
        decoding = model.components.copy()
        decoding = pd.DataFrame(decoding, index=assets)
        embedding = decoding.copy()
    else:
        raise NotImplementedError(model_type)

    data_spec = config.data_specs[cv]
    if test_set == 'test':
        _, _, test_data, _, dates, _ = get_features(data,
                                                    data_spec['start'],
                                                    data_spec['end'],
                                                    assets,
                                                    val_start=data_spec['val_start'],
                                                    test_start=data_spec.get('test_start'),
                                                    scaler=scaler)
    elif test_set == 'val':
        _, test_data, _, _, dates, _ = get_features(data,
                                                    data_spec['start'],
                                                    data_spec['end'],
                                                    assets,
                                                    val_start=data_spec['val_start'],
                                                    test_start=data_spec.get('test_start'),
                                                    scaler=scaler)
    elif test_set == 'train':
        # For first cv: predict on train data then for the others used previous validation data for prediction
        if cv == 0:
            test_data, _, _, _, dates, _ = get_features(data,
                                                        data_spec['start'],
                                                        data_spec['end'],
                                                        assets,
                                                        val_start=data_spec['val_start'],
                                                        test_start=data_spec.get('test_start'),
                                                        scaler=scaler)
        else:
            data_spec = config.data_specs[cv - 1]
            _, test_data, _, _, dates, _ = get_features(data,
                                                        data_spec['start'],
                                                        data_spec['end'],
                                                        assets,
                                                        val_start=data_spec['val_start'],
                                                        test_start=data_spec.get('test_start'),
                                                        scaler=scaler)
    else:
        raise NotImplementedError(test_set)

    # Prediction
    if "ae" in model_type:
        pred = model.predict(test_data)
        test_features = encoder.predict(test_data)
        relu_activation_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer('encoder').output)
        relu_activation = relu_activation_layer.predict(test_data)
    elif "nmf" in model_type:
        test_features = model.transform(test_data)
        pred = model.inverse_transform(test_features)
    else:
        raise NotImplementedError(model_type)

    pred *= np.sqrt(scaler['attributes']['var_'])
    pred += scaler['attributes']['mean_']
    if test_set == "train" and cv > 0:
        index = dates["val"]
    else:
        index = dates[test_set]
    pred = pd.DataFrame(pred, columns=assets, index=index)
    test_features = pd.DataFrame(test_features, index=index)
    if "ae" in model_type:
        relu_activation = pd.DataFrame(relu_activation, index=index)
    else:
        relu_activation = None

    if reorder_features:
        if config.dataset == "bond":
            base_order = BASE_FACTOR_ORDER_BOND
        elif config.dataset == "raffinot_bloomberg_comb_update_2021":
            base_order = BASE_FACTOR_ORDER_RAFFINOT
        else:
            raise NotImplementedError()
        new_order = [embedding.loc[c].idxmax() for c in base_order]
        test_features = reorder_columns(test_features, new_order)
        test_features.columns = base_order
        embed = reorder_columns(embedding, new_order)
        embed.columns = base_order
        decoding = reorder_columns(decoding, new_order)
        decoding.columns = base_order
        if relu_activation is not None:
            relu_activation = reorder_columns(relu_activation, new_order)
            relu_activation.columns = base_order

    return model, scaler, dates, test_data, test_features, pred, embedding, decoding, relu_activation


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


def config_setter(run, config, params: Dict):
    if run == "ae":
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
    elif run == "nmf":
        for k in params:
            if k == 'encoding_dim':
                config.encoding_dim = params[k]
    else:
        raise NotImplementedError(run)

    return config

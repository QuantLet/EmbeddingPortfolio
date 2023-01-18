import datetime as dt
import os
import pickle
from typing import Dict, List

import pandas as pd
import tensorflow as tf
from tensorflow.keras import activations

from dl_portfolio.data import get_features
from dl_portfolio.pca_ae import build_model
from dl_portfolio.regularizers import WeightsOrthogonality
from dl_portfolio.regressors.nonnegative_linear.ridge import NonnegativeRidge
from dl_portfolio.regressors.nonnegative_linear.base import NonnegativeLinear
from dl_portfolio.constant import (
    BASE_FACTOR_ORDER_DATASET1_3,
    BASE_FACTOR_ORDER_DATASET1_4,
    BASE_FACTOR_ORDER_DATASET2_5,
)

from sklearn.linear_model import LinearRegression, Lasso

LOG_BASE_DIR = "./dl_portfolio/log"


def build_linear_model(ae_config, reg_type: str, **kwargs):
    if reg_type == "nn_ridge":
        if ae_config.l_name == "l2":
            alpha = kwargs.get("alpha", ae_config.l)
            kwargs["alpha"] = alpha
        else:
            alpha = kwargs.get("alpha")
            assert alpha is not None
        model = NonnegativeRidge(**kwargs)
    elif reg_type == "nn_ls_custom":
        model = NonnegativeLinear()
    elif reg_type == "nn_ls":
        model = LinearRegression(positive=True, fit_intercept=False, **kwargs)
    elif reg_type == "nn_lasso":
        if ae_config.l_name == "l1":
            alpha = kwargs.get("alpha", ae_config.l)
            kwargs["alpha"] = alpha
        else:
            alpha = kwargs.get("alpha")
            assert alpha is not None
        model = Lasso(positive=True, fit_intercept=False, **kwargs)
    else:
        raise NotImplementedError(reg_type)

    return model


def reorder_columns(data, new_order):
    return data.iloc[:, new_order]


def load_result_wrapper(
    config,
    test_set: str,
    data: pd.DataFrame,
    assets: List[str],
    base_dir: str,
    reorder_features: bool = True,
    first_cv=None,
):
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
        (
            model,
            scaler,
            dates,
            t_data,
            f,
            pred,
            embed,
            decod,
            relu_act,
        ) = load_result(
            config, test_set, data, assets, base_dir, cv, reorder_features
        )
        t_data = pd.DataFrame(t_data, columns=pred.columns, index=pred.index)
        if scaler:
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

    return (
        test_data,
        prediction,
        features,
        residuals,
        embedding,
        decoding,
        relu_activation,
    )


def get_linear_encoder(
    config,
    test_set: str,
    data: pd.DataFrame,
    assets: List[str],
    base_dir: str,
    cv: str,
    reorder_features=True,
):
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

    if config.scaler_func:
        scaler = pickle.load(open(f"{base_dir}/{cv}/scaler.p", "rb"))
    else:
        scaler = None
    input_dim = len(assets)

    model, encoder, extra_features = build_model(
        config.model_type,
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
        loss=config.loss,
        uncorrelated_features=config.uncorrelated_features,
        weightage=config.weightage,
    )
    model.load_weights(f"{base_dir}/{cv}/model.h5")
    layer_name = list(
        filter(
            lambda x: "uncorrelated_features_layer" in x,
            [l.name for l in model.layers],
        )
    )[0]
    encoder = tf.keras.Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    dense_layer = tf.keras.Model(
        inputs=model.input, outputs=model.get_layer("encoder").output
    )
    dense_layer.layers[-1].activation = activations.linear

    assert dense_layer.layers[-1].activation == activations.linear
    assert encoder.layers[1].activation == activations.linear

    data_spec = config.data_specs[cv]
    if test_set == "test":
        _, _, test_data, _, dates = get_features(
            data,
            data_spec["start"],
            data_spec["end"],
            assets,
            val_start=data_spec["val_start"],
            test_start=data_spec.get("test_start"),
            scaler=scaler,
        )
    elif test_set == "val":
        _, test_data, _, _, dates = get_features(
            data,
            data_spec["start"],
            data_spec["end"],
            assets,
            val_start=data_spec["val_start"],
            test_start=data_spec.get("test_start"),
            scaler=scaler,
        )
    elif test_set == "train":
        # For first cv: predict on train data then for the others used previous validation data for prediction
        test_data, _, _, _, dates = get_features(
            data,
            data_spec["start"],
            data_spec["end"],
            assets,
            val_start=data_spec["val_start"],
            test_start=data_spec.get("test_start"),
            scaler=scaler,
        )
    else:
        raise NotImplementedError(test_set)

    # Prediction
    test_features = encoder.predict(test_data)
    lin_activation = dense_layer.predict(test_data)
    index = dates[test_set]
    test_features = pd.DataFrame(test_features, index=index)
    lin_activation = pd.DataFrame(lin_activation, index=index)

    if reorder_features:
        embedding = pd.read_pickle(f"{base_dir}/{cv}/encoder_weights.p")
        if config.dataset == "dataset1":
            if config.encoding_dim == 3:
                base_order = BASE_FACTOR_ORDER_DATASET1_3
            elif config.encoding_dim == 4:
                base_order = BASE_FACTOR_ORDER_DATASET1_4
            else:
                raise NotImplementedError()

        elif config.dataset == "dataset2":
            if config.encoding_dim == 5:
                base_order = BASE_FACTOR_ORDER_DATASET2_5
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        assert len(embedding.columns) == len(base_order)
        new_order = [embedding.loc[c].idxmax() for c in base_order]
        test_features = reorder_columns(test_features, new_order)
        test_features.columns = base_order
        lin_activation = reorder_columns(lin_activation, new_order)
        lin_activation.columns = base_order

    return model, test_features, lin_activation


def load_result(
    config,
    test_set: str,
    data: pd.DataFrame,
    assets: List[str],
    base_dir: str,
    cv: str,
    reorder_features=True,
):
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

    if config.scaler_func is not None:
        scaler = pickle.load(open(f"{base_dir}/{cv}/scaler.p", "rb"))
    else:
        scaler = None
    input_dim = len(assets)

    if "ae" in model_type:
        embedding = pd.read_pickle(f"{base_dir}/{cv}/encoder_weights.p")
        if model_type == "pca_ae_model":
            decoding = embedding.copy()
        elif model_type == "ae_model":
            decoding = pd.read_pickle(f"{base_dir}/{cv}/decoder_weights.p")
        else:
            pass
        model, encoder, extra_features = build_model(
            config.model_type,
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
            loss=config.loss,
            uncorrelated_features=config.uncorrelated_features,
            weightage=config.weightage,
        )
        model.load_weights(f"{base_dir}/{cv}/model.h5")
        layer_name = list(
            filter(
                lambda x: "uncorrelated_features_layer" in x,
                [l.name for l in model.layers],
            )
        )[0]
        encoder = tf.keras.Model(
            inputs=model.input, outputs=model.get_layer(layer_name).output
        )
    elif model_type == "convex_nmf":
        model = pickle.load(open(f"{base_dir}/{cv}/model.p", "rb"))
        embedding = model.encoding.copy()
        embedding = pd.DataFrame(embedding, index=assets)
        decoding = model.components.copy()
        decoding = pd.DataFrame(decoding, index=assets)
    elif model_type == "semi_nmf":
        model = pickle.load(open(f"{base_dir}/{cv}/model.p", "rb"))
        decoding = model.components.copy()
        decoding = pd.DataFrame(decoding, index=assets)
        embedding = decoding.copy()
    else:
        raise NotImplementedError(model_type)

    data_spec = config.data_specs[cv]
    if test_set == "test":
        _, _, test_data, _, dates = get_features(
            data,
            data_spec["start"],
            data_spec["end"],
            assets,
            val_start=data_spec["val_start"],
            test_start=data_spec.get("test_start"),
            scaler=scaler,
        )
    elif test_set == "val":
        _, test_data, _, _, dates = get_features(
            data,
            data_spec["start"],
            data_spec["end"],
            assets,
            val_start=data_spec["val_start"],
            test_start=data_spec.get("test_start"),
            scaler=scaler,
        )
    elif test_set == "train":
        # For first cv: predict on train data then for the others used previous validation data for prediction
        if cv == 0:
            test_data, _, _, _, dates = get_features(
                data,
                data_spec["start"],
                data_spec["end"],
                assets,
                val_start=data_spec["val_start"],
                test_start=data_spec.get("test_start"),
                scaler=scaler,
            )
        else:
            data_spec = config.data_specs[cv - 1]
            _, test_data, _, _, dates = get_features(
                data,
                data_spec["start"],
                data_spec["end"],
                assets,
                val_start=data_spec["val_start"],
                test_start=data_spec.get("test_start"),
                scaler=scaler,
            )
    else:
        raise NotImplementedError(test_set)

    # Prediction
    if "ae" in model_type:
        pred = model.predict(test_data)
        test_features = encoder.predict(test_data)
        relu_activation_layer = tf.keras.Model(
            inputs=model.input, outputs=model.get_layer("encoder").output
        )
        relu_activation = relu_activation_layer.predict(test_data)
    elif "nmf" in model_type:
        test_features = model.transform(test_data)
        pred = model.inverse_transform(test_features)
    else:
        raise NotImplementedError(model_type)

    if scaler is not None:
        std = scaler["attributes"]["scale_"]
        if std is None:
            std = 1.0
        pred *= std
        pred += scaler["attributes"]["mean_"]
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
        if config.dataset == "dataset1":
            if config.encoding_dim == 3:
                base_order = BASE_FACTOR_ORDER_DATASET1_3
            elif config.encoding_dim == 4:
                base_order = BASE_FACTOR_ORDER_DATASET1_4
            else:
                raise NotImplementedError()
        elif config.dataset == "dataset2":
            if config.encoding_dim == 5:
                base_order = BASE_FACTOR_ORDER_DATASET2_5
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        new_order = [embedding.loc[c].idxmax() for c in base_order]
        test_features = reorder_columns(test_features, new_order)
        test_features.columns = base_order
        embedding = reorder_columns(embedding, new_order)
        embedding.columns = base_order
        decoding = reorder_columns(decoding, new_order)
        decoding.columns = base_order
        if relu_activation is not None:
            relu_activation = reorder_columns(relu_activation, new_order)
            relu_activation.columns = base_order

    return (
        model,
        scaler,
        dates,
        test_data,
        test_features,
        pred,
        embedding,
        decoding,
        relu_activation,
    )


def create_log_dir(model_name, model_type):
    subdir = dt.datetime.strftime(dt.datetime.now(), "%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_BASE_DIR, model_name, model_type, subdir)
    if not os.path.isdir(
        log_dir
    ):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    return log_dir


def get_best_model_from_dir(dir_):
    files = os.listdir(dir_)
    files = list(filter(lambda x: "model" in x, files))
    files = [[f, f.split("e_")[-1].split(".")[0]] for f in files]
    files.sort(key=lambda x: x[1])
    file = files[-1][0]
    return file


def config_setter(run, config, params: Dict):
    if run == "ae":
        for k in params:
            if k == "encoding_dim":
                config.encoding_dim = params[k]
            elif k == "ortho_weightage":
                config.ortho_weightage = params[k]
                config.kernel_regularizer = WeightsOrthogonality(
                    config.encoding_dim,
                    weightage=config.ortho_weightage,
                    axis=0,
                    regularizer={
                        "name": config.l_name,
                        "params": {config.l_name: config.l},
                    },
                )
            elif k == "weightage":
                config.weightage = params[k]
            else:
                raise NotImplementedError()
    elif run == "nmf":
        for k in params:
            if k == "encoding_dim":
                config.encoding_dim = params[k]
    else:
        raise NotImplementedError(run)

    return config

import datetime as dt
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import activations

from dl_portfolio.cluster import compute_serial_matrix
from dl_portfolio.data import get_features
from dl_portfolio.pca_ae import build_model
from dl_portfolio.constant import (
    BASE_FACTOR_ORDER_DATASET1_4,
    BASE_FACTOR_ORDER_DATASET2_5,
    DATASET1_REF_CLUSTER,
    DATASET2_REF_CLUSTER,
)
from sklearn import metrics


LOG_BASE_DIR = "./dl_portfolio/log"


def reorder_columns(data, new_order):
    return data.iloc[:, new_order]


def get_intercept_from_model(model, scaler=None, batch_norm=False) -> list:
    model_layers = [layer.name for layer in model.layers]
    assert "decoder" in model_layers

    beta_tilde = model.get_layer("decoder").get_weights()[1]
    if scaler is not None:
        beta_tilde = (beta_tilde * scaler["scale_"]) + scaler["mean_"]

    if batch_norm:
        batch_norm_layer = [x for x in model_layers if "batch_normalization"
                            in x][0]
        batch_norm = model.get_layer(batch_norm_layer)
        epsilon = batch_norm.epsilon
        gamma_bn = np.array(batch_norm.gamma)
        beta_bn = np.array(batch_norm.beta)
        mu_bn = np.array(batch_norm.moving_mean)
        scale_bn = np.sqrt(np.array(batch_norm.moving_variance) + epsilon)

        W = model.get_layer("decoder").get_weights()[0].T
        if scaler is not None:
            W_tilde = np.dot(np.diag(scaler["scale_"]), W)
        else:
            W_tilde = W

        beta = beta_tilde + np.dot(
            W_tilde,
            (beta_bn - (gamma_bn / scale_bn) * mu_bn)
        )
    else:
        beta = beta_tilde

    return beta.tolist()


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
            decoder_bias,
            _,
            _,
            _,
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
        decoder_bias,
    )


def get_linear_encoder(
    config,
    test_set: str,
    data: pd.DataFrame,
    assets: List[str],
    base_dir: str,
    cv: str,
    layer="encoder",
    reorder_features=False,
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
    assert model_type in ["ae_model", "convex_nmf", "semi_nmf"]
    assert test_set in ["train", "val", "test"]

    if config.scaler_func:
        scaler = pickle.load(open(f"{base_dir}/{cv}/scaler.p", "rb"))
    else:
        scaler = None
    input_dim = len(assets)

    if config.encoding_dim is None:
        embedding = pd.read_pickle(f"{base_dir}/{cv}/encoder_weights.p")
        encoding_dim = embedding.shape[-1]
        # Set encoding_dim of kernel_regularizer
    else:
        encoding_dim = config.encoding_dim

    kernel_regularizer = config.kernel_regularizer
    kernel_regularizer.encoding_dim = encoding_dim

    model, encoder, extra_features = build_model(
        config.model_type,
        input_dim,
        encoding_dim,
        n_features=None,
        extra_features_dim=1,
        activation=config.activation,
        batch_normalization=config.batch_normalization,
        kernel_initializer=config.kernel_initializer,
        kernel_constraint=config.kernel_constraint,
        kernel_regularizer=kernel_regularizer,
        activity_regularizer=config.activity_regularizer,
        loss=config.loss,
        uncorrelated_features=config.uncorrelated_features,
        weightage=config.weightage,
        encoder_bias=config.encoder_bias,
        decoder_bias=config.decoder_bias,
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
    if layer in ["batch_norm", "uncorrelated_features_layer"]:
        # Take activations after batch normalization
        activation_layer = tf.keras.Model(
            inputs=model.input, outputs=model.get_layer(layer_name).output
        )
        batch_norm_layer = model.layers[2]
        boundary = np.array(
            batch_norm_layer.gamma * (
                    - batch_norm_layer.moving_mean /
                    np.sqrt(
                        batch_norm_layer.moving_variance + batch_norm_layer.epsilon)
            ) + batch_norm_layer.beta
        )
    elif layer == "relu":
        # Take activations after Relu layer
        activation_layer = tf.keras.Model(
            inputs=model.input, outputs=model.get_layer("encoder").output
        )
        boundary = np.zeros(model.layers[1].output.shape[-1])
    else:
        raise NotImplementedError(layer)
    activation_layer.get_layer("encoder").activation = activations.linear

    assert encoder.layers[1].activation == activations.linear
    intercept = get_intercept_from_model(model, scaler=scaler["attributes"],
                                         batch_norm=config.batch_normalization)

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
            excess_ret=config.excess_ret,
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
            excess_ret=config.excess_ret,
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
            excess_ret=config.excess_ret,
        )
    else:
        raise NotImplementedError(test_set)

    # Prediction
    test_features = encoder.predict(test_data)
    lin_activation = activation_layer.predict(test_data)
    index = dates[test_set]
    test_features = pd.DataFrame(test_features, index=index)
    lin_activation = pd.DataFrame(lin_activation, index=index)

    if reorder_features:
        decoding = pd.read_pickle(f"{base_dir}/{cv}/decoder_weights.p")
        if config.dataset == "dataset1":
            if config.encoding_dim == 4:
                base_order = BASE_FACTOR_ORDER_DATASET1_4
                ref_cluster = DATASET1_REF_CLUSTER
            else:
                raise NotImplementedError()
        elif config.dataset == "dataset2":
            if config.encoding_dim == 5:
                base_order = BASE_FACTOR_ORDER_DATASET2_5
                ref_cluster = DATASET2_REF_CLUSTER
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        new_order = get_features_order(decoding, ref_cluster)
        test_features = reorder_columns(test_features, new_order)
        test_features.columns = base_order
        lin_activation = reorder_columns(lin_activation, new_order)
        lin_activation.columns = base_order
        boundary = boundary[new_order]
        boundary = {c: float(boundary[i]) for i, c in enumerate(base_order)}
    else:
        boundary = {i: float(boundary[i]) for i in range(len(boundary))}
        new_order = None

    return model, test_features, lin_activation, intercept, boundary, new_order


def get_features_order(loadings: pd.DataFrame, ref_cluster: pd.DataFrame):
    """

    :param loadings:
    :param ref_cluster:
    :return:
    """
    encoding_dim = ref_cluster.shape[-1]
    dist = metrics.pairwise_distances(pd.concat([loadings.T, ref_cluster.T]))
    # Keep only the distances between loadings and ref_cluster
    dist = dist[:encoding_dim, encoding_dim:]
    return np.argmin(dist, axis=0).tolist()


def get_average_factor_loadings_over_runs(weights: pd.DataFrame,
                                          encoding_dim: int):
    """
    Group the different factors obtained from various NMReLu runs, into
    n_encoding groups and then average the factor loadings to get a unique d
    * p matrix of factor loadings.

    :param weights: typically: weights = pd.concat(
        [pd.read_pickle(f"{d}/{cv}/decoder_weights.p").T for d in dirs]
    )
    :param encoding_dim:
    :return:
    """
    # Get the distance between each factor loading vector
    dist = metrics.pairwise_distances(weights)
    assert all(np.diag(dist) < 1e-6)
    # Make the diagonal zero
    dist[np.diag_indices(len(dist))] = 0
    # Reorder the distance matrix
    seriated_dist, res_order, res_linkage = compute_serial_matrix(dist,
                                                                  method="single")
    new_order = [weights.index[i] for i in res_order]
    weights = weights.loc[new_order]
    # Now group by factor and take the average
    n_runs = int(len(weights) / encoding_dim)
    factor = np.array([[i] * n_runs for i in range(encoding_dim)]).flatten()
    weights["factor"] = factor
    weights = weights.groupby("factor").mean().T

    return weights


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
    assert model_type in ["ae_model", "convex_nmf", "semi_nmf"]
    assert test_set in ["train", "val", "test"]

    if config.scaler_func is not None:
        scaler = pickle.load(open(f"{base_dir}/{cv}/scaler.p", "rb"))
    else:
        scaler = None
    input_dim = len(assets)

    decoder_bias = None
    if "ae" in model_type:
        embedding = pd.read_pickle(f"{base_dir}/{cv}/encoder_weights.p")
        if model_type == "pca_ae_model":
            decoding = embedding.copy()
        elif model_type == "ae_model":
            decoding = pd.read_pickle(f"{base_dir}/{cv}/decoder_weights.p")
        else:
            pass
        if config.encoding_dim is None:
            encoding_dim = embedding.shape[-1]
            # Set encoding_dim of kernel_regularizer
            kernel_regularizer = config.kernel_regularizer
            kernel_regularizer.encoding_dim = encoding_dim
        else:
            encoding_dim = config.encoding_dim
            kernel_regularizer = config.kernel_regularizer

        model, encoder, extra_features = build_model(
            config.model_type,
            input_dim,
            encoding_dim,
            n_features=None,
            extra_features_dim=1,
            activation=config.activation,
            batch_normalization=config.batch_normalization,
            kernel_initializer=config.kernel_initializer,
            kernel_constraint=config.kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=config.activity_regularizer,
            loss=config.loss,
            uncorrelated_features=config.uncorrelated_features,
            weightage=config.weightage,
            encoder_bias=config.encoder_bias,
            decoder_bias=config.decoder_bias,
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
        if config.decoder_bias:
            decoder_bias = model.get_weights()[-1]
    elif model_type == "convex_nmf":
        model = pickle.load(open(f"{base_dir}/{cv}/model.p", "rb"))
        embedding = model.encoding.copy()
        embedding = pd.DataFrame(embedding, index=assets)
        decoding = model.decoding.copy()
        decoding = pd.DataFrame(decoding, index=assets)
    elif model_type == "semi_nmf":
        model = pickle.load(open(f"{base_dir}/{cv}/model.p", "rb"))
        decoding = model.G.copy()
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
            excess_ret=config.excess_ret,
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
            excess_ret=config.excess_ret,
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
                excess_ret=config.excess_ret,
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
                excess_ret=config.excess_ret,
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
            if config.encoding_dim == 4:
                base_order = BASE_FACTOR_ORDER_DATASET1_4
                ref_cluster = DATASET1_REF_CLUSTER
            else:
                raise NotImplementedError()
        elif config.dataset == "dataset2":
            if config.encoding_dim == 5:
                base_order = BASE_FACTOR_ORDER_DATASET2_5
                ref_cluster = DATASET2_REF_CLUSTER
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        new_order = get_features_order(decoding, ref_cluster)
        test_features = reorder_columns(test_features, new_order)
        test_features.columns = base_order
        embedding = reorder_columns(embedding, new_order)
        embedding.columns = base_order
        decoding = reorder_columns(decoding, new_order)
        decoding.columns = base_order
        if relu_activation is not None:
            relu_activation = reorder_columns(relu_activation, new_order)
            relu_activation.columns = base_order
    else:
        new_order = None

    if "ae" in model_type:
        if config.batch_normalization:
            batch_norm_layer = model.layers[2]
            u_batchnorm = np.array(
                batch_norm_layer.gamma * (
                        - batch_norm_layer.moving_mean /
                        np.sqrt(
                            batch_norm_layer.moving_variance + batch_norm_layer.epsilon)
                ) + batch_norm_layer.beta
            )
            u_batchnorm = pd.DataFrame(
                np.repeat(u_batchnorm.reshape(1, -1), len(test_data), axis=0),
                index=test_features.index,
                columns=test_features.columns)
        else:
            u_batchnorm = None
        u_relu = model.layers[1].get_weights()[-1]
        u_relu = pd.DataFrame(
            np.repeat(u_relu.reshape(1, -1), len(test_data), axis=0),
            index=test_features.index,
            columns=test_features.columns)
    else:
        u_batchnorm = None
        u_relu = None

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
        decoder_bias,
        u_relu,
        u_batchnorm,
        new_order,
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


def optimal_target_vol_test(returns: pd.Series, risk_free: pd.Series = 0.):
    """
    cf On the Optimality of Target Volatility Strategies, Kais Dachraoui,
    The Journal of Portfolio Management Apr 2018, 44 (5) 58-67;
    DOI: 10.3905/jpm.2018.44.5.058

    :param returns:
    :param risk_free:
    :return:
    """
    assert isinstance(returns, pd.Series)
    std_ = pd.DataFrame([returns.rolling(22).std(),
                         returns.rolling(60).std()]).T
    std_ = std_.dropna(how="all", axis=0).fillna(0.).max(1)
    std_ = std_.shift(1).dropna()
    sr_ = (returns - risk_free) / std_
    sr_.dropna(inplace=True)

    tvs = np.cov(sr_.values, std_.values)[0, 1]

    return tvs

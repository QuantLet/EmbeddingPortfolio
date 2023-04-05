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

from dl_portfolio.cluster import get_optimal_p_silhouette
from dl_portfolio.logger import LOGGER
from dl_portfolio.pca_ae import get_layer_by_name, heat_map, build_model
from dl_portfolio.data import get_features
from dl_portfolio.train import (
    fit,
    embedding_visualization,
    plot_history,
    create_dataset,
    build_model_input,
)
from dl_portfolio.constant import LOG_DIR
from dl_portfolio.nmf.semi_nmf import SemiNMF
from dl_portfolio.nmf.convex_nmf import ConvexNMF


def run_ae(
    config,
    data,
    assets,
    log_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    """

    :param config:
    :param data:
    :param assets:
    :param log_dir: if given save the result in log_dir folder, if not given
    use LOG_DIR
    :param seed: if given use specific seed
    :return:
    """

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
        if config.model_name is not None and config.model_name != "":
            subdir = f"m_{iter}_" + config.model_name + f"_seed_{seed}"
        else:
            subdir = f"m_{iter}_seed_{seed}"
        subdir = (
            subdir
            + "_"
            + str(dt.datetime.timestamp(dt.datetime.now())).replace(".", "")
        )
        save_dir = f"{log_dir}/{subdir}"
        os.makedirs(save_dir)
        copyfile(
            "./dl_portfolio/config/ae_config.py",
            os.path.join(save_dir, "ae_config.py"),
        )

    base_asset_order = assets.copy()
    assets_mapping = {
        i: base_asset_order[i] for i in range(len(base_asset_order))
    }

    if config.scaler_func is not None:
        scaler_method = config.scaler_func["name"]
        scaler_params = config.scaler_func.get("params", {})
    else:
        scaler_method = None
        scaler_params = {}

    for cv in config.data_specs:
        LOGGER.debug(f"Starting with cv: {cv}")
        if config.save:
            save_path = f"{save_dir}/{cv}"
            os.mkdir(f"{save_dir}/{cv}")
        else:
            save_path = None
        # Build model
        if config.encoding_dim is None:
            if config.nmf_model is not None:
                nmf_model = pickle.load(
                    open(f"{config.nmf_model}/{cv}/model.p", "rb")
                )
                encoding_dim = nmf_model.G.shape[-1]
            else:
                p_range = config.p_range
                assert max(p_range) < train_data.shape[-1]
                assert p_range is not None
                n_exp = config.n_exp
                if n_exp is None:
                    n_exp = 1000
                encoding_dim = get_optimal_p_silhouette(
                    train_data, p_range=p_range, n_exp=n_exp,
                    savepath=f"{save_path}/decision_curve.png",
                    show=config.show_plot,
                )
                LOGGER.info(f"Selected {encoding_dim} factors!")
        else:
            encoding_dim = config.encoding_dim

        # Set encoding_dim of kernel_regularizer
        kernel_regularizer = config.kernel_regularizer
        kernel_regularizer.encoding_dim = encoding_dim

        LOGGER.debug(f"Assets: {assets}")
        input_dim = len(assets)
        n_features = None
        model, encoder, extra_features = build_model(
            config.model_type,
            input_dim,
            encoding_dim,
            n_features=n_features,
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
        if config.nmf_model is not None:
            train_data, _, _, _, _ = get_features(
                data,
                config.data_specs[cv]["start"],
                config.data_specs[cv]["end"],
                assets,
                val_start=config.data_specs[cv]["val_start"],
                test_start=config.data_specs[cv].get("test_start"),
                scaler=scaler_method,
                resample=config.resample,
                excess_ret=config.excess_ret,
                **scaler_params,
            )

            LOGGER.info(
                f"Initilize weights with NMF model from {config.nmf_model}/"
                f"{cv}"
            )
            nmf_model = pickle.load(
                open(f"{config.nmf_model}/{cv}/model.p", "rb")
            )
            # Set encoder weights
            weights = nmf_model.encoding.copy()
            # Add small constant to avoid 0 weights at beginning of
            # training
            weights += 0.2
            # Make it unit norm
            weights = weights ** 2
            weights /= np.sum(weights, axis=0)
            weights = weights.astype(np.float32)

            if config.encoder_bias:
                # set bias
                bias = model.layers[1].get_weights()[1]
                model.layers[1].set_weights([weights, bias])
            else:
                model.layers[1].set_weights([weights])

            # Set decoder weights
            weights = nmf_model.G.copy()
            # Add small constant to avoid 0 weights at beginning of
            # training
            weights += 0.2
            # Make it unit norm
            weights = weights / np.linalg.norm(weights, axis=0)
            weights = weights.T.astype(np.float32)
            if config.decoder_bias:
                # set bias
                F = nmf_model.transform(train_data)
                bias = np.mean(train_data) - np.mean(
                    F.dot(nmf_model.G.T), 0
                )
                model.layers[-1].set_weights([weights, bias])
            else:
                model.layers[-1].set_weights([weights])

        # Create dataset:
        shuffle = False
        if config.resample is not None:
            if config.resample.get("when", None) != "each_epoch":
                train_dataset, val_dataset = create_dataset(
                    data,
                    assets,
                    config.data_specs[cv],
                    config.model_type,
                    batch_size=config.batch_size,
                    scaler_func=config.scaler_func,
                    resample=config.resample,
                    excess_ret=config.excess_ret,
                )

            else:
                shuffle = True

        # Set extra loss parameters
        if shuffle:
            model, history = fit(
                model,
                None,
                config.epochs,
                config.learning_rate,
                callbacks=config.callbacks,
                val_dataset=None,
                extra_features=n_features is not None,
                save_path=f"{save_path}" if config.save else None,
                shuffle=True,
                cv=cv,
                data=data,
                assets=assets,
                config=config,
            )
        else:
            model, history = fit(
                model,
                train_dataset,
                config.epochs,
                config.learning_rate,
                callbacks=config.callbacks,
                val_dataset=val_dataset,
                extra_features=n_features is not None,
                save_path=f"{save_path}" if config.save else None,
                shuffle=False,
            )

        if config.save:
            embedding_visualization(
                model, assets, log_dir=f"{save_path}/tensorboard/"
            )
            LOGGER.debug(f"Loading weights from {save_path}/model.h5")
            model.load_weights(f"{save_path}/model.h5")

        plot_history(history, save_path=save_path, show=config.show_plot)

        # Get results for later analysis
        data_spec = config.data_specs[cv]
        (train_data, val_data, test_data, scaler, dates,) = get_features(
            data,
            data_spec["start"],
            data_spec["end"],
            assets,
            val_start=data_spec["val_start"],
            test_start=data_spec.get("test_start"),
            scaler=scaler_method,
            resample=config.resample,
            excess_ret=config.excess_ret,
            **scaler_params,
        )

        LOGGER.debug(f"Train shape: {train_data.shape}")
        LOGGER.debug(f"Validation shape: {val_data.shape}")

        train_input = build_model_input(
            train_data,
            config.model_type,
            features=None,
        )
        val_input = build_model_input(
            val_data,
            config.model_type,
            features=None,
        )
        if test_data is not None:
            test_input = build_model_input(
                test_data,
                config.model_type,
                features=None,
            )

        # Get prediction
        if n_features:
            train_features = encoder.predict(train_input[0])
            val_features = encoder.predict(val_input[0])
        else:
            train_features = encoder.predict(train_input)
            val_features = encoder.predict(val_input)

        train_features = pd.DataFrame(train_features, index=dates["train"])
        LOGGER.info(f"Train features correlation:\n{train_features.corr()}")
        val_features = pd.DataFrame(val_features, index=dates["val"])
        LOGGER.info(f"Val features correlation:\n{val_features.corr()}")
        val_prediction = model.predict(val_input)
        if scaler:
            val_prediction = scaler.inverse_transform(val_prediction)
        val_prediction = pd.DataFrame(
            val_prediction, columns=assets, index=dates["val"]
        )

        # Get encoder weights
        decoder_weights = None
        if config.model_type in ["ae_model2", "nl_pca_ae_model"]:
            encoder_layer1 = get_layer_by_name(name="encoder1", model=model)
            encoder_weights1 = encoder_layer1.get_weights()[0]
            encoder_layer2 = get_layer_by_name(name="encoder2", model=model)
            encoder_weights2 = encoder_layer2.get_weights()[0]
            encoder_weights = np.dot(encoder_weights1, encoder_weights2)
            encoder_weights = pd.DataFrame(encoder_weights, index=assets)
            heat_map(
                pd.DataFrame(encoder_weights1), show=True, vmin=0.0, vmax=1.0
            )
            heat_map(
                pd.DataFrame(encoder_weights2), show=True, vmin=0.0, vmax=1.0
            )
            heat_map(encoder_weights, show=True)
        elif config.model_type == "pca_ae_model":
            encoder_layer = get_layer_by_name(name="encoder", model=model)
            encoder_weights = encoder_layer.get_weights()
            encoder_weights = pd.DataFrame(encoder_weights[0], index=assets)
        elif config.model_type == "ae_model":
            encoder_layer = get_layer_by_name(name="encoder", model=model)
            decoder_layer = get_layer_by_name(name="decoder", model=model)
            encoder_weights = encoder_layer.get_weights()
            decoder_weights = decoder_layer.get_weights()
            encoder_weights = pd.DataFrame(encoder_weights[0], index=assets)
            decoder_weights = pd.DataFrame(decoder_weights[0].T, index=assets)
            LOGGER.debug(f"Decoder weights:\n{decoder_weights}")

        LOGGER.debug(f"Encoder weights:\n{encoder_weights}")
        # Get prediction on test_data
        if test_data is not None:
            test_prediction = model.predict(test_input)
            if scaler:
                test_prediction = scaler.inverse_transform(test_prediction)
            test_prediction = pd.DataFrame(
                test_prediction, columns=assets, index=dates["test"]
            )

            if n_features:
                test_features = encoder.predict(test_input[0])
            else:
                test_features = encoder.predict(test_input)
            test_features = pd.DataFrame(test_features, index=dates["test"])

        # Rescale back input data
        if scaler:
            train_data = scaler.inverse_transform(train_data)
        train_data = pd.DataFrame(
            train_data, index=dates["train"], columns=assets
        )
        if scaler:
            val_data = scaler.inverse_transform(val_data)
        val_data = pd.DataFrame(val_data, index=dates["val"], columns=assets)
        if test_data is not None:
            if scaler:
                test_data = scaler.inverse_transform(test_data)
            test_data = pd.DataFrame(
                test_data, index=dates["test"], columns=assets
            )

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
            if type(config.kernel_constraint).__name__ == "NonNegAndUnitNorm":
                vmax = 1
                vmin = 0.0
            elif type(config.kernel_constraint).__name__ == "UnitNorm":
                vmax = 1
                vmin = -1
            else:
                vmax = None
                vmin = 0.0
        else:
            vmax = None
            vmin = None

        if config.show_plot:
            heat_map(
                encoder_weights, show=config.show_plot, vmax=vmax, vmin=vmin
            )
            if decoder_weights is not None:
                heat_map(
                    decoder_weights,
                    show=config.show_plot,
                    vmax=vmax,
                    vmin=vmin,
                )

        LOGGER.debug(f"Unit norm constraint:\n{(encoder_weights ** 2).sum(0)}")
        LOGGER.debug(
            "Orthogonality constraint:\n"
            f"{np.dot(encoder_weights.T, encoder_weights)}"
        )
        if decoder_weights is not None:
            LOGGER.debug(
                "Unit norm constraint (decoder):\n"
                f"{(decoder_weights ** 2).sum(0)}"
            )
            LOGGER.debug(
                f"Orthogonality constraint (decoder)"
                f"\n{np.dot(decoder_weights.T, encoder_weights)}"
            )

        if config.show_plot:
            if val_data is not None:
                for c in val_data.columns:
                    plt.plot(val_data[c], label="true")
                    plt.plot(val_prediction[c], label="pred")
                    plt.title(c)
                    plt.show()
            if test_data is not None:
                for c in test_data.columns:
                    plt.plot(test_data[c], label="true")
                    plt.plot(test_prediction[c], label="pred")
                    plt.title(c)
                    plt.show()

        if config.save:
            encoder_weights.to_pickle(f"{save_path}/encoder_weights.p")
            if decoder_weights is not None:
                decoder_weights.to_pickle(f"{save_path}/decoder_weights.p")
            if config.scaler_func is not None:
                config.scaler_func["attributes"] = scaler.__dict__
                pickle.dump(
                    config.scaler_func, open(f"{save_path}/scaler.p", "wb")
                )
            if test_data is not None:
                pass

        if config.show_plot:
            for c in train_features.columns:
                plt.plot(train_features[c], label="true")
                plt.title(c)
                plt.show()
            for c in val_features.columns:
                plt.plot(val_features[c], label="true")
                plt.title(c)
                plt.show()


def run_kmeans(config, data, assets, seed=None):
    if config.seed:
        seed = config.seed
    if seed is None:
        seed = np.random.randint(0, 1000)

    np.random.seed(seed)
    LOGGER.info(f"Set seed: {seed}")

    if config.save:
        if not os.path.isdir("log_kmeans"):
            os.mkdir("log_kmeans")
        iter = len(os.listdir("log_kmeans"))
        save_dir = (
            f"log_kmeans/m_{iter}_seed_{seed}"
            f"_{dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')}"
        )
        os.makedirs(save_dir)
        copyfile(
            "./dl_portfolio/config/ae_config.py",
            os.path.join(save_dir, "ae_config.py"),
        )

    for cv in config.data_specs:
        LOGGER.info(f"Starting with cv: {cv}")
        if config.save:
            save_path = f"{save_dir}/{cv}"
            os.mkdir(f"{save_dir}/{cv}")
        else:
            save_path = None

        LOGGER.info(f"Assets: {assets}")
        data_spec = config.data_specs[cv]
        (train_data, val_data, test_data, scaler, dates,) = get_features(
            data,
            data_spec["start"],
            data_spec["end"],
            assets,
            val_start=data_spec["val_start"],
            test_start=data_spec.get("test_start"),
            scaler="StandardScaler",
            resample=config.resample,
            excess_ret=config.excess_ret,
        )
        if config.encoding_dim is None:
            p_range = config.p_range
            assert max(p_range) < train_data.shape[-1]
            assert p_range is not None
            n_exp = config.n_exp
            if n_exp is None:
                n_exp = 1000
            n_components = get_optimal_p_silhouette(
                train_data, p_range=p_range, n_exp=n_exp,
                savepath=f"{save_path}/decision_curve.png",
                show=config.show_plot,
            )
        else:
            n_components = config.encoding_dim

        kmeans = KMeans(n_clusters=n_components, random_state=seed)
        kmeans.fit(train_data.T)
        labels = pd.DataFrame(kmeans.labels_.reshape(1, -1), columns=assets).T
        labels.columns = ["label"]
        clusters = {
            i: list(labels[labels["label"] == i].index)
            for i in range(n_components)
        }

        if config.save:
            pickle.dump(
                config.scaler_func, open(f"{save_path}/scaler.p", "wb")
            )
            pickle.dump(kmeans, open(f"{save_path}/model.p", "wb"))
            pickle.dump(clusters, open(f"{save_path}/clusters.p", "wb"))
            labels.to_pickle(f"{save_path}/labels.p")


def run_nmf(
    config,
    data,
    assets,
    log_dir: Optional[str] = None,
    seed: Optional[int] = None,
    verbose=0,
):
    if config.model_type == "convex_nmf":
        LOG_DIR = "log_convex_nmf"
    elif config.model_type == "semi_nmf":
        LOG_DIR = "log_semi_nmf"
    else:
        raise NotImplementedError(config.model_type)

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
        save_dir = (
            f"{log_dir}/m_{iter}_seed_{seed}_"
            f"{dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')}"
        )
        os.makedirs(save_dir)
        copyfile(
            "./dl_portfolio/config/nmf_config.py",
            os.path.join(save_dir, "nmf_config.py"),
        )

    if config.scaler_func is not None:
        scaler_method = config.scaler_func["name"]
        scaler_params = config.scaler_func.get("params", {})
    else:
        scaler_method = None
        scaler_params = {}

    mse = {}
    for cv in config.data_specs:
        LOGGER.info(f"Starting with cv: {cv}")
        if config.save:
            save_path = f"{save_dir}/{cv}"
            os.mkdir(f"{save_dir}/{cv}")
        else:
            save_path = None

        LOGGER.info(f"Assets: {assets}")
        data_spec = config.data_specs[cv]
        (train_data, val_data, test_data, scaler, dates,) = get_features(
            data,
            data_spec["start"],
            data_spec["end"],
            assets,
            val_start=data_spec["val_start"],
            test_start=data_spec.get("test_start"),
            scaler=scaler_method,
            resample=config.resample,
            excess_ret=config.excess_ret,
            **scaler_params,
        )
        if config.encoding_dim is None:
            p_range = config.p_range
            assert max(p_range) < train_data.shape[-1]
            assert p_range is not None
            n_exp = config.n_exp
            if n_exp is None:
                n_exp = 1000
            n_components = get_optimal_p_silhouette(
                train_data, p_range=p_range, n_exp=n_exp,
                savepath=f"{save_path}/decision_curve.png",
                show=config.show_plot,
            )
            LOGGER.info(f"Selected {n_components} factors!")
        else:
            n_components = config.encoding_dim

        if config.model_type == "convex_nmf":
            LOGGER.debug("Initiate convex NMF model")
            nmf = ConvexNMF(
                n_components=n_components,
                random_state=seed,
                verbose=verbose,
                norm_G=config.norm_G,
                norm_W=config.norm_W,
            )
        elif config.model_type == "semi_nmf":
            raise NotImplementedError("You must verify the logic here")
            LOGGER.debug("Initiate semi NMF model")
            nmf = SemiNMF(
                n_components=n_components,
                random_state=seed,
                verbose=verbose,
                norm=config.norm,
            )
        else:
            raise NotImplementedError(config.model_type)

        nmf.fit(train_data)
        mse[cv] = {
            "train": nmf.evaluate(train_data),
            "test": nmf.evaluate(val_data)
            if test_data is None
            else nmf.evaluate(test_data),
        }

        if config.save:
            LOGGER.debug(f"Saving result at cv {cv} at {save_path} ...")
            nmf.save(f"{save_path}/model.p")
            if scaler:
                config.scaler_func["attributes"] = scaler.__dict__
                pickle.dump(
                    config.scaler_func, open(f"{save_path}/scaler.p", "wb")
                )
            LOGGER.debug("Done")

        if config.show_plot:
            heat_map(pd.DataFrame(nmf.G, index=assets), show=config.show_plot)

    if config.save:
        json.dump(mse, open(f"{save_dir}/evaluation.json", "w"))

import numpy as np
import matplotlib.pyplot as plt
from dl_portfolio.custom_layer import UncorrelatedFeaturesLayer
from dl_portfolio.constraints import NonNegAndUnitNorm
from dl_portfolio.regularizers import WeightsOrthogonality
from typing import List, Optional
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.utils import CustomObjectScope


def create_linear_encoder_with_constraint(input_dim, encoding_dim):
    asset_input = tf.keras.layers.Input(
        input_dim, dtype=tf.float32, name="asset_input"
    )
    kernel_constraint = NonNegAndUnitNorm(
        max_value=1.0, axis=0
    )  # tf.keras.constraints.NonNeg()#
    kernel_regularizer = WeightsOrthogonality(
        encoding_dim,
        weightage=1,
        axis=0,
        regularizer={"name": "l2", "params": {"l2": 1e-3}},
    )
    encoder_layer = tf.keras.layers.Dense(
        encoding_dim,
        activation="linear",
        kernel_initializer=tf.keras.initializers.HeNormal(),
        kernel_regularizer=kernel_regularizer,
        kernel_constraint=kernel_constraint,
        use_bias=True,
        name="encoder",
        dtype=tf.float32,
    )
    encoder = tf.keras.models.Model(asset_input, encoder_layer(asset_input))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    encoder.compile(optimizer, "mse")

    return encoder


def create_decoder(pca_ae_model, weights: Optional[List[np.ndarray]] = None):
    output_dim = pca_ae_model.layers[0].input_shape[0][-1]
    encoding_dim = pca_ae_model.layers[1].output_shape[-1]

    factors = tf.keras.layers.Input(
        encoding_dim, dtype=tf.float32, name="relu_factor"
    )
    batch_norm = pca_ae_model.layers[2]
    output = tf.keras.layers.Dense(
        output_dim, activation="linear", dtype=tf.float32
    )
    if weights is None:
        W = pca_ae_model.layers[1].get_weights()[0].T
        b = pca_ae_model.layers[-1].get_weights()[0]
        weights = [W, b]

    decoder = tf.keras.models.Model(factors, output(batch_norm(factors)))
    decoder.layers[-1].set_weights(weights)

    return decoder


def build_model(model_type, input_dim, encoding_dim, **kwargs):
    if model_type == "ae_model":
        model, encoder, extra_features = ae_model(
            input_dim, encoding_dim, **kwargs
        )

    else:
        raise NotImplementedError()

    return model, encoder, extra_features


def get_layer_by_name(name, model):
    return [l for l in model.layers if l.name == name][0]


def ae_model(
    input_dim: int,
    encoding_dim: int,
    n_features: int = None,
    extra_features_dim: int = 1,
    activation: str = "linear",
    kernel_initializer: str = "glorot_uniform",
    activity_regularizer=None,
    kernel_constraint=None,
    kernel_regularizer=None,
    use_bias=True,
    **kwargs,
):
    uncorrelated_features = kwargs.get("uncorrelated_features", True)
    batch_size = kwargs.get("batch_size", None)
    weightage = kwargs.get("weightage", 1.0)
    batch_normalization = kwargs.get("batch_normalization", False)
    dropout = kwargs.get("dropout", None)

    if type(kernel_regularizer).__name__ == "WeightsOrthogonality":
        dkernel_regularizer = WeightsOrthogonality(
            input_dim, weightage=kernel_regularizer.weightage, axis=0
        )
        dkernel_regularizer.regularizer = dkernel_regularizer.regularizer

    if type(kernel_constraint).__name__ == "NonNegAndUnitNorm":
        dkernel_constraint = NonNegAndUnitNorm(max_value=1.0, axis=1)

    with CustomObjectScope(
        {"MyActivityRegularizer": activity_regularizer}
    ):  # required for Keras to recognize
        asset_input = tf.keras.layers.Input(
            input_dim,
            batch_size=batch_size,
            dtype=tf.float32,
            name="asset_input",
        )
        encoder_layer = tf.keras.layers.Dense(
            encoding_dim,
            activation=activation,
            kernel_initializer=kernel_initializer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            use_bias=use_bias,
            name="encoder",
            dtype=tf.float32,
        )
        decoder_layer = tf.keras.layers.Dense(
            input_dim,
            activation="linear",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=dkernel_regularizer,
            kernel_constraint=dkernel_constraint,
            use_bias=use_bias,
            name="decoder",
            dtype=tf.float32,
        )
        encoding = encoder_layer(asset_input)

        if dropout is not None:
            dropout_layer = tf.keras.layers.Dropout(dropout)
            encoding = dropout_layer(encoding)

        if batch_normalization:
            batch_norm_layer = tf.keras.layers.BatchNormalization()
            encoding = batch_norm_layer(encoding)

        if uncorrelated_features:
            activity_regularizer_layer = UncorrelatedFeaturesLayer(
                encoding_dim, norm="1", use_cov=True, weightage=weightage
            )

            encoding = activity_regularizer_layer(encoding)
        encoder = tf.keras.models.Model(asset_input, encoding)

        # Extra input
        if n_features is not None:
            extra_input = tf.keras.layers.Input(
                n_features,
                batch_size=batch_size,
                dtype=tf.float32,
                name="extra_input",
            )
            extra_features_layer = tf.keras.layers.Dense(
                extra_features_dim,
                activation="linear",
                use_bias=True,
                name="extra_features",
                dtype=tf.float32,
            )
            extra_features = extra_features_layer(extra_input)
            hidden_layer = tf.keras.layers.concatenate(
                [encoding, extra_features]
            )
            output = decoder_layer(hidden_layer)
            autoencoder = tf.keras.models.Model(
                [asset_input, extra_input], output
            )
        else:
            output = decoder_layer(encoding)
            autoencoder = tf.keras.models.Model(asset_input, output)
            extra_features = None

        return autoencoder, encoder, extra_features


def heat_map(encoder_weights, show=False, save_dir=None, **kwargs):
    n_clusters = len(encoder_weights.columns)
    yticks = list(encoder_weights.index)

    fig, axs = plt.subplots(1, n_clusters, figsize=(10, 10), sharey=True)

    for j, c in enumerate(list(encoder_weights.columns)):
        ax = sns.heatmap(
            encoder_weights[c].values.reshape(-1, 1),
            xticklabels=[c],
            yticklabels=yticks,
            ax=axs[j],
            cbar=j == n_clusters - 1,
            **kwargs,
        )
    if save_dir:
        plt.savefig(
            f"{save_dir}/clusters_heatmap.png",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )
    if show:
        plt.show()

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Union
import os
from dl_portfolio.data import get_features
from dl_portfolio.logger import LOGGER
from dl_portfolio.pca_ae import get_layer_by_name
from tensorflow.keras import backend as K
from tensorboard.plugins import projector
import matplotlib.pyplot as plt


def build_model_input(
    data: np.ndarray,
    model_type: str,
    features: Optional[np.ndarray] = None,
) -> Union[np.ndarray, List]:
    assert model_type in [
        "ae_model",
        "ae_model2",
        "pca_ae_model",
        "nl_pca_ae_model",
    ]
    if features:
        data = [data, features]

    return data


def create_dataset(
    data,
    assets: List,
    data_spec: Dict,
    model_type: str,
    batch_size: int,
    rescale: Optional[float] = None,
    scaler_func: Optional[Dict] = None,
    resample: Optional[Dict] = None,
    excess_ret=True,
):
    if scaler_func is not None:
        scaler_method = scaler_func["name"]
        scaler_params = scaler_func.get("params", {})
    else:
        scaler_method = None
        scaler_params = {}

    train_data, val_data, test_data, scaler, dates = get_features(
        data,
        data_spec["start"],
        data_spec["end"],
        assets,
        val_start=data_spec["val_start"],
        test_start=data_spec.get("test_start"),
        rescale=rescale,
        scaler=scaler_method,
        resample=resample,
        excess_ret=excess_ret,
        **scaler_params,
    )

    LOGGER.debug(f"Train shape: {train_data.shape}")
    LOGGER.debug(f"Validation shape: {val_data.shape}")

    train_input = build_model_input(train_data, model_type)
    val_input = build_model_input(val_data, model_type)
    if test_data is not None:
        test_input = build_model_input(test_data, model_type, features=None)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_input, train_data)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((val_input, val_data))

    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset


def EarlyStopping(MetricList, min_delta=0.1, patience=20, mode="min"):
    # https://stackoverflow.com/questions/59438904/applying-callbacks-in-a
    # -custom-training-loop-in-tensorflow-2-0
    # No early stopping for the first patience epochs
    if len(MetricList) <= patience:
        return False

    min_delta = abs(min_delta)
    if mode == "min":
        min_delta *= -1
    else:
        min_delta *= 1

    # last patience epochs
    last_patience_epochs = [
        x + min_delta for x in MetricList[::-1][1 : patience + 1]
    ]
    current_metric = MetricList[::-1][0]

    if mode == "min":
        if current_metric >= max(last_patience_epochs):
            LOGGER.debug(
                f"Metric did not decrease for the last {patience} epochs."
            )
            return True
        else:
            return False
    else:
        if current_metric <= min(last_patience_epochs):
            LOGGER.debug(
                f"Metric did not increase for the last {patience} epochs."
            )
            return True
        else:
            return False


# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def fit(
    model: tf.keras.models.Model,
    train_dataset: Optional[tf.data.Dataset],
    epochs,
    learning_rate: float,
    callbacks: Dict = None,
    val_dataset: tf.data.Dataset = None,
    extra_features: bool = False,
    save_path: str = None,
    shuffle: bool = False,
    cv=None,
    data=None,
    assets=None,
    config=None,
):
    """

    :param model:
    :param train_dataset:
    :param epochs:
    :param learning_rate:
    :param callbacks:
    :param val_dataset:
    :param extra_features:
    :param save_path:
    :param shuffle:
    :param cv:
    :param data:
    :param assets:
    :param config:
    :param df_sample_weights:
    :return:
    """

    loss_fn = tf.keras.losses.MeanSquaredError(name="mse_loss")

    # Train
    LOGGER.info("Start training")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Prepare the metrics.
    train_metric = [
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    ]
    val_metric = [
        tf.keras.metrics.MeanSquaredError(name="mse"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    ]

    @tf.function
    def train_step(x, y, *args, **kwargs):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss_value = loss_fn(y, pred, *args, **kwargs)
            # Add any extra losses created during the forward pass.
            reg_loss = tf.reduce_sum(model.losses)
            loss_value = loss_value + reg_loss
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if isinstance(train_metric, list):
            [m.update_state(y, pred) for m in train_metric]
        else:
            train_metric.update_state(y, pred)

        return float(loss_value), float(reg_loss)

    @tf.function
    def test_step(x, y, *args, **kwargs):
        pred = model(x, training=False)
        loss_value = loss_fn(y, pred, *args, **kwargs)

        # Add any extra losses created during the forward pass.
        reg_loss = tf.reduce_sum(model.losses)
        loss_value = loss_value + reg_loss

        if isinstance(val_metric, list):
            [m.update_state(y, pred) for m in val_metric]
        else:
            val_metric.update_state(y, pred)

        return loss_value, reg_loss

    early_stopping = callbacks.get("EarlyStopping")
    if early_stopping is not None:
        restore_best_weights = early_stopping["restore_best_weights"]
    else:
        restore_best_weights = False

    history = {
        "loss": [],
        "reg_loss": [],
        "mse": [],
        "rmse": [],
        "val_loss": [],
        "val_reg_loss": [],
        "val_mse": [],
        "val_rmse": [],
    }
    best_weights = None
    stop_training = False
    for epoch in range(epochs):
        if epoch % 10 == 0:
            LOGGER.info(f"Epochs to go: {epochs - epoch}")
        if shuffle:
            LOGGER.debug(f"Shuffling data at epoch {epoch}")
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

        # Iterate over the batches of the dataset.
        batch_loss = []
        batch_reg_loss = []
        if extra_features:
            for step, (
                x_batch_train_0,
                x_batch_train_1,
                y_batch_train,
            ) in enumerate(train_dataset):
                loss_value, reg_loss = train_step(
                    [x_batch_train_0, x_batch_train_1], y_batch_train
                )
                batch_loss.append(loss_value)
                batch_reg_loss.append(reg_loss)
        else:
            for step, (x_batch_train, y_batch_train) in enumerate(
                train_dataset
            ):
                loss_value, reg_loss = train_step(x_batch_train, y_batch_train)
                batch_loss.append(loss_value)
                batch_reg_loss.append(reg_loss)

        # Compute loss over epoch
        epoch_loss = np.mean(batch_loss)
        epoch_reg_loss = np.mean(batch_reg_loss)
        history["loss"].append(epoch_loss)
        history["reg_loss"].append(epoch_reg_loss)

        # Run a validation loop at the end of each epoch.
        batch_loss = []
        batch_reg_loss = []
        if extra_features:
            for x_batch_val_0, x_batch_val_1, y_batch_val in val_dataset:
                val_loss_value, val_reg_loss_value = test_step(
                    [x_batch_val_0, x_batch_val_1], y_batch_val
                )
                batch_loss.append(float(val_loss_value))
                batch_reg_loss.append(float(val_reg_loss_value))
        else:
            for x_batch_val, y_batch_val in val_dataset:
                val_loss_value, val_reg_loss_value = test_step(
                    x_batch_val, y_batch_val
                )
                batch_loss.append(float(val_loss_value))
                batch_reg_loss.append(float(val_reg_loss_value))
        # Compute loss over epoch
        val_epoch_loss = np.mean(batch_loss)
        val_epoch_reg_loss = np.mean(batch_reg_loss)
        history["val_loss"].append(val_epoch_loss)
        history["val_reg_loss"].append(val_epoch_reg_loss)

        # Display metrics at the end of each epoch and reset
        if isinstance(train_metric, list):
            history["mse"].append(float(train_metric[0].result().numpy()))
            history["rmse"].append(float(train_metric[1].result().numpy()))
            [m.reset_states() for m in train_metric]
        else:
            train_eval = train_metric.result().numpy()
            train_metric.reset_states()
        if isinstance(val_metric, list):
            history["val_mse"].append(float(val_metric[0].result().numpy()))
            history["val_rmse"].append(float(val_metric[1].result().numpy()))
            [m.reset_states() for m in val_metric]
        else:
            val_eval = val_metric.result().numpy()
            val_metric.reset_states()

        LOGGER.debug(
            f"Epoch {epoch}: loss = {np.round(history['loss'][-1], 4)} -reg_loss = {np.round(history['reg_loss'][-1], 4)} - mse = {np.round(history['mse'][-1], 4)} - rmse = {np.round(history['rmse'][-1], 4)} "
            f"- val_loss = {np.round(history['val_loss'][-1], 4)} - val_reg_loss = {np.round(history['val_reg_loss'][-1], 4)} - val_mse = {np.round(history['val_mse'][-1], 4)} - val_rmse = {np.round(history['val_rmse'][-1], 4)}"
        )

        # Early stopping
        if early_stopping:
            best_epoch = 0
            epoch_val_rmse = history["val_rmse"][-1]
            if early_stopping["monitor"] == "val_rmse":
                epoch_metric_stop = epoch_val_rmse
            elif early_stopping["monitor"] == "val_loss":
                epoch_metric_stop = val_epoch_loss
            else:
                raise NotImplementedError()

            if epoch_metric_stop <= np.min(history[early_stopping["monitor"]]):
                LOGGER.debug(
                    "Model has improved from {0:8.4f} to {1:8.4f}".format(
                        np.min(history[early_stopping["monitor"]]),
                        val_epoch_loss,
                    )
                )
                best_epoch = epoch
                LOGGER.debug(f"Restoring best model from epoch: {best_epoch}")
                if restore_best_weights:
                    best_weights = model.get_weights()
                    if save_path:
                        LOGGER.debug("Saving best model")
                        model.save(f"{save_path}/model.h5")
            else:
                LOGGER.debug(
                    "Model has not improved from {0:8.4f}".format(
                        np.min(history[early_stopping["monitor"]])
                    )
                )
            if epoch >= early_stopping["patience"]:
                stop_training = EarlyStopping(
                    history[early_stopping["monitor"]],
                    min_delta=early_stopping["min_delta"],
                    patience=early_stopping["patience"],
                    mode=early_stopping["mode"],
                )
            else:
                stop_training = False

        if stop_training:
            LOGGER.debug(f"Stopping training at epoch {epoch}")
            break

    if restore_best_weights:
        LOGGER.info(
            f"Training finished. Restoring best model from epoch: {best_epoch}"
        )
        model.set_weights(best_weights)
    else:
        if save_path:
            LOGGER.info(f"Training finished. Save model at last epoch")
            model.save(f"{save_path}/model.h5")
        else:
            LOGGER.info(f"Training finished at last epoch")

    return model, history


def embedding_visualization(model, labels, log_dir):
    # Tensorboard Embedding visualization
    # Set up a logs directory, so Tensorboard knows where to look for files.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, "metadata.tsv"), "w") as f:
        for asset in labels:
            f.write("{}\n".format(asset))

    # Save the weights we want to analyze as a variable.
    encoder_layer = get_layer_by_name(name="encoder", model=model)
    encoder_weights = tf.Variable(encoder_layer.get_weights()[0])
    # Create a checkpoint from embedding, the filename and key are the name
    # ofb the tensor.
    checkpoint = tf.train.Checkpoint(embedding=encoder_weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = "metadata.tsv"
    projector.visualize_embeddings(log_dir, config)


def plot_history(history, save_path=None, show=False):
    if "reg_loss" in history.keys():
        fix, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].plot(history["loss"], label="train")
        axs[0].plot(history["val_loss"], label="val")
        axs[0].set_title("loss")
        axs[0].legend()
        axs[1].plot(history["reg_loss"], label="train")
        axs[1].plot(history["val_reg_loss"], label="val")
        axs[1].set_title("reg loss")
        axs[1].legend()
        axs[2].plot(history["mse"], label="train")
        axs[2].plot(history["val_mse"], label="val")
        axs[2].set_title("mse")
        axs[2].legend()
        axs[3].plot(history["rmse"], label="train")
        axs[3].plot(history["val_rmse"], label="val")
        axs[3].set_title("rmse")
        axs[3].legend()
    else:
        fix, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(history["loss"], label="train")
        axs[0].plot(history["val_loss"], label="val")
        axs[0].set_title("loss")
        axs[0].legend()
        axs[1].plot(history["mse"], label="train")
        axs[1].plot(history["val_mse"], label="val")
        axs[1].set_title("mse")
        axs[1].legend()
        axs[2].plot(history["rmse"], label="train")
        axs[2].plot(history["val_rmse"], label="val")
        axs[2].set_title("rmse")
        axs[2].legend()

    if save_path:
        plt.savefig(f"{save_path}/history.png", transparent=True)

    if show:
        plt.show()
    plt.close()

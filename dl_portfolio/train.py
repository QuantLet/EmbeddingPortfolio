import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional
import os
from dl_portfolio.ae_data import get_features
from dl_portfolio.data import drop_remainder
from dl_portfolio.logger import LOGGER
from dl_portfolio.pca_ae import get_layer_by_name
from tensorflow.keras import backend as K
from tensorboard.plugins import projector
from dl_portfolio.losses import weighted_mse
import matplotlib.pyplot as plt


def build_model_input(train_data, val_data, test_data, model_type, features=None):
    if model_type == 'pca_permut_ae_model':
        raise NotImplementedError()
        train_input = [train_data[:, i].reshape(-1, 1) for i in range(len(assets))]
        val_input = [val_data[:, i].reshape(-1, 1) for i in range(len(assets))]
        if test_data is not None:
            test_input = [test_data[:, i].reshape(-1, 1) for i in range(len(assets))]
    elif model_type in ['ae_model', 'pca_ae_model']:
        train_input = train_data
        val_input = val_data
        test_input = test_data
    else:
        raise NotImplementedError()
    if features:
        train_input = [train_data, features['train']]
        val_input = [val_data, features['val']]

    return train_input, val_input, test_input


def create_dataset(data, assets: List, data_spec: Dict, model_type: str, batch_size: int,
                   rescale: Optional[float] = None,
                   features_config: Optional[Dict] = None, scaler_func: Optional[Dict] = None,
                   resample: Optional[Dict] = None, loss: Optional[str] = None, drop_remainder_obs: bool = True,
                   df_sample_weights=None):
    train_data, val_data, test_data, scaler, dates, features = get_features(data,
                                                                            data_spec['start'],
                                                                            data_spec['end'],
                                                                            assets,
                                                                            val_start=data_spec['val_start'],
                                                                            test_start=data_spec.get('test_start'),
                                                                            rescale=rescale,
                                                                            scaler=scaler_func['name'],
                                                                            resample=resample,
                                                                            features_config=features_config,
                                                                            **scaler_func.get('params',
                                                                                              {}))

    if drop_remainder_obs:
        indices = list(range(train_data.shape[0]))
        indices = drop_remainder(indices, batch_size, last=False)
        train_data = train_data[indices, :]
        features['train'] = features['train'][indices, :]
        dates['train'] = dates['train'][indices]

        indices = list(range(val_data.shape[0]))
        indices = drop_remainder(indices, batch_size, last=False)
        train_data = train_data[indices, :]
        features['val'] = features['val'][indices, :]
        dates['val'] = dates['val'][indices]

    # if shuffle_columns_while_training:
    #     train_data = np.transpose(train_data)
    #     np.random.seed(random_seed)
    #     np.random.shuffle(train_data)
    #     np.random.seed(seed)
    #     train_data = np.transpose(train_data)

    LOGGER.info(f'Train shape: {train_data.shape}')
    LOGGER.info(f'Validation shape: {val_data.shape}')

    train_input, val_input, test_input = build_model_input(train_data, val_data, test_data, model_type,
                                                           features=features)

    if features:
        n_features = features['train'].shape[-1]
    else:
        n_features = None

    if loss == 'weighted_mse':
        sample_weights = df_sample_weights.loc[dates['train']]
        sample_weights = tf.Variable(
            sample_weights.values,
            dtype=tf.float32
        )
        if n_features:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_input[0], train_input[1], train_data, sample_weights))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_data, sample_weights))
    else:
        if n_features:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_input[0], train_input[1], train_data))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_data))

    if n_features:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_input[0], val_input[1], val_data))
    else:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_input, val_data))

    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset


def EarlyStopping(MetricList, min_delta=0.1, patience=20, mode='min'):
    # https://stackoverflow.com/questions/59438904/applying-callbacks-in-a-custom-training-loop-in-tensorflow-2-0
    # No early stopping for the first patience epochs
    if len(MetricList) <= patience:
        return False

    min_delta = abs(min_delta)
    if mode == 'min':
        min_delta *= -1
    else:
        min_delta *= 1

    # last patience epochs
    last_patience_epochs = [x + min_delta for x in MetricList[::-1][1:patience + 1]]
    current_metric = MetricList[::-1][0]

    if mode == 'min':
        if current_metric >= max(last_patience_epochs):
            print(f'Metric did not decrease for the last {patience} epochs.')
            return True
        else:
            return False
    else:
        if current_metric <= min(last_patience_epochs):
            print(f'Metric did not increase for the last {patience} epochs.')
            return True
        else:
            return False


# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def fit(model: tf.keras.models.Model, train_dataset: tf.data.Dataset, epochs, learning_rate: float,
        loss: str = None, loss_asset_weights: Optional[tf.Tensor] = None, callbacks: Dict = None,
        val_dataset: tf.data.Dataset = None, extra_features: bool = False, save_path: str = None,
        shuffle: bool = False, cv=None, data= None, assets = None, ae_config=None, df_sample_weights=None):
    """

    :param model: keras model to train
    :param train_dataset: train dataset
    :param epochs: number of epochs
    :param learning_rate: learning rate
    :param loss: loss function instance
    :param callbacks: Dictionary of callbacks
    :param val_dataset: validation dataset
    :param extra_features: boolean, flag if model use extra features, dataset is then a list
    :param save_path: Base directory for logs and saved object
    :return:
    """
    # x = None, y = None, batch_size = None, epochs = 1, verbose = 'auto',
    # callbacks = None, validation_split = 0.0, validation_data = None, shuffle = True,
    # class_weight = None, sample_weight = None, initial_epoch = 0, steps_per_epoch = None,
    # validation_steps = None, validation_batch_size = None, validation_freq = 1,
    # max_queue_size = 10, workers = 1, use_multiprocessing = False

    if loss == 'mse_with_covariance_penalty':
        pass
    elif loss == 'mse':
        loss_fn = tf.keras.losses.MeanSquaredError(name='mse_loss')
        if loss_asset_weights is not None:
            loss_fn = weighted_mse
    elif loss == 'weighted_mse':
        raise NotImplementedError(
            'Verify implementation: check sample_weight parameter: https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError')
        loss_fn = weighted_mse
        if loss_asset_weights is not None:
            raise NotImplementedError('Do not support both sample weights and output weights for now')


    else:
        raise NotImplementedError()

    if callbacks.get('ActivityRegularizer'):
        tf.config.run_functions_eagerly(True)
        # cv_callbacks.append(ActivityRegularizer(model))

    # Train
    LOGGER.info('Start training')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Prepare the metrics.
    train_metric = [tf.keras.metrics.MeanSquaredError(name='mse'),
                    tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    val_metric = [tf.keras.metrics.MeanSquaredError(name='mse'),
                  tf.keras.metrics.RootMeanSquaredError(name='rmse')]

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

        return loss_value, reg_loss

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

    early_stopping = callbacks.get('EarlyStopping')
    if early_stopping is not None:
        restore_best_weights = early_stopping['restore_best_weights']
    else:
        restore_best_weights = False

    history = {'loss': [], 'reg_loss': [], 'mse': [], 'rmse': [], 'val_loss': [], 'val_reg_loss': [], 'val_mse': [],
               'val_rmse': []}
    best_weights = None
    stop_training = False
    for epoch in range(epochs):
        if shuffle:
            LOGGER.info(f'Shuffling data at epoch {epoch}')
            train_dataset, val_dataset = create_dataset(data,
                                                        assets,
                                                        ae_config.data_specs[cv],
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

        # Iterate over the batches of the dataset.
        batch_loss = []
        batch_reg_loss = []
        if loss == 'weighted_mse':
            if extra_features:
                for step, (x_batch_train_0, x_batch_train_1, y_batch_train, weights_batch) in enumerate(
                        train_dataset):
                    loss_value, reg_loss = train_step([x_batch_train_0, x_batch_train_1], y_batch_train,
                                                      weights_batch)
                    batch_loss.append(float(loss_value))
                    batch_reg_loss.append(float(reg_loss))
            else:
                for step, (x_batch_train, y_batch_train, weights_batch) in enumerate(train_dataset):
                    loss_value, reg_loss = train_step(x_batch_train, y_batch_train, weights_batch)
                    batch_loss.append(float(loss_value))
                    batch_reg_loss.append(float(reg_loss))
        else:
            if loss_asset_weights is None:
                if extra_features:
                    for step, (x_batch_train_0, x_batch_train_1, y_batch_train) in enumerate(train_dataset):
                        loss_value, reg_loss = train_step([x_batch_train_0, x_batch_train_1], y_batch_train)
                        batch_loss.append(float(loss_value))
                        batch_reg_loss.append(float(reg_loss))
                else:
                    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                        loss_value, reg_loss = train_step(x_batch_train, y_batch_train)
                        batch_loss.append(float(loss_value))
                        batch_reg_loss.append(float(reg_loss))
            else:
                if extra_features:
                    for step, (x_batch_train_0, x_batch_train_1, y_batch_train) in enumerate(train_dataset):
                        loss_value, reg_loss = train_step([x_batch_train_0, x_batch_train_1], y_batch_train,
                                                          loss_asset_weights)
                        batch_loss.append(float(loss_value))
                        batch_reg_loss.append(float(reg_loss))
                else:
                    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                        loss_value, reg_loss = train_step(x_batch_train, y_batch_train,
                                                          loss_asset_weights)
                        batch_loss.append(float(loss_value))
                        batch_reg_loss.append(float(reg_loss))

        # Compute loss over epoch
        epoch_loss = np.mean(batch_loss)
        epoch_reg_loss = np.mean(batch_reg_loss)
        history['loss'].append(epoch_loss)
        history['reg_loss'].append(epoch_reg_loss)

        # Run a validation loop at the end of each epoch.
        batch_loss = []
        batch_reg_loss = []
        if extra_features:
            for x_batch_val_0, x_batch_val_1, y_batch_val in val_dataset:
                val_loss_value, val_reg_loss_value = test_step([x_batch_val_0, x_batch_val_1], y_batch_val)
                batch_loss.append(float(val_loss_value))
                batch_reg_loss.append(float(val_reg_loss_value))
        else:
            for x_batch_val, y_batch_val in val_dataset:
                val_loss_value, val_reg_loss_value = test_step(x_batch_val, y_batch_val)
                batch_loss.append(float(val_loss_value))
                batch_reg_loss.append(float(val_reg_loss_value))
        # Compute loss over epoch
        val_epoch_loss = np.mean(batch_loss)
        val_epoch_reg_loss = np.mean(batch_reg_loss)
        history['val_loss'].append(val_epoch_loss)
        history['val_reg_loss'].append(val_epoch_reg_loss)

        # Display metrics at the end of each epoch and reset
        if isinstance(train_metric, list):
            history['mse'].append(float(train_metric[0].result().numpy()))
            history['rmse'].append(float(train_metric[1].result().numpy()))
            [m.reset_states() for m in train_metric]
        else:
            train_eval = train_metric.result().numpy()
            train_metric.reset_states()
        if isinstance(val_metric, list):
            history['val_mse'].append(float(val_metric[0].result().numpy()))
            history['val_rmse'].append(float(val_metric[1].result().numpy()))
            [m.reset_states() for m in val_metric]
        else:
            val_eval = val_metric.result().numpy()
            val_metric.reset_states()

        LOGGER.info(
            f"Epoch {epoch}: loss = {np.round(history['loss'][-1], 4)} - reg_loss = {np.round(history['reg_loss'][-1], 4)} - mse = {np.round(history['mse'][-1], 4)} - rmse = {np.round(history['rmse'][-1], 4)} "
            f"- val_loss = {np.round(history['val_loss'][-1], 4)} - val_reg_loss = {np.round(history['val_reg_loss'][-1], 4)} - val_mse = {np.round(history['val_mse'][-1], 4)} - val_rmse = {np.round(history['val_rmse'][-1], 4)}")

        # Early stopping
        if early_stopping:
            best_epoch = 0
            epoch_val_rmse = history['val_rmse'][-1]
            if early_stopping['monitor'] == 'val_rmse':
                epoch_metric_stop = epoch_val_rmse
            elif early_stopping['monitor'] == 'val_loss':
                epoch_metric_stop = val_epoch_loss
            else:
                raise NotImplementedError()

            if epoch_metric_stop <= np.min(history[early_stopping['monitor']]):
                LOGGER.info("Model has improved from {0:8.4f} to {1:8.4f}".format(
                    np.min(history[early_stopping['monitor']]),
                    val_epoch_loss))
                best_epoch = epoch
                LOGGER.info(f"Restoring best model from epoch: {best_epoch}")
                if restore_best_weights:
                    best_weights = model.get_weights()
                    if save_path:
                        LOGGER.info('Saving best model')
                        model.save(f"{save_path}/model.h5")
            else:
                LOGGER.info(
                    "Model has not improved from {0:8.4f}".format(np.min(history[early_stopping['monitor']])))
            if epoch >= early_stopping['patience']:
                stop_training = EarlyStopping(history[early_stopping['monitor']],
                                              min_delta=early_stopping['min_delta'],
                                              patience=early_stopping['patience'],
                                              mode=early_stopping['mode'])
            else:
                stop_training = False

        if stop_training:
            LOGGER.info(f"Stopping training at epoch {epoch}")
            break

    if restore_best_weights:
        LOGGER.info(f"Training finished. Restoring best model from epoch: {best_epoch}")
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
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for asset in labels:
            f.write("{}\n".format(asset))

    # Save the weights we want to analyze as a variable.
    encoder_layer = get_layer_by_name(name='encoder', model=model)
    encoder_weights = tf.Variable(encoder_layer.get_weights()[0])
    # Create a checkpoint from embedding, the filename and key are the name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=encoder_weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def plot_history(history, save_path=None, show=False):
    if 'reg_loss' in history.keys():
        fix, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].plot(history['loss'], label='train')
        axs[0].plot(history['val_loss'], label='val')
        axs[0].set_title('loss')
        axs[0].legend()
        axs[1].plot(history['reg_loss'], label='train')
        axs[1].plot(history['val_reg_loss'], label='val')
        axs[1].set_title('reg loss')
        axs[1].legend()
        axs[2].plot(history['mse'], label='train')
        axs[2].plot(history['val_mse'], label='val')
        axs[2].set_title('mse')
        axs[2].legend()
        axs[3].plot(history['rmse'], label='train')
        axs[3].plot(history['val_rmse'], label='val')
        axs[3].set_title('rmse')
        axs[3].legend()
    else:
        fix, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(history['loss'], label='train')
        axs[0].plot(history['val_loss'], label='val')
        axs[0].set_title('loss')
        axs[0].legend()
        axs[1].plot(history['mse'], label='train')
        axs[1].plot(history['val_mse'], label='val')
        axs[1].set_title('mse')
        axs[1].legend()
        axs[2].plot(history['rmse'], label='train')
        axs[2].plot(history['val_rmse'], label='val')
        axs[2].set_title('rmse')
        axs[2].legend()

    if save_path:
        plt.savefig(f"{save_path}/history.png")

    if show:
        plt.show()
    plt.close()

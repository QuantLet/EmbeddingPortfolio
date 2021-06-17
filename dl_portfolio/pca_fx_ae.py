import pandas as pd
import matplotlib.pyplot as plt
import os, pickle
from dl_portfolio.logger import LOGGER
import datetime as dt
from dl_portfolio.pca_ae import ActivityRegularizer, NonNegAndUnitNormInit, heat_map_cluster, pca_ae_model, \
    get_layer_by_name, heat_map, pca_permut_ae_model, ae_model
from tensorflow.keras import backend as K
from shutil import copyfile
from dl_portfolio.data import drop_remainder
from tensorboard.plugins import projector
from dl_portfolio.losses import weighted_mae, weighted_mse
from dl_portfolio.ae_data import get_features, load_data, get_sample_weights_from_df, labelQuantile

LOG_DIR = 'dl_portfolio/log_fx_AE'


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


if __name__ == "__main__":
    from dl_portfolio.config.ae_config import *

    random_seed = np.random.randint(0, 100)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    LOGGER.info(f"Set seed: {seed}")

    if save:
        if model_name is not None and model_name != '':
            subdir = model_name
        else:
            subdir = 'model'
        subdir = subdir + '_' + dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')
        save_dir = f"{LOG_DIR}/{subdir}"
        os.mkdir(save_dir)
        copyfile('./dl_portfolio/config/ae_config.py',
                 os.path.join(save_dir, 'ae_config.py'))
    data, assets = load_data(type=data_type, dropnan=dropnan, freq=freq)

    base_asset_order = assets.copy()
    assets_mapping = {i: base_asset_order[i] for i in range(len(base_asset_order))}

    if loss == 'weighted_mse':
        file_name = f"./data/sample_weights_lq_{label_param['lq']}_uq_{label_param['uq']}_w_{label_param['window']}.p"
        if os.path.isfile(file_name):
            LOGGER.info(f'Loading sample weights from {file_name}')
            df_sample_weights = pd.read_pickle(file_name)
            df_sample_weights = df_sample_weights[assets]
        else:
            LOGGER.info('Computing sample weights ...')
            d, _ = load_data(type=['indices', 'forex', 'forex_metals', 'commodities'], dropnan=True)
            t_sample_weights, _ = get_sample_weights_from_df(d, labelQuantile, **label_param)
            d, _ = load_data(type=['crypto'], dropnan=False)
            c_sample_weights, _ = get_sample_weights_from_df(d, labelQuantile, **label_param)
            df_sample_weights = pd.concat([t_sample_weights, c_sample_weights], 1)
            df_sample_weights = df_sample_weights.fillna(0.0)
            df_sample_weights = df_sample_weights[assets]
            del d
            del c_sample_weights
            del t_sample_weights
            LOGGER.info(f'Saving sample weights to {file_name}')
            df_sample_weights.to_pickle(
                file_name)
            LOGGER.info('Done')

    for cv in data_specs:
        LOGGER.info(f'Starting with cv: {cv}')
        if save:
            os.mkdir(f"{save_dir}/{cv}")
        data_spec = data_specs[cv]

        if shuffle_columns:
            LOGGER.info('Shuffle assets order')
            if cv == 0:
                random_assets = assets.copy()
                np.random.seed(random_seed)
                np.random.shuffle(random_assets)
                np.random.seed(seed)
                assets = random_assets
            else:
                np.random.seed(random_seed)
                np.random.shuffle(assets)
                np.random.seed(seed)

        LOGGER.info(f'Assets order: {assets}')
        if loss == 'weighted_mse':
            # reorder columns
            df_sample_weights = df_sample_weights[assets]

        train_data, val_data, test_data, scaler, dates, features = get_features(data, data_spec['start'],
                                                                                data_spec['end'],
                                                                                assets, val_size=val_size,
                                                                                rescale=rescale,
                                                                                scaler=scaler_func['name'],
                                                                                features_config=features_config,
                                                                                **scaler_func.get('params', {}))

        # if shuffle_columns_while_training:
        #     train_data = np.transpose(train_data)
        #     np.random.seed(random_seed)
        #     np.random.shuffle(train_data)
        #     np.random.seed(seed)
        #     train_data = np.transpose(train_data)

        LOGGER.info(f'Train shape: {train_data.shape}')
        LOGGER.info(f'Validation shape: {val_data.shape}')
        # Build model
        input_dim = len(assets)
        n_features = None
        if model_type == 'pca_permut_ae_model':
            model, encoder = pca_permut_ae_model(input_dim, encoding_dim, activation=activation,
                                                 kernel_initializer=kernel_initializer,
                                                 kernel_constraint=kernel_constraint,
                                                 kernel_regularizer=kernel_regularizer,
                                                 activity_regularizer=activity_regularizer,
                                                 pooling=pooling
                                                 )
            train_input = [train_data[:, i].reshape(-1, 1) for i in range(len(assets))]
            val_input = [val_data[:, i].reshape(-1, 1) for i in range(len(assets))]
            test_input = [test_data[:, i].reshape(-1, 1) for i in range(len(assets))]

        elif model_type == 'ae_model':
            if features:
                n_features = features['train'].shape[-1]
            else:
                n_features = None
            model, encoder, extra_features = ae_model(input_dim,
                                                      encoding_dim,
                                                      n_features=n_features,
                                                      extra_features_dim=1,
                                                      activation=activation,
                                                      kernel_initializer=kernel_initializer,
                                                      kernel_constraint=kernel_constraint,
                                                      kernel_regularizer=kernel_regularizer,
                                                      activity_regularizer=activity_regularizer,
                                                      batch_size=batch_size if drop_remainder_obs else None,
                                                      loss=loss
                                                      )
            train_input = train_data
            val_input = val_data
            test_input = test_data

        elif model_type == 'pca_ae_model':
            if features:
                n_features = features['train'].shape[-1]
            else:
                n_features = None
            model, encoder, extra_features = pca_ae_model(input_dim,
                                                          encoding_dim,
                                                          n_features=n_features,
                                                          extra_features_dim=1,
                                                          activation=activation,
                                                          kernel_initializer=kernel_initializer,
                                                          kernel_constraint=kernel_constraint,
                                                          kernel_regularizer=kernel_regularizer,
                                                          activity_regularizer=activity_regularizer,
                                                          batch_size=batch_size if drop_remainder_obs else None,
                                                          loss=loss
                                                          )
            train_input = train_data
            val_input = val_data
            test_input = test_data

        else:
            raise NotImplementedError()

        if callback_activity_regularizer:
            tf.config.run_functions_eagerly(True)
            cv_callbacks.append(ActivityRegularizer(model))

        if drop_remainder_obs:
            indices = list(range(train_input.shape[0]))
            indices = drop_remainder(indices, batch_size, last=False)
            train_input = train_input[indices, :]
            train_data = train_data[indices, :]
            dates['train'] = dates['train'][indices]
            if features:
                train_input = [train_data, features['train'][indices, :]]
            indices = list(range(val_input.shape[0]))
            indices = drop_remainder(indices, batch_size, last=False)
            val_input = val_input[indices, :]
            val_data = val_data[indices, :]
            dates['val'] = dates['val'][indices]
            if features:
                val_input = [val_data, features['val'][indices, :]]
        else:
            if features:
                train_input = [train_data, features['train']]
                val_input = [val_data, features['val']]

        print(model.summary())
        # Train
        LOGGER.info('Start training')
        print(input_dim)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Prepare the metrics.
        train_metric = [tf.keras.metrics.MeanSquaredError(name='mse'),
                        # tfa.metrics.r_square.RSquare(),  # r_square
                        tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        val_metric = [tf.keras.metrics.MeanSquaredError(name='mse'),
                      # tfa.metrics.r_square.RSquare(),  # r_square
                      tf.keras.metrics.RootMeanSquaredError(name='rmse')]

        if loss == 'mse_with_covariance_penalty':
            pass
        elif loss == 'mse':
            loss_fn = tf.keras.losses.MeanSquaredError(name='mse_loss')
        elif loss == 'weighted_mse':
            loss_fn = weighted_mse

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


        # test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

        # history = model.fit(train_dataset, # train_input, train_data,
        #                     epochs=epochs,
        #                     batch_size=batch_size,
        #                     validation_data=val_dataset, # (val_input, val_data),
        #                     validation_batch_size=batch_size,
        #                     callbacks=cv_callbacks,
        #                     shuffle=False,
        #                     verbose=1)

        @tf.function
        def train_step(x, y, *args, **kwargs):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                loss_value = loss_fn(y, pred, *args, **kwargs)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if isinstance(train_metric, list):
                [m.update_state(y, pred) for m in train_metric]
            else:
                train_metric.update_state(y, pred)

            return loss_value


        @tf.function
        def test_step(x, y, *args, **kwargs):
            pred = model(x, training=False)
            loss_value = loss_fn(y, pred, *args, **kwargs)

            if isinstance(val_metric, list):
                [m.update_state(y, pred) for m in val_metric]
            else:
                val_metric.update_state(y, pred)

            return loss_value


        early_stopping = callbacks.get('EarlyStopping')
        if early_stopping is not None:
            restore_best_weights = early_stopping['restore_best_weights']
        else:
            restore_best_weights = False
        history = {'loss': [], 'mse': [], 'rmse': [], 'val_loss': [], 'val_mse': [], 'val_rmse': []}
        best_weights = None
        stop_training = False
        for epoch in range(epochs):
            # Iterate over the batches of the dataset.
            batch_loss = []
            if loss == 'weighted_mse':
                if n_features:
                    for step, (x_batch_train_0, x_batch_train_1, y_batch_train, weights_batch) in enumerate(
                            train_dataset):
                        loss_value = train_step([x_batch_train_0, x_batch_train_1], y_batch_train, weights_batch)
                        batch_loss.append(float(loss_value))
                        # Log every 200 batches.
                        # if step % 200 == 0 and step > 0:
                        #     print(
                        #         "Training loss (for one batch) at step %d: %.4f"
                        #         % (step, float(loss_value))
                        #     )
                        #     print("Seen so far: %d samples" % ((step + 1) * batch_size))
                else:
                    for step, (x_batch_train, y_batch_train, weights_batch) in enumerate(train_dataset):
                        loss_value = train_step(x_batch_train, y_batch_train, weights_batch)
                        batch_loss.append(float(loss_value))
                        # Log every 200 batches.
                        # if step % 200 == 0 and step > 0:
                        #     print(
                        #         "Training loss (for one batch) at step %d: %.4f"
                        #         % (step, float(loss_value))
                        #     )
                        #     print("Seen so far: %d samples" % ((step + 1) * batch_size))
            else:
                if n_features:
                    for step, (x_batch_train_0, x_batch_train_1, y_batch_train) in enumerate(train_dataset):
                        loss_value = train_step([x_batch_train_0, x_batch_train_1], y_batch_train)
                        batch_loss.append(float(loss_value))
                        # Log every 200 batches.
                        # if step % 200 == 0 and step > 0:
                        #     print(
                        #         "Training loss (for one batch) at step %d: %.4f"
                        #         % (step, float(loss_value))
                        #     )
                        #     print("Seen so far: %d samples" % ((step + 1) * batch_size))
                else:
                    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                        loss_value = train_step(x_batch_train, y_batch_train)
                        batch_loss.append(float(loss_value))
                        # Log every 200 batches.
                        # if step % 200 == 0 and step > 0:
                        #     print(
                        #         "Training loss (for one batch) at step %d: %.4f"
                        #         % (step, float(loss_value))
                        #     )
                        #     print("Seen so far: %d samples" % ((step + 1) * batch_size))
            if restore_best_weights and best_weights is None:
                best_epoch = epoch
                best_weights = model.get_weights()

            # Compute loss over epoch
            epoch_loss = np.mean(batch_loss)
            history['loss'].append(epoch_loss)

            # Run a validation loop at the end of each epoch.
            batch_loss = []
            if n_features:
                for x_batch_val_0, x_batch_val_1, y_batch_val in val_dataset:
                    val_loss_value = test_step([x_batch_val_0, x_batch_val_1], y_batch_val)
                    batch_loss.append(float(val_loss_value))
            else:
                for x_batch_val, y_batch_val in val_dataset:
                    val_loss_value = test_step(x_batch_val, y_batch_val)
                    batch_loss.append(float(val_loss_value))
            # Compute loss over epoch
            val_epoch_loss = np.mean(batch_loss)

            # Early stopping
            if early_stopping:
                if epoch >= early_stopping['patience']:
                    if val_epoch_loss <= np.min(history['val_loss']):
                        LOGGER.info("Model has improved from {0:8.4f} to {1:8.4f}".format(np.min(history['val_loss']),
                                                                                          val_epoch_loss))
                        best_epoch = epoch
                        best_loss = val_epoch_loss
                        if restore_best_weights:
                            LOGGER.info(
                                f"Restoring best weights from epoch {best_epoch} with loss {np.round(best_loss, 4)}")
                            best_weights = model.get_weights()
                            if save:
                                model.save(f"{save_dir}/{cv}/best_model_stopped.h5")
                    else:
                        LOGGER.info(
                            "Model has not improved from {0:8.4f}".format(np.min(history['val_loss'])))

                    stop_training = EarlyStopping(history[early_stopping['monitor']],
                                                  min_delta=early_stopping['min_delta'],
                                                  patience=early_stopping['patience'],
                                                  mode=early_stopping['mode'])

            history['val_loss'].append(val_epoch_loss)
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
                f"Epoch {epoch}: loss = {np.round(history['loss'][-1], 4)} - mse = {np.round(history['mse'][-1], 4)} - rmse = {np.round(history['rmse'][-1], 4)} "
                f"- val_loss = {np.round(history['val_loss'][-1], 4)} - val_mse = {np.round(history['val_mse'][-1], 4)} - val_rmse = {np.round(history['val_rmse'][-1], 4)}")

            if save:
                epoch_val_rmse = history['val_rmse'][-1]
                if epoch_val_rmse == np.min(history['val_rmse']):
                    LOGGER.info('Saving best model')
                    model.save(f"{save_dir}/{cv}/best_model.h5")

            if stop_training:
                LOGGER.info(f"Stopping training at epoch {epoch}")
                break

        if save:
            model.save(f"{save_dir}/{cv}/model.h5")

            # Tensorboard Embedding visualization
            # Set up a logs directory, so Tensorboard knows where to look for files.
            log_dir = f"{save_dir}/{cv}/tensorboard/"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Save Labels separately on a line-by-line manner.
            with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
                for asset in assets:
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

        fix, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(history['loss'], label='loss')
        axs[0].plot(history['val_loss'], label='val loss')
        axs[0].legend()
        axs[1].plot(history['mse'], label='mse')
        axs[1].plot(history['val_mse'], label='val_mse')
        axs[1].legend()
        axs[2].plot(history['rmse'], label='rmse')
        axs[2].plot(history['val_rmse'], label='val_rmse')
        axs[2].legend()

        if save:
            plt.savefig(f"{save_dir}/{cv}/history.png")
        plt.show()

        if save:
            LOGGER.info(f"Loading weights from {save_dir}/{cv}/best_model.h5")
            model.load_weights(f"{save_dir}/{cv}/best_model.h5")

        # Evaluate
        # model.evaluate(train_input, train_data)
        # model.evaluate(val_input, val_data)

        # Results
        if n_features:
            train_features = encoder.predict(train_input[0])
        else:
            train_features = encoder.predict(train_input)
        train_features = pd.DataFrame(train_features, index=dates['train'])

        val_prediction = model.predict(val_input)
        val_prediction = scaler.inverse_transform(val_prediction)
        val_prediction = pd.DataFrame(val_prediction, columns=assets, index=dates['val'])

        if n_features:
            val_features = encoder.predict(val_input[0])
        else:
            val_features = encoder.predict(val_input)
        val_features = pd.DataFrame(val_features, index=dates['val'])


        encoder_layer = get_layer_by_name(name='encoder', model=model)
        encoder_weights = encoder_layer.get_weights()
        # decoder_weights = model.layers[2].get_weights()
        encoder_weights = pd.DataFrame(encoder_weights[0], index=assets)
        LOGGER.info(f"Encoder weights:\n{encoder_weights}")

        if features:
            test_input = [test_data, features['test']]
        test_prediction = model.predict(test_input)
        test_prediction = scaler.inverse_transform(test_prediction)
        test_prediction = pd.DataFrame(test_prediction, columns=assets, index=dates['test'])

        if n_features:
            test_features = encoder.predict(test_input[0])
        else:
            test_features = encoder.predict(test_input)
        test_features = pd.DataFrame(test_features, index=dates['test'])

        # train_cluster_portfolio = encoder.predict(train_data)
        # train_cluster_portfolio = pd.DataFrame(train_cluster_portfolio, index=dates['train'])
        train_cluster_portfolio = pd.DataFrame(np.dot(train_data, encoder_weights / encoder_weights.sum()),
                                               index=dates['train'])

        # val_cluster_portfolio = encoder.predict(val_data)
        # val_cluster_portfolio = pd.DataFrame(val_cluster_portfolio, index=dates['val'])
        val_cluster_portfolio = pd.DataFrame(np.dot(val_data, encoder_weights / encoder_weights.sum()),
                                             index=dates['val'])

        coskewness = PositiveSkewnessConstraint(encoding_dim, weightage=1, norm='1', normalize=False)
        LOGGER.info(
            f'Coskewness on validation set: {coskewness(tf.constant(val_cluster_portfolio.values, dtype=tf.float32)).numpy()}')

        # test_cluster_portfolio = encoder.predict(test_data)
        # test_cluster_portfolio = pd.DataFrame(test_cluster_portfolio, index=dates['test'])
        test_cluster_portfolio = pd.DataFrame(np.dot(test_data, encoder_weights / encoder_weights.sum()),
                                              index=dates['test'])

        LOGGER.info(
            f'Coskewness on test set: {coskewness(tf.constant(test_cluster_portfolio.values, dtype=tf.float32)).numpy()}')
        train_data = scaler.inverse_transform(train_data)
        train_data = pd.DataFrame(train_data, index=dates['train'], columns=assets)
        val_data = scaler.inverse_transform(val_data)
        val_data = pd.DataFrame(val_data, index=dates['val'], columns=assets)
        test_data = scaler.inverse_transform(test_data)
        test_data = pd.DataFrame(test_data, index=dates['test'], columns=assets)

        if shuffle_columns:
            LOGGER.info('Reorder results with base asset order')
            val_prediction = val_prediction.loc[:, base_asset_order]
            train_data = train_data.loc[:, base_asset_order]
            val_data = val_data.loc[:, base_asset_order]
            test_data = test_data.loc[:, base_asset_order]
            test_prediction = test_prediction.loc[:, base_asset_order]
            encoder_weights = encoder_weights.loc[base_asset_order, :]

        if kernel_constraint is not None:
            vmax = 1.
            vmin = 0.
        else:
            vmax = None
            vmin = None

        if save:
            heat_map(encoder_weights, show=True, save=save, save_dir=f"{save_dir}/{cv}", vmax=vmax, vmin=vmin)
        else:
            heat_map(encoder_weights, show=True, vmax=vmax, vmin=vmin)

        cluster_portfolio = {
            'train': train_cluster_portfolio,
            'val': val_cluster_portfolio,
            'test': test_cluster_portfolio
        }

        LOGGER.info(f"Encoder feature correlation:\n{np.corrcoef(val_cluster_portfolio.T)}")
        LOGGER.info(f"Unit norm constraint:\n{encoder.layers[-1].kernel.numpy().sum(0)}")

        if save:
            train_data.to_pickle(f"{save_dir}/{cv}/train_returns.p")
            val_data.to_pickle(f"{save_dir}/{cv}/val_returns.p")
            test_data.to_pickle(f"{save_dir}/{cv}/test_returns.p")
            val_prediction.to_pickle(f"{save_dir}/{cv}/val_prediction.p")
            test_prediction.to_pickle(f"{save_dir}/{cv}/test_prediction.p")
            encoder_weights.to_pickle(f"{save_dir}/{cv}/encoder_weights.p")
            train_features.to_pickle(f"{save_dir}/{cv}/train_features.p")
            val_features.to_pickle(f"{save_dir}/{cv}/val_features.p")
            test_features.to_pickle(f"{save_dir}/{cv}/test_features.p")
            # encoding_pca.to_pickle(f"{save_dir}/{cv}/encoding_pca.p")
            pickle.dump(cluster_portfolio, open(f"{save_dir}/{cv}/cluster_portfolio.p", "wb"))
            # pickle.dump(pca_cluster_portfolio, open(f"{save_dir}/{cv}/pca_cluster_portfolio.p", "wb"))

            scaler_func['attributes'] = scaler.__dict__
            pickle.dump(scaler_func, open(f"{save_dir}/{cv}/scaler.p", "wb"))

    if save:
        heat_map_cluster(save_dir, show=True, save=save, vmax=1., vmin=0.)

import tensorflow as tf
import numpy as np
from dl_portfolio.metrics import volatility, portfolio_returns, sharpe_ratio, \
    penalized_volatility_returns, average_return, cum_return, sortino_ratio
from dl_portfolio.logger import LOGGER
import matplotlib.pyplot as plt
from typing import List, Union
from dl_portfolio.evaluate import plot_train_history
import pandas as pd
import os


def set_learning_rate(epoch, lr_scheduler):
    learning_rate_strategy = 'step'
    if learning_rate_strategy == 'step':
        max_step = -1
        learning_rate = 0.0
        for step, lr in lr_scheduler.items():
            if epoch >= step > max_step:
                learning_rate = lr
                max_step = step
        if max_step == -1:
            raise ValueError('cannot find learning rate for step %d' % epoch)
    elif learning_rate_strategy == 'cosine':
        raise NotImplementedError()
    return learning_rate


def train(train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, model, model_type: str, loss_name: str,
          optimizer: tf.keras.optimizers, learning_rate: Union[float, dict], n_epochs: int, assets: List[str],
          trading_fee: float, log_every: int, plot_every: int, no_cash: bool = False, train_returns=None,
          test_returns=None, **kwargs):
    save = kwargs.get('save', False)
    n_assets = len(assets)
    if no_cash:
        n_pairs = n_assets
    else:
        n_pairs = n_assets - 1

    if save:
        log_dir = kwargs.get('log_dir')
        assert log_dir is not None

    if loss_name == 'sharpe_ratio':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32),
                       'annual_period': tf.constant(kwargs.get('annual_period', 1), dtype=tf.float32),
                       'epsilon': tf.constant(kwargs.get('epsilon', 1e-6), dtype=tf.float32),
                       }
        loss_function = sharpe_ratio
    elif loss_name == 'sortino_ratio':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32),
                       'annual_period': tf.constant(kwargs.get('annual_period', 1), dtype=tf.float32),
                       }
        loss_function = sortino_ratio
    elif loss_name == 'penalized_volatility_returns':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32),
                       'alpha': tf.constant(kwargs.get('alpha', 1), dtype=tf.float32)}
        loss_function = penalized_volatility_returns
    elif loss_name == 'average_return':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32)}
        loss_function = average_return
    elif loss_name == 'cum_return':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32)}
        loss_function = cum_return
    elif loss_name == 'volatility':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32)}
        loss_function = volatility
    else:
        raise NotImplementedError()

    train_history = {'loss': [], 'avg_ret': [], 'cum_ret': [], 'total_loss': []}
    test_history = {'loss': [], 'avg_ret': [], 'cum_ret': [], 'total_loss': []}

    for epoch in range(n_epochs):
        train_epoch_stats = {'loss': tf.keras.metrics.Mean(), 'avg_ret': [],
                             'volatility': [],
                             'cum_ret': [], 'fee': tf.keras.metrics.Mean()}
        test_epoch_stats = {'loss': tf.keras.metrics.Mean(), 'avg_ret': [],
                            'volatility': [],
                            'cum_ret': []}
        # Get learning rate
        optimizer.lr = set_learning_rate(epoch, learning_rate)
        if epoch in learning_rate.keys():
            LOGGER.info(f'Setting learning rate to: {optimizer.lr.numpy()}')

        # Training loop
        epoch_actions = []
        epoch_losses = []
        epoch_returns = []
        epoch_grads = []
        counter = 0

        for indices, features, returns in train_dataset:
            if counter == 0:
                if no_cash:
                    initial_position = tf.Variable([[1 / n_assets] * n_assets], dtype=tf.float32)
                else:
                    initial_position = tf.Variable(np.array([[0] * (n_assets - 1)]), dtype=tf.float32)
            else:
                if no_cash:
                    initial_position = actions[-1:, :]
                else:
                    initial_position = actions[-1:, :-1]

            if model_type == "EIIE":
                features = tf.transpose(features, [0, 3, 1, 2])
            elif model_type in ['asset_independent_model', 'stacked_asset_model']:
                features = [features[:, :, :, i] for i in range(n_pairs)]

            # Optimize the model
            with tf.GradientTape() as tape:
                actions = model(features, training=True)
                port_return_no_fee, port_return = portfolio_returns(actions, returns, initial_position,
                                                                    trading_fee=trading_fee,
                                                                    cash_bias=not no_cash)
                # print(np.sum(fee < 0))
                # print(port_return_no_fee - port_return)
                # if loss_name in ['sharpe_ratio', 'penalized_volatility_returns']:
                #     benchmark = tf.reduce_mean(returns[:, :-1], axis=-1)
                #     loss_params['benchmark'] = benchmark
                loss_value = loss_function(port_return, **loss_params)
            # if np.isnan(loss_value.numpy()):
            #     if len(epoch_actions) > 0:
            #         print(epoch_actions[-1])
            #         print(epoch_grads[-1])
            #         print(epoch_losses[-1])
            #     else:
            #         print(actions)
            #         print(loss_value)
            #
            #     nans = np.isnan(features.numpy()).tolist()
            #     nans = [i for sub in nans for i in sub]
            #     LOGGER.debug(f'Features is nan: {any(nans)}')
            #     LOGGER.debug(f'ACTION: {actions}')
            #     LOGGER.debug(f'PORT RETURN: {port_return}')
            #     raise ValueError('Tf returned NaN')

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            fee = tf.reduce_mean(port_return_no_fee - port_return)
            epoch_returns.append(port_return.numpy())
            train_epoch_stats['loss'].update_state(loss_value)  # Add current batch loss
            train_epoch_stats['fee'].update_state(fee)

            epoch_losses.append(loss_value.numpy())
            epoch_grads.append(grads)

            # update actions
            # print(portfolio_weights[indices+1])
            epoch_actions.append(actions.numpy())

            counter += 1

        epoch_actions = np.array([i for sub in epoch_actions for i in sub])
        epoch_returns = np.array([i for sub in epoch_returns for i in sub])
        train_avg_ret = np.mean(epoch_returns)
        train_cum_ret = np.exp(np.cumsum(epoch_returns)[-1]) - 1
        train_volatility = np.std(epoch_returns)
        train_total_loss = loss_function(tf.Variable(epoch_returns), **loss_params)

        # Inference
        counter = 0
        test_actions = []
        test_returns = []
        for indices, features, returns in test_dataset:
            if model_type == "EIIE":
                features = tf.transpose(features, [0, 3, 1, 2])
            elif model_type in ['asset_independent_model', 'stacked_asset_model']:
                features = [features[:, :, :, i] for i in range(n_pairs)]

            if counter == 0:
                initial_position = epoch_actions[-1:, :]
            else:
                initial_position = actions[-1:, :]
            if not no_cash:
                initial_position = initial_position[-1:, :-1]

            actions = model(features, training=False)
            # update actions
            test_actions.append(actions.numpy())
            port_return_no_fee, port_return = portfolio_returns(actions, returns, initial_position,
                                                                trading_fee=trading_fee,
                                                                cash_bias=not no_cash)
            test_returns.append(port_return.numpy())
            # if loss_name in ['sharpe_ratio', 'penalized_volatility_returns']:
            #     benchmark = tf.reduce_mean(returns[:, :-1], axis=-1)
            #     loss_params['benchmark'] = benchmark
            loss_value = loss_function(port_return, **loss_params)
            # Track progress
            test_epoch_stats['loss'].update_state(loss_value)
            counter += 1

        test_actions = np.array([i for sub in test_actions for i in sub])
        test_returns = np.array([i for sub in test_returns for i in sub])
        test_avg_ret = np.mean(test_returns)
        test_cum_ret = np.exp(np.cumsum(test_returns)[-1]) - 1
        test_volatility = np.std(test_returns)
        test_total_loss = loss_function(tf.Variable(test_returns), **loss_params)

        # Update history
        train_history['total_loss'].append(train_total_loss)
        train_history['loss'].append(train_epoch_stats['loss'].result())
        train_history['cum_ret'].append(train_cum_ret)
        train_history['avg_ret'].append(train_avg_ret)
        test_history['total_loss'].append(test_total_loss)
        test_history['loss'].append(test_epoch_stats['loss'].result())
        test_history['avg_ret'].append(test_avg_ret)
        test_history['cum_ret'].append(test_cum_ret)

        if epoch % log_every == 0:
            LOGGER.info(
                "Epoch {:03d}: Loss: {:.6f}, Mean Loss: {:.6f}, Returns: {:.6f}, Vol: {:.6f}, Cum Returns: {:.6f}, Trading fee: {:.6f}, Test Loss: {:.6f}, Mean Test Loss: {:.6f}, Test Returns: {:.6f}, Test Vol: {:.6f}, Test Cum Returns: {:.6f}".format(
                    epoch,
                    train_total_loss,
                    train_epoch_stats[
                        'loss'].result(),
                    train_avg_ret,
                    train_volatility,
                    train_cum_ret,
                    train_epoch_stats[
                        'fee'].result(),
                    test_total_loss,
                    test_epoch_stats[
                        'loss'].result(),
                    test_avg_ret,
                    test_volatility,
                    test_cum_ret
                )
            )

        if epoch % plot_every == 0:
            plt.plot(epoch_actions)
            plt.legend(assets)
            plt.title(f'prediction at epoch {epoch}')
            plt.show()

            """if no_cash:
                initial_position = tf.Variable([[1 / n_assets] * n_assets], dtype=tf.float32)
            else:
                initial_position = tf.Variable(np.array([[0] * (n_assets - 1)]), dtype=tf.float32)
            strat_perf_no_fee, strat_perf = portfolio_returns(tf.Variable(epoch_actions),
                                                              tf.Variable(train_returns.values),
                                                              initial_position, trading_fee=trading_fee,
                                                              cash_bias=not no_cash)

            strat_perf = strat_perf.numpy()
            strat_perf = pd.Series(strat_perf, index=train_returns.index)
            strat_perf_no_fee = strat_perf_no_fee.numpy()
            strat_perf_no_fee = pd.Series(strat_perf_no_fee, index=train_returns.index)

            if not no_cash:
                eq_port = train_returns.drop('cash', 1)
            else:
                eq_port = train_returns.copy()
            eq_port = eq_port.mean(1)
            plt.plot((eq_port + 1).cumprod(), label='equally weighted')
            plt.plot((strat_perf + 1).cumprod(), label='strategy')
            # plt.plot((strat_perf_no_fee + 1).cumprod(), label='strategy without fee')
            plt.legend()
            plt.title(f'Train perf at epoch {epoch}')
            plt.show()"""

            if not no_cash:
                eq_port = train_returns.drop('cash', 1)
            else:
                eq_port = train_returns.copy()
            eq_port = eq_port.mean(1)

            plt.plot(np.exp(np.cumsum(eq_port)) - 1, label='equally weighted')
            plt.plot(np.exp(np.cumsum(pd.Series(epoch_returns, index=eq_port.index))) - 1, label='strategy')
            # plt.plot((strat_perf_no_fee + 1).cumprod(), label='strategy without fee')
            plt.legend()
            plt.title(f'Train perf at epoch {epoch}')
            plt.show()

            plot_train_history(train_history, test_history, show=True)

        if save:
            if test_history['total_loss'][-1] == np.min(test_history['total_loss']):
                LOGGER.info(f"Saving model at epoch {epoch}: {os.path.join(log_dir, f'model_e_{epoch}.ckpt')}")
                model.save_weights(os.path.join(log_dir, f'model_e_{epoch}.ckpt'))

    return model, train_history, test_history


def pretrain(train_dataset: tf.data.Dataset, model, model_type: str, loss_name: str,
             optimizer: tf.keras.optimizers, learning_rate: Union[float, dict], n_epochs: int, assets: List[str],
             trading_fee: float, log_every: int, plot_every: int, no_cash: bool = False, train_returns=None, **kwargs):
    save = kwargs.get('save', False)
    feed_prev_weights = kwargs.get('feed_prev_weights', False)

    n_assets = len(assets)
    if no_cash:
        n_pairs = n_assets
    else:
        n_pairs = n_assets - 1

    if feed_prev_weights:
        n_samples = sum([b.shape[0] for b, _, _ in train_dataset])
        portfolio_weights = np.zeros((n_samples + 1, n_pairs))
        portfolio_weights[:] = 1 / n_pairs
    if save:
        log_dir = kwargs.get('log_dir')
        assert log_dir is not None

    if loss_name == 'sharpe_ratio':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32),
                       'annual_period': tf.constant(kwargs.get('annual_period', 1), dtype=tf.float32),
                       'epsilon': tf.constant(kwargs.get('epsilon', 1e-6), dtype=tf.float32),
                       }
        loss_function = sharpe_ratio
    elif loss_name == 'sortino_ratio':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32),
                       'annual_period': tf.constant(kwargs.get('annual_period', 1), dtype=tf.float32),
                       }
        loss_function = sortino_ratio
    elif loss_name == 'penalized_volatility_returns':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32),
                       'alpha': tf.constant(kwargs.get('alpha', 1), dtype=tf.float32)}
        loss_function = penalized_volatility_returns
    elif loss_name == 'average_return':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32)}
        loss_function = average_return
    elif loss_name == 'cum_return':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32)}
        loss_function = cum_return
    elif loss_name == 'volatility':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32)}
        loss_function = volatility
    else:
        raise NotImplementedError()

    train_history = {'loss': [], 'avg_ret': [], 'cum_ret': [], 'total_loss': []}
    test_history = {'loss': [], 'avg_ret': [], 'cum_ret': [], 'total_loss': []}

    for epoch in range(n_epochs):
        train_epoch_stats = {'loss': tf.keras.metrics.Mean(), 'avg_ret': [],
                             'volatility': [],
                             'cum_ret': [], 'fee': tf.keras.metrics.Mean()}
        # Get learning rate
        optimizer.lr = set_learning_rate(epoch, learning_rate)
        if epoch in learning_rate.keys():
            LOGGER.info(f'Setting learning rate to: {optimizer.lr.numpy()}')

        # Training loop
        epoch_actions = []
        epoch_losses = []
        epoch_returns = []
        epoch_grads = []
        counter = 0
        for indices, features, returns in train_dataset:
            indices = indices.numpy()
            prev_weights = portfolio_weights[indices, :]
            if counter == 0:
                if no_cash:
                    initial_position = tf.Variable([[1 / n_assets] * n_assets], dtype=tf.float32)
                else:
                    initial_position = tf.Variable(np.array([[0] * (n_assets - 1)]), dtype=tf.float32)
            else:
                if no_cash:
                    initial_position = actions[-1:, :]
                else:
                    initial_position = actions[-1:, :-1]

            if model_type == "EIIE":
                features = tf.transpose(features, [0, 3, 1, 2])
            elif model_type in ['asset_independent_model', 'stacked_asset_model']:
                features = [features[:, :, :, i] for i in range(n_pairs)]
                if feed_prev_weights:
                    features.append(prev_weights)

            # Optimize the model
            with tf.GradientTape() as tape:
                actions = model(features, training=True)
                port_return_no_fee, port_return = portfolio_returns(actions, returns, initial_position,
                                                                    trading_fee=trading_fee,
                                                                    cash_bias=not no_cash)
                loss_value = loss_function(port_return, **loss_params)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            fee = tf.reduce_mean(port_return_no_fee - port_return)
            epoch_returns.append(port_return.numpy())
            train_epoch_stats['loss'].update_state(loss_value)  # Add current batch loss
            train_epoch_stats['fee'].update_state(fee)

            epoch_losses.append(loss_value.numpy())
            epoch_grads.append(grads)

            # update actions
            portfolio_weights[indices + 1] = actions.numpy()
            # print(portfolio_weights[indices+1])
            epoch_actions.append(actions.numpy())

            counter += 1

        epoch_actions = np.array([i for sub in epoch_actions for i in sub])
        epoch_returns = np.array([i for sub in epoch_returns for i in sub])
        train_avg_ret = np.mean(epoch_returns)
        train_cum_ret = np.exp(np.cumsum(epoch_returns)[-1]) - 1
        train_volatility = np.std(epoch_returns)
        train_total_loss = loss_function(tf.Variable(epoch_returns), **loss_params)

        # Update history
        train_history['total_loss'].append(train_total_loss)
        train_history['loss'].append(train_epoch_stats['loss'].result())
        train_history['cum_ret'].append(train_cum_ret)
        train_history['avg_ret'].append(train_avg_ret)
        if epoch % log_every == 0:
            LOGGER.info(
                "Epoch {:03d}: Loss: {:.6f}, Mean Loss: {:.6f}, Returns: {:.6f}, Vol: {:.6f}, Cum Returns: {:.6f}, Trading fee: {:.6f}".format(
                    epoch,
                    train_total_loss,
                    train_epoch_stats[
                        'loss'].result(),
                    train_avg_ret,
                    train_volatility,
                    train_cum_ret,
                    train_epoch_stats[
                        'fee'].result()
                )
            )

        if epoch % plot_every == 0:
            plt.plot(epoch_actions)
            plt.legend(assets)
            plt.title(f'prediction at epoch {epoch}')
            plt.show()

            if not no_cash:
                eq_port = train_returns.drop('cash', 1)
            else:
                eq_port = train_returns.copy()
            eq_port = eq_port.mean(1)

            plt.plot(np.exp(np.cumsum(eq_port)) - 1, label='equally weighted')
            plt.plot(np.exp(np.cumsum(pd.Series(epoch_returns, index=eq_port.index))) - 1, label='strategy')
            # plt.plot((strat_perf_no_fee + 1).cumprod(), label='strategy without fee')
            plt.legend()
            plt.title(f'Train perf at epoch {epoch}')
            plt.show()

            plot_train_history(train_history, test_history, show=True)

        if save:
            model.save_weights(os.path.join(log_dir, f'model_e_{epoch}.ckpt'))

    return model, train_history, test_history, portfolio_weights


def online_training(train_examples: np.array, train_returns: np.array, test_examples: np.array, test_returns: np.array,
                    portfolio_weights: np.array, model, model_type: str, loss_name: str, optimizer: tf.keras.optimizers,
                    learning_rate: Union[float, dict], n_epochs: int, batch_size, assets: List[str],
                    trading_fee: float, log_every: int, plot_every: int, no_cash: bool = False,
                    **kwargs):
    save = kwargs.get('save', False)
    feed_prev_weights = kwargs.get('feed_prev_weights', False)

    n_assets = len(assets)
    if no_cash:
        n_pairs = n_assets
    else:
        n_pairs = n_assets - 1

    if save:
        log_dir = kwargs.get('log_dir')
        assert log_dir is not None

    if loss_name == 'sharpe_ratio':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32),
                       'annual_period': tf.constant(kwargs.get('annual_period', 1), dtype=tf.float32),
                       'epsilon': tf.constant(kwargs.get('epsilon', 0.001), dtype=tf.float32),
                       }
        loss_function = sharpe_ratio
    elif loss_name == 'penalized_volatility_returns':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32),
                       'alpha': tf.constant(kwargs.get('alpha', 1), dtype=tf.float32)}
        loss_function = penalized_volatility_returns
    elif loss_name == 'average_return':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32)}
        loss_function = average_return
    elif loss_name == 'cum_return':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32)}
        loss_function = cum_return
    elif loss_name == 'volatility':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32)}
        loss_function = volatility
    else:
        raise NotImplementedError()

    train_history = {'loss': [], 'avg_ret': [], 'cum_ret': [], 'total_loss': []}
    test_history = {'loss': [], 'avg_ret': [], 'cum_ret': [], 'total_loss': []}

    test_examples = np.transpose(test_examples, (1, 2, 3, 0))
    n_test = len(test_examples)
    assert n_test == len(test_returns)
    strat_return = []
    test_prediction = []
    for step in range(n_test):
        if step == 0:
            train_features = np.transpose(train_examples, (1, 2, 3, 0))
        else:
            train_features = np.concatenate([train_features[1:, :], np.expand_dims(test_example, 0)], 0)
            train_returns = np.concatenate([train_returns[1:, :], np.expand_dims(test_return, 0)], 0)
            portfolio_weights = np.concatenate([portfolio_weights[1:, :], new_weights], 0)
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (list(range(len(train_features))), train_features, train_returns))
            train_dataset = train_dataset.batch(batch_size, drop_remainder=False)

            for epoch in range(n_epochs):
                train_epoch_stats = {'loss': tf.keras.metrics.Mean(), 'avg_ret': [],
                                     'volatility': [],
                                     'cum_ret': [], 'fee': tf.keras.metrics.Mean()}
                # Get learning rate
                optimizer.lr = set_learning_rate(epoch, learning_rate)
                if epoch in learning_rate.keys():
                    LOGGER.info(f'Setting learning rate to: {optimizer.lr.numpy()}')

                # Training loop
                epoch_actions = []
                epoch_losses = []
                epoch_returns = []
                epoch_grads = []
                counter = 0
                for indices, features, returns in train_dataset:
                    indices = indices.numpy()
                    prev_weights = portfolio_weights[indices, :]
                    if counter == 0:
                        if no_cash:
                            initial_position = tf.Variable([[1 / n_assets] * n_assets], dtype=tf.float32)
                        else:
                            initial_position = tf.Variable(np.array([[0] * (n_assets - 1)]), dtype=tf.float32)
                    else:
                        if no_cash:
                            initial_position = actions[-1:, :]
                        else:
                            initial_position = actions[-1:, :-1]

                    if model_type == "EIIE":
                        features = tf.transpose(features, [0, 3, 1, 2])
                    elif model_type in ['asset_independent_model', 'stacked_asset_model']:
                        features = [features[:, :, :, i] for i in range(n_pairs)]
                        if feed_prev_weights:
                            features.append(prev_weights)

                    # Optimize the model
                    with tf.GradientTape() as tape:
                        actions = model(features, training=True)
                        port_return_no_fee, port_return = portfolio_returns(actions, returns, initial_position,
                                                                            trading_fee=trading_fee,
                                                                            cash_bias=not no_cash)
                        loss_value = loss_function(port_return, **loss_params)

                    grads = tape.gradient(loss_value, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    # Track progress
                    fee = tf.reduce_mean(port_return_no_fee - port_return)
                    epoch_returns.append(port_return.numpy())
                    train_epoch_stats['loss'].update_state(loss_value)  # Add current batch loss
                    train_epoch_stats['fee'].update_state(fee)

                    epoch_losses.append(loss_value.numpy())
                    epoch_grads.append(grads)

                    # update actions
                    portfolio_weights[indices + 1] = actions.numpy()
                    # print(portfolio_weights[indices+1])
                    epoch_actions.append(actions.numpy())
                    counter += 1

                epoch_actions = np.array([i for sub in epoch_actions for i in sub])
                epoch_returns = np.array([i for sub in epoch_returns for i in sub])
                train_avg_ret = np.mean(epoch_returns)
                train_cum_ret = np.exp(np.cumsum(epoch_returns)[-1]) - 1
                train_volatility = np.std(epoch_returns)
                train_total_loss = loss_function(tf.Variable(epoch_returns), **loss_params)

                # Update history
                train_history['total_loss'].append(train_total_loss)
                train_history['loss'].append(train_epoch_stats['loss'].result())
                train_history['cum_ret'].append(train_cum_ret)
                train_history['avg_ret'].append(train_avg_ret)

        test_return = test_returns[step]
        test_example = test_examples[step]
        test_feature = [np.expand_dims(test_example[:, :, i], 0) for i in range(n_pairs)]
        test_feature.append(np.expand_dims(portfolio_weights[-1], 0))
        new_weights = model(test_feature, training=True)
        port_return_no_fee, port_return = portfolio_returns(new_weights, tf.Variable(np.expand_dims(test_return, 0),
                                                                                     dtype=tf.float32),
                                                            tf.Variable(np.expand_dims(portfolio_weights[-1], 0),
                                                                        dtype=tf.float32),
                                                            trading_fee=trading_fee,
                                                            cash_bias=not no_cash)
        strat_return.append(port_return.numpy())
        test_prediction.append(new_weights.numpy()[0])

        # Stats
        cum_ret = np.exp(np.cumsum(strat_return)[-1]) - 1
        avg_ret = np.exp(np.mean(strat_return)) - 1
        vol = np.exp(np.std(strat_return)) - 1

        if step % log_every == 0:
            LOGGER.info(
                "Step {:03d}: Returns: {:.6f}, Vol: {:.6f}, Cum Returns: {:.6f}".format(
                    step,
                    avg_ret,
                    vol,
                    cum_ret
                )
            )

        if save:
            model.save_weights(os.path.join(log_dir, f'model_e_{epoch}.ckpt'))

    test_prediction = np.array(test_prediction)

    return model, strat_return, test_prediction

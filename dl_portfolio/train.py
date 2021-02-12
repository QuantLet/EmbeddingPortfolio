import tensorflow as tf
import numpy as np
from dl_portfolio.metrics import portfolio_returns, sharpe_ratio, penalized_volatility_returns, average_return, \
    cum_return
from dl_portfolio.logger import LOGGER
import matplotlib.pyplot as plt
from typing import List, Union
from dl_portfolio.evaluate import plot_train_history
import pandas as pd


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
          trading_fee: float, log_every: int, plot_every: int, no_cash: bool = False, train_returns=None, **kwargs):
    n_assets = len(assets)
    if loss_name == 'sharpe_ratio':
        loss_params = {'benchmark': tf.constant(kwargs.get('benchmark', 0), dtype=tf.float32),
                       'annual_period': tf.constant(kwargs.get('annual_period', 1), dtype=tf.float32)}
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
    else:
        raise NotImplementedError()

    """if clip_value is not None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=momentum, clipvalue=clip_value)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=momentum)
    """
    train_history = {'loss': [], 'avg_ret': [], 'cum_ret': []}
    test_history = {'loss': [], 'avg_ret': [], 'cum_ret': []}

    for epoch in range(n_epochs):
        train_epoch_stats = {'loss': tf.keras.metrics.Mean(), 'avg_ret': tf.keras.metrics.Mean(),
                             'cum_ret': tf.keras.metrics.Sum()}
        test_epoch_stats = {'loss': tf.keras.metrics.Mean(), 'avg_ret': tf.keras.metrics.Mean(),
                            'cum_ret': tf.keras.metrics.Sum()}
        # Get learning rate
        optimizer.lr = set_learning_rate(epoch, learning_rate)
        if epoch in learning_rate.keys():
            LOGGER.info(f'Setting learning rate to: {optimizer.lr.numpy()}')

        # Training loop
        epoch_actions = []
        epoch_losses = []
        epoch_grads = []
        counter = 0
        for features, returns in train_dataset:
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
            elif model_type == 'asset_independent_model':
                features = [features[:, :, :, i] for i in range(n_assets)]

            # Optimize the model
            with tf.GradientTape() as tape:
                actions = model(features, training=True)

                port_return_no_fee, port_return = portfolio_returns(actions, returns, initial_position,
                                                                    trading_fee=trading_fee,
                                                                    cash_bias=not no_cash)
                # print(port_return_no_fee - port_return)
                # if loss_name in ['sharpe_ratio', 'penalized_volatility_returns']:
                #     benchmark = tf.reduce_mean(returns[:, :-1], axis=-1)
                #     loss_params['benchmark'] = benchmark
                loss_value = loss_function(port_return, **loss_params)

            if np.isnan(loss_value.numpy()):
                if len(epoch_actions) > 0:
                    print(epoch_actions[-1])
                    print(epoch_grads[-1])
                    print(epoch_losses[-1])
                else:
                    print(actions)
                    print(loss_value)

                nans = np.isnan(features.numpy()).tolist()
                nans = [i for sub in nans for i in sub]
                LOGGER.debug(f'Features is nan: {any(nans)}')
                LOGGER.debug(f'ACTION: {actions}')
                LOGGER.debug(f'PORT RETURN: {port_return}')
                raise ValueError('Tf returned NaN')

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            ret_t = tf.reduce_sum(returns * actions, axis=-1)
            avg_reg = tf.reduce_mean(ret_t)
            cum_ret = tf.reduce_sum(ret_t)
            train_epoch_stats['loss'].update_state(loss_value)  # Add current batch loss
            train_epoch_stats['cum_ret'].update_state(cum_ret)
            train_epoch_stats['avg_ret'].update_state(avg_reg)

            epoch_losses.append(loss_value.numpy())
            epoch_grads.append(grads)

            # update actions
            epoch_actions.append(actions.numpy())

            counter += 1

        epoch_actions = np.array([i for sub in epoch_actions for i in sub])

        # Inference
        counter = 0
        for features, returns in test_dataset:
            if model_type == "EIIE":
                features = tf.transpose(features, [0, 3, 1, 2])
            elif model_type == 'asset_independent_model':
                features = [features[:, :, :, i] for i in range(n_assets)]

            if counter == 0:
                initial_position = epoch_actions[-1:, :]
            else:
                initial_position = actions[-1:, :]
            if not no_cash:
                initial_position = initial_position[-1:, :-1]

            actions = model(features, training=False)
            port_return_no_fee, port_return = portfolio_returns(actions, returns, initial_position,
                                                                trading_fee=trading_fee,
                                                                cash_bias=not no_cash)

            # if loss_name in ['sharpe_ratio', 'penalized_volatility_returns']:
            #     benchmark = tf.reduce_mean(returns[:, :-1], axis=-1)
            #     loss_params['benchmark'] = benchmark
            loss_value = loss_function(port_return, **loss_params)

            # Track progress
            test_epoch_stats['loss'].update_state(loss_value)
            ret_t = tf.reduce_sum(returns * actions, axis=-1)
            avg_reg = tf.reduce_mean(ret_t)
            cum_ret = tf.reduce_sum(ret_t)
            test_epoch_stats['cum_ret'].update_state(cum_ret)
            test_epoch_stats['avg_ret'].update_state(avg_reg)

            counter += 1

        # Update history
        train_history['loss'].append(train_epoch_stats['loss'].result())
        train_history['avg_ret'].append(train_epoch_stats['avg_ret'].result())
        train_history['cum_ret'].append(train_epoch_stats['cum_ret'].result())
        test_history['loss'].append(test_epoch_stats['loss'].result())
        test_history['avg_ret'].append(test_epoch_stats['avg_ret'].result())
        test_history['cum_ret'].append(test_epoch_stats['cum_ret'].result())

        if epoch % log_every == 0:
            LOGGER.info(
                "Epoch {:03d}: Loss: {:.6f}, Returns: {:.6f}, Cum Returns: {:.6f}, Test Loss: {:.6f}, Test Returns: {:.6f}, Test Cum Returns: {:.6f}".format(
                    epoch,
                    train_epoch_stats[
                        'loss'].result(),
                    train_epoch_stats[
                        'avg_ret'].result(),
                    train_epoch_stats[
                        'cum_ret'].result(),
                    test_epoch_stats[
                        'loss'].result(),
                    test_epoch_stats[
                        'avg_ret'].result(),
                    test_epoch_stats[
                        'cum_ret'].result())
            )
            print()

        if epoch % plot_every == 0:
            plt.plot(epoch_actions)
            plt.legend(assets)
            plt.title(f'prediction at epoch {epoch}')
            plt.show()

            if no_cash:
                initial_position = tf.Variable([[1 / n_assets] * n_assets], dtype=tf.float32)
            else:
                initial_position = tf.Variable(np.array([[0] * (n_assets - 1)]), dtype=tf.float32)
            strat_perf_no_fee, strat_perf = portfolio_returns(tf.Variable(epoch_actions),
                                                              tf.Variable(train_returns.values),
                                                              initial_position, trading_fee=trading_fee,
                                                              cash_bias=not no_cash)

            strat_perf = strat_perf.numpy()
            strat_perf = pd.Series(strat_perf, index=train_returns.index)
            if not no_cash:
                eq_port = train_returns.drop('cash', 1)
            else:
                eq_port = train_returns.copy()
            eq_port = eq_port.mean(1)
            plt.plot((eq_port + 1).cumprod(), label='equally weighted')
            plt.plot((strat_perf + 1).cumprod(), label='strategy')
            plt.legend()
            plt.title(f'Train perf at epoch {epoch}')
            plt.show()

            plot_train_history(train_history, test_history, show=True)

    return model, train_history, test_history

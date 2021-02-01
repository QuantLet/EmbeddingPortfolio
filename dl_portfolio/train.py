import tensorflow as tf
import numpy as np
from dl_portfolio.metrics import portfolio_returns, sharpe_ratio
from dl_portfolio.logger import LOGGER
import matplotlib.pyplot as plt
from typing import List, Union
from dl_portfolio.evaluate import plot_train_history


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


def train(train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, model, learning_rate: Union[float, dict],
          momentum: float,
          n_epochs: int, assets: List[str], benchmark: float, annual_period: int, trading_fee: float,
          log_every: int, plot_every: int, no_cash: bool = False, clip_value: float = None, train_returns=None):
    n_assets = len(assets)

    if isinstance(learning_rate, dict):
        initial_lr = learning_rate[0]
    if clip_value is not None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=momentum, clipvalue=clip_value)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=momentum)

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
                initial_position = actions[-1:, :-1]

            # Optimize the model

            with tf.GradientTape() as tape:
                actions = model(features, training=True)
                port_return_no_fee, port_return = portfolio_returns(actions, returns, initial_position,
                                                                    trading_fee=trading_fee,
                                                                    cash_bias=not no_cash)
                # print(port_return_no_fee - port_return)
                loss_value = sharpe_ratio(port_return, benchmark=tf.constant(benchmark, dtype=tf.float32),
                                          annual_period=tf.constant(annual_period, dtype=tf.float32))

            if np.isnan(loss_value.numpy()):
                if len(epoch_actions) > 0:
                    print(epoch_actions[-1])
                    print(epoch_grads[-1])
                    print(epoch_losses[-1])
                else:
                    print(actions)
                    print(loss_value)
                exit()
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
            loss_value = sharpe_ratio(port_return, benchmark=tf.constant(benchmark, dtype=tf.float32),
                                      annual_period=tf.constant(annual_period, dtype=tf.float32))

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

        if epoch % plot_every == 0:
            plt.plot(epoch_actions)
            plt.legend(assets)
            plt.title(f'prediction at epoch {epoch}')
            plt.show()

            if no_cash:
                initial_position = tf.Variable([[1 / n_assets] * n_assets], dtype=tf.float32)
            else:
                initial_position = tf.Variable(np.array([[0] * (n_assets - 1)]), dtype=tf.float32)
            strat_perf_no_fee, strat_perf = portfolio_returns(tf.Variable(epoch_actions), tf.Variable(train_returns),
                                           initial_position, trading_fee=trading_fee, cash_bias=not no_cash)

            strat_perf = strat_perf.numpy()
            eq_port = train_returns[:len(epoch_actions), :].mean(1)
            plt.plot((eq_port + 1).cumprod(), label='equally weighted')
            plt.plot((strat_perf + 1).cumprod(), label='strategy')
            plt.legend()
            plt.title(f'Train perf at epoch {epoch}')
            plt.show()

            plot_train_history(train_history, test_history, show=True)

    return model, train_history, test_history

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from utils.config import LOGGER
from typing import Optional, Tuple
from first_etf_strat.model import MLP


def build_model(input_dim: Tuple, output_dim: int, batch_size: int, cash_bias: bool = True,
                n_hidden: int = 1, cash_initializer: tf.initializers = tf.ones_initializer(),
                dropout: Optional[float] = None, training: bool = False):
    assert n_hidden > 0
    if cash_bias:
        cash_weight = tf.Variable(initial_value=cash_initializer(shape=(batch_size, 1), dtype='float32'),
                                  trainable=True)

    input_ = tf.keras.layers.Input(input_dim, dtype=tf.float32)
    for i in range(n_hidden):
        if i == 0:
            hidden = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float32)(input_)
        else:
            hidden = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float32)(hidden)
        if dropout:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)

    if cash_bias:
        output = tf.keras.layers.Dense(output_dim - 1, activation='linear', dtype=tf.float32)(hidden)
        output = tf.keras.layers.Concatenate(axis=-1)([output, cash_weight])
        output = tf.keras.layers.Activation('softmax')(output)
    else:
        output = tf.keras.layers.Dense(output_dim, activation='softmax', dtype=tf.float32)(hidden)

    model = tf.keras.models.Model(input_, output)
    return model


def sharpe_ratio(model, x: np.ndarray, returns: np.ndarray, training: bool, benchmark: float = 0.0093):
    """

    :param model: tf model
    :param x: input features
    :param returns: corresponding returns for next period
    :param training: is training or inference
    :param benchmark: constant: risk free rate: 10Y US treasury bond 0.93% 31.12.2020
    :return:
    """
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    # take log maybe ??
    ret = tf.math.reduce_sum(returns * y_, axis=-1)
    sr = - tf.reduce_mean(ret - tf.constant(benchmark, dtype=tf.float32)) / (
            tf.math.reduce_std(ret - tf.constant(benchmark, dtype=tf.float32)) + 10e-12)
    # sr = tf.math.reduce_variance(ret - tf.constant(benchmark, dtype=tf.float32)) / (tf.math.square(tf.reduce_mean(ret - tf.constant(benchmark, dtype=tf.float32))) + 10e-12)
    return y_, sr


def build_delayed_window(data: np.ndarray, seq_len: int, return_3d: bool = False):
    """

    :param data: data
    :param seq_len: length of past window
    :param return_3d: if True then return  (n, seq_len, n_features)
    :return:
    """
    n_features = data.shape[-1]
    # sequence data: (n, seq_len, n_features)
    seq_data = np.array([data[i - seq_len:i, :] for i in range(seq_len, len(data))], dtype=np.float32)
    if not return_3d:
        # concatenate columns: (n, seq_len * n_features)
        data = np.zeros((seq_data.shape[0], seq_len * n_features))
        for i in range(n_features - 1):
            data[:, i * seq_len:seq_len * (i + 1)] = seq_data[:, :, i]
    else:
        data = seq_data

    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("ETF FNN")
    parser.add_argument("--seq-len", type=int, default=5, help="Input sequence length")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1, help="Epoch logs frequency")
    parser.add_argument("--plot-every", type=int, default=20, help="Plots frequency")
    parser.add_argument("--no-cash", action='store_true', help="Implement a portfolio without cash")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate to apply after each hidden layer")
    # parser.add_argument("--long-short", action='store_true', help="Implement a long-short portfolio stategy")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum parameter in SGD")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility, if not set use default")
    parser.add_argument("--test-size", type=int, default=300, help="Test size")
    parser.add_argument("--benchmark", type=float, default=0., help="Risk free rate for excess Sharpe Ratio")

    args = parser.parse_args()

    LOGGER.info('Set seed')
    if args.seed is None:
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
    LOGGER.info(f'Seed: {seed}')

    LOGGER.info('Load data')
    dfdata = pd.read_pickle('data/clean_VTI_AGG_DBC.p')
    assets = np.unique(dfdata.columns.get_level_values(0)).tolist()
    if not args.no_cash:
        assets = assets + ['cash']
    n_assets = len(assets)
    LOGGER.info(dfdata.head())

    plt.plot(dfdata.loc[:, pd.IndexSlice[:, 'close']].droplevel(1, 1))
    plt.show()

    LOGGER.info('Create input and returns array ...')
    # get input data
    data = dfdata.loc[:, pd.IndexSlice[:, 'close']].droplevel(0, 1).values
    # drop last price since returns and features should be shifted by one
    data = data[:-1, :]
    # get returns for next period
    returns = dfdata.loc[:, pd.IndexSlice[:, 'returns']].droplevel(0, 1).values
    # drop first row since it is nan
    returns = returns[1:, :]
    # add cash in returns
    returns = np.concatenate([returns, np.zeros(len(returns)).reshape(-1, 1)], 1)

    LOGGER.info('Preprocessing ...')
    # Preprocessing
    LOGGER.info('Train / test split')
    train_returns = returns[:-args.test_size].astype(np.float32)
    train_examples = data[:-args.test_size].astype(np.float32)
    test_returns = returns[-args.test_size:].astype(np.float32)
    test_examples = data[-args.test_size:].astype(np.float32)

    LOGGER.info('MinMaxScaler with range [-1,1]')
    scaler = preprocessing.MinMaxScaler([-1, 1])
    LOGGER.info('Fit to train set and transform')
    scaler.fit(train_examples)
    train_examples = scaler.transform(train_examples)
    LOGGER.info('Transform test set')
    test_examples = scaler.transform(test_examples)

    for i in range(train_examples.shape[-1]):
        plt.plot(train_examples[:, i])
        plt.show()

    LOGGER.info('Build delayed input sequence and corresponding returns')
    # Train set
    train_examples = build_delayed_window(train_examples, seq_len=args.seq_len, return_3d=False)
    train_returns = build_delayed_window(train_returns, seq_len=args.seq_len, return_3d=True)
    # Returns correspond to last value of sequence
    train_returns = train_returns[:, -1, :]

    # Test set
    test_examples = build_delayed_window(test_examples, seq_len=args.seq_len, return_3d=False)
    test_returns = build_delayed_window(test_returns, seq_len=args.seq_len, return_3d=True)
    # Train returns correspond to last value of sequence
    test_returns = test_returns[:, -1, :]

    LOGGER.info(f'Train data shape: {train_examples.shape}')
    LOGGER.info(f'Train returns shape: {train_returns.shape}')
    LOGGER.info(f'Test data shape: {test_examples.shape}')
    LOGGER.info(f'Test returns shape: {test_returns.shape}')

    # model
    LOGGER.info('Create model')
    n_features = train_examples.shape[-1]


    model = build_model(input_dim=(n_features),
                        output_dim=n_assets,
                        batch_size=args.batch_size,
                        n_hidden=args.n_hidden,
                        dropout=args.dropout,
                        cash_bias=not args.no_cash,
                        cash_initializer=tf.ones_initializer())
    LOGGER.info(model.summary())
    """
    model = MLP(input_dim=(None, n_features), output_dim=n_assets, batch_size=args.batch_size, n_hidden=args.n_hidden,
                cash_bias=not args.no_cash, cash_initializer=tf.ones_initializer(), dropout=args.dropout)
    """

    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum)
    # Traning pipeline
    LOGGER.info('Create tf.data.Dataset')
    # Train
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_returns))
    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)
    # Test
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_returns))
    test_dataset = test_dataset.batch(args.batch_size, drop_remainder=True)

    # Training loop
    LOGGER.info('Start training ...')
    weights = []
    train_history = {'loss': [], 'avg_ret': [], 'cum_ret': []}
    test_history = {'loss': [], 'avg_ret': [], 'cum_ret': []}
    for epoch in range(args.n_epochs):
        train_epoch_stats = {'loss': tf.keras.metrics.Mean(), 'avg_ret': tf.keras.metrics.Mean(),
                             'cum_ret': tf.keras.metrics.Sum()}
        test_epoch_stats = {'loss': tf.keras.metrics.Mean(), 'avg_ret': tf.keras.metrics.Mean(),
                            'cum_ret': tf.keras.metrics.Sum()}

        # Training loop
        epoch_actions = []
        counter = 0
        for features, returns in train_dataset:
            # Optimize the model
            with tf.GradientTape() as tape:
                actions, loss_value = sharpe_ratio(model, features, returns, training=True, benchmark=args.benchmark)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            weights.append([var.numpy() for var in model.trainable_variables])

            # Track progress
            ret_t = tf.reduce_sum(returns * actions, axis=-1)
            avg_reg = tf.reduce_mean(ret_t)
            cum_ret = tf.reduce_sum(ret_t)
            train_epoch_stats['loss'].update_state(loss_value)  # Add current batch loss
            train_epoch_stats['cum_ret'].update_state(cum_ret)
            train_epoch_stats['avg_ret'].update_state(avg_reg)

            # update actions
            epoch_actions.append(actions.numpy())
            counter += 1

        epoch_actions = np.array([i for sub in epoch_actions for i in sub])

        # Inference
        for features, returns in test_dataset:
            actions, loss_value = sharpe_ratio(model, features, returns, training=False, benchmark=args.benchmark)
            # Track progress
            test_epoch_stats['loss'].update_state(loss_value)
            ret_t = tf.reduce_sum(returns * actions, axis=-1)
            avg_reg = tf.reduce_mean(ret_t)
            cum_ret = tf.reduce_sum(ret_t)
            test_epoch_stats['cum_ret'].update_state(cum_ret)
            test_epoch_stats['avg_ret'].update_state(avg_reg)

        # Update history
        train_history['loss'].append(train_epoch_stats['loss'].result())
        train_history['avg_ret'].append(train_epoch_stats['avg_ret'].result())
        train_history['cum_ret'].append(train_epoch_stats['cum_ret'].result())
        test_history['loss'].append(test_epoch_stats['loss'].result())
        test_history['avg_ret'].append(test_epoch_stats['avg_ret'].result())
        test_history['cum_ret'].append(test_epoch_stats['cum_ret'].result())

        if epoch % args.log_every == 0:
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

        if epoch % args.plot_every == 0:
            fig, axs = plt.subplots(1, 3, figsize=(15, 3))
            axs[0].plot([w[0][:, 0] for w in weights])
            axs[0].set_title('weight 0')
            axs[1].plot([w[1][0] for w in weights])
            axs[1].set_title('weight 1')
            axs[2].plot(epoch_actions)
            axs[2].legend(assets)
            axs[2].set_title('prediction')
            fig.suptitle(f'Epoch {epoch}')
            plt.show()

            strat_perf = (train_returns[:len(epoch_actions), :] * epoch_actions).sum(1)
            eq_port = train_returns[:len(epoch_actions), :].mean(1)
            plt.plot((eq_port + 1).cumprod(), label='equally weighted')
            plt.plot((strat_perf + 1).cumprod(), label='strategy')
            plt.legend()
            plt.title(f'Train perf at epoch {epoch}')
            plt.show()

            fig, axs = plt.subplots(1, 3, figsize=(15, 3))
            axs[0].plot(train_history['loss'])
            axs[0].plot(test_history['loss'])
            axs[0].set_title('Loss')
            axs[1].plot(train_history['avg_ret'])
            axs[1].plot(test_history['avg_ret'])
            axs[1].set_title('Average return')
            axs[2].plot(train_history['cum_ret'])
            axs[2].plot(test_history['cum_ret'])
            axs[2].set_title('Cum return')
            fig.suptitle(f'Epoch {epoch}')
            plt.show()

    # Inference on train
    # TODO: parallelization
    train_predictions = []
    for features, returns in train_dataset:
        pred, _ = sharpe_ratio(model, features, returns, training=False, benchmark=args.benchmark)
        train_predictions.append(pred)
    train_predictions = np.array([i for sub in train_predictions for i in sub])

    # Inference on test
    # TODO: parallelization
    test_predictions = []
    for features, returns in test_dataset:
        pred, _ = sharpe_ratio(model, features, returns, training=False, benchmark=args.benchmark)
        test_predictions.append(pred)
    test_predictions = np.array([i for sub in test_predictions for i in sub])

    # Final prediction
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    axs[0].plot(train_predictions)
    axs[0].legend(assets)
    axs[0].set_title('Train prediction')
    axs[1].plot(test_predictions)
    axs[1].legend(assets)
    axs[1].set_title('Test prediction')
    plt.show()

    # Final perf
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    strat_perf = (train_returns[:len(train_predictions), :] * train_predictions).sum(1)
    eq_port = train_returns[:len(train_predictions), :].mean(1)
    axs[0].plot((eq_port + 1).cumprod(), label='equally weighted')
    axs[0].plot((strat_perf + 1).cumprod(), label='equally strategy')
    axs[0].legend()
    axs[0].set_title('Final train performance')
    strat_perf = (test_returns[:len(test_predictions), :] * test_predictions).sum(1)
    eq_port = test_returns[:len(test_predictions), :].mean(1)
    axs[1].plot((eq_port + 1).cumprod(), label='equally weighted')
    axs[1].plot((strat_perf + 1).cumprod(), label='equally strategy')
    axs[1].legend()
    axs[1].set_title('Final test performance')
    plt.show()

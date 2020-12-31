import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from utils.config import LOGGER


def build_model(input_dim: tuple, output_dim: int, batch_size: int, cash_bias: bool = True,
                n_hidden: int = 1, cash_initializer: tf.initializers = tf.ones_initializer()):
    if cash_bias:
        cash_weight = tf.Variable(initial_value=cash_initializer(shape=(batch_size, 1), dtype='float32'),
                                trainable=True)

    input_ = tf.keras.layers.Input(input_dim, dtype=tf.float32)
    for i in range(n_hidden):
        if i == 0:
            hidden = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float32)(input_)
        else:
            hidden = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float32)(hidden)

    if cash_bias:
        output = tf.keras.layers.Dense(output_dim - 1, activation='linear', dtype=tf.float32)(hidden)
        output = tf.keras.layers.Concatenate(axis=-1)([output, cash_weight])
        output = tf.keras.layers.Activation('softmax')(output)
    else:
        output = tf.keras.layers.Dense(output_dim, activation='softmax', dtype=tf.float32)(hidden)

    model = tf.keras.models.Model(input_, output)
    return model


def sharpe_ratio(model, x, returns, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x)  # model(x, training=training)
    # take log maybe ??
    ret = tf.math.reduce_sum(returns * y_, axis=-1)
    sr = - tf.reduce_mean(ret) / (tf.math.reduce_std(ret) + 10e-12)
    return y_, sr


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("ETF FNN")
    parser.add_argument("--seq-len", type=int, default=5, help="Input sequence length")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1, help="Epoch logs frequency")
    parser.add_argument("--plot-every", type=int, default=20, help="Plots frequency")
    parser.add_argument("--no-cash", action='store_true', help="Implement a portfolio without cash")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden layers")
    # parser.add_argument("--long-short", action='store_true', help="Implement a long-short portfolio stategy")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.975, help="Momentum parameter in SGD")
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility, if not set use default")
    args = parser.parse_args()

    LOGGER.info('Set seed')
    np.random.seed(1)

    LOGGER.info('Load data')
    dfdata = pd.read_pickle('data/clean_VTI_AGG_DBC.p')
    assets = np.unique(dfdata.columns.get_level_values(0)).tolist()
    if args.no_cash:
        n_assets = len(assets)
    else:
        n_assets = len(assets) + 1
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
    train_returns = returns.copy()
    train_examples = data.copy()
    LOGGER.info('MinMaxScaler with range [-1,1]')
    scaler = preprocessing.MinMaxScaler([-1, 1])
    scaler.fit(train_examples)
    train_examples = scaler.transform(train_examples)
    train_examples = train_examples.astype(np.float32)
    train_returns = train_returns.astype(np.float32)

    for i in range(train_examples.shape[-1]):
        plt.plot(train_examples[:, i])
        plt.show()

    LOGGER.info('Build delayed input sequence')
    # sequence data
    seq_train_examples = np.array([data[i - args.seq_len:i, :] for i in range(args.seq_len, len(train_examples))],
                                  dtype=np.float32)
    seq_train_returns = np.array([returns[i - args.seq_len:i, :] for i in range(args.seq_len, len(train_returns))],
                                 dtype=np.float32)

    LOGGER.info('Reshape array to 2d for MLP')
    # contaneate columns
    train_examples = np.zeros((seq_train_examples.shape[0], seq_train_examples.shape[1] * seq_train_examples.shape[2]))
    for i in range(n_assets - 1):
        train_examples[:, i * args.seq_len:args.seq_len * (i + 1)] = seq_train_examples[:, :, i]
    train_returns = seq_train_returns[:, -1, :]

    LOGGER.info(f'Train data shape: {train_examples.shape}')
    LOGGER.info(f'Train returns shape: {train_returns.shape}')

    # model
    LOGGER.info('Create model')
    n_features = train_examples.shape[-1]

    model = build_model(input_dim=(n_features),
                        output_dim=n_assets,
                        batch_size=args.batch_size,
                        n_hidden=args.n_hidden,
                        cash_bias=not args.no_cash,
                        cash_initializer=tf.ones_initializer())
    LOGGER.info(model.summary())
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum)

    # Traning pipeline
    LOGGER.info('Create tf.data.Dataset')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_returns))
    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=True)

    # Training loop
    LOGGER.info('Start training ...')
    weights = []
    loss_history = []
    avg_ret_history = []
    cum_ret_history = []
    for epoch in range(args.n_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_cum_returns = tf.keras.metrics.Sum()
        epoch_returns = tf.keras.metrics.Mean()

        # Training loop
        mean_loss = []
        epoch_actions = []
        counter = 0

        # ACTIONS ARE DELAYED BY BATCH SIZE NO ??
        # MAYBE POSSIBLE TO USE BATCH_ISZE = TIME_STEP and return sequence = True
        for features, returns in train_dataset:

            # Optimize the model
            with tf.GradientTape() as tape:
                actions, loss_value = sharpe_ratio(model, features, returns, training=True)
            grads = tape.gradient(loss_value, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            weights.append([var.numpy() for var in model.trainable_variables])

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss

            ret_t = tf.reduce_sum(returns * actions, axis=-1)
            avg_reg = tf.reduce_mean(ret_t)
            cum_ret = tf.reduce_sum(ret_t)

            epoch_cum_returns.update_state(cum_ret)
            epoch_returns.update_state(avg_reg)
            mean_loss.append(cum_ret.numpy())

            # update actions
            epoch_actions.append(actions.numpy())

            if counter % 10000 == 0 and counter > 0:
                print('########################')
                print(f'Batch {counter}')
                print(f'Epoch avg returns: {epoch_returns.result().numpy()}')
                print(f'Epoch cum returns: {epoch_cum_returns.result().numpy()}')
                print(f'Epoch loss: {epoch_loss_avg.result().numpy()}')
                print('########################')

                fix, axs = plt.subplots(1, 2, figsize=(10, 3))
                axs[0].plot(np.cumprod(returns.numpy() + 1))
                axs[0].set_title('B&H')
                axs[1].plot(np.cumprod(ret_t.numpy() + 1))
                axs[1].set_title('Cumret')
                plt.show()

            counter += 1

        epoch_actions = np.array([i for sub in epoch_actions for i in sub])

        if epoch % args.log_every == 0:
            LOGGER.info("Epoch {:03d}: Loss: {:.6f}, Returns: {:.6f}, Cum Returns: {:.6f}".format(epoch,
                                                                                            epoch_loss_avg.result(),
                                                                                            epoch_returns.result(),
                                                                                            epoch_cum_returns.result())
                  )
        if epoch % args.plot_every == 0:
            fix, axs = plt.subplots(1, 3, figsize=(15, 3))
            axs[0].plot([w[0][:, 0] for w in weights])
            axs[0].set_title('weight 0')
            axs[1].plot([w[1][0] for w in weights])
            axs[1].set_title('weight 1')
            axs[2].plot(epoch_actions)
            axs[2].set_title('prediction')
            plt.show()

            fix, axs = plt.subplots(1, 3, figsize=(15, 3))
            axs[0].plot(loss_history)
            axs[0].set_title('Loss')
            axs[1].plot(avg_ret_history)
            axs[1].set_title('Average return')
            axs[2].plot(cum_ret_history)
            axs[2].set_title('Cum return')
            plt.show()


        loss_history.append(epoch_loss_avg.result())
        avg_ret_history.append(epoch_returns.result())
        cum_ret_history.append(epoch_cum_returns.result())



    strat_perf = (train_returns[:len(epoch_actions), :] * epoch_actions).sum(1)
    eq_port = train_returns[:len(epoch_actions), :].mean(1)
    plt.plot(strat_perf)
    plt.show()
    plt.plot((eq_port + 1).cumprod(), label='equally weighted')
    plt.plot((strat_perf + 1).cumprod(), label='strategy')
    plt.legend()
    plt.show()

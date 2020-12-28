from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tensorflow.keras import initializers
from loss import log_avg_ret, avg_ret, excess_log_avg_ret, cum_ret, sharpe_ratio, volatility
import tensorflow as tf
from model import ann_model, lstm_model


def grad(model, inputs, returns):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, returns, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def generate_sinus_price(n, price0=100):
    # sinus price
    # Get x values of the sine wave
    step = 1000/n
    time = np.arange(0, 1000, 0.1)
    print(len(time))
    # Amplitude of the sine wave is sine of a variable like time
    price = 10 * np.sin(time) + price0 + np.random.normal(loc=0.0, scale=3, size=len(time))
    price = pd.Series(price).rolling(3).mean().dropna()

    return price


def generate_2ddata(n=1000, price0=100, lags=5):
    price = generate_sinus_price(n, price0)
    dfdata = pd.DataFrame(dtype=np.float32)
    dfdata['price'] = price
    for i in range(0, lags):
        dfdata['feature_%s' % (lags - 1 - i)] = dfdata['price'].shift(-i)
    dfdata['returns_pct'] = dfdata['feature_0'].pct_change().shift(-1)
    dfdata['returns_diff'] = dfdata['feature_0'].diff().shift(-1)
    dfdata.dropna(inplace=True)
    dfdata.drop('price', 1, inplace=True)
    dfdata = dfdata.astype(np.float32)
    features_names = [f'feature_{i}' for i in range(lags)]

    return dfdata, features_names


def get_3d_sequences(dfdata: pd.DataFrame, feature_names: str, returns: str, seq_len: int):
    input = dfdata[feature_names].values
    returns = dfdata[returns].values.reshape(-1, 1)
    seq_input = np.array([input[i - seq_len:i, :] for i in range(seq_len, len(input))])
    seq_returns = np.array([returns[i - seq_len:i, :] for i in range(seq_len, len(returns))])

    return seq_input, seq_returns


if __name__ == "__main__":
    MODEL = 'LSTM'
    seed = random.randint(0, 1000)
    print(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    log_every=10
    initializer = 'normal_uniform'  # zeros, 'random_uniform', normal_uniform, 'glorot_uniform'
    BATCH_SIZE = 256
    loss = sharpe_ratio #avg_ret
    RETURNS = 'returns_pct'  # 'diff'
    output_activation = 'sigmoid'
    hidden_activation = 'tanh'
    stateful = True
    reset_state_every = 200
    learning_rate = 0.01
    num_epochs = 2000
    test_size = 200
    seq_len = 1
    lags = 32

    if MODEL == 'ANN':
        stateful = False

    # generate data
    dfdata, feature_names = generate_2ddata(n=1000, price0=100, lags=lags)
    print(dfdata.head())
    print(len(dfdata))
    plt.plot(dfdata['feature_0'])
    plt.show()
    #exit()
    if MODEL == 'ANN':
        # Create train / test dataset
        train_examples = dfdata[feature_names].values[:-test_size]
        train_returns = dfdata[RETURNS].values[:-test_size].reshape(-1, 1)
        test_examples = dfdata[feature_names].values[-test_size:]
        test_returns = dfdata[RETURNS].values[-test_size:].reshape(-1, 1)
    elif MODEL == 'LSTM':
        seq_input, seq_returns = get_3d_sequences(dfdata, feature_names, RETURNS, seq_len)
        train_examples = seq_input[:-test_size, :, :]
        train_returns = seq_returns[:-test_size, -1, :]
        test_examples = seq_input[-test_size:, :, :]
        test_returns = seq_returns[-test_size:, -1, :]
    else:
        raise NotImplementedError()

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_returns))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_returns))

    # batch
    if stateful:
        drop_remainder = True
    else:
        drop_remainder = False
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder)
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder)

    # perfect strategy
    perfect_signals = train_returns + 1
    perfect_signals[perfect_signals >= 1] = 1
    perfect_signals[perfect_signals < 1] = -1
    max_perf = np.cumsum((perfect_signals * train_returns))[-1]

    print(f'######## MAX PERF: {max_perf}')

    ##### training
    # Build model
    input_dim = train_dataset.element_spec[0].shape[1:]
    if MODEL == 'ANN':

        model = ann_model(input_dim,
                          output_activation=output_activation,
                          hidden_activation=hidden_activation)
    elif MODEL == 'LSTM':
        if stateful:
            input_dim = (BATCH_SIZE, input_dim[0], input_dim[1])

        model = lstm_model(input_dim,
                           stateful=stateful,
                           output_activation=output_activation,
                           hidden_activation=hidden_activation)
    print(model.summary())
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # model.compile(optimizer=optimizer, loss=loss)

    ## Note: Rerunning this cell uses the same model variables

    # Keep results for plotting
    train_loss_results = []
    train_cum_returns_results = []
    train_returns_results = []

    weights = []
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_cum_returns = tf.keras.metrics.Sum()
        epoch_returns = tf.keras.metrics.Mean()

        # Training loop
        mean_loss = []
        counter = 0
        for features, returns in train_dataset:
            counter+=1
            # Optimize the model
            loss_value, grads = grad(model, features, returns)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            weights.append([var.numpy() for var in model.trainable_variables])

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            cum_ret = tf.reduce_sum(returns * model(features, training=True))
            ret = tf.reduce_mean(returns * model(features, training=True))

            epoch_cum_returns.update_state(cum_ret)
            epoch_returns.update_state(ret)
            mean_loss.append(cum_ret.numpy())

            if stateful and counter % reset_state_every == 0:
                #print('Resetting state')
                model.reset_states()

        pred = model(features, training=True)

        # plt.hist(pred.numpy(), bins = 30)
        # plt.show()
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_cum_returns_results.append(epoch_cum_returns.result())
        train_returns_results.append(epoch_returns.result())

        if epoch % log_every == 0:
            fix, axs = plt.subplots(1, 3, figsize=(15, 3))
            axs[0].plot([w[0][:, 0] for w in weights])
            axs[0].set_title('weight 0')
            axs[1].plot([w[1][0] for w in weights])
            axs[1].set_title('weight 1')
            predict_dataset = tf.convert_to_tensor(train_examples)
            if stateful:
                predict_dataset = tf.data.Dataset.from_tensor_slices((predict_dataset))
                predict_dataset = predict_dataset.batch(BATCH_SIZE, drop_remainder)
                predictions = []
                for features in predict_dataset:
                    pred = model(features, training=False)
                    pred = pred.numpy().reshape(-1)
                    predictions.append(pred)
                predictions = np.array([i for sub in predictions for i in sub])
            else:
                predictions = model(predict_dataset, training=False)
                predictions = predictions.numpy().reshape(-1)
            axs[2].plot(predictions)
            axs[2].set_title('prediction')
            plt.show()

            print("Epoch {:03d}: Loss: {:.6f}, Returns: {:.6f}, Cum Returns: {:.6f}".format(epoch,
                                                                                            epoch_loss_avg.result(),
                                                                                            epoch_returns.result(),
                                                                                            epoch_cum_returns.result())
                  )

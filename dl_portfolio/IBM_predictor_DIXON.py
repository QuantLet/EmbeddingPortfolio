from dl_portfolio.custom_layer import DynamicSmoothRNN
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from dl_portfolio.logger import LOGGER


def build_seq_data(data, seq_len, nb_sequence=None, horizon=1):
    LOGGER.info(f'Data shape: {data.shape}')
    n_sample = data.shape[0] - horizon
    max_nb_sequence = n_sample - seq_len +1
    LOGGER.info(f'Max nb of sequences: {max_nb_sequence}')
    if nb_sequence is None:
        nb_sequence = max_nb_sequence
    if nb_sequence > max_nb_sequence:
        nb_sequence = max_nb_sequence
    LOGGER.info(f"NB sequence: {nb_sequence}")
    seq_data = np.zeros((nb_sequence, seq_len, 1))
    seq_data[:] = np.nan
    label = np.zeros((nb_sequence, 1))
    label[:] = np.nan
    for i in range(nb_sequence):
        seq_data[i] = data[i:i + seq_len]
        label[i] = data[i + seq_len + horizon - 1]

    seq_data = seq_data.astype(np.float32)
    label = label.astype(np.float32)
    assert np.sum(np.isnan(seq_data)) == 0, np.sum(np.isnan(seq_data))
    assert np.sum(label[:-1, 0] != seq_data[1:, -1, 0]) == 0

    return seq_data, label


if __name__ == "__main__":
    start_date = None  # '2006-01-03'
    start_test = '2017-08-09'
    batch_size = 256
    l1 = 1e-3 # 10−3, 10−2, 10−1, 0
    seq_len = 5 # 2
    neurons = 50  # 5, 10, 15, 20, 25
    epochs = 1000
    learning_rate = 0.001


    def scheduler(epoch, lr):
        if epoch > 1000:
            return 0.0001
        elif epoch > 5000:
            return 0.0001
        else:
            return lr


    callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    data = pd.read_pickle('./data/stock/IBM.p')
    data = data.loc[:'2017-12-29', :] if start_date is None else data.loc[start_date:'2017-12-29', :]
    data = data.loc[:, pd.IndexSlice['IBM', ['close']]].droplevel(1, 1)  # date from paper
    train_data = data.loc[:start_test].iloc[:-1, :]
    test_data = data.loc[start_test:]
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    range_scaler = preprocessing.MinMaxScaler(feature_range=[-1, 1])
    train_data = range_scaler.fit_transform(train_data)
    test_data = range_scaler.transform(test_data)

    train_seq_data, train_label = build_seq_data(train_data, seq_len=seq_len)
    test_seq_data, test_label = build_seq_data(test_data, seq_len=seq_len)

    input_dim = train_seq_data.shape[1:]

    # Model
    input_ = tf.keras.layers.Input(input_dim)
    layer_1 = DynamicSmoothRNN(neurons, return_sequences=False, activation='tanh',
                               kernel_initializer=tf.keras.initializers.random_normal(),
                               kernel_regularizer=tf.keras.regularizers.L1(l1))
    output_layer = tf.keras.layers.Dense(1, activation='linear')

    hidden_1 = layer_1(input_)
    output = output_layer(hidden_1)
    model = tf.keras.models.Model(input_, output)
    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    history = model.fit(train_seq_data, train_label, batch_size=batch_size, verbose=1, epochs=epochs,
                        validation_data=(test_seq_data, test_label), callbacks=[callback])
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('History')
    plt.legend()
    plt.show()

    # Prediction
    train_prediction = model.predict(train_seq_data)

    plt.plot(train_label, label='truth')
    plt.plot(train_prediction, label='pred')
    plt.legend()
    plt.title('Train pred')
    plt.show()
    plt.plot((train_label - train_prediction) ** 2)
    plt.show()

    plt.plot(train_label[:30], label='truth')
    plt.plot(train_prediction[:30], label='pred')
    plt.legend()
    plt.title('Train pred')
    plt.show()
    plt.plot((train_label - train_prediction)[:30] ** 2)
    plt.title('Train error')
    plt.show()

    test_prediction = model.predict(test_seq_data)
    LOGGER.info(f'Evaluate on test: {model.evaluate(test_seq_data, test_prediction)}')
    LOGGER.info(f'Evaluate on test: {model.evaluate(test_seq_data, test_label)}')

    test_label = range_scaler.inverse_transform(test_label)
    test_prediction = range_scaler.inverse_transform(test_prediction)
    test_label = scaler.inverse_transform(test_label)
    test_prediction = scaler.inverse_transform(test_prediction)



    print('test pred shape', test_prediction.shape)

    plt.plot(test_label, label='truth')
    plt.plot(test_prediction, label='pred')
    plt.legend()
    plt.title('Test pred')
    plt.show()
    plt.plot((test_label - test_prediction) ** 2)
    plt.title('Test error')
    plt.show()
    plt.plot(test_label[:30], label='truth')
    plt.plot(test_prediction[:30], label='pred')
    plt.legend()
    plt.title('Test pred')
    plt.show()
    plt.plot((test_label - test_prediction)[:30] ** 2)
    plt.title('Test error')
    plt.show()


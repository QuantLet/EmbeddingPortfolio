import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import backend as K
from dl_portfolio.custom_layer import DenseTied
from dl_portfolio.constraints import WeightsOrthogonalityConstraint, NonNegAndUnitNorm, UncorrelatedFeaturesConstraint

if __name__ == "__main__":
    np.random.seed(10)
    learning_rate = 1e-2
    epochs = 200
    activation = 'linear'
    encoding_dim = 2

    asset_to_train_on = ['bitcoin', 'dogecoin', 'litecoin', 'ripple', 'ethereum', 'cardano']
    # build main data
    val_until = '2021-01-01'
    data = pd.DataFrame()
    for asset in asset_to_train_on:
        coin_data = pd.read_csv(f'data/crypto_data/coingecko/market/{asset}.csv', index_col=0)
        coin_data.index = pd.to_datetime(coin_data.index)
        coin_data = coin_data[['price']]
        coin_data.columns = [asset]
        data = pd.concat([data, coin_data], 1)
    data = data.loc[:val_until, :]

    data_specs = {}
    data_specs[0] = {'start': str(data.index[0]), 'end': str(data.drop('cardano', 1).dropna().index[0]),
                     'assets': ['bitcoin', 'dogecoin', 'litecoin', 'ripple']}
    data_specs[1] = {'start': str(data.drop('cardano', 1).dropna().index[0]), 'end': str(data.dropna().index[0]),
                     'assets': ['bitcoin', 'dogecoin', 'litecoin', 'ripple', 'ethereum']}

    data_specs[2] = {'start': str(data.dropna().index[0]), 'end': str(data.index[-1]),
                     'assets': ['bitcoin', 'dogecoin', 'litecoin', 'ripple', 'ethereum', 'cardano']}

    # train / val split

    set_ = 0
    data_spec = data_specs[set_]
    # train_data = data.loc[data_spec['start']:data_spec['end'], data_spec['assets']].iloc[:-1, :]
    # val_size = int(0.2 * len(train_data))
    # val_data = data.loc[data_spec['end']:, data_spec['assets']]
    # val_data = val_data.iloc[:val_size]

    val_size = int(0.2 * len(data))
    train_data = data.iloc[:-val_size, :]
    val_data = data.iloc[-val_size:, :]

    # featurization
    train_data = train_data.pct_change(1).dropna().values
    val_data = val_data.pct_change(1).dropna().values

    # standardization
    scaler = preprocessing.StandardScaler(with_std=True, with_mean=False)
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)

    input_dim = train_data.shape[1]
    input_ = tf.keras.layers.Input(input_dim, dtype=tf.float32, name='input')
    encoder_layer = tf.keras.layers.Dense(encoding_dim,
                                          activation=activation,
                                          kernel_initializer='random_normal',
                                          kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim,
                                                                                            axis=0),
                                          activity_regularizer=UncorrelatedFeaturesConstraint(encoding_dim,
                                                                                              weightage=0.2),
                                          kernel_constraint=NonNegAndUnitNorm(axis=0),
                                          dtype=tf.float32, name='encoder')
    decoder_layer = DenseTied(input_dim, tied_to=encoder_layer, activation=activation,
                              kernel_initializer='random_normal',
                              kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, axis=1),
                              use_bias=True,
                              dtype=tf.float32, name='decoder')

    output = decoder_layer(encoder_layer(input_))
    encoding = encoder_layer(input_)
    model = tf.keras.models.Model(input_, output)
    encoder = tf.keras.models.Model(input_, encoding)
    print(model.summary())

    # Train
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Very low learning rate
                  loss='mse')
    history = model.fit(train_data, train_data,
                        epochs=epochs,
                        validation_data=(val_data, val_data), verbose=1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()

    val_prediction = model.predict(val_data)
    indices = np.random.choice(list(range(len(val_data))), 20)
    xticks = list(range(val_data.shape[-1]))

    for i in indices:
        plt.figure()
        plt.scatter(xticks, val_data[i], label = 'truth')
        plt.scatter(xticks, val_prediction[i], label = 'prediction')
        plt.legend()
        plt.show()
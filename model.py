from typing import Tuple
import tensorflow as tf


def ann_model(input_dim: Tuple, initializer='zeros', hidden_activation='tanh', output_activation='tanh'):
    inputs = tf.keras.layers.Input(shape=input_dim)
    # hidden = tf.keras.layers.Dense(1, activation=hidden_activation)(inputs)
    outputs = tf.keras.layers.Dense(1,
                                    kernel_initializer=initializer,
                                    bias_initializer=initializer,
                                    activation=output_activation)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def lstm_model(input_dim: Tuple, initializer='zeros', hidden_activation='tanh', output_activation='tanh'):
    # Build model
    inputs = tf.keras.layers.Input(shape=input_dim)
    # hidden = tf.keras.layers.Dense(10, activation='tanh')(inputs)
    # hidden = tf.keras.layers.LSTM(1,
    #                               kernel_initializer=initializer,
    #                               bias_initializer=initializer,
    #                               return_sequences=True,
    #                               activation=output_activation)(inputs)
    outputs = tf.keras.layers.LSTM(1,
                                   kernel_initializer=initializer,
                                   bias_initializer=initializer,
                                   return_sequences=False,
                                   activation=output_activation)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

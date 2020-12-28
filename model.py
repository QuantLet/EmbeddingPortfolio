from typing import Tuple
import tensorflow as tf


def ann_model(input_dim: Tuple, initializer='zeros', hidden_activation='tanh', output_activation='tanh'):
    inputs = tf.keras.layers.Input(shape=input_dim)
    # hidden = tf.keras.layers.Dense(1,
    #                                kernel_initializer=initializer,
    #                                bias_initializer=initializer,
    #                                activation=hidden_activation)(inputs)
    outputs = tf.keras.layers.Dense(1,
                                    kernel_initializer=initializer,
                                    bias_initializer=initializer,
                                    activation=output_activation)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def lstm_model(input_dim: Tuple, initializer='zeros', hidden_activation='tanh', stateful=False,
               output_activation='tanh'):
    # Build model
    if stateful:
        inputs = tf.keras.layers.Input(batch_shape=input_dim)
    else:
        inputs = tf.keras.layers.Input(shape=input_dim)


    outputs = tf.keras.layers.LSTM(1,
                                   kernel_initializer=initializer,
                                   bias_initializer=initializer,
                                   return_sequences=False,
                                   stateful=stateful,
                                   activation=output_activation)(inputs)
    # outputs = tf.keras.layers.Dense(1,
    #                                 kernel_initializer=initializer,
    #                                 bias_initializer=initializer,
    #                                 activation=output_activation)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

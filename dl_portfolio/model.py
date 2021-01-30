import tensorflow as tf
from typing import Tuple, Optional
from first_etf_strat.logger import LOGGER


class CashBias(tf.keras.layers.Layer):

    def __init__(self, initializer):
        super(CashBias, self).__init__()
        self.initializer = initializer

    def build(self, input_shape):
        w_init = self.initializer
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1),
                                 dtype='float32'),
            trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        print('input', tf.shape(inputs))
        print('w', self.w)
        return tf.tile(self.w, tf.shape(inputs)[0])


def build_etf_mlp(input_dim: Tuple, output_dim: int, n_hidden: int = 1, dropout: Optional[float] = None,
                  training: bool = False):
    """

    :param input_dim:
    :param output_dim:
    :param batch_size:
    :param cash_bias:
    :param n_hidden:
    :param cash_initializer:
    :param dropout:
    :param training: to control dropout layer behavior during inference
    :return:
    """

    assert n_hidden > 0

    input_ = tf.keras.layers.Input(input_dim, dtype=tf.float32)

    for i in range(n_hidden):
        if i == 0:
            hidden = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float32)(input_)
        else:
            hidden = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float32)(hidden)
        if dropout:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)

    output = tf.keras.layers.Dense(output_dim, activation='softmax', dtype=tf.float32)(hidden)

    return tf.keras.models.Model(input_, output)


def build_etf_mlp_with_cash_bias(input_dim: Tuple, output_dim: int, batch_size: int,
                                 n_hidden: int = 1, cash_initializer: tf.initializers = tf.ones_initializer(),
                                 dropout: Optional[float] = None, training: bool = False):
    """

    :param input_dim:
    :param output_dim:
    :param batch_size:
    :param cash_bias:
    :param n_hidden:
    :param cash_initializer:
    :param dropout:
    :param training: to control dropout layer behavior during inference
    :return:
    """

    assert n_hidden > 0

    input_ = tf.keras.layers.Input(input_dim, dtype=tf.float32)
    cash_weight = tf.Variable(initial_value=cash_initializer(shape=(batch_size, 1), dtype='float32'),
                              trainable=True)

    for i in range(n_hidden):
        if i == 0:
            hidden = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float32)(input_)
        else:
            hidden = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float32)(hidden)
        if dropout:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)

    output = tf.keras.layers.Dense(output_dim - 1, activation='linear', dtype=tf.float32)(hidden)
    output = tf.keras.layers.Concatenate(axis=-1)([output, cash_weight])
    output = tf.keras.layers.Activation('softmax')(output)

    model = tf.keras.models.Model(input_, output)
    return model


class MLP(tf.keras.Model):
    def __init__(self, output_dim: int, batch_size: int, cash_bias: bool = True,
                 n_hidden: int = 1, cash_initializer: tf.initializers = tf.ones_initializer(),
                 dropout: Optional[float] = None):
        raise NotImplementedError('Subclass gives weird result in inference')
        super(MLP, self).__init__()

        assert n_hidden > 0
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.cash_bias = cash_bias
        self.n_hidden = n_hidden
        self.cash_initializer = cash_initializer
        self.dropout = dropout

        if cash_bias:
            self.cash_weight = tf.Variable(initial_value=cash_initializer(shape=(batch_size, 1), dtype='float32'),
                                           trainable=True)

        # Hidden layers
        self.hidden_layers = {}
        self.dropout_layers = {}

        for i in range(self.n_hidden):
            if i == 0:
                LOGGER.info(f'Adding hidden layer {i}')
                self.hidden_layers[f"hidden_{i}"] = tf.keras.layers.Dense(64,
                                                                          activation='tanh',
                                                                          dtype=tf.float32,
                                                                          name=f"hidden_{i}")
            else:
                self.hidden_layers[f"hidden_{i}"] = tf.keras.layers.Dense(64, activation='tanh', dtype=tf.float32)
            if self.dropout:
                exit()
                LOGGER.info(f'Adding dropout layer {i}')
                self.dropout_layers[f"dropout_{i}"] = tf.keras.layers.Dropout(self.dropout)

        # Output layer
        if cash_bias:
            self.asset_weights = tf.keras.layers.Dense(self.output_dim - 1, activation='linear', dtype=tf.float32,
                                                       name='asset_weights')
        else:
            self.asset_weights = tf.keras.layers.Dense(self.output_dim, activation='linear', dtype=tf.float32,
                                                       name='asset_weights')

        self.output_layer = tf.keras.layers.Activation('softmax', name='output')

    def call(self, inputs, training=False):
        for i in range(self.n_hidden):
            if i == 0:
                network = self.hidden_layers[f"hidden_{i}"](inputs)
            else:
                network = self.hidden_layers[f"hidden_{i}"](network)
            if training and self.dropout:
                exit()
                network = self.dropout_layers[f"dropout_{i}"](network, training=training)

        if self.cash_bias:
            network = tf.keras.layers.Concatenate(axis=-1)([self.asset_weights(network), self.cash_weight])

        return self.output_layer(network)


if __name__ == "__main__":
    import numpy as np

    model = MLP(input_dim=(None, 5), output_dim=4, batch_size=64, n_hidden=1, cash_initializer=tf.ones_initializer(),
                dropout=0.2)

    pred = model(np.zeros((64, 5)), training=True)
    print(pred.shape)
    pred = model(np.zeros((64, 5)), training=False)
    print(pred.shape)

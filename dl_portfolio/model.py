import tensorflow as tf
from typing import Tuple, Optional, List, Dict
from dl_portfolio.logger import LOGGER
from dl_portfolio.custom_layer import DynamicSmoothRNN, SmoothRNN


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


def build_mlp(input_dim: Tuple, layers: List[Dict], output_dim: int, dropout: Optional[float] = None,
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
    input_ = tf.keras.layers.Input(input_dim, dtype=tf.float32)

    for i, layer in enumerate(layers):
        if i == 0:
            hidden = tf.keras.layers.Dense(layer['neurons'], **layer['params'], dtype=tf.float32)(input_)
        else:
            hidden = tf.keras.layers.Dense(layer['neurons'], **layer['params'], dtype=tf.float32)(hidden)
        if dropout:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)

    output = tf.keras.layers.Dense(output_dim, activation='softmax', dtype=tf.float32)(hidden)

    return tf.keras.models.Model(input_, output)


def build_layer(config: Dict, **kwargs):
    if config['type'] == 'conv2d':
        layer = tf.keras.layers.Conv2D(config['filters'], config['kernel_size'], **config['params'], dtype=tf.float32)

        # strides = layer['strides'], padding = 'valid',
        # data_format = None, dilation_rate = (1, 1), groups = 1, activation = 'tanh',
        # use_bias = True, kernel_initializer = 'glorot_uniform',
        # bias_initializer = 'zeros', kernel_regularizer = None,
        # bias_regularizer = None, activity_regularizer = None, kernel_constraint = None,
        # bias_constraint = None)(input_)
    elif config['type'] == 'conv1d':
        layer = tf.keras.layers.Conv1D(config['filters'], config['kernel_size'], **config['params'], dtype=tf.float32,
                                       **kwargs)
    elif config['type'] == 'lstm':
        layer = tf.keras.layers.LSTM(config['neurons'], **config['params'], dtype=tf.float32, **kwargs)

    elif config['type'] == 'gru':
        layer = tf.keras.layers.GRU(config['neurons'], **config['params'], dtype=tf.float32, **kwargs)

    elif config['type'] == 'DynamicSmoothRNN':
        layer = DynamicSmoothRNN(config['neurons'], **config['params'], dtype=tf.float32, **kwargs)

    elif config['type'] == 'SmoothRNN':
        layer = SmoothRNN(config['neurons'], config['alpha'], **config['params'], dtype=tf.float32, **kwargs)

    elif config['type'] == 'EIIE_dense':
        # from pgportfolio
        width = kwargs.get('width')
        layer = tf.keras.layers.Conv2D(config['filters'], [1, width], strides=(1, 1), dtype=tf.float32,
                                       **config['params'])
    elif config['type'] == 'dense':
        layer = tf.keras.layers.Dense(config['neurons'], **config['params'], dtype=tf.float32, **kwargs)

    elif config['type'] == 'BatchNormalization':
        layer = tf.keras.layers.BatchNormalization(**config.get('params', {}), dtype=tf.float32, **kwargs)

    elif config['type'] == 'RepeatVector':
        layer = tf.keras.layers.RepeatVector(config['neurons'])

    elif config['type'] == 'TimeDistributed':
        layer = tf.keras.layers.RepeatVector(config['neurons'])

    else:
        raise NotImplementedError(f"'{config['type']}' type layer has not been implemented")

    return layer


def EIIE_model(input_dim: Tuple, output_dim: int, layers: List[Dict], dropout: Optional[float] = 0.,
               training: bool = False):
    """

    :param input_dim: [assets, window, features]
    :param output_dim:
    :param batch_size:
    :param cash_bias:
    :param n_hidden:
    :param cash_initializer:
    :param dropout:
    :param training: to control dropout layer behavior during inference
    :return:
    """
    assert len(input_dim) == 3
    input_ = tf.keras.layers.Input(input_dim, dtype=tf.float32)
    for i, layer in enumerate(layers):

        if layer['type'] == 'EIIE_dense':
            width = hidden.get_shape()[2]
            layer = build_layer(layer, width=width)
        else:
            layer = build_layer(layer)
        if i == 0:
            hidden = layer(input_)
        else:
            hidden = layer(hidden)
        if dropout != 0:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)

    width = hidden.get_shape()[2]
    hidden = tf.keras.layers.Conv2D(1, [1, width], padding="valid", kernel_regularizer=None)(hidden)
    hidden = hidden[:, :, 0, 0]
    # btc_bias = tf.ones((1,1))
    # hidden = tf.concat([hidden, btc_bias], 1)
    # output = tf.keras.layers.Activation(hidden, activation="softmax")
    #
    output = tf.keras.layers.Dense(output_dim, activation='softmax', dtype=tf.float32)(hidden)
    # output = tf.keras.layers.Activation(hidden, activation="softmax")

    return tf.keras.models.Model(input_, output)


def asset_independent_model(input_dim: Tuple, output_dim: int, n_pairs: int, layers: List[Dict],
                            dropout: Optional[float] = 0., training: bool = False, cash_bias=False, batch_size=None):
    if cash_bias:
        assert batch_size is not None
    output_layer = layers[-1]
    assert output_layer['type'] in ['softmax', 'simple_long_only', 'softmax_with_weights']
    asset_graph = []
    inputs = []
    for k in range(n_pairs):
        input_ = tf.keras.layers.Input((input_dim), dtype=tf.float32)
        inputs.append(input_)
        for i, layer in enumerate(layers[:-1]):
            layer_name = f'asset_{k}_layer_{i}'
            layer = build_layer(layer, name=layer_name)
            if i == 0:
                hidden = layer(input_)
            else:
                hidden = layer(hidden)
            if dropout != 0:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)
        asset_graph.append(hidden)

    all_asset = tf.keras.layers.concatenate(asset_graph, axis=-1)
    shape = all_asset.shape
    if len(shape) > 2:
        assert len(shape) == 3
        assert shape[1] == 1
        all_asset = tf.keras.layers.Reshape([shape[2]])(all_asset)

    if output_layer['type'] == 'softmax':
        output = tf.keras.layers.Dense(output_dim, activation='softmax', dtype=tf.float32)(all_asset)

    elif output_layer['type'] == 'simple_long_only':
        # apply sigmoid to get positive weights
        all_asset = tf.keras.layers.Activation('sigmoid', dtype=tf.float32)(all_asset)
        output = all_asset / tf.reshape(tf.reduce_sum(all_asset, axis=-1), (-1, 1))
    elif output_layer['type'] == 'softmax_with_weights':
        prev_weights = tf.keras.layers.Input((n_pairs), dtype=tf.float32, name='previous_weights')
        inputs.append(prev_weights)
        all_asset_with_w = tf.keras.layers.concatenate([all_asset, prev_weights], axis=-1)
        output = tf.keras.layers.Dense(n_pairs, activation='softmax', dtype=tf.float32)(all_asset_with_w)
    elif output_layer['type'] == 'cash_bias':
        assert len(all_asset.shape) == 2
        assert all_asset.shape[-1] == n_pairs
        cash_bias = build_cash_bias(batch_size)
        all_asset_with_cash = tf.keras.layers.concatenate([all_asset, cash_bias], axis=-1)
        output = tf.keras.layers.Activation('softmax')(all_asset_with_cash)

    return tf.keras.models.Model(inputs, output)


def stacked_asset_model(input_dim: Tuple, output_dim: int, n_pairs: int, layers: List[Dict],
                        dropout: Optional[float] = 0., training: bool = False, cash_bias=False, batch_size=None):
    if cash_bias:
        assert batch_size is not None
    output_layer = layers[-1]
    assert output_layer['type'] in ['softmax', 'simple_long_only', 'softmax_with_weights', 'cash_bias']
    asset_graph = []
    inputs = []
    for k in range(n_pairs):
        input_ = tf.keras.layers.Input((input_dim), dtype=tf.float32)
        inputs.append(input_)

    all_asset = tf.keras.layers.concatenate(inputs, axis=-1)
    for i, layer in enumerate(layers[:-1]):
        layer_name = f'asset_{k}_layer_{i}'
        layer = build_layer(layer, name=layer_name)
        if i == 0:
            hidden = layer(all_asset)
        else:
            hidden = layer(hidden)
        if dropout != 0:
            hidden = tf.keras.layers.Dropout(dropout)(hidden)

    shape = hidden.shape
    if len(shape) > 2:
        assert len(shape) == 3
        assert shape[1] == 1
        hidden = tf.keras.layers.Reshape([shape[2]])(hidden)

    if output_layer['type'] == 'softmax':
        output = tf.keras.layers.Dense(output_dim, activation='softmax', dtype=tf.float32)(hidden)

    elif output_layer['type'] == 'simple_long_only':
        # apply sigmoid to get positive weights
        output = tf.keras.layers.Dense(output_dim, activation='sigmoid', dtype=tf.float32, **output_layer['params'])(hidden)
        output = output / tf.reshape(tf.reduce_sum(output, axis=-1), (-1, 1))
    elif output_layer['type'] == 'softmax_with_weights':
        raise NotImplementedError()
        prev_weights = tf.keras.layers.Input((n_pairs), dtype=tf.float32, name='previous_weights')
        inputs.append(prev_weights)
        all_asset_with_w = tf.keras.layers.concatenate([all_asset, prev_weights], axis=-1)
        output = tf.keras.layers.Dense(n_assets, activation='softmax', dtype=tf.float32)(all_asset_with_w)
    elif output_layer['type'] == 'cash_bias':
        assert len(hidden.shape) == 2
        output = tf.keras.layers.Dense(output_dim - 1, activation='linear', dtype=tf.float32)(hidden)
        cash_bias = build_cash_bias(batch_size)
        output = tf.keras.layers.concatenate([output, cash_bias], axis=-1)
        output = tf.keras.layers.Activation('softmax')(output)

    return tf.keras.models.Model(inputs, output)


def build_mlp_with_cash_bias(input_dim: Tuple, output_dim: int, batch_size: int,
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
                network = self.dropout_layers[f"dropout_{i}"](network, training=training)

        if self.cash_bias:
            network = tf.keras.layers.Concatenate(axis=-1)([self.asset_weights(network), self.cash_weight])

        return self.output_layer(network)


class CashBias(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(CashBias, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        print(input_shape)
        self.bias = self.add_weight('cash_bias',
                                    shape=(input_shape[-1], 1),
                                    initializer=tf.keras.initializers.Ones(),
                                    trainable=True)

    def call(self, x):
        print(x.shape)

        return tf.stack([x, self.bias])  # tf.concat([x, self.bias], -1)
        # tf.keras.layers.Concatenate

    def test(self):
        print('THIS WORK', tf.TensorShape((None, 1)))

        # cash_bias = CashBias()(all_asset)
        exit()

        # print(tf.Variable(tf.ones_initializer()(shape=tf.TensorShape((None, 1)), dtype=tf.float32), validate_shape=False))
        # exit()
        # print(tf.ones_initializer()(tf.TensorShape([None, 1])))
        exit()
        cash_bias = tf.Variable(initial_value=1, shape=tf.TensorShape([None, 1]), validate_shape=False)
        exit()

        print(cash_bias.shape)  # ], validate_shape=False))
        exit()
        cash_bias = tf.Variable(1., shape=1)
        print(cash_bias.shape)
        exit()


def build_cash_bias(batch_size, initial_value=tf.initializers.Ones()):
    return tf.Variable(initial_value=initial_value((batch_size, 1)), dtype=tf.float32, trainable=True)


if __name__ == "__main__":
    import numpy as np

    # model = MLP(input_dim=(None, 5), output_dim=4, batch_size=64, n_hidden=1, cash_initializer=tf.ones_initializer(),
    #             dropout=0.2)
    #
    # pred = model(np.zeros((64, 5)), training=True)
    # print(pred.shape)
    # pred = model(np.zeros((64, 5)), training=False)
    # print(pred.shape)

    input_ = tf.keras.layers.Input((10, 2))

    output = CashBias()(input_)
    exit()

    layer_1 = tf.keras.layers.LSTM(1)
    output_layer = tf.keras.layers.Dense(3)
    cash_bias = CashBias()

    output_no_cash = output_layer(layer_1(input_))
    model = tf.keras.models.Model(input_, output_no_cash)
    print(model.summary())

    output = cash_bias(output_no_cash)

    model = tf.keras.models.Model(input_, output)
    print(model.summary())

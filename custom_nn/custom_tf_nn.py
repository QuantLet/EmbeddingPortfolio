# https://www.tensorflow.org/tutorials/customization/custom_layers
# https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/keras/layers/recurrent.py#L1222
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
# from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


def _config_for_enable_caching_device(rnn_cell):
    """Return the dict config for RNN cell wrt to enable_caching_device field.
    Since enable_caching_device is a internal implementation detail for speed up
    the RNN variable read when running on the multi remote worker setting, we
    don't want this config to be serialized constantly in the JSON. We will only
    serialize this field when a none default value is used to create the cell.
    Args:
      rnn_cell: the RNN cell for serialize.
    Returns:
      A dict which contains the JSON config for enable_caching_device value or
      empty dict if the enable_caching_device value is same as the default value.
    """
    default_enable_caching_device = ops.executing_eagerly_outside_functions()
    if rnn_cell._enable_caching_device != default_enable_caching_device:
        return {'enable_caching_device': rnn_cell._enable_caching_device}
    return {}


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = array_ops.shape(inputs)[0]
        dtype = inputs.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            'batch_size and dtype cannot be None while constructing initial state: '
            'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))


class ARLayerCell(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        """
        # By default use cached variable under v2 mode, see b/143699808.
        if ops.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        """
        super(ARLayerCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        print('INPUT SHAPE: ', input_shape)
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

        print('Parameter shapes:')
        print(f'Kernel: {self.kernel.shape}')
        print(f'Recurrent Kernel: {self.recurrent_kernel.shape}')
        print(f'Bias: {self.bias.shape}')

    def call(self, inputs, prev_output, training=None):
        print(inputs.shape)
        print(self.kernel.shape)
        h = K.dot(inputs, self.kernel)
        print('i', inputs)
        print('h', h)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        output = h + K.dot(prev_output, self.recurrent_kernel)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        # config.update(_config_for_enable_caching_device(self))
        base_config = super(ARLayerCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        print('HERE')
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


# First, let's define a RNN Cell, as a layer subclass.

class MinimalRNNCell(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


if __name__ == "__main__":
    #https://adventuresinmachinelearning.com/policy-gradient-tensorflow-2/
    #https://keras.io/guides/customizing_what_happens_in_fit/

    # Let's use this cell in a RNN layer:

    cell = MinimalRNNCell(32)
    x = tf.keras.Input((None, 5))
    layer = tf.keras.layers.RNN(cell)
    y = layer(x)

    cell = MinimalRNNCell(32)
    inputs = np.array([[[1], [1]], [[1], [1]]]).astype(np.float32)
    print(inputs.shape)
    layer = tf.keras.layers.RNN(cell)
    output, new_state = layer(inputs)
    print(output)
    print(output.shape)
    print(new_state)
    print(new_state.shape)

    cells = [MinimalRNNCell(32), MinimalRNNCell(64)]

    inputs = np.array([[[1], [1]], [[1], [1]]]).astype(np.float32)
    print(inputs.shape)
    layer = tf.keras.layers.RNN(cells)
    output, new_state = layer(inputs)
    print(output)
    print(output.shape)
    print(new_state)
    print(new_state.shape)

    exit()

    # Here's how to use the cell to build a stacked RNN:

    cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
    x = tf.keras.Input((None, 5))
    layer = tf.keras.layers.RNN(cells)
    y = layer(x)


    exit()

    # layer = MyDenseLayer(10)
    # _ = layer(tf.zeros([10, 5]))  # Calling the layer `.builds` it.

    cell = ARLayerCell(1,
                        kernel_initializer='ones',
                        recurrent_initializer='ones',
                        bias_initializer='ones'
                        )

    """
      w_i = np.array([1.], dtype=np.float32)
    x_t = np.array([1.], dtype=np.float32)
    b_h = np.array([1.], dtype=np.float32)
    y_t_1 = np.array([1.], dtype=np.float32)
    w_r = np.array([1.], dtype=np.float32)
    o = tf.math.tanh(h_o)
    
    """


    def forward(x_t, y_t_1=np.array([1.], dtype=np.float32), w_i=np.array([1.], dtype=np.float32),
                b_h=np.array([1.], dtype=np.float32),
                w_r=np.array([1.], dtype=np.float32)):
        h = w_i * x_t + b_h
        h_o = h + w_r * y_t_1
        o = tf.math.tanh(h_o)

        return o


    o_1 = forward(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float32))
    print('OUTPUT 1: ', o_1)
    o_2 = forward(np.array([1.], dtype=np.float32), o_1)
    print('OUTPUT 2: ', o_2)
    # exit()
    tf.ones([2, 1])

    inputs = np.array([[1], [1]]).astype(np.float32)

    output, new_state = cell(inputs, tf.ones([1, 1]))
    print(output)
    print(output.shape)
    print(new_state)
    print(new_state.shape)


    print('######################################## HERE')
    cell = ARLayerCell(1,
                       kernel_initializer='ones',
                       recurrent_initializer='ones',
                       bias_initializer='ones'
                       )
    inputs = np.array([[[1], [1]], [[1], [1]]]).astype(np.float32)
    print(inputs.shape)
    #layer = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True, stateful=True)

    layer = tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(1,
                                                            kernel_initializer='ones',
                                                            recurrent_initializer='ones',
                                                            bias_initializer='ones'
                                                            ),
                              return_sequences=True,
                              return_state=True,
                              stateful=True)

    output, new_state = layer(inputs)

    print('Sequences')
    print(output)
    print(output.shape)

    print('States')
    print(new_state)
    print(new_state.shape)
    exit('DONE')

    ######################################## HERE
    """
    (2, 2, 1)
    Sequences
    tf.Tensor(
    [[[0.9950547 ]
      [0.99500567]]
    
     [[0.9993292 ]
      [0.9993283 ]]], shape=(2, 2, 1), dtype=float32)
    (2, 2, 1)
    States
    tf.Tensor(
    [[0.99500567]
     [0.9993283 ]], shape=(2, 1), dtype=float32)
    (2, 1)
    
    
    (2, 2, 1)
    Sequences
    tf.Tensor(
    [[[0.9640276 ]
      [0.99468684]]
    
     [[0.9950547 ]
      [0.9993226 ]]], shape=(2, 2, 1), dtype=float32)
    (2, 2, 1)
    States
    tf.Tensor(
    [[0.99468684]
     [0.9993226 ]], shape=(2, 1), dtype=float32)
    """

    print('################# KERAS #################')
    inputs = np.array([[1], [1]]).astype(np.float32)
    layer = tf.keras.layers.SimpleRNNCell(1,
                                          kernel_initializer='ones',
                                          recurrent_initializer='ones',
                                          bias_initializer='ones'
                                          )
    output, new_state = layer(inputs, tf.ones([1, 1]))

    print(output)
    print(output.shape)
    print(new_state)
    print(new_state.shape)

    print('##### RNN #####')
    inputs = np.array([[[1], [1]]]).astype(np.float32)
    print(inputs.shape)

    rnn = tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(1,
                                                            kernel_initializer='ones',
                                                            recurrent_initializer='ones',
                                                            bias_initializer='ones'
                                                            ),
                              return_sequences=True,
                              return_state=True,
                              stateful=True)
    output, new_state = rnn(inputs)
    print(output)
    print(output.shape)
    print(new_state)
    print(new_state.shape)


    print('##### Simple RNN #####')

    rnn = tf.keras.layers.SimpleRNN(1,
                                    kernel_initializer='ones',
                                    recurrent_initializer='ones',
                                    bias_initializer='ones',
                                    return_sequences=True,
                                    return_state=True)
    output, new_state = rnn(inputs)
    print(output)
    print(output.shape)
    print(new_state)
    print(new_state.shape)

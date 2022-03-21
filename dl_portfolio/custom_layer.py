import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec
import tensorflow_probability as tfp


class UncorrelatedFeaturesLayer(tf.keras.layers.Layer):
    def __init__(self, encoding_dim: int, weightage: float = 1., norm: str = '1/2', use_cov: bool = True, **kwargs):
        super(UncorrelatedFeaturesLayer, self).__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.use_cov = use_cov
        self.norm = norm
        self.m = None

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        x_centered = K.transpose(x_centered)
        covariance = tfp.stats.covariance(
            x_centered
        )

        return covariance

    def get_corr(self, x):
        x_centered_list = []
        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))
        x_centered = tf.stack(x_centered_list)
        x_centered = K.transpose(x_centered)
        correlation = tfp.stats.correlation(
            x_centered
        )

        return correlation

    def get_config(self):
        return {
            "encoding_dim": self.encoding_dim,
            "weightage": self.weightage,
            "norm": self.norm,
            "use_cov": self.use_cov
        }

    def uncorrelated_feature(self, x):
        if self.use_cov:
            self.m = self.get_covariance(x)
        else:
            self.m = self.get_corr(x)

        if self.encoding_dim <= 1:
            return 0.0
        else:
            output = K.sum(K.square(self.m - tf.math.multiply(self.m, tf.eye(self.encoding_dim)))) / 2
            if self.norm == '1':
                return output
            elif self.norm == '1/2':
                # avoid overweighting fat tails
                return K.sqrt(output)
            else:
                raise NotImplementedError("norm must be '1' or '1/2' ")

    def call(self, x):
        self.pen = self.weightage * self.uncorrelated_feature(x)
        self.add_loss(self.pen)
        return x


class DenseTied(tf.keras.layers.Dense):
    def __init__(self, units, tied_to, n_features=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if n_features is not None:
            assert n_features > 0
        self.tied_to = tied_to
        self.n_features = n_features
        super(DenseTied, self).__init__(units, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                        bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                        kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        if self.n_features:
            self.features_kernel = self.add_weight("kernel",
                                                   shape=[int(self.n_features), self.units],
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint,
                                                   dtype=self.dtype,
                                                   trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        if self.n_features:
            weights = tf.concat([K.transpose(self.tied_to.kernel), self.features_kernel], 0, name='concat')
            output = K.dot(inputs, weights)
        else:
            output = K.dot(inputs, K.transpose(self.tied_to.kernel))

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class TransposeDense(tf.keras.layers.Dense):
    # https://medium.com/@lmayrandprovencher/building-an-autoencoder-with-tied-weights-in-keras-c4a559c529a2
    def __init__(self, units, ref_layer, **kwargs):
        super(TransposeDense, self).__init__(units, **kwargs)
        self.ref_layer = ref_layer

    def build(self, input_shape):
        self.b = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            name='bias',
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint
        )

    def call(self, inputs):
        x = tf.matmul(inputs, self.ref_layer.weights[0], transpose_b=True)
        return self.activation(x + self.b)

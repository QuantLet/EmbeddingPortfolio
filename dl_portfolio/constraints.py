import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint
import tensorflow_probability as tfp
from tensorflow.python.ops import math_ops


class NonNegAndUnitNorm(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.
    Also available via the shortcut function `tf.keras.constraints.unit_norm`.
    Args:
    axis: integer, axis along which to calculate weight norms.
      For instance, in a `Dense` layer the weight matrix
      has shape `(input_dim, output_dim)`,
      set `axis` to `0` to constrain each weight vector
      of length `(input_dim,)`.
      In a `Conv2D` layer with `data_format="channels_last"`,
      the weight tensor has shape
      `(rows, cols, input_depth, output_depth)`,
      set `axis` to `[0, 1, 2]`
      to constrain the weights of each filter tensor of size
      `(rows, cols, input_depth)`.
    """

    def __init__(self, max_value=1.0, axis=0, max_dim=None, norm='l2'):
        self.axis = axis
        self.max_dim = max_dim
        self.norm = norm
        self.max_value = max_value

    def __call__(self, w):
        if self.max_dim is not None:
            assert self.axis == 0
            w_reg = w[:, :self.max_dim]
            w_reg = w_reg * math_ops.cast(math_ops.greater_equal(w_reg, 0.), K.floatx())
            w_reg = w_reg * math_ops.cast(math_ops.greater_equal(self.max_value, w_reg), K.floatx())

            output = w_reg / (K.epsilon() + K.sqrt(tf.reduce_sum(tf.square(w_reg), axis=self.axis, keepdims=True)))
            w = tf.concat([output, w[:, self.max_dim:]], axis=-1)
        else:
            # w = w * math_ops.cast(math_ops.greater_equal(w, 0.), K.floatx())
            # w = w * math_ops.cast(math_ops.greater_equal(self.max_value, w), K.floatx())
            w = K.clip(w, 0, self.max_value)
            if self.norm == 'l2':
                w = w / (K.epsilon() + K.sqrt(math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True)))
            elif self.norm == 'l1':
                w = w / (K.epsilon() + K.sqrt(math_ops.reduce_sum(w, axis=self.axis, keepdims=True)))
        return w

    def get_config(self):
        return {
            'axis': self.axis,
            'max_value': self.max_value,
            'max_dim': self.max_dim,
            'norm': self.norm
        }


class UncorrelatedFeaturesConstraint(Constraint):
    def __init__(self, encoding_dim: int, weightage: float = 1., norm: str = '1', use_cov: bool = True):
        raise NotImplementedError('You must use the Layer implementation')
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

        # random_perturb = tf.random.normal(tf.shape(x_centered), 0, 1e-6, dtype=tf.float32)
        # x_centered = x_centered + random_perturb

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

        # random_perturb = tf.random.normal(tf.shape(x_centered), 0, 1e-6, dtype=tf.float32)
        # x_centered = x_centered + random_perturb

        correlation = tfp.stats.correlation(
            x_centered
        )

        return correlation

    # Constraint penalty => Could we look at conditional covariance: on tail
    # Y
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

    def __call__(self, x):
        self.pen = self.weightage * self.uncorrelated_feature(x)
        return self.pen

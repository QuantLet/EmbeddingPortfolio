import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint


class WeightsOrthogonalityConstraint(Constraint):
    def __init__(self, encoding_dim, weightage=1.0, axis=0, max_dim=None):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        self.max_dim = max_dim

    def weights_orthogonality(self, w):
        if self.axis == 1:
            w = tf.transpose(w)
        if self.encoding_dim > 1:
            m = K.dot(K.transpose(w), w) - tf.eye(self.encoding_dim)
            return self.weightage * K.sqrt(K.sum(K.square(m)))
        else:
            m = K.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        if self.max_dim is not None:
            w_reg = w[:, :self.max_dim]
            m = self.weights_orthogonality(w_reg)
        else:
            m = self.weights_orthogonality(w)
        return m


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

    def __init__(self, axis=0, max_dim=None):
        self.axis = axis
        self.max_dim = max_dim

    def __call__(self, w):
        if self.max_dim is not None:
            assert self.axis == 0
            w_reg = w[:, :self.max_dim]
            non_neg = w_reg * tf.cast(tf.greater_equal(w_reg, 0.), K.floatx())
            output = non_neg / (K.epsilon() + K.sqrt(tf.reduce_sum(tf.square(non_neg), axis=self.axis, keepdims=True)))
            w = tf.concat([output, w[:, self.max_dim:]], axis=-1)
        else:
            non_neg = w * tf.cast(tf.greater_equal(w, 0.), K.floatx())
            w = non_neg / (K.epsilon() + K.sqrt(tf.reduce_sum(tf.square(non_neg), axis=self.axis, keepdims=True)))
        return w

    def get_config(self):
        return {'axis': self.axis}


class UncorrelatedFeaturesConstraint(Constraint):
    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / \
                     tf.cast(x_centered.get_shape()[0], tf.float32)

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self, x):
        covariance = self.get_covariance(x)
        if (self.encoding_dim <= 1):
            return 0.0
        else:
            output = K.sum(K.square(
                covariance - tf.math.multiply(covariance, tf.eye(self.encoding_dim))))
            return output

    def __call__(self, x):
        return self.weightage * self.uncorrelated_feature(x)

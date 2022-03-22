import tensorflow as tf
from tensorflow.keras import backend as K
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

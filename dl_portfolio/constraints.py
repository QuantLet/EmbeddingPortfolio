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
            w = w * math_ops.cast(math_ops.greater_equal(w, 0.), K.floatx())
            w = w * math_ops.cast(math_ops.greater_equal(self.max_value, w), K.floatx())
            if self.norm == 'l2':
                w = w / (K.epsilon() + K.sqrt(math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True)))
            elif self.norm == 'l1':
                w = w / (K.epsilon() + K.sqrt(math_ops.reduce_sum(w, axis=self.axis, keepdims=True)))
        return w

    def get_config(self):
        return {'axis': self.axis}


class OldUncorrelatedFeaturesConstraint(Constraint):
    def __init__(self, encoding_dim, weightage=1.0, use_cov=True, **kwargs):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.use_cov = use_cov

    def get_covariance(self, x):
        # print(x)
        x_centered_list = []
        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))
        x_centered = tf.stack(x_centered_list)
        # random_perturb = tf.random.normal(tf.shape(x_centered), 0, 1e-6, dtype=tf.float32)
        # x_centered = x_centered + random_perturb

        covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(x_centered.get_shape()[0], tf.float32)

        return covariance

    def get_correlation(self, x):
        raise NotImplementedError()

    # Constraint penalty
    def uncorrelated_feature(self, x):
        if self.use_cov:
            self.covariance = self.get_covariance(x)
        else:
            self.covariance = self.get_correlation(x)

        if self.encoding_dim <= 1:
            return 0.0
        else:
            # lower triangular part minus diagonal
            output = K.square(tf.linalg.band_part(self.covariance, -1, 0) - tf.linalg.band_part(self.covariance, 0, 0))
            output = K.sum(output)
            # output = K.mean(output)
            # K.sum(K.square(covariance - tf.math.multiply(covariance, tf.eye(self.encoding_dim))))

            return output

    def __call__(self, x):
        self.pen = self.weightage * self.uncorrelated_feature(x)
        return self.pen

    # def get_config(self):  # required class method
    #     return {"uncorr": float(K.get_value(self.pen)),
    #             "covariance": float(K.get_value(self.covariance))}


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

    # def get_config(self):  # required class method
    #     return {"uncorr": float(K.get_value(self.pen)),
    #             "covariance": float(K.get_value(self.m))}


class TailUncorrelatedFeaturesConstraint(Constraint):
    def __init__(self, encoding_dim: int, q: float, side: str, weightage: float = 1., norm: str = '1',
                 use_cov: bool = True):
        raise NotImplementedError('You must create activaity reg layer')
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.use_cov = use_cov
        self.norm = norm
        self.m = None
        self.q = q
        self.side = side
        self.num_quantiles = int(100 / (self.q * 100))
        assert side in ['left', 'right']

    def get_quantiles(self, x):
        quantiles = []
        for i in range(self.encoding_dim):
            quantiles.append(tfp.stats.quantiles(x[:, i], num_quantiles=self.num_quantiles)[1])
        return quantiles

    def get_covariance(self, x):
        x_centered_list = []
        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        x_centered = K.transpose(x_centered)

        # random_perturb = tf.random.normal(tf.shape(x_centered), 0, 1e-6, dtype=tf.float32)
        # x_centered = x_centered + random_perturb

        quantiles = self.get_quantiles(x_centered)
        x_centered_list = []
        for i in range(self.encoding_dim):
            if self.side == 'left':
                x_centered_list.append(tf.clip_by_value(x_centered[:, i], -1e12, quantiles[i]))
            elif self.side == 'right':
                x_centered_list.append(tf.clip_by_value(x_centered[:, i], quantiles[i], 1e12))
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

        # random_perturb = tf.random.normal(tf.shape(x_centered), 0, 1e-6, dtype=tf.float32)
        # x_centered = x_centered + random_perturb

        correlation = tfp.stats.correlation(
            K.transpose(x_centered)
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

    # def get_config(self):  # required class method
    #     return {"uncorr": float(K.get_value(self.pen)),
    #             "covariance": float(K.get_value(self.m))}


class PositiveSkewnessConstraint(Constraint):
    def __init__(self, encoding_dim, normalize=True, weightage=1.0, norm='1'):
        raise NotImplementedError('You must create activaity reg layer')
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.normalize = normalize
        self.norm = norm
        self.m = None

    def get_coskew(self, x):
        x = K.transpose(x)
        x = x - K.mean(x, axis=1, keepdims=True)
        num_obs = x.shape[1]
        num_assets = x.shape[0]

        operator_1 = tf.linalg.LinearOperatorFullMatrix(K.transpose(x))
        operator_2 = tf.linalg.LinearOperatorFullMatrix(K.transpose(x))
        operator = tf.linalg.LinearOperatorKronecker([operator_1, operator_2])
        kron_mat = operator.to_dense()[::num_obs + 1, :]
        coskew = K.dot(x, kron_mat)

        if self.normalize:
            std_mat = tf.reshape(K.std(x, axis=1), (-1, 1))
            operator_1 = tf.linalg.LinearOperatorFullMatrix(K.transpose(std_mat))
            operator_2 = tf.linalg.LinearOperatorFullMatrix(K.transpose(std_mat))
            operator = tf.linalg.LinearOperatorKronecker([operator_1, operator_2])

            kron_mat = operator.to_dense()[::num_assets + 1, :]
            std_mat = K.dot(std_mat, kron_mat)
            coskew = coskew / std_mat

        return coskew

    def skewed_features(self, x):
        """
        Idea: penalize for coskewness near 0: get two clusters: one of highly positive skew, one of highly negative skew
        :param x:
        :return:
        """
        self.m = self.get_coskew(x)
        self.m = 1 / self.m

        output = []
        for i in range(self.encoding_dim):
            block = self.m[:, i * self.encoding_dim:(i + 1) * self.encoding_dim] \
                    - tf.math.multiply(self.m[:, i * self.encoding_dim:(i + 1) * self.encoding_dim],
                                       tf.eye(self.encoding_dim))
            block = K.sum(K.square(block))
            output.append(block)

        if self.norm == '1':
            pass
        elif self.norm == '1/2':
            output = [K.sqrt(out) for out in output]

        return K.sum(output)  # K.reduce_mean ?

    def positive_skewed_features(self, x):
        """
        Idea: penalized for negative coskewness
        :param x:
        :return:
        """
        self.m = self.get_coskew(x)
        output = []
        for i in range(self.encoding_dim):
            block = self.m[:, i * self.encoding_dim:(i + 1) * self.encoding_dim] \
                    - tf.math.multiply(self.m[:, i * self.encoding_dim:(i + 1) * self.encoding_dim],
                                       tf.eye(self.encoding_dim))
            block = tf.clip_by_value(block, -1e6, 1e6)
            block = tf.clip_by_value(- block, 0, 1e6)
            block = K.sum(block) / 2  # tf.reduce_mean(block)  # K.sum(K.square(block)) / 2  # since symmetric matrix
            output.append(block)

        if self.norm == '1':
            pass
        elif self.norm == '1/2':
            output = [K.sqrt(out) for out in output]

        return tf.reduce_mean(output)  # K.mean(output) # K.sum(output)  # K.reduce_mean ?

    def __call__(self, x):
        self.pen = self.weightage * self.positive_skewed_features(x)
        return self.pen


class PositiveSkewnessUncorrConstraint(PositiveSkewnessConstraint, UncorrelatedFeaturesConstraint):
    def __init__(self, encoding_dim, coske_weightage=1.0, uncorr_weightage=1.0):
        raise NotImplementedError('You must create activaity reg layer')

        super(PositiveSkewnessConstraint).__init__(encoding_dim, normalize=False, weightage=coske_weightage,
                                                   norm='1')
        super(UncorrelatedFeaturesConstraint).__init__(encoding_dim, weightage=uncorr_weightage, norm='1/2')

    def __call__(self, x):
        coskew = self.coske_weightage * self.positive_skewed_features(x)
        uncorr = self.uncorr_weightage * self.uncorrelated_feature(x)
        self.pen = (coskew + uncorr) / 2
        return self.pen

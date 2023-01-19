import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint
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

    def __init__(self, max_value=1.0, axis=0, max_dim=None, norm="l2"):
        self.axis = axis
        self.max_dim = max_dim
        self.norm = norm
        self.max_value = max_value

    def __call__(self, w):
        if self.max_dim is not None:
            assert self.axis == 0
            w_reg = w[:, : self.max_dim]
            w_reg = w_reg * math_ops.cast(
                math_ops.greater_equal(w_reg, 0.0), K.floatx()
            )
            w_reg = w_reg * math_ops.cast(
                math_ops.greater_equal(self.max_value, w_reg), K.floatx()
            )

            output = w_reg / (
                K.epsilon()
                + K.sqrt(
                    tf.reduce_sum(
                        tf.square(w_reg), axis=self.axis, keepdims=True
                    )
                )
            )
            w = tf.concat([output, w[:, self.max_dim :]], axis=-1)
        else:
            # w = w * math_ops.cast(math_ops.greater_equal(w, 0.), K.floatx())
            # w = w * math_ops.cast(math_ops.greater_equal(self.max_value, w),
            # K.floatx())
            w = K.clip(w, 0, self.max_value)
            if self.norm == "l2":
                w = w / (
                    K.epsilon()
                    + K.sqrt(
                        math_ops.reduce_sum(
                            math_ops.square(w), axis=self.axis, keepdims=True
                        )
                    )
                )
            elif self.norm == "l1":
                w = w / (
                    K.epsilon()
                    + K.sqrt(
                        math_ops.reduce_sum(w, axis=self.axis, keepdims=True)
                    )
                )
        return w

    def get_config(self):
        return {
            "axis": self.axis,
            "max_value": self.max_value,
            "max_dim": self.max_dim,
            "norm": self.norm,
        }

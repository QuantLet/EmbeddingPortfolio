import tensorflow as tf
import tensorflow.keras.backend as K
from typing import Optional, Dict
from tensorflow.keras.regularizers import Regularizer


class WeightsOrthogonality(Regularizer):
    def __init__(
        self,
        encoding_dim: int,
        weightage: str = 1.0,
        axis: int = 0,
        max_dim: Optional[int] = None,
        regularizer: Optional[Dict] = None,
    ):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        self.max_dim = max_dim
        self.regularizer = regularizer
        if self.regularizer:
            if self.regularizer.get("name") == "l2":
                self.regularizer = tf.keras.regularizers.l2(
                    **self.regularizer["params"]
                )
            elif self.regularizer.get("name") == "l1":
                self.regularizer = tf.keras.regularizers.l1(
                    **self.regularizer["params"]
                )
            elif self.regularizer.get("name") == "l1_l2":
                self.regularizer = tf.keras.regularizers.l1_l2(
                    **self.regularizer["params"]
                )
            else:
                raise NotImplementedError()

    def weights_orthogonality(self, w):
        if self.axis == 1:
            w = tf.transpose(w)
        if self.encoding_dim > 1:
            m = K.dot(K.transpose(w), w) - tf.eye(self.encoding_dim)
            return self.weightage * K.sqrt(K.sum(K.square(m)))
            # return self.weightage * K.sum(K.square(m))
        else:
            m = K.sum(w ** 2) - 1.0
            return m

    def __call__(self, w):
        regularization = K.constant(0.0, dtype=w.dtype)
        if self.max_dim:
            w_reg = w[:, : self.max_dim]

            if self.regularizer:
                regularization += self.regularizer(w_reg)
            regularization += self.weights_orthogonality(w_reg)
        else:
            if self.regularizer:
                regularization += self.regularizer(w)
            regularization += self.weights_orthogonality(w)
        return regularization

    def get_config(self):
        return {
            "encoding_dim": self.encoding_dim,
            "weightage": self.weightage,
            "axis": self.axis,
            "max_dim": self.max_dim,
            "regularizer": self.regularizer,
        }

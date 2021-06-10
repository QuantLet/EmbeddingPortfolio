import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from tensorflow.python.ops import math_ops


def weighted_mse(y_true, y_pred, weights):
    diff = tf.math.squared_difference(y_pred, y_true)  # squared difference
    weighted_diff = tf.math.multiply(weights, diff)  # weights * diff
    loss = K.mean(weighted_diff, axis=-1)  # mean over last dimension
    return loss


def weighted_mae(y_true, y_pred, weights):
    diff = tf.math.abs(y_pred - y_true)  # squared difference
    weighted_diff = tf.math.multiply(weights, diff)  # weights * diff
    loss = K.mean(weighted_diff, axis=-1)  # mean over last dimension
    return loss

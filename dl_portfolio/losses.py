import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from tensorflow.python.ops import math_ops


def weighted_mse(y_true, y_pred, weights=None):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    diff = tf.math.squared_difference(y_pred, y_true)  # squared difference
    if weights is not None:
        weights = tf.cast(weights, dtype=tf.float32)
        diff = tf.math.multiply(weights, diff)  # weights * diff
    loss = K.mean(diff, axis=-1)  # mean over last dimension: mean of individual loss for each target
    loss = K.mean(loss, axis=0)  # mean over batch
    return loss


def weighted_mae(y_true, y_pred, weights=None):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    diff = tf.math.abs(y_pred - y_true)  # squared difference
    if weights is not None:
        weights = tf.cast(weights, dtype=tf.float32)
        diff = tf.math.multiply(weights, diff)  # weights * diff
    loss = K.mean(diff, axis=-1)  # mean over last dimension: mean of individual loss for each target
    loss = K.mean(loss, axis=0)  # mean over batch
    return loss

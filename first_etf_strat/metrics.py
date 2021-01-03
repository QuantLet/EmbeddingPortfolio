import tensorflow as tf
import numpy as np


def portfolio_returns(prediction: tf.Tensor, next_returns: tf.Tensor, initial_position: tf.Tensor, trading_fee: float = 0.,
                      cash_bias: bool = True):
    ret = tf.math.reduce_sum(next_returns * prediction, axis=-1)
    if cash_bias:
        positions = tf.concat([initial_position, prediction[:, :-1]], 0)
    else:
        positions = tf.concat([initial_position, prediction], 0)
    transaction_cost = trading_fee * tf.math.reduce_sum(np.abs(positions[1:] - positions[:-1]), axis=1)
    return ret, ret - transaction_cost


def sharpe_ratio(port_returns: tf.Tensor, benchmark: float = 0.0093):
    # take log maybe ??
    sr = - tf.reduce_mean(port_returns - tf.constant(benchmark, dtype=tf.float32)) / (
            tf.math.reduce_std(port_returns - tf.constant(benchmark, dtype=tf.float32)) + 10e-12)
    # sr = tf.math.reduce_variance(ret - tf.constant(benchmark, dtype=tf.float32)) / (tf.math.square(tf.reduce_mean(ret - tf.constant(benchmark, dtype=tf.float32))) + 10e-12)
    return sr

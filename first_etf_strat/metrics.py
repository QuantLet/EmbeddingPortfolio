import tensorflow as tf
import numpy as np


def portfolio_returns(prediction: tf.Tensor, next_returns: tf.Tensor, initial_position: tf.Tensor,
                      trading_fee: float = 0.,
                      cash_bias: bool = True):
    ret = tf.math.reduce_sum(next_returns * prediction, axis=-1)
    if cash_bias:
        positions = tf.concat([initial_position, prediction[:, :-1]], 0)
    else:
        positions = tf.concat([initial_position, prediction], 0)
    transaction_cost = trading_fee * tf.math.reduce_sum(np.abs(positions[1:] - positions[:-1]), axis=1)
    return ret, ret - transaction_cost


def sharpe_ratio(port_returns: tf.Tensor, benchmark: tf.constant(0.0093, dtype=tf.float32),
                 annual_period: tf.constant = tf.constant(0, dtype=tf.float32)):
    # take log maybe ??
    excess_return = port_returns - benchmark
    if annual_period != 0:
        sr = - annual_period / np.sqrt(annual_period) * tf.reduce_mean(
            tf.math.log(excess_return + 1.)) / (
                 tf.math.reduce_std(tf.math.log(excess_return + 1.) + 1e-12))
    else:
        sr = - tf.reduce_mean(
            tf.math.log(excess_return + 1.)) / (
                 tf.math.reduce_std(tf.math.log(excess_return + 1.) + 1e-12))
    # sr = tf.math.reduce_variance(ret - tf.constant(benchmark, dtype=tf.float32)) / (tf.math.square(tf.reduce_mean(ret - tf.constant(benchmark, dtype=tf.float32))) + 10e-12)
    return sr


def diff_sr(prev_A, prev_B, ret):
    return (prev_B * (ret - prev_A) - 0.5 * prev_A * (ret ** 2 - prev_B)) / ((prev_B - prev_A ** 2) ** (3 / 2))

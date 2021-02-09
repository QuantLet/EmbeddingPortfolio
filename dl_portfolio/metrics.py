import tensorflow as tf
import numpy as np


def portfolio_returns(prediction: tf.Tensor, next_returns: tf.Tensor, initial_position: tf.Tensor,
                      trading_fee: float = 0., cash_bias: bool = True):
    ret = tf.math.reduce_sum(next_returns * prediction, axis=-1)
    if cash_bias:
        positions = tf.concat([initial_position, prediction[:, :-1]], 0)
    else:
        positions = tf.concat([initial_position, prediction], 0)

    # Moody et al (1998) https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.8437&rep=rep1&type=pdf
    ret_fee = tf.math.reduce_sum((next_returns + 1) * prediction, axis=-1) * (
            1 - trading_fee * tf.math.reduce_sum(np.abs(positions[1:] - positions[:-1]), axis=1))
    ret_fee = ret_fee - 1

    return ret, ret_fee


def penalized_volatility_returns(port_returns: tf.Tensor,
                                 benchmark: tf.constant = tf.constant(0.0093, dtype=tf.float32),
                                 alpha: tf.constant = tf.constant(1., dtype=tf.float32)):
    # take log maybe ??
    excess_return = port_returns - benchmark
    # loss = - (tf.reduce_mean(excess_return) - alpha * tf.math.sqrt(
    #     tf.reduce_mean(excess_return ** 2) - tf.reduce_mean(excess_return) ** 2))
    loss = - (tf.reduce_mean(excess_return) - alpha * tf.math.reduce_std(excess_return))
    return loss


def sharpe_ratio(port_returns: tf.Tensor, benchmark: tf.constant = tf.constant(0.0093, dtype=tf.float32),
                 annual_period: tf.constant = tf.constant(0, dtype=tf.float32)):
    # take log maybe ??
    excess_return = port_returns - benchmark
    if annual_period != 0:
        sr = - annual_period / np.sqrt(annual_period) * tf.reduce_mean(excess_return) / (
            tf.math.sqrt(tf.reduce_mean(excess_return ** 2) - tf.reduce_mean(excess_return) ** 2 + 1e-12))
    else:
        sr = - tf.reduce_mean(excess_return) / (
            tf.math.sqrt(tf.reduce_mean(excess_return ** 2) - tf.reduce_mean(excess_return) ** 2 + 1e-12))
    return sr


def diff_sr(prev_A, prev_B, ret):
    return (prev_B * (ret - prev_A) - 0.5 * prev_A * (ret ** 2 - prev_B)) / ((prev_B - prev_A ** 2) ** (3 / 2))

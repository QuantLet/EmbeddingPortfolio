import tensorflow as tf
import numpy as np


# TODO: what about edgeworth expansion for sharpe ratio => estimate variance with expansion, maybe more differentiable or look at probabilistic sharpe ratio
def portfolio_returns(prediction: tf.Tensor, next_returns: tf.Tensor, initial_position: tf.Tensor,
                      trading_fee: float = 0., cash_bias: bool = True):
    ret = tf.math.reduce_sum(next_returns * prediction, axis=-1)

    if cash_bias:
        positions = tf.concat([initial_position, prediction[:, :-1]], 0)
    else:
        positions = tf.concat([initial_position, prediction], 0)

    # Moody et al (1998) https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.8437&rep=rep1&type=pdf
    ret_fee = ret - trading_fee * tf.math.reduce_sum(np.abs(positions[1:] - positions[:-1]), axis=1)

    return ret, ret_fee


# TODO: what about edgeworth expansion for sharpe ratio => estimate variance with expansion, maybe more differentiable or look at probabilistic sharpe ratio
def np_portfolio_returns(prediction: np.array, next_returns: np.array, initial_position: np.array,
                         trading_fee: float = 0., cash_bias: bool = True):
    ret = np.sum(next_returns * prediction, axis=1)

    if cash_bias:
        positions = np.concatenate([initial_position, prediction[:, :-1]], 0)
    else:
        positions = np.concatenate([initial_position, prediction], 0)

    # Moody et al (1998) https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.8437&rep=rep1&type=pdf
    ret_fee = ret - trading_fee * np.sum(np.abs(positions[1:] - positions[:-1]), axis=1)

    return ret, ret_fee


def penalized_volatility_returns(port_returns: tf.Tensor,
                                 benchmark: tf.constant = tf.constant(0.0093, dtype=tf.float32),
                                 alpha: tf.constant = tf.constant(1., dtype=tf.float32)):
    excess_return = port_returns - benchmark
    # loss = - (tf.reduce_mean(excess_return) - alpha * tf.math.sqrt(
    #     tf.reduce_mean(excess_return ** 2) - tf.reduce_mean(excess_return) ** 2))
    loss = - (tf.reduce_mean(excess_return) - alpha * tf.math.reduce_std(excess_return))
    return loss


def sharpe_ratio(port_returns: tf.Tensor, benchmark: tf.constant = tf.constant(0.0093, dtype=tf.float32),
                 annual_period: tf.constant = tf.constant(0, dtype=tf.float32), epsilon: float = 1e-6):
    # take log maybe ??
    excess_return = port_returns - benchmark
    if annual_period != 0:
        sr = - annual_period / np.sqrt(annual_period) * tf.reduce_mean(excess_return) / (
            tf.math.sqrt(tf.reduce_mean(excess_return ** 2) - tf.reduce_mean(excess_return) ** 2 + epsilon))
    else:
        sr = - tf.reduce_mean(excess_return) / (
            tf.math.sqrt(tf.reduce_mean(excess_return ** 2) - tf.reduce_mean(excess_return) ** 2 + epsilon))
    return sr

def average_return(port_returns: tf.Tensor, benchmark: tf.constant = tf.constant(0.0093, dtype=tf.float32)):
    # take log maybe ??
    excess_return = port_returns - benchmark
    return - tf.math.reduce_mean(excess_return)


def volatility(port_returns: tf.Tensor, benchmark: tf.constant = tf.constant(0.0093, dtype=tf.float32)):
    # take log maybe ??
    excess_return = port_returns - benchmark
    return tf.math.sqrt(tf.reduce_mean(excess_return ** 2) - tf.reduce_mean(excess_return) ** 2)


def cum_return(port_returns: tf.Tensor, benchmark: tf.constant = tf.constant(0.0093, dtype=tf.float32)):
    # take log maybe ??
    excess_return = port_returns - benchmark
    return - tf.math.reduce_prod(excess_return + 1.)


def diff_sr(prev_A, prev_B, ret):
    return (prev_B * (ret - prev_A) - 0.5 * prev_A * (ret ** 2 - prev_B)) / ((prev_B - prev_A ** 2) ** (3 / 2))

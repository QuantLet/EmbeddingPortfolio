import tensorflow as tf
import numpy as np


def log_avg_ret(model, x, returns, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x)  # model(x, training=training)

    return - tf.reduce_mean(tf.math.log(y_ * (returns + 1)))


def excess_log_avg_ret(model, x, returns, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x)  # model(x, training=training)

    return - tf.reduce_mean(tf.math.log(y_ * (returns + 1)) - tf.math.log((returns + 1)))


def avg_ret(model, x, returns, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x)  # model(x, training=training)

    return - tf.reduce_mean(y_ * returns)


def sharpe_ratio(model, x, returns, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x)  # model(x, training=training)

    #return tf.math.reduce_variance(y_ * returns) / ((tf.reduce_mean(y_ * returns) + 10e-12)**2)
    return - tf.reduce_mean(y_ * returns) / (tf.math.reduce_std(y_ * returns) + 10e-12)

def volatility(model, x, returns, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x)  # model(x, training=training)
    # print(np.sum(np.isnan(returns)))
    # print(np.sum(np.isnan(y_)))
    # print(tf.math.reduce_std(y_ * returns))
    return tf.math.reduce_std(y_ * returns)


def cum_ret(model, x, returns, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x)  # model(x, training=training)

    return - tf.reduce_sum(y_ * returns)

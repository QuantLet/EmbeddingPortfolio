import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def feature_sensitivity(model, data, loss_func, features=None):
    sensitivity = []
    for i in range(data.shape[0]):
        x = tf.Variable(data[i].reshape(1, -1), dtype=tf.float32)
        if features is not None:
            f = tf.Variable(features[i].reshape(1, -1), dtype=tf.float32)
            inputs = [x, f]
        else:
            inputs = x
        with tf.GradientTape() as tape:
            y = model(inputs)
            loss = loss_func(x, y)

        dloss_dx = tape.gradient(loss, x)
        dloss_dx = dloss_dx.numpy()[0]
        s = np.abs(dloss_dx) / np.max(np.abs(dloss_dx))
        sensitivity.append(s)
    sensitivity = np.array(sensitivity)

    return sensitivity


def plot_sensitivity(sensitivity: pd.DataFrame, save_path=None, show=False, max_xticks=50):
    plt.figure(figsize=(20, 10))
    c = plt.imshow(sensitivity.T,
                   aspect="auto",
                   cmap="gray",
                   interpolation="nearest")
    plt.colorbar(c)

    xtickslabels = list(sensitivity.index)
    xticks = list(range(0, len(sensitivity), len(sensitivity) // max_xticks + 1))

    xtickslabels = np.array(xtickslabels)[xticks].tolist()
    xtickslabels = [l.strftime("%Y-%m-%d") for l in xtickslabels]
    plt.xticks(xticks, xtickslabels, rotation=45)

    ytickslabels = list(sensitivity.columns)
    plt.yticks(range(len(ytickslabels)), ytickslabels)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

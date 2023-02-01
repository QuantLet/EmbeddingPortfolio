import os
import time

import fastcluster
import scipy.cluster.hierarchy as sch
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tensorflow as tf
from scipy.cluster import hierarchy
from statsmodels.stats.correlation_tools import corr_nearest


def generate_batch_corr_mat(n_batch, n_assets, generator=None, seed=None,
                            nearest=True):
    if seed:
        tf.random.set_seed(seed)
    noise = tf.random.normal([n_batch, 100])
    # Sample with gan
    if generator is None:
        generator = tf.keras.models.load_model(
            "prado/corrgan-models/saved_model/generator_100d")
    corr_mats = generator(noise, training=False)

    # Make correlation matrix
    a, b = np.triu_indices(n_assets, k=1)
    batch = {}
    for i in range(n_batch):
        print(n_batch - i)
        corr_mat = np.array(corr_mats[i, :, :, 0])
        # set diag to 1
        np.fill_diagonal(corr_mat, 1)
        # symmetrize
        corr_mat[b, a] = corr_mat[a, b]

        if nearest:
            # nearest corr
            nearest_corr_mat = corr_nearest(corr_mat)
            # set diag to 1
            np.fill_diagonal(nearest_corr_mat, 1)
            # symmetrize
            nearest_corr_mat[b, a] = nearest_corr_mat[a, b]
        else:
            nearest_corr_mat = corr_mat.copy()

        # arrange with hierarchical clustering
        dist = 1 - nearest_corr_mat
        dim = len(dist)
        tri_a, tri_b = np.triu_indices(dim, k=1)
        t1 = time.time()
        Z = fastcluster.linkage(dist[tri_a, tri_b], method='ward')
        permutation = hierarchy.leaves_list(
            hierarchy.optimal_leaf_ordering(Z, dist[tri_a, tri_b]))

        dend = hierarchy.dendrogram(Z)
        t2 = time.time()
        print(t2-t1)
        corr_mat = nearest_corr_mat[permutation, :][:, permutation]

        col_cluster_mapper = dend["leaves_color_list"]
        cluster_mapper = pd.DataFrame(
            col_cluster_mapper, columns=["cluster"]
        ).reset_index(drop=False).groupby("cluster")[
            "index"].unique().to_dict()
        cluster_mapper = {c: cluster_mapper[c].tolist() for c in
                          cluster_mapper}

        batch[i] = [corr_mat, col_cluster_mapper, cluster_mapper]

    return batch


def make_corr_mat_from_gan(corr_mat):
    # Make correlation matrix
    a, b = np.triu_indices(corr_mat.shape[0], k=1)
    # set diag to 1
    np.fill_diagonal(corr_mat, 1)
    # symmetrize
    corr_mat[b, a] = corr_mat[a, b]
    # nearest corr
    nearest_corr_mat = corr_nearest(corr_mat)
    # set diag to 1
    np.fill_diagonal(nearest_corr_mat, 1)
    # symmetrize
    nearest_corr_mat[b, a] = nearest_corr_mat[a, b]

    # arrange with hierarchical clustering
    dist = 1 - nearest_corr_mat
    dim = len(dist)
    tri_a, tri_b = np.triu_indices(dim, k=1)
    Z = fastcluster.linkage(dist[tri_a, tri_b], method='ward')
    permutation = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(Z, dist[tri_a, tri_b]))

    dend = hierarchy.dendrogram(Z, no_plot=True)
    corr_mat = nearest_corr_mat[permutation, :][:, permutation]
    col_cluster_mapper = dend["leaves_color_list"]
    cluster_mapper = pd.DataFrame(
        col_cluster_mapper, columns=["cluster"]
    ).reset_index(drop=False).groupby("cluster")["index"].unique().to_dict()
    cluster_mapper = {c: cluster_mapper[c].tolist() for c in cluster_mapper}

    return corr_mat, col_cluster_mapper, cluster_mapper


def generate_corr_mat(n_assets, generator=None, seed=None):
    """
    Source: https://gmarti.gitlab.io/qfin/2020/08/11/corrgan-pretrained-models.html

    :param n_assets:
    :param seed:
    :return:
    """
    if seed:
        tf.random.set_seed(seed)
    noise = tf.random.normal([1, 100])
    # Sample with gan
    if generator is None:
        generator = tf.keras.models.load_model(
            f"prado/corrgan-models/saved_model/generator_{n_assets}d")
    corr_mat = generator(noise, training=False)

    corr_mat = np.array(corr_mat[0, :, :, 0])

    return make_corr_mat_from_gan(corr_mat)


def generate_data_corrgan(n_assets, n_obs, generator=None, min_sigma=0.0025,
                          max_sigma=0.015, seed=None):
    """
    - First generate a correlation matrix with CorrGAN with n_assets
    - Generate n_assets volatilies between 0.0025 and 0.015 to get a
    covariance matrix
    - Generate (n_obs, n_assets) returns from log-normal with the generated
    covariance matrix
    :param n_assets:
    :param n_obs:
    :param min_sigma:
    :param max_sigma:
    :param seed:
    :return:
    """
    corr_mat, col_cluster_mapper, cluster_mapper = generate_corr_mat(
        n_assets, generator, seed=seed)
    sigmas = np.random.uniform(min_sigma, max_sigma, n_assets).reshape(n_assets, 1)
    den = np.dot(sigmas, sigmas.T)
    cov_mat = corr_mat * den
    returns = np.exp(np.random.multivariate_normal(
        np.zeros(n_assets), cov_mat, size=n_obs)) - 1

    return returns, col_cluster_mapper, cluster_mapper


def generate_returns_from_gan(gan_mat, n_obs, min_sigma=0.0025,
                              max_sigma=0.015):
    n_assets = gan_mat.shape[0]
    corr_mat, col_cluster_mapper, cluster_mapper = make_corr_mat_from_gan(
        gan_mat)
    sigmas = np.random.uniform(min_sigma, max_sigma, n_assets).reshape(
        n_assets, 1)
    den = np.dot(sigmas, sigmas.T)
    cov_mat = corr_mat * den
    returns = np.exp(np.random.multivariate_normal(
        np.zeros(n_assets), cov_mat, size=n_obs)) - 1

    return returns, col_cluster_mapper, cluster_mapper

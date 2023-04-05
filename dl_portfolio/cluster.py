import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.cluster import KMeans
from typing import Dict, List
from scipy.spatial.distance import squareform
from fastcluster import linkage


from gap_statistic import OptimalK
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler

from dl_portfolio.data import bb_resample_sample
from dl_portfolio.logger import LOGGER
from dl_portfolio.nmf.convex_nmf import ConvexNMF


def convex_nmf_cluster(X, k):
    nmf = ConvexNMF(n_components=k, norm_G="l2")
    nmf.fit(X.T)
    centers = nmf.transform(X.T).T
    labels = np.argmax(nmf.G, axis=1)

    return centers, labels


def silhouette(X, k):
    _, labels = convex_nmf_cluster(X.T, k)
    return metrics.silhouette_score(euclidean_distances(X.T), labels)


def cluster_selection_curve(
    data: np.ndarray,
    criterion: str = "silhouette",
    p_range=list(range(1, 21)),
    resample: bool = True,
    block_length: int = 60,
):
    min_p = 1
    assert min(p_range) >= min_p
    if resample:
        data, _ = bb_resample_sample(data, block_length=block_length)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    if criterion == "silhouette":
        param_perf = [silhouette(X, k) for k in p_range]
    else:
        raise NotImplementedError(criterion)

    return param_perf


def bb_silhouette(bb_criteria, alpha=0.05, plot=False, savepath=None,
                  show=False,  min_p=2):
    assert len(bb_criteria.shape) == 2
    mean_ = np.mean(bb_criteria, axis=1)
    lower_b = np.quantile(bb_criteria, alpha, axis=1)
    upper_b = np.quantile(bb_criteria, 1-alpha, axis=1)
    best_p = np.argmax((np.roll(mean_, 1) >= lower_b)[1:]) + min_p

    if plot or savepath is not None or show:
        plt.plot(mean_)
        plt.fill_between(range(len(bb_criteria)), lower_b, upper_b,
                         alpha=0.2, color="grey")
        plt.xticks(range(len(bb_criteria)),
                   range(min_p, len(bb_criteria) + 1))
        plt.scatter(best_p - min_p, mean_[best_p - min_p], s=40, c="red")
        if savepath:
            plt.savefig(savepath, transparent=True, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    return best_p, mean_, lower_b, upper_b


def get_optimal_p_silhouette(data: np.ndarray, p_range: List,
                             n_exp: int = 1000, n_jobs: int = os.cpu_count(),
                             criterion: str = "silhouette",
                             resample: bool = True, block_length: int = 60,
                             **kwargs):
    if n_jobs <= 1:
        bb_criteria = []
        for i in range(n_exp):
            if i % 10 == 0:
                LOGGER.info(f"Steps to go: {n_exp - i}")
            criteria = cluster_selection_curve(
                data, p_range=p_range, criterion=criterion,
                resample=resample, block_length=block_length
            )
            bb_criteria.append(criteria)
    else:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            bb_criteria = _parallel_pool(
                delayed(cluster_selection_curve)(
                    data, p_range=p_range, criterion=criterion,
                    resample=resample, block_length=block_length
                )
                for i in range(n_exp)
            )

    bb_criteria = np.array(bb_criteria).T
    min_p = min(p_range)
    best_p, _, _, _ = bb_silhouette(bb_criteria, min_p=min_p, **kwargs)

    return best_p


def get_optimal_p_gap(
    X, clusterer=None, n_refs=3, cluster_array=range(3, 10)
):
    """
    cf Estimating the number of clusters in a data set via the gap
    statistic, (Tibshirani et al.)
    k̂  = smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}.

    """
    if clusterer is None:
        clusterer = convex_nmf_cluster
    optimalk = OptimalK(clusterer=clusterer)

    optimalk(X.T, n_refs=n_refs, cluster_array=cluster_array)

    decision = (
        optimalk.gap_df.gap_value
        - (optimalk.gap_df.gap_value - optimalk.gap_df.sk).shift(-1)
    ) >= 0
    decision = decision & (
        (optimalk.gap_df.gap_value - optimalk.gap_df.gap_value.shift(-1)) >= 0
    )
    n_clusters_i = np.argmax(decision)
    n_clusters = int(optimalk.gap_df.n_clusters.iloc[n_clusters_i])

    if n_clusters == 1:
        LOGGER.warning("No cluster found!!")

    return n_clusters, optimalk


def get_cluster_assignment(base_dir, cluster_names):
    models = os.listdir(base_dir)
    paths = [
        f"{base_dir}/{d}"
        for d in models
        if os.path.isdir(f"{base_dir}/{d}") and d[0] != "."
    ]
    n_folds = os.listdir(paths[0])
    n_folds = sum([d.isdigit() for d in n_folds])

    cluster_assignment = {}
    cv_labels = {}
    for cv in range(n_folds):
        cv_labels[cv] = {}
        for i, path in enumerate(paths):
            embedding = pd.read_pickle(f"{path}/{cv}/encoder_weights.p")
            c, cv_labels[cv][i] = get_cluster_labels(embedding)

        cons_mat = consensus_matrix(
            cv_labels[cv], reorder=True, method="single"
        )
        cluster_assignment[cv] = assign_cluster_from_consmat(
            cons_mat, cluster_names, t=0
        )

    return cluster_assignment


def assign_cluster_from_consmat(
    cons_mat: pd.DataFrame, cluster_names: List[str], t: float
):
    """
    First check if asset fall in one (or more) of the clusters for a certain proportion of runs (t)
    Then, select cluster which includes the asset the most
    :param cons_mat: pd.DataFrame
    :param cluster_names: list
    :param t: float
    :return:
    """
    assigned = cons_mat.loc[cluster_names].sum() > t
    cluster_assignment = pd.Series(index=cons_mat.index, dtype="object")
    cluster_assignment.loc[assigned] = cons_mat.loc[
        cluster_names, assigned
    ].idxmax()
    # reorder index
    cluster_assignment = cluster_assignment.loc[cons_mat.index]
    return cluster_assignment


def get_cluster_labels(embedding: pd.DataFrame, threshold: float = 0.1):
    encoding_dim = embedding.shape[-1]
    if threshold is None:
        kmeans = KMeans(n_clusters=encoding_dim, random_state=0).fit(
            embedding.values
        )
        labels = pd.DataFrame(
            kmeans.labels_, index=embedding.index, columns=["label"]
        )
    else:
        mask = embedding >= threshold
        labels = pd.DataFrame(columns=["label"], index=mask.index)
        for c in range(encoding_dim):
            labels.loc[mask.index[mask[c]], "label"] = c

        # When orthogonality constraint is not met, take max
        multi_c = (labels["label"] != embedding.idxmax(1)) & (
            ~labels["label"].isna()
        )
        labels.loc[multi_c, "label"] = embedding.idxmax(1).loc[multi_c]
        labels = labels.fillna(
            encoding_dim
        )  # Some assets might not be assigned to any
        # cluster, put them in another cluster

    cluster_labels = np.unique(labels["label"].values).tolist()
    cluster_labels.sort()

    clusters = {}
    for c in cluster_labels:
        assets = list(labels.loc[labels["label"] == c].index)
        clusters[c] = assets

    clusters = {
        i: clusters.get(i)
        for i in range(len(cluster_labels))
        if clusters.get(i) is not None
    }  # reorder dict

    return clusters, labels


def consensus_matrix(labels: Dict, reorder=False, method="single"):
    """

    :param method:
    :param reorder:
    :param labels:
    :return:
    """
    n_runs = len(labels)
    assets = labels[0]["label"].index
    n_assets = len(assets)

    connect_mat = []
    for i in range(n_runs):
        cm = pd.DataFrame(
            np.eye(n_assets), columns=assets, index=assets, dtype=np.uint8
        )
        classes = labels[i]["label"].unique().tolist()

        for label in classes:
            cluster = list(assets[labels[i]["label"] == label])
            cluster = {
                a: [x for j, x in enumerate(cluster) if i != j]
                for i, a in enumerate(cluster)
            }
            for c in cluster:
                cm.loc[c, cluster[c]] = 1

        connect_mat.append(cm)

    cons_mat = pd.DataFrame(0, columns=assets, index=assets, dtype=np.uint8)
    for i in range(n_runs):
        cons_mat += connect_mat[i]
    cons_mat = cons_mat / n_runs

    if reorder:
        cons_mat, res_order, res_linkage = compute_serial_matrix(
            (1 - cons_mat).values, method
        )
        new_order = [assets[i] for i in res_order]
        cons_mat = pd.DataFrame(
            1 - cons_mat, index=new_order, columns=new_order
        )

    return cons_mat


def rand_score_permutation(labels):
    """

    :param cv_labels: Dictionnary with following structure {run_i: {}, ...}
    :return:
    """
    n_runs = len(labels)
    rand = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(n_runs):
            rand[i, j] = metrics.adjusted_rand_score(
                labels[i]["label"], labels[j]["label"]
            )
    return rand


def seriation(Z, N, cur_index):
    # https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
    """
    input:
        - Z is a hierarchical tree (dendrogram)
        - N is the number of points given to the clustering process
        - cur_index is the position in the tree for the recursive traversal
    output:
        - order implied by the hierarchical tree Z

    seriation computes the order implied by a hierarchical tree (dendrogram)
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_mat, method="ward"):
    # https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
    """
    input:
        - dist_mat is a distance matrix
        - method = ["ward","single","average","complete"]
    output:
        - seriated_dist is the input dist_mat,
          but with re-ordered rows and columns
          according to the seriation, i.e. the
          order implied by the hierarchical tree
        - res_order is the order implied by
          the hierarhical tree
        - res_linkage is the hierarhical tree (dendrogram)

    compute_serial_matrix transforms a distance matrix into
    a sorted distance matrix according to the order implied
    by the hierarchical tree (dendrogram)
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[
        [res_order[i] for i in a], [res_order[j] for j in b]
    ]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage

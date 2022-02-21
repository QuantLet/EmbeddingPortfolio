import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from typing import Dict, List
from scipy.spatial.distance import squareform
from fastcluster import linkage


def get_cluster_assignment(base_dir, cluster_names):
    models = os.listdir(base_dir)
    paths = [f"{base_dir}/{d}" for d in models if os.path.isdir(f"{base_dir}/{d}") and d[0] != "."]
    n_folds = os.listdir(paths[0])
    n_folds = sum([d.isdigit() for d in n_folds])

    cluster_assignment = {}
    cv_labels = {}
    for cv in range(n_folds):
        cv_labels[cv] = {}
        for i, path in enumerate(paths):
            embedding = pd.read_pickle(f'{path}/{cv}/encoder_weights.p')
            c, cv_labels[cv][i] = get_cluster_labels(embedding)

        cons_mat = consensus_matrix(cv_labels[cv], reorder=True, method="single")
        cluster_assignment[cv] = assign_cluster_from_consmat(cons_mat, cluster_names, t=0)

    return cluster_assignment

def assign_cluster_from_consmat(cons_mat: pd.DataFrame, cluster_names: List[str], t: float):
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
    cluster_assignment.loc[assigned] = cons_mat.loc[cluster_names, assigned].idxmax()
    # reorder index
    cluster_assignment = cluster_assignment.loc[cons_mat.index]
    return cluster_assignment


def get_cluster_labels(embedding: pd.DataFrame, threshold: float = 0.1):
    encoding_dim = embedding.shape[-1]
    if threshold is None:
        kmeans = KMeans(n_clusters=encoding_dim, random_state=0).fit(embedding.values)
        labels = pd.DataFrame(kmeans.labels_, index=embedding.index, columns=['label'])
    else:
        mask = embedding >= threshold
        labels = pd.DataFrame(columns=['label'], index=mask.index)
        for c in range(encoding_dim):
            labels.loc[mask.index[mask[c]], 'label'] = c

        # When orthogonality constraint is not met, take max
        multi_c = (labels['label'] != embedding.idxmax(1)) & (~labels['label'].isna())
        labels.loc[multi_c, 'label'] = embedding.idxmax(1).loc[multi_c]
        labels = labels.fillna(
            encoding_dim)  # Some assets might not be assigned to any cluster, put them in another cluster

    cluster_labels = np.unique(labels['label'].values).tolist()
    cluster_labels.sort()

    clusters = {}
    for c in cluster_labels:
        assets = list(labels.loc[labels['label'] == c].index)
        clusters[c] = assets

    clusters = {i: clusters.get(i) for i in range(len(cluster_labels)) if clusters.get(i) is not None}  # reorder dict

    return clusters, labels


def consensus_matrix(labels: Dict, reorder=False, method='single'):
    """

    :param cv_labels: Dictionnary with following structure {run_i: {}, ...}
    :return:
    """
    n_runs = len(labels)
    assets = labels[0]['label'].index
    n_assets = len(assets)

    connect_mat = []
    for i in range(n_runs):
        cm = pd.DataFrame(np.eye(n_assets), columns=assets, index=assets, dtype=np.uint8)
        classes = labels[i]['label'].unique().tolist()

        for label in classes:
            cluster = list(assets[labels[i]['label'] == label])
            cluster = {a: [x for j, x in enumerate(cluster) if i != j] for i, a in enumerate(cluster)}
            for c in cluster:
                cm.loc[c, cluster[c]] = 1

        connect_mat.append(cm)

    cons_mat = pd.DataFrame(0, columns=assets, index=assets, dtype=np.uint8)
    for i in range(n_runs):
        cons_mat += connect_mat[i]
    cons_mat = cons_mat / n_runs

    if reorder:
        cons_mat, res_order, res_linkage = compute_serial_matrix((1 - cons_mat).values, method)
        new_order = [assets[i] for i in res_order]
        cons_mat = pd.DataFrame(1 - cons_mat, index=new_order, columns=new_order)

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
            rand[i, j] = metrics.adjusted_rand_score(labels[i]['label'], labels[j]['label'])
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
        return (seriation(Z, N, left) + seriation(Z, N, right))


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
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage

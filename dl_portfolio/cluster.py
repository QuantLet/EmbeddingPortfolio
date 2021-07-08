import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def get_cluster_labels(embedding, threshold=0.1):
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


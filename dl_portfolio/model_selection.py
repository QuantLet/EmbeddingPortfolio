import os

from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.metrics import euclidean_distances

from dl_portfolio.constant import AVAILABLE_CRITERIA
from dl_portfolio.logger import LOGGER
from dl_portfolio.nmf.convex_nmf import ConvexNMF
from sklearn.preprocessing import StandardScaler
import numpy as np
from dl_portfolio.data import bb_resample_sample

from sklearn.neighbors import KernelDensity


def nmf_cluster(X, k):
    nmf = ConvexNMF(n_components=k, norm_G="l2")
    nmf.fit(X.T)
    centers = nmf.transform(X.T).T
    labels = nmf.G.T.argmax(axis=0)

    return centers, labels


def silhouette(X, k):
    nmf = ConvexNMF(n_components=k, norm_G="l2")
    nmf.fit(X)
    W = nmf.G.copy()
    W[W < 1e-3] = 0
    labels = np.argmax(W, axis=1)

    return metrics.silhouette_score(euclidean_distances(X.T), labels,
                                    metric="precomputed")


def nmf_eps(X, k):
    nmf = ConvexNMF(n_components=k, norm_G="l2")
    nmf.fit(X)
    Z = nmf.transform(X)
    pred = nmf.inverse_transform(Z)
    eps = np.mean(X - pred, axis=1)
    if len(eps.shape) == 1:
        eps = eps.reshape(-1, 1)

    return eps


def compute_aic(ll: float, n_params: int) -> float:
    """
    Calculates Akaike Information Criterion (AIC) from the total sample
    log-likelihood (Akaike, 1973).

    Args:
        ll: Total log-likelihood of a sample
        n_params: Number of free parameters

    Returns:

    """
    return 2 * (n_params - ll)


def model_selection_curve(
    data: np.ndarray,
    criterion: str = "aic",
    p_range=list(range(1, 21)),
    resample: bool = True,
    block_length: int = 60,
    d: int = 1,
):
    min_p = 1
    assert min(p_range) >= min_p
    if resample:
        data, _ = bb_resample_sample(data, block_length=block_length)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    if criterion == "aic_id":
        eps1 = nmf_eps(X, min_p)
        kde = []
        for i in range(eps1.shape[-1]):
            kde.append(
                KernelDensity(kernel="exponential", bandwidth="silverman").fit(
                    eps1[:, i].reshape(-1, 1)
                )
            )

        param_perf = []
        for k in p_range:
            eps = nmf_eps(X, k)
            ll = 0
            for i in range(eps1.shape[-1]):
                f = kde[i].score_samples(eps[:, i].reshape(-1, 1))
                f[np.isinf(f)] = 1e-20
                ll += f.sum()
            param_perf.append(compute_aic(ll, (k - min_p) * d))
    elif criterion == "aic":
        eps1 = nmf_eps(X, min_p)
        kde = KernelDensity(kernel="exponential", bandwidth="silverman").fit(
            eps1
        )
        param_perf = []
        for k in p_range:
            eps = nmf_eps(X, k)
            f = kde.score_samples(eps)
            f[np.isinf(f)] = 1e-20
            ll = f.sum()
            param_perf.append(compute_aic(ll, (k - min_p) * d))
    elif criterion == "silhouette":
        param_perf = [silhouette(X, k) for k in p_range]
    else:
        raise NotImplementedError

    return param_perf


def select_p_from_criterion(
    data,
    p_range=list(range(1, 10)),
    n_exp=100,
    n_jobs=os.cpu_count(),
    **kwargs,
):
    min_p = min(p_range)
    if n_jobs <= 1:
        all_criteria = []
        for i in range(n_exp):
            if i % 10 == 0:
                LOGGER.info(f"Steps to go: {n_exp - i}")
            criteria = model_selection_curve(data, p_range=p_range, **kwargs)
            all_criteria.append(criteria)
    else:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            all_criteria = _parallel_pool(
                delayed(model_selection_curve)(data, p_range=p_range, **kwargs)
                for i in range(n_exp)
            )

    all_criteria = np.array(all_criteria).T

    # Mean based on average of runs
    criterion = kwargs.get("criterion", "aic")
    if criterion == "silhouette":
        best = np.argmax(np.mean(all_criteria, axis=-1)) + min_p
    else:
        best = np.argmin(np.mean(all_criteria, axis=-1)) + min_p
        mins = np.argmin(all_criteria, axis=0) + min_p
        mean_mins = np.mean(mins)
        LOGGER.info(
            f"{mean_mins - 1.96 * np.std(mins)}, {mean_mins}, "
            f"{mean_mins + 1.96 * np.std(mins)}"
        )

    return best, all_criteria


def get_optimal_p(criterion="aic", **kwargs) -> int:
    """
    Ex for criterion == "gap": kwargs = {X=train_data, n_refs=10,
    cluster_array=range(3, 10)}
    :param criterion:
    :param kwargs:
    :return:
    """
    assert criterion in AVAILABLE_CRITERIA
    if criterion == "aic":
        n_components, _ = select_p_from_criterion(
            criterion=criterion, **kwargs
        )
    elif criterion == "gap":
        n_components, _ = get_optimal_n_clusters(**kwargs)
    else:
        raise NotImplementedError

    return n_components

import os

from joblib import Parallel, delayed

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


def nmf_eps(X, k):
    nmf = ConvexNMF(n_components=k, norm_G="l2")
    nmf.fit(X)
    Z = nmf.transform(X)
    pred = nmf.inverse_transform(Z)
    eps = np.mean((X - pred)**2, axis=1)
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


def aic_curve(
        data: np.ndarray, p_range=list(range(1, 10)),
        resample: bool = True, block_length: int = 60):
    min_p = 1
    d = 1
    assert min(p_range) >= min_p
    if resample:
        data, _ = bb_resample_sample(data, block_length=block_length)
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    eps1 = nmf_eps(X, min_p)
    kde = KernelDensity(kernel="epanechnikov", bandwidth="scott").fit(
        eps1
    )
    aic = []
    for k in p_range:
        eps = nmf_eps(X, k)
        f = kde.score_samples(eps)
        f[np.isinf(f)] = 1e-20
        ll = f.sum()
        aic.append(compute_aic(ll, (k - min_p)*d))

    return aic


def find_best_p(data, p_range=list(range(1, 10)), n_exp=100,
                n_jobs=os.cpu_count(), **kwargs):
    min_p = min(p_range)
    if n_jobs <= 1:
        all_aics = []
        for i in range(n_exp):
            if i % 10 == 0:
                LOGGER.info(f"Steps to go: {n_exp - i}")
            aic = aic_curve(data, p_range=p_range, **kwargs)
            all_aics.append(aic)
    else:
        with Parallel(n_jobs=n_jobs) as _parallel_pool:
            all_aics = _parallel_pool(
                delayed(aic_curve)(
                    data, p_range=p_range, **kwargs) for i in range(n_exp)
            )

    all_aics = np.array(all_aics).T

    # Mean based on average of runs
    best = np.argmin(np.mean(all_aics, axis=-1)) + min_p

    mins = np.argmin(all_aics, axis=0) + min_p
    mean_mins = np.mean(mins)
    LOGGER.info(f"{mean_mins - 1.96 * np.std(mins)}, {mean_mins}, "
                f"{mean_mins + 1.96 * np.std(mins)}")

    return best, all_aics

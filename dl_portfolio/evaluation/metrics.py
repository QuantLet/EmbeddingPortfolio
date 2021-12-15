import numpy as np
from typing import Union
from pandas.core.frame import DataFrame


def cov_eigenvalue_accuracy(sample_cov: Union[DataFrame, np.ndarray], alt_cov: Union[DataFrame, np.ndarray]):
    sample_cov = sample_cov ** 2
    alt_cov = alt_cov ** 2
    return np.sqrt(np.trace(alt_cov.T.dot(alt_cov)) / np.trace(sample_cov.T.dot(sample_cov)))


def cov_magnitue_error(sample_cov: Union[DataFrame, np.ndarray], alt_cov: Union[DataFrame, np.ndarray]):
    if type(sample_cov) == DataFrame:
        sample_cov = sample_cov.values
    if type(alt_cov) == DataFrame:
        alt_cov = alt_cov.values
    return np.sum(np.abs(sample_cov - alt_cov)) / np.sum(np.abs(sample_cov))


def cov_direction_accuracy(sample_cov: Union[DataFrame, np.ndarray], alt_cov: Union[DataFrame, np.ndarray]):
    if type(sample_cov) == DataFrame:
        sample_cov = sample_cov.values
    if type(alt_cov) == DataFrame:
        alt_cov = alt_cov.values
    return np.sum(np.sign(alt_cov * sample_cov)).sum() / np.linalg.matrix_rank(sample_cov) ** 2

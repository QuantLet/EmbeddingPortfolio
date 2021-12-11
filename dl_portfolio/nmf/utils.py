import numpy as np


def negative_matrix(A):
    return (np.abs(A) - A) / 2


def positive_matrix(A):
    return (np.abs(A) + A) / 2

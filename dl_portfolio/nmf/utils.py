import numpy as np


def negative_matrix(A):
    return (np.abs(A) - A) / 2


def positive_matrix(A):
    return (np.abs(A) + A) / 2


def reconstruction_error(X, F, G, loss='mse'):
    X_hat = F.dot(G.T)
    if loss == 'mse':
        loss = mean_squarred_error(X, X_hat)
    else:
        raise NotImplementedError(loss)
    return loss


def mean_squarred_error(y_true, y_pred):
    errors = np.average((y_true - y_pred) ** 2, axis=0)
    return np.average(errors)

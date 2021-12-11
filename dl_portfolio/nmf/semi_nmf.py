import numbers
import numpy as np
import time

from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from dl_portfolio.logger import LOGGER
from dl_portfolio.nmf.utils import negative_matrix, positive_matrix, reconstruction_error


class SemiNMF(BaseEstimator):
    def __init__(self, n_components, max_iter=200, tol=1e-6, random_state=None, verbose=0, loss="mse", shuffle=False):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.shuffle = shuffle
        self._is_fitted = False
        self.components = None
        self.loss = loss

    def _check_params(self, X):
        # n_components
        self._n_components = self.n_components
        if self._n_components is None:
            self._n_components = X.shape[1]
        if (
                not isinstance(self._n_components, numbers.Integral)
                or self._n_components <= 0
        ):
            raise ValueError(
                "Number of components must be a positive integer; got "
                f"(n_components={self._n_components!r})"
            )

        # max_iter
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iterations must be a positive "
                f"integer; got (max_iter={self.max_iter!r})"
            )

        # tol
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got "
                f"(tol={self.tol!r})"
            )

        return self

    def fit(self, X, verbose: Optional[int] = None):
        X = X.astype(np.float32)

        if verbose is not None:
            self.verbose = verbose

        start_time = time.time()
        self._check_params(X)
        # Initialize G and F
        G = self._initilize_g(X)
        F = self._update_f(X, G)

        # used for the convergence criterion
        error_at_init = reconstruction_error(X, F, G, loss=self.loss)
        previous_error = error_at_init

        for n_iter in range(self.max_iter):
            # Update G
            G = self._update_g(X, G, F)
            # Update F
            F = self._update_f(X, G)

            if n_iter == self.max_iter - 1:
                if self.verbose:
                    LOGGER.info('Reached max iteration number, stopping')

            if self.tol > 0 and n_iter % 10 == 0:
                error = reconstruction_error(X, F, G, loss=self.loss)

                if self.verbose:
                    iter_time = time.time()
                    LOGGER.info(
                        "Epoch %02d reached after %.3f seconds, error: %f"
                        % (n_iter, iter_time - start_time, error)
                    )

                if (previous_error - error) / error_at_init < self.tol:
                    if self.verbose:
                        LOGGER.info(f"Converged at iteration: {n_iter} with tolerance: {self.tol}")
                    break
                previous_error = error

        self.components = G
        self._is_fitted = True

    def transform(self, X):
        assert self._is_fitted, "You must fit the model first"
        G = self.components.copy()
        F = X.dot(G.dot(np.linalg.inv(G.T.dot(G))))
        return F

    def inverse_transform(self, F):
        assert self._is_fitted, "You must fit the model first"
        return np.dot(F, self.components.T)

    def _initilize_g(self, X):
        d = X.shape[-1]
        G = np.zeros((d, self._n_components))
        kmeans = KMeans(n_clusters=self._n_components, random_state=self.random_state).fit(X.T)
        for i in range(d):
            G[i, kmeans.labels_[i]] = 1
        # add constant
        G += 0.2
        return G

    @staticmethod
    def _update_f(X, G):
        return X.dot(G.dot(np.linalg.inv(G.T.dot(G))))

    @staticmethod
    def _update_g(X, G, F):
        F_TF_minus = negative_matrix(F.T.dot(F))
        F_TF_plus = positive_matrix(F.T.dot(F))

        X_TF_minus = negative_matrix(X.T.dot(F))
        X_TF_plus = positive_matrix(X.T.dot(F))

        numerator = X_TF_plus + G.dot(F_TF_minus)
        denominator = X_TF_minus + G.dot(F_TF_plus)

        assert (denominator != 0).all(), "Division by 0"

        return G * np.sqrt(numerator / denominator)

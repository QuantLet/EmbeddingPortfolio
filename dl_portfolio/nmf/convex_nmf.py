import numbers
import numpy as np
import time

from typing import Optional
from sklearn.decomposition._nmf import _beta_divergence

from dl_portfolio.logger import LOGGER
from dl_portfolio.nmf.semi_nmf import SemiNMF
from dl_portfolio.nmf.utils import negative_matrix, positive_matrix


class ConvexNMF(SemiNMF):
    def __init__(self, n_components, G=None, max_iter=200, tol=1e-4, random_state=None, verbose=0, beta_loss=2,
                 shuffle=False):
        super(ConvexNMF, self).__init__(n_components, max_iter=max_iter, tol=tol, random_state=random_state,
                                        verbose=verbose, beta_loss=beta_loss, shuffle=shuffle)
        self.G = G
        self.encoding = None

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
        if verbose is not None:
            self.verbose = verbose

        start_time = time.time()
        self._check_params(X)
        # Initialize G and F
        G, W = self._initilize_g_w(X, self.G)
        print(G, W)
        F = X.dot(W)
        print(F)

        # used for the convergence criterion
        error_at_init = _beta_divergence(X, F, G.T, beta=self.beta_loss, square_root=True)
        previous_error = error_at_init

        for n_iter in range(self.max_iter):
            # Update G
            G = self._update_g(X, G, F) # self._update_g(X, G, X.dot(W))
            # Update W
            W = self._update_w(X, W, G)
            # Update F
            F = X.dot(W)

            if n_iter == self.max_iter - 1:
                if self.verbose:
                    LOGGER.info('Reached max iteration number, stopping')

            error = _beta_divergence(X, F, G.T, beta=self.beta_loss, square_root=True)

            if self.verbose:
                iter_time = time.time()
                LOGGER.info(
                    "Epoch %02d reached after %.3f seconds, error: %f"
                    % (n_iter, iter_time - start_time, error)
                )

            if self.tol > 0 and n_iter % 10 == 0:
                error = _beta_divergence(X, F, G.T, beta=self.beta_loss, square_root=True)

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
        self.encoding = W
        self._is_fitted = True

    def transform(self, X):
        assert self._is_fitted, "You must fit the model first"
        W = self.encoding.copy()
        F = X.dot(W)
        return F

    def inverse_transform(self, F):
        assert self._is_fitted, "You must fit the model first"
        return np.dot(F, self.components.T)

    def _initilize_g_w(self, X, G=None):
        if G is None:
            G = self._initilize_g(X)
            H = G - 0.2
            D_n = np.diag(H.sum(0).astype(int))
            W = np.dot(G, np.linalg.inv(D_n))
        else:
            W = G.dot(np.linalg.inv(np.dot(G.T, G)))
            W_plus = positive_matrix(W)
            W = W_plus + 0.2 * np.sum(np.abs(W_plus)) / np.sum(W_plus != 0)

        return G, W

    @staticmethod
    def _update_w(X, W, G):
        X_TX_plus = positive_matrix(X.T.dot(X))
        X_TX_minus = negative_matrix(X.T.dot(X))

        numerator = X_TX_plus.dot(G) + X_TX_minus.dot(W.dot(G.T.dot(G)))
        denominator = X_TX_minus.dot(G) + X_TX_plus.dot(W.dot(G.T.dot(G)))

        assert (denominator != 0).all(), "Division by 0"

        return W * np.sqrt(numerator / denominator)

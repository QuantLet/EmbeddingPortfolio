import numpy as np

from dl_portfolio.nmf.utils import positive_matrix, negative_matrix


def test_neg_pos_matrix():
    positives = np.random.uniform(0, 1, 10)
    negatives = -1.0 * np.random.uniform(0, 1, 10)
    A = np.array([positives, negatives]).reshape((4, 5))
    A_minus = negative_matrix(A)
    A_plus = positive_matrix(A)

    assert (A_minus >= 0).all()
    assert (A_plus >= 0).all()


if __name__ == "__main__":
    test_neg_pos_matrix()

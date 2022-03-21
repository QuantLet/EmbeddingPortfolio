import numpy as np


def id_nb_bootstrap(n_obs, block_length):
    """
    Create bootstrapped indexes with the none overlapping block bootstrap
    ('nbb') strategy given the number of observations in a timeseries and
    the length of the blocks.

    :param n_obs:
    :param block_length:
    :return:
    """
    assert block_length < n_obs
    assert block_length > 3

    n_blocks = int(np.ceil(n_obs / block_length))
    nexts = np.repeat([np.arange(0, block_length)], n_blocks, axis=0)

    blocks = np.random.permutation(
        np.arange(0, n_obs, block_length)
    ).reshape(-1, 1)

    _id = (blocks + nexts).ravel()[:n_obs]
    _id = _id[_id < n_obs]

    return _id

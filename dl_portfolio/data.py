import numpy as np


def build_delayed_window(data: np.ndarray, seq_len: int, return_3d: bool = False):
    """

    :param data: data
    :param seq_len: length of past window
    :param return_3d: if True then return  (n, seq_len, n_features)
    :return:
    """
    n_features = data.shape[-1]
    # sequence data: (n, seq_len, n_features)
    seq_data = np.array([data[i - seq_len:i, :] for i in range(seq_len, len(data))], dtype=np.float32)

    if return_3d:
        data = seq_data
    else:
        # concatenate columns: (n, seq_len * n_features)
        data = np.zeros((seq_data.shape[0], seq_len * n_features))
        data[:] = np.nan
        for i in range(n_features):
            data[:, i * seq_len:seq_len * (i + 1)] = seq_data[:, :, i]
        assert not any(np.isnan(data).sum(1).tolist())
    return data


def features_generator(dataset):
    for features, _ in dataset:
        yield features


def returns_generator(dataset):
    for _, next_returns in dataset:
        yield next_returns

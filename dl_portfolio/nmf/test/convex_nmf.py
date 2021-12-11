import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dl_portfolio.ae_data import load_data
from dl_portfolio.nmf.convex_nmf import ConvexNMF

if __name__ == "__main__":
    dfdata, assets = load_data(dataset='bond', dropnan=False, freq='D', crix=False, crypto_assets=['BTC', 'ETH', 'XRP'])
    assets = ['UK_B', 'US_B', 'GE_B', 'SPX_X', 'SX5E_X', 'UKX_X', 'BTC', 'ETH', 'XRP']
    encoding_dim = 3

    dfdata = dfdata[assets]
    dfdata = np.log(dfdata.pct_change().dropna() + 1)
    seed = 10
    data = dfdata.values
    X_train = data[:-int(0.2 * len(data))]
    X_test = data[-int(0.2 * len(data)):]
    mean_ = np.mean(X_train, 0)
    std_ = np.std(X_train, 0)
    X_train -= mean_
    X_train /= std_
    X_test -= mean_
    X_test /= std_

    # Dimensions
    n = X_train.shape[0]
    d = X_train.shape[1]
    p = encoding_dim

    convex_nmf = ConvexNMF(n_components=p, random_state=seed, verbose=1)

    # convex_nmf._check_params(X_train)
    # G = convex_nmf._initilize_g(X_train)
    # H = G - 0.2
    # D_n = np.diag(H.sum(0).astype(int))
    # W = np.dot(H, np.linalg.inv(D_n))
    # print(W)
    # W_plus = (np.abs(W) + W) / 2
    # print(W_plus)
    # W = W_plus + 0.2 * np.sum(np.abs(W_plus)) / np.sum(W_plus != 0)
    # print(W)
    # print(np.dot(G, np.linalg.inv(D_n)))
    # print(W.astype(np.float32) == np.dot(G, np.linalg.inv(D_n)).astype(np.float32))

    convex_nmf.fit(X_train, verbose=1)
    print(convex_nmf.components)
    plt.figure(figsize=(10, 10))
    sns.heatmap(convex_nmf.components, yticklabels=assets)
    plt.show()

    print(convex_nmf.encoding)
    factors = convex_nmf.transform(X_train)
    print(factors)
    pred = convex_nmf.inverse_transform(factors)

    for i in range(X_train.shape[-1]):
        plt.plot(X_train[:, i])
        plt.plot(pred[:, i])
        plt.show()

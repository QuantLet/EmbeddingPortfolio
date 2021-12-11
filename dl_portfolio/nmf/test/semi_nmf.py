import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dl_portfolio.ae_data import load_data
from dl_portfolio.nmf.semi_nmf import SemiNMF

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

    semi_nmf = SemiNMF(n_components=p, random_state=seed)

    # semi_nmf = semi_nmf._check_params(X_train)
    # G = semi_nmf._initilize_g(X_train)
    # F = semi_nmf._update_f(X_train, G)
    # print(G)
    # print(F)
    # print(F.shape)
    #
    # G = semi_nmf._update_g(X_train, G, F)
    #
    # print(G.shape)

    semi_nmf.fit(X_train, verbose=1)
    print(semi_nmf.components)
    plt.figure(figsize=(10, 10))
    sns.heatmap(semi_nmf.components, yticklabels=assets)
    plt.show()

    factors = semi_nmf.transform(X_train)
    print(factors)
    pred = semi_nmf.inverse_transform(factors)

    print(semi_nmf.evaluate(X_train))
    print(semi_nmf.evaluate(X_test))

    for i in range(X_train.shape[-1]):
        plt.plot(X_train[:, i])
        plt.plot(pred[:, i])
        plt.show()

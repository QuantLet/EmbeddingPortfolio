import pandas as pd
import os, pickle
from dl_portfolio.logger import LOGGER
import datetime as dt
from shutil import copyfile
from dl_portfolio.ae_data import get_features, load_data
import numpy as np
from sklearn.cluster import KMeans


def run(ae_config, seed=None):
    random_seed = np.random.randint(0, 100)
    if ae_config.seed:
        seed = ae_config.seed
    if seed is None:
        seed = np.random.randint(0, 1000)

    np.random.seed(seed)
    LOGGER.info(f"Set seed: {seed}")

    if ae_config.save:
        iter = len(os.listdir('log_kmeans'))
        save_dir = f"log_kmeans/m_{iter}_seed_{seed}_{dt.datetime.strftime(dt.datetime.now(), '%Y%m%d_%H%M%S')}"
        os.makedirs(save_dir)
        copyfile('./dl_portfolio/config/ae_config.py',
                 os.path.join(save_dir, 'ae_config.py'))

    if ae_config.dataset == 'bond':
        data, assets = load_data(dataset=ae_config.dataset, assets=ae_config.assets, dropnan=ae_config.dropnan,
                                 freq=ae_config.freq, crix=ae_config.crix, crypto_assets=ae_config.crypto_assets)
    else:
        data, assets = load_data(dataset=ae_config.dataset, assets=ae_config.assets, dropnan=ae_config.dropnan,
                                 freq=ae_config.freq)

    for cv in ae_config.data_specs:
        LOGGER.info(f'Starting with cv: {cv}')
        if ae_config.save:
            save_path = f"{save_dir}/{cv}"
            os.mkdir(f"{save_dir}/{cv}")
        else:
            save_path = None

        LOGGER.info(f'Assets order: {assets}')
        data_spec = ae_config.data_specs[cv]
        train_data, val_data, test_data, scaler, dates, features = get_features(data,
                                                                                data_spec['start'],
                                                                                data_spec['end'],
                                                                                assets,
                                                                                val_start=data_spec['val_start'],
                                                                                test_start=data_spec.get('test_start'),
                                                                                scaler='StandardScaler',
                                                                                resample={
                                                                                    'method': 'nbb',
                                                                                    'where': ['train'],
                                                                                    'block_length': 60
                                                                                })
        kmeans = KMeans(n_clusters=ae_config.encoding_dim, random_state=seed)
        kmeans.fit(train_data.T)
        labels = pd.DataFrame(kmeans.labels_.reshape(1, -1), columns=assets).T
        labels.columns = ['label']
        clusters = {i: list(labels[labels['label'] == i].index) for i in range(ae_config.encoding_dim)}

        if ae_config.save:
            pickle.dump(ae_config.scaler_func, open(f"{save_path}/scaler.p", "wb"))
            pickle.dump(kmeans, open(f"{save_path}/model.p", "wb"))
            pickle.dump(clusters, open(f"{save_path}/clusters.p", "wb"))
            labels.to_pickle(f"{save_path}/labels.p")

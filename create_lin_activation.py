import os
import sys

import pandas as pd

from dl_portfolio.data import load_data
from dl_portfolio.utils import get_linear_encoder
from dl_portfolio.logger import LOGGER


def create_linear_features(base_dir):
    sys.path.append(base_dir)
    import ae_config as config

    # Load test results
    data, assets = load_data(dataset=config.dataset)
    os.mkdir(f"activationProba/data/{config.dataset}")

    cvs = list(config.data_specs.keys())
    for cv in cvs:
        LOGGER.info(f"Saving to: 'activationProba/data/{config.dataset}/{cv}'")
        os.mkdir(f"activationProba/data/{config.dataset}/{cv}")
        _, _, train_act = get_linear_encoder(config, 'train', data, assets, base_dir, cv)
        _, _, val_act = get_linear_encoder(config, 'val', data, assets, base_dir, cv)
        _, _, test_act = get_linear_encoder(config, 'test', data, assets, base_dir, cv)
        train_act = pd.concat([train_act, val_act])
        train_act.to_csv(f"activationProba/data/{config.dataset}/{cv}/train_linear_activation.csv")
        test_act.to_csv(f"activationProba/data/{config.dataset}/{cv}/test_linear_activation.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",
                        type=str,
                        help="Directory with ae model logs, ex: 'final_models/ae/dataset1/m_0_dataset1_nbb_resample_bl_60_seed_0_1647953383912806'")

    args = parser.parse_args()

    LOGGER.info(f"Create linear activation for model {args.base_dir}")
    create_linear_features(args.base_dir)
    LOGGER.info("Done")

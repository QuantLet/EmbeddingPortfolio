import json
import os
import sys

import pandas as pd

from dl_portfolio.cluster import get_cluster_labels
from dl_portfolio.data import load_data
from dl_portfolio.utils import get_linear_encoder
from dl_portfolio.logger import LOGGER


def create_linear_features(base_dir, reorder_features=False, layer="encoder"):
    sys.path.append(base_dir)
    import ae_config as config

    # Load test results
    data, assets = load_data(dataset=config.dataset)
    os.makedirs(f"activationProba/data/{config.dataset}")

    test_lin_activation = pd.DataFrame()

    cvs = list(config.data_specs.keys())
    if "test_start" in config.data_specs[0]:
        include_test = True
    else:
        include_test = False
    for cv in cvs:
        LOGGER.info(f"Saving to: 'activationProba/data/{config.dataset}/{cv}'")
        os.mkdir(f"activationProba/data/{config.dataset}/{cv}")
        _, _, train_act, intercept, boundary, new_order = get_linear_encoder(
            config, "train", data, assets, base_dir, cv, layer=layer,
            reorder_features=reorder_features,
        )
        _, _, val_act, _, boundary, new_order = get_linear_encoder(
            config, "val", data, assets, base_dir, cv, layer=layer,
            reorder_features=reorder_features,
        )
        if include_test:
            _, _, test_act, _, boundary, new_order = get_linear_encoder(
                config, "test", data, assets, base_dir, cv, layer=layer,
                reorder_features=reorder_features,
            )
        train_act = pd.concat([train_act, val_act])
        train_act.to_csv(
            f"activationProba/data/{config.dataset}/"
            f"{cv}/train_linear_activation.csv"
        )
        val_act.to_csv(
            f"activationProba/data/{config.dataset}/"
            f"{cv}/val_linear_activation.csv"
        )
        if include_test:
            test_act.to_csv(
                f"activationProba/data/{config.dataset}/"
                f"{cv}/test_linear_activation.csv"
            )
            test_lin_activation = pd.concat([test_lin_activation, test_act])
        loading = pd.read_pickle(f"{base_dir}/{cv}/decoder_weights.p")
        if new_order:
            loading = loading.iloc[:, new_order]
            loading.columns = range(loading.shape[-1])
        cluster_assignment, _ = get_cluster_labels(loading)

        json.dump(cluster_assignment,
                  open(f"activationProba/data/{config.dataset}/{cv}/"
                       f"cluster_assignment.json", "w"))
        json.dump(intercept, open(f"activationProba/data/{config.dataset}/"
                                  f"{cv}/intercept.json", "w"))
        json.dump(boundary, open(f"activationProba/data/{config.dataset}/"
                                  f"{cv}/boundary.json", "w"))

    if config.encoding_dim is not None:
        _, _, train_lin_activation, intercept, boundary, new_order = get_linear_encoder(
            config, "train", data, assets, base_dir, 0, layer=layer,
            reorder_features=reorder_features,
        )
        _, _, val_lin_activation, _, boundary, new_order = get_linear_encoder(
            config, "val", data, assets, base_dir, 0, layer=layer,
            reorder_features=reorder_features,
        )
        if include_test:
            lin_act = pd.concat(
                [train_lin_activation, val_lin_activation, test_lin_activation]
            )
        else:
            lin_act = pd.concat([train_lin_activation, val_lin_activation])
        lin_act.to_csv(
            f"activationProba/data/{config.dataset}/linear_activation.csv"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Directory with AE model logs, "
             "ex: 'final_models/ae/dataset1/m_0_dataset1_nbb_resample_bl_60_seed_0_1647953383912806'",
    )
    parser.add_argument(
        "--reorder_features",
        action="store_true",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="encoder",
        help="Name of the layer from which to compute output",
    )
    args = parser.parse_args()

    LOGGER.info(f"Create linear activation for model {args.base_dir}")
    create_linear_features(args.base_dir,
                           reorder_features=args.reorder_features,
                           layer=args.layer)
    LOGGER.info("Done")

import datetime as dt
import pandas as pd
import tensorflow as tf

from dl_portfolio.constraints import NonNegAndUnitNorm
from dl_portfolio.regularizers import WeightsOrthogonality

show_plot = False
save = True
nmf_model = "./log_convex_nmf_excess_p3_val/m_0_seed_2_20230120_172626"

# Data
dataset = "dataset1"
resample = {
    "method": "nbb",
    "where": ["train"],
    "block_length": 60,
    "when": "each_epoch",
}
excess_ret = False
scaler_func = {
    "name": "StandardScaler",
    "params": {
        "with_mean": True,
        "with_std": True,
    },
}
seed = None

# Model
encoding_dim = 4
encoder_bias = True
decoder_bias = True
batch_normalization = True
uncorrelated_features = True
weightage = 1e-2
ortho_weightage = 1e-2
l_name = "l1"
l = 1e-3
activation = "relu"
features_config = None
model_name = f"{dataset}_nbb_resample_bl_{resample['block_length']}"
model_name = model_name.replace(".", "d")

shuffle_columns = False
model_type = "ae_model"

learning_rate = 1e-3
epochs = 1500
batch_size = 32
val_size = None
test_size = 0
loss = "mse"
label_param = None
rescale = None

# Constraints and regularizer
activity_regularizer = None
kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
kernel_regularizer = WeightsOrthogonality(
    encoding_dim,
    weightage=ortho_weightage,
    axis=0,
    regularizer={"name": l_name, "params": {l_name: l}},
)
callback_activity_regularizer = False
kernel_constraint = NonNegAndUnitNorm(max_value=1.0, axis=0)

callbacks = {
    "EarlyStopping": {
        "monitor": "val_loss",
        "min_delta": 1e-3,
        "mode": "min",
        "patience": 100,
        "verbose": 1,
        "restore_best_weights": True,
    }
}

val_start = pd.date_range(
    "2018-12-01", "2019-10-01", freq="1MS"
) + dt.timedelta(days=12)
val_start = [str(d.date()) for d in val_start]

val_end = pd.date_range("2019-01-01", "2019-11-01", freq="1MS") + dt.timedelta(
    days=11
)
val_end = [str(d.date()) for d in val_end]

data_specs = {}
for i in range(len(val_start)):
    data_specs[i] = {
        "start": "2016-06-30",
        "val_start": val_start[i],
        "end": val_end[i],
    }

import pandas as pd
import tensorflow as tf

from dl_portfolio.constraints import NonNegAndUnitNorm
from dl_portfolio.regularizers import WeightsOrthogonality

dataset = "dataset1"
show_plot = False
save = True
nmf_model = "./final_models/nmf/dataset1/m_0_seed_0_20230405_125943"

resample = {
    "method": "nbb",
    "where": ["train"],
    "block_length": 60,
    "when": "each_epoch",
}
excess_ret = False
seed = None

encoding_dim = None
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
model_name = None
shuffle_columns = False
scaler_func = {"name": "StandardScaler"}
model_type = "ae_model"

learning_rate = 1e-3
epochs = 1000
batch_size = 32
loss = "mse"

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

val_start = pd.date_range("2017-05-01", "2023-01-01", freq="1MS")
test_start = pd.date_range("2017-06-01", "2023-02-01", freq="1MS")
end = pd.date_range("2017-06-01", "2023-03-01", freq="M")

val_start = [str(d.date()) for d in val_start]
test_start = [str(d.date()) for d in test_start]
end = [str(d.date()) for d in end]

data_specs = {}
for i in range(len(val_start)):
    data_specs[i] = {
        "start": "2016-06-30",
        "val_start": val_start[i],
        "test_start": test_start[i],
        "end": end[i],
    }


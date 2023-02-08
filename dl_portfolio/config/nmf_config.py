import datetime as dt
import pandas as pd

model_type = "convex_nmf"
save = True
show_plot = False
seed = None

# Model
encoding_dim = 3

# Data
dataset = "dataset1"
resample = {
    "method": "nbb",
    "where": ["train"],
    "block_length": 60,
    "when": "each_epoch",
}
excess_ret = True
scaler_func = {"name": "StandardScaler"}

# Model name
model_name = f"{dataset}_nbb_resample_bl_{resample['block_length']}"
model_name = model_name.replace(".", "d")

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

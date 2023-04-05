import pandas as pd
import datetime as dt

model_type = "convex_nmf"
save = True
show_plot = True
seed = None

# Model
encoding_dim = None
p_range = list(range(2, 16))
n_exp = 10
norm_G = "l2"
norm_W = None

# Data
dataset = "dataset2"
resample = None
scaler_func = {"name": "StandardScaler"}
excess_ret = False

val_start = pd.date_range("1990-02-01", "2021-11-01", freq="1MS")
start = [d - dt.timedelta(days=365) for d in val_start]
start = [str(d.date()) for d in start]
val_start = [str(d.date()) for d in val_start]

end = pd.date_range("1990-02-01", "2021-12-01", freq="1M")
end = [str(d.date()) for d in end]

data_specs = {}
for i in range(len(val_start)):
    data_specs[i] = {
        "start": start[i],
        "val_start": val_start[i],
        "end": end[i],
    }

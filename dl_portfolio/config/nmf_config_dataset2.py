import pandas as pd

model_type = "convex_nmf"
save = True
show_plot = False
seed = None

# Model
encoding_dim = 5

# Data
dataset = 'dataset2'
resample = {
    'method': 'nbb',
    'where': ['train'],
    'block_length': 60,
    'when': 'each_epoch'
}
excess_ret = False
scaler_func = {
    'name': 'StandardScaler'
}

val_start = pd.date_range('2007-01-01', '2021-09-01', freq='1MS')
val_start = [str(d.date()) for d in val_start]

test_start = pd.date_range('2007-02-01', '2021-10-01', freq='1MS')
test_start = [str(d.date()) for d in test_start]
test_end = pd.date_range('2007-02-01', '2021-11-01', freq='1M')
test_end = [str(d.date()) for d in test_end]

data_specs = {}
for i in range(len(val_start)):
    data_specs[i] = {
        'start': '1989-02-01',
        'val_start': val_start[i],
        'test_start': test_start[i],
        'end': test_end[i]
    }

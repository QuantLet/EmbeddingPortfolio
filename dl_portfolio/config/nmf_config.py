model_type = "convex_nmf"
save = True
show_plot = False
seed = None

# Model
encoding_dim = 4

# Data
dataset = 'dataset1'
resample = {
    'method': 'nbb',
    'where': ['train'],
    'block_length': 60,
    'when': 'each_epoch'
}
scaler_func = {
    'name': 'StandardScaler'
}

# Model name
model_name = f"{dataset}_nbb_resample_bl_{resample['block_length']}"
model_name = model_name.replace('.', 'd')

data_specs = {
    0: {
        'start': '2016-06-30',
        'val_start': '2019-11-13',
        'test_start': '2019-12-12',
        'end': '2020-01-11'
    },
    1: {
        'start': '2016-06-30',
        'val_start': '2019-12-13',
        'test_start': '2020-01-12',
        'end': '2020-02-11'
    }
}

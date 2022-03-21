model_type = "convex_nmf"
save = True
show_plot = False
seed = None  # np.random.randint(0, 1000)

# Model
encoding_dim = 4

# Data
dataset = 'bond'
dropnan = False
freq = "1D"

resample = {
    'method': 'nbb',
    'where': ['train'],
    'block_length': 60,
    'when': 'each_epoch'
}
crix = False
crypto_assets = ['BTC', 'DASH', 'ETH', 'LTC', 'XRP']
assets = None
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
    },
    2: {
        'start': '2016-06-30',
        'val_start': '2020-01-13',
        'test_start': '2020-02-12',
        'end': '2020-03-11'
    },
    3: {
        'start': '2016-06-30',
        'val_start': '2020-02-13',
        'test_start': '2020-03-12',
        'end': '2020-04-11'
    },
    4: {
        'start': '2016-06-30',
        'val_start': '2020-03-13',
        'test_start': '2020-04-12',
        'end': '2020-05-11'
    },
    5: {
        'start': '2016-06-30',
        'val_start': '2020-04-13',
        'test_start': '2020-05-12',
        'end': '2020-06-11'
    },
    6: {
        'start': '2016-06-30',
        'val_start': '2020-05-13',
        'test_start': '2020-06-12',
        'end': '2020-07-11'
    },
    7: {
        'start': '2016-06-30',
        'val_start': '2020-06-13',
        'test_start': '2020-07-12',
        'end': '2020-08-11'
    },
    8: {
        'start': '2016-06-30',
        'val_start': '2020-07-13',
        'test_start': '2020-08-12',
        'end': '2020-09-11'
    },
    9: {
        'start': '2016-06-30',
        'val_start': '2020-08-13',
        'test_start': '2020-09-12',
        'end': '2020-10-11'
    },
    10: {
        'start': '2016-06-30',
        'val_start': '2020-09-13',
        'test_start': '2020-10-12',
        'end': '2020-11-11'
    },
    11: {
        'start': '2016-06-30',
        'val_start': '2020-10-13',
        'test_start': '2020-11-12',
        'end': '2020-12-11'
    },
    12: {
        'start': '2016-06-30',
        'val_start': '2020-11-13',
        'test_start': '2020-12-12',
        'end': '2021-01-11'
    },
    13: {
        'start': '2016-06-30',
        'val_start': '2020-12-13',
        'test_start': '2021-01-12',
        'end': '2021-02-11'
    },
    14: {
        'start': '2016-06-30',
        'val_start': '2021-01-13',
        'test_start': '2021-02-12',
        'end': '2021-03-11'
    },
    15: {
        'start': '2016-06-30',
        'val_start': '2021-02-13',
        'test_start': '2021-03-12',
        'end': '2021-04-11'
    },
    16: {
        'start': '2016-06-30',
        'val_start': '2021-03-13',
        'test_start': '2021-04-12',
        'end': '2021-05-11'
    },
    17: {
        'start': '2016-06-30',
        'val_start': '2021-04-13',
        'test_start': '2021-05-12',
        'end': '2021-06-11'
    },
    18: {
        'start': '2016-06-30',
        'val_start': '2021-05-13',
        'test_start': '2021-06-12',
        'end': '2021-07-11'
    },
    19: {
        'start': '2016-06-30',
        'val_start': '2021-06-13',
        'test_start': '2021-07-12',
        'end': '2021-08-11'
    },
    20: {
        'start': '2016-06-30',
        'val_start': '2021-07-13',
        'test_start': '2021-08-12',
        'end': '2021-09-11'
    },
    21: {
        'start': '2016-06-30',
        'val_start': '2021-08-13',
        'test_start': '2021-09-12',
        'end': '2021-10-11'
    }
}

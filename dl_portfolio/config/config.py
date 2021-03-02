import tensorflow as tf
from dl_portfolio.metrics import penalized_volatility_returns, sharpe_ratio

# Random seed
seed = 9

# Data
path = 'crypto_data/price/train_data_1800.p'  # './first_etf_strat/data/clean_VTI_AGG_DBC.p'
pairs = ['BTC', 'DASH', 'ETH', 'LTC', 'XEM', 'XMR', 'XRP']  # ['VTI', 'AGG', 'DBC'] #
start_date = '2017-01-01'
seq_len = 2
lookfront = 1
features = [
    {'name': 'close', 'lookback': 0},
    {'name': 'open_close_returns', 'lookback': 0},
    {'name': 'realized_volatility', 'params': {'time_period': 30}, 'lookback': 29}
    # {'name': 'returns', 'params': {'time_period': 1}}
]
preprocess = {
    'open_close_returns': {'method': 'mean_std', 'params': {'with_std': True, 'with_mean': False}},
    'close': {'method': 'seq_normalization', 'params': {'base': -1}}
}
# preprocess = {
#     'close': {'method': 'minmax', 'params': {'feature_range': [-1, 1]}}
# }
# preprocess = {
#     'close': {'method': 'seq_normalization', 'params': {'base': -1}}
# }
# Model
model_type = "asset_independent_model"
layers = [
    {
        'type': 'gru',
        'neurons': 6,
        'params': {
            'activation': 'tanh',
            'kernel_regularizer': tf.keras.regularizers.L2(5e-8)
        }

    },
    {
        'type': 'dense',
        'neurons': 1,
        'params': {
            'activation': 'linear'
        }
    },
    {
        'type': 'softmax'
    }
]
dropout = 0

# model_type = "asset_independent_model"
# layers = [
#     {
#         'type': 'lstm',
#         'neurons': 6,
#         'params': {
#             'activation': 'tanh',
#             'return_sequences': True,
#             'kernel_regularizer': tf.keras.regularizers.L2(5e-4)
#         }
#
#     },
#     {
#         'type': 'BatchNormalization'
#     },
#     {
#         'type': 'lstm',
#         'neurons': 6,
#         'params': {
#             'activation': 'tanh',
#             'kernel_regularizer': tf.keras.regularizers.L2(5e-4)
#         }
#     },
#     {
#         'type': 'BatchNormalization'
#     },
#     {
#         'type': 'dense',
#         'neurons': 1,
#         'params': {
#             'activation': 'linear'
#         }
#     },
#     {
#         'type': 'simple_long_only'
#     }
# ]
# dropout = 0
#
# model_type = "EIIE"
# layers = [
#     {
#         'type': 'conv2d',
#         'filters': 2,
#         'kernel_size': [1, 5],
#         'params': {
#             'strides': (1, 1),
#             'activation': 'relu',
#             'kernel_regularizer': tf.keras.regularizers.L2(5e-4)
#         }
#     },
#     {
#         'type': 'EIIE_dense',
#         'filters': len(pairs),
#         'params': {
#             'activation': 'relu',  'kernel_regularizer': tf.keras.regularizers.L2(5e-4)
#         }
#     },
# ]
# dropout = 0#.05

# Training
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# TODO implement early stopping
n_epochs = 100
# lr_scheduler = {
#     0: 0.01,
#     1: 0.001,
#     12: 0.0005,
#     14: 0.0001
# }

lr_scheduler = {
    0: 0.001,
    10: 0.0005,
    25: 0.0001,
    55: 0.00005
}

# lr_scheduler = {
#     0: 0.005,
#     50: 0.001
# }
batch_size = 128
log_every = 1
plot_every = 10
cv_type = 'incremental'
load_model = None  # 'CV'
nb_folds = 1
val_size = 1  # in months

# Strategy, loss function
freq = 14400 # 14400 # 43200 # 14400 # 14400 #* 6  # 4h
no_cash = True
trading_fee = 0.00075

# loss_config = {
#     'name': 'penalized_volatility_returns',
#     'params': {'benchmark': 0., 'alpha': 1.5}
# }

# loss_config = {
#     'name': 'sharpe_ratio',
#     'params': {'benchmark': 0., 'annual_period': 365*4}
# }
loss_config = {
    'name': 'sortino_ratio',
    'params': {'benchmark': 0.}
}

# loss_config = {
#     'name': 'sharpe_ratio',
#     'params': {'benchmark': 0., 'annual_period': 1}
# }
# loss_config = {
#     'name': 'penalized_volatility_returns',
#     'params': {'benchmark': 0.0093, 'alpha': 0.8}
# }
# Benchmark
benchmark = 'marketcap'  # 'equally_weighted'  # 'marketcap'
# Output
model_name = "asset_independent_SR"
save = False

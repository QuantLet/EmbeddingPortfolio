import tensorflow as tf
from dl_portfolio.metrics import penalized_volatility_returns, sharpe_ratio

# Random seed
seed = None

# Data
path = 'data/forex/daily_train.p'   # 'data/crypto_data/price/train_data_1800.p'  # './first_etf_strat/data/clean_VTI_AGG_DBC.p'
pairs = ['AUDUSD', 'CADUSD', 'EURUSD', 'GBPUSD', 'JPYUSD'] # ['BTC', 'DASH', 'ETH', 'LTC', 'XEM', 'XMR', 'XRP']  # ['VTI', 'AGG', 'DBC'] #
start_date = '2016-01-01'
seq_len = 7
lookfront = 1
features = [
    {'name': 'close', 'lookback': 0},
    {'name': 'open_close_returns', 'lookback': 0},
    {'name': 'realized_volatility', 'params': {'time_period': 22}, 'lookback': 21}
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
# model_type = "asset_independent_model"
# layers = [
#     {
#         'type': 'gru',
#         'neurons': 1,
#         'params': {
#             'activation': 'tanh',
#             'kernel_regularizer': tf.keras.regularizers.L2(5e-9)
#         }
#
#     },
#     {
#         'type': 'dense',
#         'neurons': 1,
#         'params': {
#             'activation': 'linear'
#         }
#     },
#     {
#         'type': 'softmax'
#     }
# ]
# dropout = 0

# model_type = "asset_independent_model"
# layers = [
#     {
#         'type': 'lstm',
#         'neurons': 64,
#         'params': {
#             'activation': 'tanh',
#             'return_sequences': False,
#             'kernel_regularizer': tf.keras.regularizers.L2(5e-4)
#         }
#
#     },
#     {
#         'type': 'softmax'
#     }
# ]
# dropout = 0
#
model_type = "stacked_asset_model"
layers = [
    {
        'type': 'lstm',
        'neurons': 64,
        'params': {
            'activation': 'tanh',
            'return_sequences': False,
            'kernel_regularizer': tf.keras.regularizers.L1(5e-8)
        }

    },
    {
        'type': 'simple_long_only'
    }
]
dropout = 0

#
# model_type = "EIIE"
# layers = [
#     {
#         'type': 'conv2d',
#         'filters': 4,
#         'kernel_size': [1, 3],
#         'params': {
#             'strides': (1, 1),
#             'activation': 'relu',
#             'kernel_initializer': 'ones',  # 'glorot_uniform',
#             'kernel_regularizer': tf.keras.regularizers.L2(5e-8)
#         }
#     },
#     {
#         'type': 'EIIE_dense',
#         'filters': len(pairs),
#         'params': {
#             'activation': 'relu',
#             'kernel_initializer': 'random_normal',  # 'glorot_uniform',
#             'kernel_regularizer': tf.keras.regularizers.L2(5e-8)
#         }
#     },
# ]
# dropout = 0.05  # .05

# Training
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# TODO implement early stopping
n_epochs = 2000
# lr_scheduler = {
#     0: 0.01,
#     1: 0.001,
#     12: 0.0005,
#     14: 0.0001
# }

lr_scheduler = {
    0: 0.0001,
    300: 0.00001,
    300: 0.000005
}
callbacks = {
    'early_stopping': {
        'patience': 5,
        'min_epoch': 100,
        'monitor': 'loss' # 'total_loss'
    }
}
# lr_scheduler = {
#     0: 0.005,
#     50: 0.001
# }
batch_size = 256
log_every = 1
plot_every = 20
cv_type = 'incremental'
load_model = None  # 'CV'
nb_folds = 3
val_size = 12  # in months

# Strategy, loss function
freq = 1  # 14400 # 43200 # 14400 # 14400 #* 6  # 4h
no_cash = True
# crypto_trading_fee = 0.00075
trading_fee = 0.00002

# loss_config = {
#     'name': 'penalized_volatility_returns',
#     'params': {'benchmark': 0., 'alpha': 1.5}
# }

# loss_config = {
#     'name': 'sharpe_ratio',
#     'params': {'benchmark': 0., 'annual_period': 365*4}
# }
loss_config = {
    'name': 'sharpe_ratio',
    'params': {'benchmark': 0.}
}

# loss_config = {
#     'name': 'sortino_ratio',
#     'params': {'benchmark': 0.00075}
# }

# loss_config = {
#     'name': 'sharpe_ratio',
#     'params': {'benchmark': 0., 'annual_period': 1}
# }
# loss_config = {
#     'name': 'penalized_volatility_returns',
#     'params': {'benchmark': 0.0093, 'alpha': 0.8}
# }
# Benchmark
benchmark = 'equally_weighted'  # 'equally_weighted'  # 'marketcap'

# Output
model_name = "stacked_asset_model_DSRNN_SR"
save = False


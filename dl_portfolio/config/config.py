import tensorflow as tf
from dl_portfolio.metrics import penalized_volatility_returns, sharpe_ratio

# Random seed
seed = None

# Data
start_date = '2017-01-01'
seq_len = 24
features = [
    {'name': 'close'},
    {'name': 'returns', 'params': {'time_period': 1}}
]
preprocess = {
    'close': {'method': 'seq_normalization', 'params': {'base': -1}}
}

# Model
model_type = "asset_independent_model"
layers = [
    {
        'type': 'lstm',
        'neurons': 2,
        'params': {
            'activation': 'tanh',
            'return_sequences': True
        }

    },
    {
        'type': 'lstm',
        'neurons': 1,
        'params': {
            'activation': 'linear'
        }
    },
    {
        'type': 'simple_long_only'
    }
]


# model_type = "EIIE"
# layers = [
#     {
#         'type': 'conv2d',
#         'filters': 2,
#         'kernel_size': [1, 3],
#         'params': {
#             'strides': (1, 1),
#             'activation': 'relu',
#             'kernel_regularizer': tf.keras.regularizers.L2(5e-4)
#         }
#     },
#     {
#         'type': 'EIIE_dense',
#         'filters': 8,
#         'params': {
#             'activation': 'relu',  'kernel_regularizer': tf.keras.regularizers.L2(5e-4)
#         }
#     },
# ]
dropout = 0

# Training
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.8)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# TODO implement early stopping
n_epochs = 150
lr_scheduler = {
    0: 0.01
}
batch_size = 128
log_every = 1
plot_every = 20
cv_type='fold'
load_model = 'CV'
nb_folds = 5
val_size = 6  # in months

# Strategy, loss function
freq = 14400  # 4h
no_cash = True
trading_fee = 0.0075

loss_config = {
    'name': 'cum_return',
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

# Output
model_name = "EIIE_SR"
save = False

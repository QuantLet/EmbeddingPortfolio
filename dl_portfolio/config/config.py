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
    'close': {'method': 'minmax', 'feature_range': [-1, 1]}
}

# Model
model_type = "EIIE"
layers = [
    {
        'type': 'conv',
        'filters': 2,
        'kernel_size': [1, 3],
        'params': {
            'strides': (1, 1),
            'activation': 'tanh'
        }
    },
    {
        'type': 'EIIE_dense',
        'filters': 10,
        'params': {
            'activation': 'tanh',
            'kernel_regularizer': tf.keras.regularizers.L2(5e-9)
        }
    },
]

model_name = "EIIE_vol_pen_returns"
n_hidden = 1
dropout = 0

# Training
load_model = None
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.95)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# TODO implement early stopping
n_epochs = 100
lr_scheduler = {
    0: 0.01,
    40: 0.001,
    80: 0.0001
}
batch_size = 16
log_every = 1
plot_every = 10
nb_folds = 5
test_size = 300

# Strategy, loss function
freq = 14400  # 4h
no_cash = True
trading_fee = 0.0075
# loss_config = {
#     'name': 'sharpe_ratio',
#     'params': {'benchmark': 0., 'annual_period': 1}
# }
loss_config = {
    'name': 'penalized_volatility_returns',
    'params': {'benchmark': 0., 'alpha': 0.2}
}
# Output
save = True

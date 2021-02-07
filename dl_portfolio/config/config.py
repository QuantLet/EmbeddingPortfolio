import tensorflow as tf

# Random seed
seed = None

# Data
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

model_name = "first_EIIE"
n_hidden = 1
dropout = 0

# Training
load_model = None
learning_rate = 0.001

lr_scheduler = {
    0: 0.01,
    15: 0.001,
    40: 0.0001
}

momentum = 0.8
n_epochs = 100
batch_size = 64
log_every = 1
plot_every = 20
nb_folds = 5
test_size = 300

# Strategy, loss function
freq = 14400 # 4h
no_cash = False
benchmark = 0.
annual_period = 1
trading_fee = 0.0075

# Output
save = False

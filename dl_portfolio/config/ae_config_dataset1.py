import tensorflow as tf
from dl_portfolio.constraints import NonNegAndUnitNorm
from dl_portfolio.regularizers import WeightsOrthogonality

dataset = 'dataset1'
show_plot = False
save = True
nmf_model = "./final_models/nmf/dataset1/m_0_seed_7_20220322_122627"


resample = {
    'method': 'nbb',
    'where': ['train'],
    'block_length': 60,
    'when': 'each_epoch'
}

seed = None

encoding_dim = 4
batch_normalization = True
uncorrelated_features = True
weightage = 1e-2
ortho_weightage = 1e-2
l_name = 'l1'
l = 1e-3
activation = 'relu'
features_config = None
model_name = f"{dataset}_nbb_resample_bl_{resample['block_length']}"
model_name = model_name.replace('.', 'd')

shuffle_columns = False
scaler_func = {
    'name': 'StandardScaler'
}
model_type = 'ae_model'

learning_rate = 1e-3
epochs = 1000
batch_size = 32
drop_remainder_obs = False
val_size = None
test_size = 0
loss = 'mse'
label_param = None
rescale = None

# Constraints and regularizer
activity_regularizer = None
kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
kernel_regularizer = WeightsOrthogonality(
    encoding_dim,
    weightage=ortho_weightage,
    axis=0,
    regularizer={'name': l_name, 'params': {l_name: l}}
)
callback_activity_regularizer = False
kernel_constraint = NonNegAndUnitNorm(max_value=1., axis=0)

callbacks = {
    'EarlyStopping': {
        'monitor': 'val_loss',
        'min_delta': 1e-3,
        'mode': 'min',
        'patience': 100,
        'verbose': 1,
        'restore_best_weights': True
    }
}

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
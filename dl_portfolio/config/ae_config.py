import tensorflow as tf
import numpy as np
from dl_portfolio.pca_ae import NonNegAndUnitNormInit
from dl_portfolio.constraints import (UncorrelatedFeaturesConstraint, NonNegAndUnitNorm,
                                      PositiveSkewnessConstraint, TailUncorrelatedFeaturesConstraint,
                                      PositiveSkewnessUncorrConstraint)
from dl_portfolio.regularizers import WeightsOrthogonality

# tf.config.run_functions_eagerly(True)

# seed = np.random.randint(100)
data_type = ['indices', 'forex', 'forex_metals', 'commodities', 'crypto']
shuffle_columns = False  # True
dropnan = False
freq = "1H"
drop_weekends = False
shuffle_columns_while_training = False
scaler_func = {
    'name': 'StandardScaler'
}
# features_config=None
features_config = [{'name': 'hour_in_week'}]
activation = 'relu'
model_type = 'pca_ae_model'
seed = np.random.randint(0, 100)
fx = True
save = False
model_name = f'RELU_encoding4_mse_time_feature_weightage_1_NO_HKUSD'
encoding_dim = 5
learning_rate = 1e-3
epochs = 1000
batch_size = 256
drop_remainder_obs = True
val_size = 30 * 3 * 24
loss = 'mse'
label_param = None
# label_param = {
#     'lq': 0.05,
#     'uq': 0.95,
#     'window': 24 * 7
# }
# cov_loss = 'mse_with_covariance_penalty'
rescale = None

# Constraints and regularizer
# activity_regularizer = UncorrelatedFeaturesConstraint(encoding_dim, norm='1/2', weightage=1.)
activity_regularizer = None  # tf.keras.regularizers.l1(2e-5)
uncorrelated_features = True
weightage = 1e-1

kernel_initializer = NonNegAndUnitNormInit(initializer='glorot_uniform')
kernel_regularizer = WeightsOrthogonality(encoding_dim,
                                          weightage=1e-2,
                                          axis=0,
                                          regularizer={'name': 'l2', 'params': {'l2': 1e-2}})
callback_activity_regularizer = False
kernel_constraint = NonNegAndUnitNorm(axis=0)  # tf.keras.constraints.NonNeg()#


def scheduler(epoch):
    return 1e-3 * np.exp(-epoch / 5000)


callbacks = {
    'EarlyStopping': {
        'monitor': 'val_loss',
        'min_delta': 1e-2,
        'mode': 'min',
        'patience': 200,
        'verbose': 1,
        'restore_best_weights': True
    }
}

data_specs = {
    0: {
        'start': '2015-08-07',
        'end': '2019-10-30'
    },
    1: {
        'start': '2015-08-07',
        'end': '2020-01-30'
    },
    2: {
        'start': '2015-08-07',
        'end': '2020-04-30'
    },
    3: {
        'start': '2015-08-07',
        'end': '2020-07-30'
    },
    4: {
        'start': '2015-08-07',
        'end': '2020-10-30'
    },
    5: {
        'start': '2015-08-07',
        'end': '2021-01-30'
    }
}

# data_specs = {
#     0: {
#         'start': '2015-08-07',
#         'end': '2019-10-30'
#     }
# }

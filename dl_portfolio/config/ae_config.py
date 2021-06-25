import tensorflow as tf
import numpy as np
from dl_portfolio.pca_ae import NonNegAndUnitNormInit
from dl_portfolio.constraints import (UncorrelatedFeaturesConstraint, NonNegAndUnitNorm,
                                      PositiveSkewnessConstraint, TailUncorrelatedFeaturesConstraint,
                                      PositiveSkewnessUncorrConstraint)
from dl_portfolio.regularizers import WeightsOrthogonality
from dl_portfolio.constant import CRYPTO_ASSETS, COMMODITIES, FX_ASSETS, FX_METALS_ASSETS, INDICES

assets = COMMODITIES + FX_ASSETS + FX_METALS_ASSETS + INDICES + ['CRIX']
encoding_dim = 6
uncorrelated_features = True
weightage = 1e-2
ortho_weightage = 1e-2
l_name = 'l2'
l = 1e-3
activation = 'relu'
features_config = [{'name': 'hour_in_week'}]
model_name = f'CRIX_activation_{activation}_encoding_{encoding_dim}_mse_time_feature_wu_{weightage}_wo_{ortho_weightage}_{l_name}_{l}'
model_name = model_name.replace('.', 'd')

# seed = np.random.randint(100)
# data_type = ['indices', 'forex', 'forex_metals', 'commodities', 'crypto']
shuffle_columns = False  # True
dropnan = False
freq = "1H"
drop_weekends = False
shuffle_columns_while_training = False
scaler_func = {
    'name': 'StandardScaler'
}
# features_config=None
model_type = 'pca_ae_model'
seed = np.random.randint(0, 100)

show_plot = True
save = True
learning_rate = 1e-3
epochs = 1000
batch_size = 256
drop_remainder_obs = True
val_size = 30 * 24
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
activity_regularizer = None
kernel_initializer = NonNegAndUnitNormInit(initializer='glorot_uniform')
kernel_regularizer = WeightsOrthogonality(
    encoding_dim,
    weightage=ortho_weightage,
    axis=0,
    regularizer={'name': l_name, 'params': {l_name: l}}
)
callback_activity_regularizer = False
kernel_constraint = NonNegAndUnitNorm(axis=0)  # tf.keras.constraints.NonNeg()#


def scheduler(epoch):
    return 1e-3 * np.exp(-epoch / 5000)


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
        'start': '2015-08-07',
        'end': '2019-12-11'
    },
    1: {
        'start': '2015-08-07',
        'end': '2020-01-11'
    },
    2: {
        'start': '2015-08-07',
        'end': '2020-02-11'
    },
    3: {
        'start': '2015-08-07',
        'end': '2020-03-11'
    },
    4: {
        'start': '2015-08-07',
        'end': '2020-04-11'
    },
    5: {
        'start': '2015-08-07',
        'end': '2020-05-11'
    },
    6: {
        'start': '2015-08-07',
        'end': '2020-06-11'
    },
    7: {
        'start': '2015-08-07',
        'end': '2020-07-11'
    },
    8: {
        'start': '2015-08-07',
        'end': '2020-08-11'
    },
    9: {
        'start': '2015-08-07',
        'end': '2020-09-11'
    },
    10: {
        'start': '2015-08-07',
        'end': '2020-10-11'
    },
    11: {
        'start': '2015-08-07',
        'end': '2020-11-11'
    },
    12: {
        'start': '2015-08-07',
        'end': '2020-12-11'
    },
    13: {
        'start': '2015-08-07',
        'end': '2021-01-11'
    },
    14: {
        'start': '2015-08-07',
        'end': '2021-02-11'
    },
    15: {
        'start': '2015-08-07',
        'end': '2021-03-11'
    },
    16: {
        'start': '2015-08-07',
        'end': '2021-04-11'
    },
    17: {
        'start': '2015-08-07',
        'end': '2021-05-11'
    },
    18: {
        'start': '2015-08-07',
        'end': '2021-06-11'
    }

}

# data_specs = {
#     0: {
#         'start': '2015-08-07',
#         'end': '2019-12-11'
#     }
# }
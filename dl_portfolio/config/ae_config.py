import tensorflow as tf
import numpy as np
from dl_portfolio.pca_ae import NonNegAndUnitNormInit
from dl_portfolio.constraints import (UncorrelatedFeaturesConstraint, NonNegAndUnitNorm, WeightsOrthogonalityConstraint,
                                      PositiveSkewnessConstraint, TailUncorrelatedFeaturesConstraint, PositiveSkewnessUncorrConstraint)

# seed = np.random.randint(100)
data_type = ['indices', 'forex', 'forex_metals', 'crypto', 'commodities']
shuffle_columns = False  # True
drop_weekends = False
shuffle_columns_while_training = False
model_type = 'pca_ae_model'
seed = np.random.randint(0, 100)
fx = True
save = True
model_name = f'more_assets_encoding_5_norm_sqrt_REFACTOR'
encoding_dim = 5
learning_rate = 1e-3
epochs = 1000
batch_size = 256
drop_remainder_obs = True
activation = 'elu'
val_size = 30 * 3 * 24
loss = 'mse'
# cov_loss = 'mse_with_covariance_penalty'
rescale = None

# Constraints and regularizer
activity_regularizer = UncorrelatedFeaturesConstraint(encoding_dim, norm='1/2', weightage=1.)
# activity_regularizer = tf.keras.regularizers.l1(2e-5)
# activity_regularizer = PositiveSkewnessConstraint(encoding_dim, weightage=1, norm='1', normalize=False)
# activity_regularizer = PositiveSkewnessUncorrConstraint(encoding_dim, coske_weightage=1.0, uncorr_weightage=1.0)
# activity_regularizer = TailUncorrelatedFeaturesConstraint(encoding_dim, q=0.05, side='left', norm='1/2', weightage=1000.)
callback_activity_regularizer = False
kernel_initializer = NonNegAndUnitNormInit(initializer='glorot_uniform')
kernel_regularizer = WeightsOrthogonalityConstraint(encoding_dim, weightage=1e-2, axis=0)
kernel_constraint = NonNegAndUnitNorm(axis = 0) # tf.keras.constraints.NonNeg()#


# pooling = 'average'

def scheduler(epoch):
    return 1e-3 * np.exp(-epoch / 5000)



callbacks = {
    'EarlyStopping': {
        'monitor': 'val_loss',
        'min_delta': 1e-3,
        'mode': 'min',
        'patience': 50,
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
#         'end': '2021-01-30'
#     }
# }

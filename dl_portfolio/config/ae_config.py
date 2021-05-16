import tensorflow as tf
import numpy as np
from dl_portfolio.pca_ae import NonNegAndUnitNormInit
from dl_portfolio.constraints import (UncorrelatedFeaturesConstraint, NonNegAndUnitNorm, WeightsOrthogonalityConstraint,
                                      PositiveSkewnessConstraint)

# seed = np.random.randint(100)
data_type = ['indices', 'forex', 'forex_metals', 'crypto', 'commodities']
shuffle_columns = True  # True
drop_weekends = False
shuffle_columns_while_training = False
model_type = 'pca_ae_model'
seed = np.random.randint(0, 100)
fx = True
save = True
model_name = f'coskew_encoding_4_nokernel_reg_no_const'
encoding_dim = 4
learning_rate = 1e-3
epochs = 1000
batch_size = 256
drop_remainder_obs = True
activation = 'elu'
val_size = 30 * 3 * 24
loss = 'mae'
rescale = None

# Constraints and regularizer
# activity_regularizer = UncorrelatedFeaturesConstraint(encoding_dim, norm='1/2', weightage=1.)
# activity_regularizer = tf.keras.regularizers.l1(1e-3)
activity_regularizer = PositiveSkewnessConstraint(encoding_dim, weightage=1, norm='1')
callback_activity_regularizer = False
kernel_initializer = NonNegAndUnitNormInit(initializer='glorot_uniform')
kernel_regularizer = None # WeightsOrthogonalityConstraint(encoding_dim, weightage=1e-2, axis=0)
kernel_constraint = tf.keras.constraints.NonNeg() # NonNegAndUnitNorm(axis=0)  # tf.keras.constraints.NonNeg()


# pooling = 'average'

def scheduler(epoch):
    return 1e-3 * np.exp(-epoch / 5000)


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_r_square', min_delta=1e-3, patience=50, verbose=1,
        mode='max', restore_best_weights=True
    ),
    # tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
]

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

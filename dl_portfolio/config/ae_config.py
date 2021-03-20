import numpy as np
from dl_portfolio.pca_ae import NonNegAndUnitNormInit


seed = np.random.randint(100)
data_type = ['indices', 'forex', 'forex_metals', 'crypto']
seed = 69
fx = True
save = True
model_name = f'first_run_{seed}_encoding_3_weightage_1_batch_512_epoch_2000'
learning_rate = 1e-3
epochs = 2000
batch_size = 512
activation = 'elu'
encoding_dim = 3
val_size = 30 * 3 * 24
uncorr_features = True
weightage = 1.#1e-4  # 1e-2
activity_regularizer = None  # tf.keras.regularizers.l1(1e-3)
loss = 'mae'
rescale = None
kernel_initializer = NonNegAndUnitNormInit(initializer='glorot_uniform')
ortho_weights = True
non_neg_unit_norm = True
non_neg = False


def scheduler(epoch):
    if epoch < 5:
        return 1e-2
    else:
        return 1e-3


callbacks = [
    # tf.keras.callbacks.EarlyStopping(
    #     monitor='val_r_square', min_delta=1e-4, patience=300, verbose=1,
    #     mode='max', baseline=None, restore_best_weights=True
    # ),
    # tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
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
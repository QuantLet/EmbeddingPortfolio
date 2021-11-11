import tensorflow as tf
import numpy as np
from dl_portfolio.pca_ae import NonNegAndUnitNormInit
from dl_portfolio.constraints import (UncorrelatedFeaturesConstraint, NonNegAndUnitNorm,
                                      PositiveSkewnessConstraint, TailUncorrelatedFeaturesConstraint,
                                      PositiveSkewnessUncorrConstraint)
from dl_portfolio.regularizers import WeightsOrthogonality
from dl_portfolio.constant import CRYPTO_ASSETS, COMMODITIES, FX_ASSETS, FX_METALS_ASSETS, INDICES

# VALIDATION = 1 month from 2019-01-11 to 2019-12-11, THEN OUT OF SAMPLE TESTs

dataset = 'cac'
show_plot = False
save = True

resample = {
    'method': 'nbb',
    'where': ['train'],
    'block_length': 60,
    'when': 'each_epoch'
}
loss_asset_weights = None
crix = False
crypto_assets = ['BTC', 'DASH', 'ETH', 'LTC', 'XRP']

# tf.config.run_functions_eagerly(True)
seed = None  # np.random.randint(0, 100)
assets = None  # COMMODITIES + FX_ASSETS + FX_METALS_ASSETS + INDICES + CRYPTO_ASSETS  # ['CRIX']
encoding_dim = 4
uncorrelated_features = True
weightage = 1e-2
ortho_weightage = 1e-2
l_name = 'l2'
l = 1e-3
activation = 'relu'
features_config = None
model_name = f"{dataset}_nbb_resample_bl_{resample['block_length']}"
model_name = model_name.replace('.', 'd')

shuffle_columns = False  # True
dropnan = False
freq = "1D"
drop_weekends = False
shuffle_columns_while_training = False
scaler_func = {
    'name': 'StandardScaler'
}
# features_config=None
model_type = 'pca_ae_model'

learning_rate = 1e-3
epochs = 1000
batch_size = 32
drop_remainder_obs = False
val_size = None  # 22*6 # 30 * 24
test_size = 0
loss = 'mse'
label_param = None
# label_param = {
#     'lq': 0.05,
#     'uq': 0.95,
#     'window': 24
# }
# cov_loss = 'mse_with_covariance_penalty'
rescale = None

# Constraints and regularizer
batch_normalization = True
activity_regularizer = None
kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
# kernel_initializer = NonNegAndUnitNormInit(initializer='he_normal', seed=seed)
kernel_regularizer = WeightsOrthogonality(
    encoding_dim,
    weightage=ortho_weightage,
    axis=0,
    regularizer={'name': l_name, 'params': {l_name: l}}
)
# kernel_regularizer = None
callback_activity_regularizer = False
kernel_constraint = NonNegAndUnitNorm(max_value=1., axis=0)  # tf.keras.constraints.NonNeg()#


def scheduler(epoch):
    return 1e-3 * np.exp(-epoch / 5000)


callbacks = {
    'EarlyStopping': {
        'monitor': 'val_loss',
        'min_delta': 1e-3,
        'mode': 'min',
        'patience': 300,
        'verbose': 1,
        'restore_best_weights': True
    }
}

data_specs = {
    0: {
        'start': '2000-07-19',
        'val_start': '2005-01-01',
        'end': '2007-01-01'
    }
}
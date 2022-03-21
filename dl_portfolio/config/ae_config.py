import tensorflow as tf
from dl_portfolio.constraints import NonNegAndUnitNorm
from dl_portfolio.regularizers import WeightsOrthogonality

dataset = 'bond'
show_plot = False
save = False
nmf_model = "log_convex_nmf_bond_CRYPTO_encoding_4_nbb_60_test_shuffle_nov_update/m_0_seed_0_20211212_181153"

resample = {
    'method': 'nbb',
    'where': ['train'],
    'block_length': 60,
    'when': 'each_epoch'
}
loss_asset_weights = None
crix = False
crypto_assets = ['BTC', 'DASH', 'ETH', 'LTC', 'XRP']

seed = None
assets = None
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
dropnan = False
freq = "1D"
drop_weekends = False
shuffle_columns_while_training = False
scaler_func = {
    'name': 'StandardScaler'
}
# features_config=None
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
    }
}

import pandas as pd

save = True
show_plot = False
seed = None  # np.random.randint(0, 1000)

# Model
encoding_dim = 4

# Data
dataset = 'cac'
dropnan = False
freq = "1D"
resample = {
    'method': 'nbb',
    'where': ['train'],
    'block_length': 60,
    'when': 'each_epoch'
}
assets = None
scaler_func = {
    'name': 'StandardScaler'
}

# Model name
model_name = f"{dataset}_nbb_resample_bl_{resample['block_length']}"
model_name = model_name.replace('.', 'd')

# Data split
# data_specs = {
#     0: {
#         'start': '2000-07-19',
#         'val_start': '2005-01-01',
#         'end': '2007-01-01'
#     }
# }
val_start = pd.date_range('2005-01-01', '2006-12-01', freq='1MS')
val_start = [str(d.date()) for d in val_start]
val_end = pd.date_range('2005-02-01', '2007-01-01', freq='1MS')
val_end = [str(d.date()) for d in val_end]

data_specs = {}
for i in range(len(val_start)):
    data_specs[i] = {
        'start': '1989-02-01',
        'val_start': val_start[i],
        'end': val_end[i]
    }

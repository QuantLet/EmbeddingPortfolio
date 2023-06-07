import datetime as dt
import json
import numpy as np
import os
import pandas as pd
import pickle

from dl_portfolio.logger import LOGGER
from dl_portfolio.varspread.varspread import (
    get_var_spread_param,
    get_reg_coef,
    get_var_spread_signal
)
from dl_portfolio.constant import (
    DATA_SPECS_AE_DATASET1,
    DATA_SPECS_AE_DATASET2
)

dataset = "dataset1"
window_size = 250
config = json.load(open('dl_portfolio/varspread/config.json', 'r'))
comments = config['comments']
datenow = str(dt.datetime.now()).split('.')[0].replace('-', '').replace(':', '').replace(' ', '')


save_dir = 'varspread/%s_c-%s' % (datenow, comments)
# save dir
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

json.dump(config, open("%s/config.p" % (save_dir), 'w'))

qs_name = config["qs_name"]
qfit = config['qfit']
coefs = np.arange(config['coefs'][0], config['coefs'][1], config['coefs'][2])
lags = range(config['lags'][0], config['lags'][1])

result = {}

if dataset == "dataset1":
    data_specs = DATA_SPECS_AE_DATASET1
elif dataset == "dataset2":
    data_specs = DATA_SPECS_AE_DATASET2
else:
    raise NotImplementedError(dataset)

# Load econ data
data = pd.read_csv(
    "/Users/brunospilak/Documents/HU/quantlet/MLvsGARCH/MLvsGARCHecon/saved_models/20230607112758_prediction_qfit_0.05.csv",
    index_col=0,
    parse_dates=True
)
data['returns_t+1'] = - data['std_losses']

# Rescale estimates
cols = list(filter(lambda x: 'var_' in x, data.columns))
cols = cols + list(filter(lambda x: 'es_' in x, data.columns))
data[cols] = data[cols] * data[['norm_sd']].values

for qname in qs_name:
    LOGGER.info(f"Alpha: {qname}")
    quantile = float('0.%s' % qname)

    # Select one quantile
    q_data = data[['returns_t+1', 'std_losses', 'norm_sd'] +
                list(filter(lambda x: '_%s' % quantile in x, data.columns))]
    q_data.columns = [c.replace('_%s' % quantile, '') for c in q_data.columns]

    # Get labels
    varspread = pd.DataFrame()
    for cv in data_specs.keys():
        LOGGER.info(f"CV: {cv}")
        test_start = data_specs[cv]["test_start"]
        test_end = data_specs[cv]["end"]
        train_data = q_data.loc[:test_start].iloc[-window_size - 1:-1]

        p, coef = get_var_spread_param(
            train_data, train_data['returns_t+1'],  lags, coefs
        )
        # need to concatenate train set
        test_data = q_data.loc[test_start:test_end]
        test_dates = test_data.index
        test_window = pd.concat([train_data.iloc[-p:], test_data])
        qw_result = get_reg_coef(test_window, p)
        label = get_var_spread_signal(qw_result, coef)
        qw_result['label'] = label
        assert all(qw_result.index == test_dates)
        varspread = pd.concat([varspread, qw_result])

    result[qname] = varspread
    LOGGER.info('Saving temp result...')
    pickle.dump(result, open(f'{save_dir}/{qname}_vs_result.p', 'wb'))
    LOGGER.info(f"Done with alpha {qname}")

LOGGER.info('Saving result...')
pickle.dump(result, open('%s/vs_result.p' % save_dir, 'wb'))
LOGGER.info('Done')
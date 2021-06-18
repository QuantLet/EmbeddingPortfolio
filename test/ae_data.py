from dl_portfolio.ae_data import load_data
import sys
import pandas as pd
import numpy as np

data_type = ['indices', 'forex', 'forex_metals', 'commodities', 'crypto']
freq = "1D"
dropnan = False

# data, assets = load_data(type=data_type, dropnan=dropnan, freq=freq)
indices = ['UKXUSD', 'FRXUSD', 'JPXUSD', 'SPXUSD', 'NSXUSD', 'HKXUSD', 'AUXUSD']
data = pd.read_pickle('./data/histdatacom/indices_f_3600_2014_2021_close_index.p')
data = data.loc[:, pd.IndexSlice[indices, 'close']].droplevel(1, 1)


index = list(data.index)
hour = np.unique([d.hour for d in index], return_counts = True)
print(hour)

weekday = pd.DataFrame(np.array([[d.weekday(), d.hour] for d in index]), columns = ['weekday', 'hour'])


print(np.unique(weekday.loc[weekday['hour'] == 3, 'weekday'].values))
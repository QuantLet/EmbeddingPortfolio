#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dl_portfolio.data import impute_missing_risk_free, load_risk_free
from dl_portfolio.backtest import (
    get_factors_rc_and_weights,
    get_number_of_nmf_bets,
    get_number_of_pc_bets
)
import pickle
import os
import seaborn as sns


# # Save directory

# In[6]:


SAVE=True
save=SAVE
savedir = "paper_result_update/nmf"
if save:
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
   


# # Load backtest result

# In[7]:


perf_nmf_dir_1 = "./performance/test_final_models/nmf/dataset1_20230412_171402"
markowitz_dir1 = "final_models/run_11_dataset1_20230408_145352"
perf_nmf_dir_2 = "./performance/test_final_models/nmf/dataset2_20230412_171420"
markowitz_dir2 = "final_models/run_12_dataset2_20230408_145946"

perf1, port_weights1, leverage1, stats1 = load_all_backtest_result(perf_nmf_dir_1,
                                                                   markowitz_dir1,
                                                                   "dataset1")
dates1 = perf1.index

perf2, port_weights2, leverage2, stats2 = load_all_backtest_result(perf_nmf_dir_2,
                                                                   markowitz_dir2,
                                                                   "dataset2")
dates2 = perf2.index


# In[9]:


benchmark  = pd.read_csv("data/benchmarks.csv", index_col=0, parse_dates=True)
benchmark = benchmark.pct_change().dropna()

risk_free = load_risk_free()
risk_free = risk_free.reindex(benchmark.index)
risk_free = impute_missing_risk_free(risk_free)

benchmark1 = benchmark.reindex(dates1)
benchmark2 = benchmark.reindex(dates2)

leverage1 = 0.05 / (benchmark1.std() * np.sqrt(252))
lev_cost = np.repeat(risk_free.reindex(benchmark1.index).values,
                     len(leverage1), axis=-1) * (1 - leverage1.values.reshape(1,-1))
benchmark1 = benchmark1*leverage1 + lev_cost

leverage2 = 0.05 / (benchmark2.std() * np.sqrt(252))
lev_cost = np.repeat(risk_free.reindex(benchmark2.index).values, 
                     len(leverage2), axis=-1) * (1 - leverage2.values.reshape(1,-1))
benchmark2 = benchmark2*leverage2 + lev_cost



# # Dataset 1

# ## Backtest performance

# In[10]:


perf1.head()


# In[11]:


save = SAVE
save_path = f"{savedir}/performance_all_dataset1.png"
plot_final_backtest_performance(perf1, benchmark1, save=save, save_path=save_path)


# ## Backtest stats

# In[14]:


stats1


# In[23]:


save = SAVE
ORDER = [ "nmfrp", "rb_factor", "hrp", "erc", "hcaa", "markowitz", "GMV_robust",
         "equal", "equal_class", "SP500", "EuroStoxx50"]

YTICKLABELS = ["NMFRP", "NMFRB", "HRP", "ERC",
               "HCAA",  "Markowitz", "Robust-M", 
               "Equal", "Equal\nclass", "SP500", "EuroStoxx50"]
stats1.loc[['SP500', 'EuroStoxx50'], 'TTO'] = np.nan
METRICS_ORDER = [
 'Return',
 'Volatility',
 'VaR-5%',
 'ES-5%',
 'SR',
 'PSR',
 'minTRL',
 'MDD',
 'CR',
 'CEQ',
 'TTO',
 'SSPW',
 'NMF Bets',
 'PC Bets'
]
pstats = stats1.loc[ORDER, METRICS_ORDER]

metrics = [c for c in list(pstats.columns) if c not in ['Skewness', 'Excess kurtosis']]
fig, axs = plt.subplots(1, len(metrics), figsize = (23,10))
i = 0
for i, c in enumerate(metrics):
    if c in ['Return', 'VaR-5%', 'ES-5%', 'Volatility', 'MDD', 'CEQ', 'TTO']:
        xticklabel = c + '(%)'
    else:
        xticklabel = c
        
    temp = pstats[[c]] 
    min_ = min(temp[c])
    if min_ < 0:
        center = 0
    else:
        center=min(temp[c]) - np.std(temp[c])
    sns.heatmap(np.abs(temp) if c == "SSPW" else temp,
                cmap='bwr', 
                center=center,
                vmin=min(temp[c]) - np.std(temp[c]),
                vmax=max(temp[c]) + np.std(temp[c]),
                annot=True,
                fmt='.2f' if c != "minTRL" else '.1g',
                annot_kws={'color': 'black', 'size': 'xx-large', 'ha': 'center'},
                yticklabels = YTICKLABELS,
                xticklabels = [xticklabel],
                cbar=False,
                ax=axs[i])
    
    if i == 0:
        axs[i].tick_params(axis='y', labelrotation = 0, labelsize=15)
    else:
        axs[i].set_yticks([])
        
    axs[i].tick_params(axis='x', labelrotation = 0,  labelsize=15, labeltop=True, labelbottom=False)

plt.subplots_adjust(wspace=0, hspace=0)

if save:
    plt.savefig(f'{savedir}/stats_heatmap_dataset1.png', bbox_inches='tight', transparent=True)
   
    


# ## Weights

# In[17]:


save=SAVE
weights = port_weights1.copy()        
if save:
    savepath = f'{savedir}/weights_barplot_dataset1.png'
else:
    savepath = None
plot_weights(weights, savepath=savepath)


# # Dataset 2
# ## Backtest performance

# In[19]:


save = SAVE
save_path = f"{savedir}/performance_all_dataset2.png"
plot_final_backtest_performance(perf2, benchmark2, save=save, save_path=save_path)


# ## Backtest stats

# In[24]:


save = SAVE
stats2.loc[['SP500', 'EuroStoxx50'], 'TTO'] = np.nan
pstats = stats1.loc[ORDER, METRICS_ORDER]

metrics = [c for c in list(pstats.columns) if c not in ['Skewness', 'Excess kurtosis']]
fig, axs = plt.subplots(1, len(metrics), figsize = (23,10))
i = 0
for i, c in enumerate(metrics):
    if c in ['Return', 'VaR-5%', 'ES-5%', 'Volatility', 'MDD', 'CEQ', 'TTO']:
        xticklabel = c + '(%)'
    else:
        xticklabel = c
        
    temp = pstats[[c]] 
    min_ = min(temp[c])
    if min_ < 0:
        center = 0
    else:
        center=min(temp[c]) - np.std(temp[c])
    sns.heatmap(np.abs(temp) if c == "SSPW" else temp,
                cmap='bwr', 
                center=center,
                vmin=min(temp[c]) - np.std(temp[c]),
                vmax=max(temp[c]) + np.std(temp[c]),
                annot=True,
                fmt='.2f' if c != "minTRL" else '.1g',
                annot_kws={'color': 'black', 'size': 'xx-large', 'ha': 'center'},
                yticklabels = YTICKLABELS,
                xticklabels = [xticklabel],
                cbar=False,
                ax=axs[i])
    
    if i == 0:
        axs[i].tick_params(axis='y', labelrotation = 0, labelsize=15)
    else:
        axs[i].set_yticks([])
        
    axs[i].tick_params(axis='x', labelrotation = 0,  labelsize=15, labeltop=True, labelbottom=False)

plt.subplots_adjust(wspace=0., hspace=0.)

if save:
    plt.savefig(f'{savedir}/stats_heatmap_dataset2.png', bbox_inches='tight', transparent=True)
   
    


# ## Weights

# In[21]:


save=SAVE
weights = port_weights2.copy()        
if save:
    savepath = f'{savedir}/weights_barplot_dataset2.png'
else:
    savepath = None
plot_weights(weights, savepath=savepath, x_step=12)

pstatsf = pstats.drop(["Volatility", "Skewness", "Excess kurtosis", "CR"], axis=1)
pstatsf = pstatsf.astype(float).round(2)
pstatsf.index = XTICKLABELS
pstatsf.columns = ['Return (\%)', 'VaR-5% (\%)', 'ES-5% (\%)', 'SR', 'PSR', 'minTRL', 'MDD (\%)', 'CEQ (\%)', 'SSPW (\%)', 'TTO (\%)']
pstatsf["minTRL"] = pstatsf["minTRL"].round(0).astype(int)
pstatsf.to_csv(f"{savedir}/stats.csv")
# In[ ]:





import pandas as pd
import datetime as dt
import numpy as np

AVAILABLE_CRITERIA = ["gap", "aic"]

CRYPTO_ASSETS = ["BTC", "DASH", "DOGE", "ETH", "LTC", "XEM", "XMR", "XRP"]
FX_ASSETS = ["CADUSD", "CHFUSD", "EURUSD", "GBPUSD", "JPYUSD", "AUDUSD"]
FX_METALS_ASSETS = ["XAUUSD", "XAGUSD"]
INDICES = [
    "UKXUSD",
    "FRXUSD",
    "JPXUSD",
    "SPXUSD",
    "NSXUSD",
    "HKXUSD",
    "AUXUSD",
]
COMMODITIES = ["WTIUSD", "BCOUSD"]

LOG_DIR = "log_AE"

PORTFOLIOS = [
    "equal",
    "equal_class",
    "erc",
    "aerp",
    "hrp",
    "hcaa",
    "rb_factor",
    "rb_factor_full_erc"
]

BASE_FACTOR_ORDER_DATASET1_4 = ["Crypto", "Bonds", "Stocks", "Forex"]
BASE_FACTOR_ORDER_DATASET2_5 = ["USB", "FRB", "CD", "StocksUS", "Stocks"]

DATASET1_ASSETS = ['UK_B', 'US_B', 'JP_B', 'GE_B', 'CNY_FX', 'EUR_FX',
                   'GBP_FX', 'JPY_FX',
                   'GOLDS_C', 'EPRA_X', 'MXWD_X', 'NKY_X', 'SHCOMP_X', 'SPX_X',
                   'SX5E_X',
                   'UKX_X', 'ETH', 'LTC', 'DASH', 'XRP', 'BTC']
DATASET2_ASSETS = ['SP500',
                   'FTSE',
                   'EuroStoxx50',
                   'Russel2000',
                   'EuroStox_Small',
                   'FTSE_Small',
                   'MSCI_EM',
                   'CRB',
                   'Gold',
                   'US-2Y',
                   'US-5Y',
                   'US-10Y',
                   'US-30Y',
                   'French-2Y',
                   'French-5Y',
                   'French-10Y',
                   'French-30Y']


def dataset2_reference_cluster():
    ref_cluster_usb = pd.Series([0.] * len(DATASET2_ASSETS),
                                index=DATASET2_ASSETS)
    ref_cluster_usb.loc[["US-2Y", "US-5Y", "US-10Y", "US-30Y"]] = 1.

    ref_cluster_frb = pd.Series([0.] * len(DATASET2_ASSETS),
                                index=DATASET2_ASSETS)
    ref_cluster_frb.loc[
        ["French-2Y", "French-5Y", "French-10Y", "French-30Y"]] = 1.

    ref_cluster_com = pd.Series([0.] * len(DATASET2_ASSETS),
                                index=DATASET2_ASSETS)
    ref_cluster_com.loc[["CRB", "Gold"]] = 1.

    ref_cluster_us_stocks = pd.Series([0.] * len(DATASET2_ASSETS),
                                      index=DATASET2_ASSETS)
    ref_cluster_us_stocks.loc[["SP500", "Russel2000"]] = 1.

    ref_cluster_stocks = pd.Series([0.] * len(DATASET2_ASSETS),
                                   index=DATASET2_ASSETS)
    ref_cluster_stocks.loc[['FTSE', 'EuroStoxx50', 'EuroStox_Small',
                            'FTSE_Small', 'MSCI_EM', ]] = 1.

    ref_cluster = pd.concat([ref_cluster_usb, ref_cluster_frb,
                             ref_cluster_com, ref_cluster_us_stocks,
                             ref_cluster_stocks],
                            axis=1)
    assert all(ref_cluster.sum(1) == 1)
    ref_cluster = ref_cluster / np.linalg.norm(ref_cluster, axis=0)
    return ref_cluster


DATASET2_REF_CLUSTER = dataset2_reference_cluster()


def dataset1_reference_cluster():
    ref_cluster_crypto = pd.Series([0.] * len(DATASET1_ASSETS),
                                index=DATASET1_ASSETS)
    ref_cluster_crypto.loc[['ETH', 'LTC', 'DASH', 'XRP', 'BTC']] = 1.

    ref_cluster_bonds = pd.Series([0.] * len(DATASET1_ASSETS),
                                index=DATASET1_ASSETS)
    ref_cluster_bonds.loc[
        ['UK_B', 'US_B', 'JP_B', 'GE_B']] = 1.

    ref_cluster_stocks = pd.Series([0.] * len(DATASET1_ASSETS),
                                index=DATASET1_ASSETS)
    ref_cluster_stocks.loc[['EPRA_X', 'MXWD_X', 'NKY_X', 'SHCOMP_X', 'SPX_X', 'SX5E_X',
       'UKX_X']] = 1.

    ref_cluster_fx = pd.Series([0.] * len(DATASET1_ASSETS),
                                   index=DATASET1_ASSETS)
    ref_cluster_fx.loc[['CNY_FX', 'EUR_FX', 'GBP_FX', 'JPY_FX', 'GOLDS_C']] = 1.

    ref_cluster = pd.concat([ref_cluster_crypto, ref_cluster_bonds,
                             ref_cluster_stocks, ref_cluster_fx],
                            axis=1)
    assert all(ref_cluster.sum(1) == 1)
    ref_cluster = ref_cluster / np.linalg.norm(ref_cluster, axis=0)
    ref_cluster.columns = BASE_FACTOR_ORDER_DATASET1_4
    return ref_cluster


DATASET1_REF_CLUSTER = dataset1_reference_cluster()


AVAILABLE_METHODS = [
    "calibrated_exceedance",
    "hedged_strat_cum_excess_return_cluster",
    "hedged_equal_cum_excess_return_cluster",
]
METHODS_MAPPING = {k: i for i, k in enumerate(AVAILABLE_METHODS)}

# Cv dates
val_start = pd.date_range(
    "2017-06-01", "2023-02-01", freq="1MS"
)
start = [d - dt.timedelta(days=365) for d in val_start]
start = [str(d.date()) for d in start]
val_start = [str(d.date()) for d in val_start]

end = pd.date_range("2017-06-01", "2023-03-01", freq="M")
end = [str(d.date()) for d in end]

data_specs = {}
for i in range(len(val_start)):
    data_specs[i] = {
        "start": start[i],
        "val_start": val_start[i],
        "end": end[i],
    }
DATA_SPECS_NMF_DATASET1 = data_specs.copy()

val_start = pd.date_range("2017-05-01", "2023-01-01", freq="1MS")
test_start = pd.date_range("2017-06-01", "2023-02-01", freq="1MS")
end = pd.date_range("2017-06-01", "2023-03-01", freq="M")

val_start = [str(d.date()) for d in val_start]
test_start = [str(d.date()) for d in test_start]
end = [str(d.date()) for d in end]

data_specs = {}
for i in range(len(val_start)):
    data_specs[i] = {
        "start": "2016-06-30",
        "val_start": val_start[i],
        "test_start": test_start[i],
        "end": end[i],
    }
DATA_SPECS_AE_DATASET1 = data_specs.copy()

# Dataset 2
val_start = pd.date_range("1990-01-01", "2021-10-01", freq="1MS")
test_start = pd.date_range("1990-02-01", "2021-11-01", freq="1MS")
val_start = [str(d.date()) for d in val_start]
test_start = [str(d.date()) for d in test_start]

end = pd.date_range("1990-02-01", "2021-12-01", freq="1M")
end = [str(d.date()) for d in end]

data_specs = {}
for i in range(len(val_start)):
    data_specs[i] = {
        "start": "1989-02-01",
        "val_start": val_start[i],
        "test_start": test_start[i],
        "end": end[i],
    }
DATA_SPECS_AE_DATASET2 = data_specs.copy()

val_start = pd.date_range("1990-02-01", "2021-11-01", freq="1MS")
start = [d - dt.timedelta(days=365) for d in val_start]
start = [str(d.date()) for d in start]
val_start = [str(d.date()) for d in val_start]

end = pd.date_range("1990-02-01", "2021-12-01", freq="1M")
end = [str(d.date()) for d in end]

data_specs = {}
for i in range(len(val_start)):
    data_specs[i] = {
        "start": start[i],
        "val_start": val_start[i],
        "end": end[i],
    }

for i in range(len(val_start)):
    data_specs[i] = {
        "start": start[i],
        "val_start": val_start[i],
        "end": end[i],
    }
DATA_SPECS_NMF_DATASET2 = data_specs.copy()

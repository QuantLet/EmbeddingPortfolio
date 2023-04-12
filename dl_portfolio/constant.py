import pandas as pd
import datetime as dt

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

BASE_FACTOR_ORDER_DATASET1_4 = ["GE_B", "SPX_X", "EUR_FX", "BTC"]
BASE_FACTOR_ORDER_DATASET1_3 = ["GE_B", "SPX_X", "BTC"]
BASE_FACTOR_ORDER_DATASET2_5 = [
    "SP500",
    "EuroStox_Small",
    "Gold",
    "US-5Y",
    "French-5Y",
]

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

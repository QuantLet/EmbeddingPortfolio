import pandas as pd

BASE_FREQ = 1800
BASE_COLUMNS = ["open", "high", "low", "close", "volume", "quoteVolume"]
RESAMPLE_DICT = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quoteVolume": "sum",
}

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
    "markowitz",
    "shrink_markowitz",
    "ivp",
    "aerp",
    "rp",
    "sector_erc",
    "aeerc",
    "ae_rp_c",
    "kmaa",
    "aeaa",
    "hrp",
    "hcaa",
    "drp",
    "principal"
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

DATA_SPECS_BOND = {
    0: {
        "start": "2016-06-30",
        "val_start": "2019-11-13",
        "test_start": "2019-12-12",
        "end": "2020-01-11",
    },
    1: {
        "start": "2016-06-30",
        "val_start": "2019-12-13",
        "test_start": "2020-01-12",
        "end": "2020-02-11",
    },
    2: {
        "start": "2016-06-30",
        "val_start": "2020-01-13",
        "test_start": "2020-02-12",
        "end": "2020-03-11",
    },
    3: {
        "start": "2016-06-30",
        "val_start": "2020-02-13",
        "test_start": "2020-03-12",
        "end": "2020-04-11",
    },
    4: {
        "start": "2016-06-30",
        "val_start": "2020-03-13",
        "test_start": "2020-04-12",
        "end": "2020-05-11",
    },
    5: {
        "start": "2016-06-30",
        "val_start": "2020-04-13",
        "test_start": "2020-05-12",
        "end": "2020-06-11",
    },
    6: {
        "start": "2016-06-30",
        "val_start": "2020-05-13",
        "test_start": "2020-06-12",
        "end": "2020-07-11",
    },
    7: {
        "start": "2016-06-30",
        "val_start": "2020-06-13",
        "test_start": "2020-07-12",
        "end": "2020-08-11",
    },
    8: {
        "start": "2016-06-30",
        "val_start": "2020-07-13",
        "test_start": "2020-08-12",
        "end": "2020-09-11",
    },
    9: {
        "start": "2016-06-30",
        "val_start": "2020-08-13",
        "test_start": "2020-09-12",
        "end": "2020-10-11",
    },
    10: {
        "start": "2016-06-30",
        "val_start": "2020-09-13",
        "test_start": "2020-10-12",
        "end": "2020-11-11",
    },
    11: {
        "start": "2016-06-30",
        "val_start": "2020-10-13",
        "test_start": "2020-11-12",
        "end": "2020-12-11",
    },
    12: {
        "start": "2016-06-30",
        "val_start": "2020-11-13",
        "test_start": "2020-12-12",
        "end": "2021-01-11",
    },
    13: {
        "start": "2016-06-30",
        "val_start": "2020-12-13",
        "test_start": "2021-01-12",
        "end": "2021-02-11",
    },
    14: {
        "start": "2016-06-30",
        "val_start": "2021-01-13",
        "test_start": "2021-02-12",
        "end": "2021-03-11",
    },
    15: {
        "start": "2016-06-30",
        "val_start": "2021-02-13",
        "test_start": "2021-03-12",
        "end": "2021-04-11",
    },
    16: {
        "start": "2016-06-30",
        "val_start": "2021-03-13",
        "test_start": "2021-04-12",
        "end": "2021-05-11",
    },
    17: {
        "start": "2016-06-30",
        "val_start": "2021-04-13",
        "test_start": "2021-05-12",
        "end": "2021-06-11",
    },
    18: {
        "start": "2016-06-30",
        "val_start": "2021-05-13",
        "test_start": "2021-06-12",
        "end": "2021-07-11",
    },
    19: {
        "start": "2016-06-30",
        "val_start": "2021-06-13",
        "test_start": "2021-07-12",
        "end": "2021-08-11",
    },
    20: {
        "start": "2016-06-30",
        "val_start": "2021-07-13",
        "test_start": "2021-08-12",
        "end": "2021-09-11",
    },
    21: {
        "start": "2016-06-30",
        "val_start": "2021-08-13",
        "test_start": "2021-09-12",
        "end": "2021-10-11",
    },
}

val_start = pd.date_range("2007-01-01", "2021-09-01", freq="1MS")
val_start = [str(d.date()) for d in val_start]

test_start = pd.date_range("2007-02-01", "2021-10-01", freq="1MS")
test_start = [str(d.date()) for d in test_start]
test_end = pd.date_range("2007-02-01", "2021-11-01", freq="1M")
test_end = [str(d.date()) for d in test_end]

DATA_SPECS_MULTIASSET_TRADITIONAL = {}
for i in range(len(val_start)):
    DATA_SPECS_MULTIASSET_TRADITIONAL[i] = {
        "start": "1989-02-01",
        "val_start": val_start[i],
        "test_start": test_start[i],
        "end": test_end[i],
    }

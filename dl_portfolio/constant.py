BASE_FREQ = 1800
BASE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'quoteVolume']
RESAMPLE_DICT = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
                 'quoteVolume': 'sum'}

CRYPTO_ASSETS = ['BTC', 'DASH', 'DOGE', 'ETH', 'LTC', 'XEM', 'XMR', 'XRP']
FX_ASSETS = ['CADUSD', 'CHFUSD', 'EURUSD', 'GBPUSD', 'JPYUSD', 'AUDUSD']
FX_METALS_ASSETS = ['XAUUSD', 'XAGUSD']
INDICES = ['UKXUSD', 'FRXUSD', 'JPXUSD', 'SPXUSD', 'NSXUSD', 'HKXUSD', 'AUXUSD']
COMMODITIES = ['WTIUSD', 'BCOUSD']

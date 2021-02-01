
BASE_FREQ = 1800
BASE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'quoteVolume']
RESAMPLE_DICT = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
                 'quoteVolume': 'sum'}
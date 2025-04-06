import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from config.settings import SYMBOLS, TIME_FRAME
from config.oanda import ACCOUNT_ID, ACCESS_TOKEN, ENVIRONMENT

class DataFetcher:
    def __init__(self):
        self.client = API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)
        
    def fetch_multi_symbol_data(self, count=1000):
        data = {}
        for symbol in SYMBOLS:
            params = {
                "count": count,
                "granularity": TIME_FRAME,
                "price": "BA"
            }

            endpoint = InstrumentsCandles(instrument=symbol, params=params)
            response = self.client.request(endpoint)
            candles = response['candles']

            df = pd.DataFrame([{
                "time": c["time"],
                "open": float(c["bid"]["o"]),
                "high": float(c["bid"]["h"]),
                "low": float(c["bid"]["l"]),
                "close": float(c["bid"]["c"]),
                "volume": int(c["volume"])
            } for c in candles])
            
            data[symbol] = df.set_index("time")
        
        return data
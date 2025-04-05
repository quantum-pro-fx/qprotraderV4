from oandapyV20 import API
from oandapyV20.endpoints import instruments
import pandas as pd
from src.config import Config

def fetch_historical_data(count=500):
    """Fetch historical candle data from Oanda"""
    client = API(access_token=Config.ACCESS_TOKEN)
    params = {
        "count": count,
        "granularity": Config.TIMEFRAME,
        "price": "BA"
    }
    
    r = instruments.InstrumentsCandles(
        instrument=Config.SYMBOL,
        params=params
    )
    client.request(r)
    
    # Convert to pandas DataFrame
    candles = r.response['candles']
    data = []
    for candle in candles:
        data.append({
            'time': candle['time'],
            'open': float(candle['bid']['o']),
            'high': float(candle['bid']['h']),
            'low': float(candle['bid']['l']),
            'close': float(candle['bid']['c']),
            'volume': int(candle['volume'])
        })
    
    return pd.DataFrame(data)
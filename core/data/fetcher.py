import os
import pickle
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.instruments import InstrumentsCandles
from config.settings import SYMBOLS, TIME_FRAME
from config.oanda import ACCOUNT_ID, ACCESS_TOKEN, ENVIRONMENT
import time

class DataFetcher:
    def __init__(self, max_retries=3, retry_delay=1, cache_dir=".market_data_cache"):
        self.client = API(access_token=ACCESS_TOKEN, environment=ENVIRONMENT)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)  # Ensure cache directory exists
        
        # Timeframe to timedelta mapping for freshness checks
        self.timeframe_map = {
            'S5': timedelta(seconds=5),
            'S10': timedelta(seconds=10),
            'S15': timedelta(seconds=15),
            'S30': timedelta(seconds=30),
            'M1': timedelta(minutes=1),
            'M2': timedelta(minutes=2),
            'M4': timedelta(minutes=4),
            'M5': timedelta(minutes=5),
            'M10': timedelta(minutes=10),
            'M15': timedelta(minutes=15),
            'M30': timedelta(minutes=30),
            'H1': timedelta(hours=1),
            'H2': timedelta(hours=2),
            'H3': timedelta(hours=3),
            'H4': timedelta(hours=4),
            'H6': timedelta(hours=6),
            'H8': timedelta(hours=8),
            'H12': timedelta(hours=12),
            'D': timedelta(days=1),
            'W': timedelta(weeks=1),
            'M': timedelta(days=30)
        }

    def _get_cache_file(self, symbol):
        """Generate cache file path for a symbol"""
        return self.cache_dir / f"{symbol}_{TIME_FRAME}.pkl"

    def _is_data_fresh(self, df):
        """
        Check if cached data is still fresh based on timeframe.
        Returns True if data is fresh and should be used.
        """
        if df is None or df.empty:
            return False
            
        # Get expected cadence for this timeframe
        cadence = self.timeframe_map.get(TIME_FRAME, timedelta(minutes=5))
        
        # Ensure we're working with timezone-aware datetimes
        last_candle_time = df.index[-1].to_pydatetime()
        if last_candle_time.tzinfo is None:
            last_candle_time = last_candle_time.replace(tzinfo=timezone.utc)
        
        # Calculate when the next candle should arrive
        next_expected_candle = last_candle_time + cadence
        
        # Add buffer (2x timeframe) to account for potential delays
        buffer_time = 2 * cadence
        fresh_until = next_expected_candle + buffer_time
        
        # Compare with current time (timezone-aware)
        now = datetime.now(timezone.utc)
        return now < fresh_until

    def _load_from_cache(self, symbol):
        """Load cached data if available and fresh"""
        cache_file = self._get_cache_file(symbol)
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
                
                # Ensure DataFrame index is timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                    
                if self._is_data_fresh(df):
                    print(f"Using fresh cached data for {symbol}")
                    return df
                print(f"Cached data for {symbol} is stale")
        except Exception as e:
            print(f"Error loading cache for {symbol}: {e}")
        return None

    def _save_to_cache(self, symbol, df):
        """Save data to cache"""
        if df is None or df.empty:
            return
            
        try:
            # Ensure DataFrame index is timezone-aware before saving
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
                
            cache_file = self._get_cache_file(symbol)
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            print(f"Error saving cache for {symbol}: {e}")

    def fetch_symbol_data(self, symbol, count=5000):
        """Fetch data for a single symbol with caching"""
        # Try to load from cache first
        cached_data = self._load_from_cache(symbol)
        if cached_data is not None:
            return cached_data
            
        # If no fresh cache, fetch from API
        for attempt in range(self.max_retries):
            try:
                params = {
                    "count": min(count, 5000),
                    "granularity": TIME_FRAME,
                    "price": "BA"
                }

                endpoint = InstrumentsCandles(instrument=symbol, params=params)
                response = self.client.request(endpoint)
                
                if 'candles' not in response or not response['candles']:
                    raise ValueError(f"No candle data returned for {symbol}")
                
                processed = []
                for c in response['candles']:
                    if not c['complete']:
                        continue
                        
                    try:
                        processed.append({
                            "time": c["time"],
                            "bid_open": float(c["bid"]["o"]),
                            "bid_high": float(c["bid"]["h"]),
                            "bid_low": float(c["bid"]["l"]),
                            "bid_close": float(c["bid"]["c"]),
                            "ask_open": float(c["ask"]["o"]),
                            "ask_high": float(c["ask"]["h"]),
                            "ask_low": float(c["ask"]["l"]),
                            "ask_close": float(c["ask"]["c"]),
                            "volume": int(c["volume"]),
                            "complete": c["complete"]
                        })
                    except KeyError as e:
                        print(f"Malformed candle data for {symbol}, skipping: {e}")
                        continue
                
                if not processed:
                    raise ValueError(f"No complete candles for {symbol}")
                
                df = pd.DataFrame(processed)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                # Ensure timezone-aware index
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                # Calculate mid prices
                df['open'] = (df['bid_open'] + df['ask_open']) / 2
                df['high'] = (df['bid_high'] + df['ask_high']) / 2
                df['low'] = (df['bid_low'] + df['ask_low']) / 2
                df['close'] = (df['bid_close'] + df['ask_close']) / 2
                
                # Save to cache before returning
                self._save_to_cache(symbol, df)
                return df
                
            except V20Error as e:
                print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt == self.max_retries - 1:
                    print(f"Max retries reached for {symbol}")
                    return None
                time.sleep(self.retry_delay)
            except Exception as e:
                print(f"Unexpected error fetching {symbol}: {e}")
                return None

    def fetch_multi_symbol_data(self, count=5000):
        """Fetch data for all symbols with caching"""
        data = {}
        for symbol in SYMBOLS:
            symbol_data = self.fetch_symbol_data(symbol, count)
            if symbol_data is not None:
                data[symbol] = symbol_data
            else:
                print(f"Warning: No data retrieved for {symbol}")
        return data
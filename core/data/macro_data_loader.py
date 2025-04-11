from collections import deque
import numpy as np
import pandas as pd
import logging
import time
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
from config.settings import FRED_API_KEY


class MacroDataLoader:
    def __init__(self, fred_api_key: Optional[str] = None):
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        self.fred_api_key = fred_api_key or FRED_API_KEY
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def get_current_vix(self) -> float:
        """Fetch VIX from FRED API with caching"""
        cache_key = "vix"
        if cache_key not in self.cache or time.time() - self.cache[cache_key]['timestamp'] > self.cache_duration:
            try:
                # FRED API implementation
                self.cache[cache_key] = {
                    'value': self._fetch_fred_data("VIXCLS"),
                    'timestamp': time.time()
                }
            except Exception as e:
                logging.error(f"VIX fetch failed: {str(e)}")
                return 20.0  # Fallback to average volatility
        return self.cache[cache_key]['value']
    
    def get_yield_spread(self) -> float:
        """10Y-2Y Treasury spread"""
        cache_key = "yield_spread"
        if cache_key not in self.cache or time.time() - self.cache[cache_key]['timestamp'] > self.cache_duration:
            try:
                ten_year = self._fetch_fred_data("DGS10")
                two_year = self._fetch_fred_data("DGS2")
                self.cache[cache_key] = {
                    'value': ten_year - two_year,
                    'timestamp': time.time()
                }
            except Exception:
                logging.error("Yield curve fetch failed")
                return 0.5  # Normal slope fallback
        return self.cache[cache_key]['value']
    
    def _fetch_fred_data(self, series_id: str) -> float:
        """Generic FRED API fetcher"""
        if not self.fred_api_key:
            raise ValueError("FRED API key not configured")
            
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'observation_start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'observation_end': datetime.now().strftime('%Y-%m-%d'),
            'limit': 1
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return float(data['observations'][0]['value'])
        except Exception as e:
            logging.error(f"FRED API fetch failed for {series_id}: {str(e)}")
            raise

    def get_economic_calendar(self, days: int = 7) -> List[Dict]:
        """Fetch upcoming economic events"""
        cache_key = f"economic_calendar_{days}"
        if cache_key not in self.cache or time.time() - self.cache[cache_key]['timestamp'] > self.cache_duration:
            try:
                url = f"https://economic-calendar.tradingview.com/events?minImportance=1&days={days}"
                response = self.session.get(url)
                response.raise_for_status()
                self.cache[cache_key] = {
                    'value': response.json(),
                    'timestamp': time.time()
                }
            except Exception as e:
                logging.error(f"Economic calendar fetch failed: {str(e)}")
                return []
        return self.cache[cache_key]['value']

    def clear_cache(self):
        """Clear all cached data"""
        self.cache = {}
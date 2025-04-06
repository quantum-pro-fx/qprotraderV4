import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

class FeatureEngineer:
    def __init__(self, min_data_points: int = 50):
        """
        Enhanced feature engineering with:
        - Robust error handling
        - Input validation
        - Parallel processing support
        - Feature validation
        
        Args:
            min_data_points: Minimum observations required for reliable indicators
        """
        self.feature_config = {
            "price": ["close", "returns", "log_returns"],
            "trend": ["ema_10", "ema_30", "macd_line", "macd_signal"],
            "momentum": ["rsi_14", "stoch_k", "stoch_d"],
            "volatility": ["atr_14", "bb_upper", "bb_middle", "bb_lower"],
            "volume": ["obv", "volume_delta", "volume_ma"]
        }
        self.min_data_points = max(min_data_points, 50)  # Never less than 50
        self._validate_indicators()

        # Initialize with FIXED feature set
        self.expected_features = [
            # Price features
            'close', 'returns', 'log_returns',
            # Trend features
            'ema_10', 'ema_30', 'macd_line', 'macd_signal',
            # Momentum features
            'rsi_14', 'stoch_k', 'stoch_d',
            # Volatility features
            'atr_14', 'bb_upper', 'bb_middle', 'bb_lower',
            # Volume features
            'obv', 'volume_delta', 'volume_ma'
        ]
        self.min_data_points = max(min_data_points, 50)
        
    def _validate_indicators(self):
        """Verify all configured indicators are available"""
        missing = []
        for indicator in set().union(*self.feature_config.values()):
            if not hasattr(ta, indicator.split('_')[0]):
                missing.append(indicator)
        if missing:
            warnings.warn(f"Unavailable indicators: {missing}")

    def _validate_input(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Validate input DataFrame with improved checks"""
        if df.empty:
            raise ValueError("Empty DataFrame received")
            
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
            
        validated_df = df[list(required_cols)].copy()
        
        # Forward fill OHLCV data
        validated_df[['open', 'high', 'low', 'close']] = validated_df[['open', 'high', 'low', 'close']].ffill()
        validated_df['volume'] = validated_df['volume'].fillna(0)
        
        close_series = validated_df["close"].dropna()
        
        if len(close_series) < 1:  # We need at least current observation
            raise ValueError("No valid price data available")
            
        return close_series, validated_df

    def _safe_ta_indicator(self, func, *args, **kwargs) -> Optional[pd.Series]:
        """Compute TA indicator with comprehensive error handling"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(*args, **kwargs)
                
            if result is None:
                return None
                
            if isinstance(result, pd.DataFrame):
                return result.iloc[:, 0] if len(result.columns) == 1 else result
            return result
        except Exception as e:
            warnings.warn(f"Indicator {func.__name__} failed: {str(e)}")
            return None

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core feature calculation logic"""
        close_series, df = self._validate_input(df)
        
        # ===== Price Features =====
        df["returns"] = close_series.pct_change()
        df["log_returns"] = np.log(close_series).diff()
        
        # ===== Trend Features =====
        df["ema_10"] = ta.ema(close_series, length=10) if len(close_series) >= 10 else np.nan
        df["ema_30"] = ta.ema(close_series, length=30) if len(close_series) >= 30 else np.nan
        
        if len(close_series) >= 26:
            macd = ta.macd(close_series, fast=12, slow=26, signal=9)
            if macd is not None:
                df["macd_line"] = macd.get("MACD_12_26_9", np.nan)
                df["macd_signal"] = macd.get("MACDs_12_26_9", np.nan)

        # ===== Momentum Features =====
        df["rsi_14"] = ta.rsi(close_series, length=14)  if len(close_series) >= 14 else np.nan
        
        stoch = ta.stoch(df["high"], df["low"], close_series, k=14, d=3)
        if stoch is not None:
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]
        
        # ===== Volatility Features =====
        df["atr_14"] = ta.atr(df["high"], df["low"], close_series, length=14) if len(close_series) >= 14 else np.nan
        
        if len(close_series) >= 20:
            bb = ta.bbands(close_series, length=20)
            if bb is not None:
                df["bb_upper"] = bb["BBU_20_2.0"]
                df["bb_middle"] = bb["BBM_20_2.0"] 
                df["bb_lower"] = bb["BBL_20_2.0"]
            
        # ===== Volume Features =====
        df["obv"] = ta.obv(close_series, df["volume"])
        df["volume_delta"] = df["volume"].diff()
        df["volume_ma"] = df["volume"].rolling(20).mean()
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent output features"""
        try:
            # Calculate features
            result = self._calculate_features(df)
            
            # Ensure all expected features are present
            for feature in self.expected_features:
                if feature not in result.columns:
                    result[feature] = np.nan
            
            # Return only expected features in consistent order
            return result[self.expected_features].iloc[[-1]]
            
        except Exception as e:
            raise ValueError(f"Feature transformation failed: {str(e)}") from e

    def transform_multi(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process multiple symbols with parallel processing option"""
        from concurrent.futures import ThreadPoolExecutor
        
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                symbol: executor.submit(self.transform, df)
                for symbol, df in data.items() if not df.empty
            }
            
            for symbol, future in futures.items():
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    warnings.warn(f"Failed processing {symbol}: {str(e)}")
                    
        return results
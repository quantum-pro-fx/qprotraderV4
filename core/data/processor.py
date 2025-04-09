import numpy as np
import pandas as pd
import talib
from scipy.stats import zscore
from typing import Dict, List, Tuple, Optional
import logging
import warnings

class InstitutionalFeatureEngineer:
    """Enhanced institutional feature engineering with 25 market features"""
    
    def __init__(self, 
                 min_data_points: int = 100, 
                 expected_shape: Tuple[int, int] = (5, 25),  # Updated for 25 features
                 atr_window: int = 14):
        self.logger = logging.getLogger(__name__)
        self.min_data_points = max(min_data_points, 100)
        self.expected_shape = expected_shape
        self.atr_window = atr_window
        
        warnings.filterwarnings('once', category=RuntimeWarning)
        
        # Updated feature groups with 25 total features
        self.feature_groups = {
            'price': [
                ('close', lambda d: d['close'].values),
                ('normalized_close', self._calculate_normalized_close),
                ('log_ret', self._calculate_log_returns),
                ('zscore_50', self._calculate_zscore_50),
                ('hilo_ratio', self._calculate_hilo_ratio)
            ],
            'momentum': [
                ('rsi', self._calculate_rsi),
                ('macd', self._calculate_macd),
                ('stoch', self._calculate_stoch),
                ('cci', self._calculate_cci),
                ('adx', self._calculate_adx)
            ],
            'volatility': [
                ('atr', self._calculate_atr),
                ('bb_width', self._calculate_bb_width),
                ('keltner', self._calculate_keltner),
                ('chandelier', self._calculate_chandelier),
                ('vix', self._calculate_vix)
            ],
            'volume': [
                ('obv', self._calculate_obv),
                ('vwap', self._calculate_vwap),
                ('volume_osc', self._calculate_volume_osc),
                ('eom', self._calculate_eom),
                ('mfi', self._calculate_mfi)
            ],
            'patterns': [
                ('doji', self._calculate_doji),
                ('engulfing', self._calculate_engulfing),
                ('hammer', self._calculate_hammer),
                ('evening_star', self._calculate_evening_star),
                ('piercing', self._calculate_piercing)
            ]
        }

    # ======== Core Calculation Methods ========
    def _safe_calculation(self, func, df: pd.DataFrame, default=0.0) -> np.ndarray:
        try:
            if func is None:
                return np.full(len(df), default)
            result = func(df)
            return np.nan_to_num(result.astype(np.float64), nan=default)
        except Exception as e:
            func_name = getattr(func, '__name__', 'unnamed_function')
            self.logger.debug(f"Feature failed: {func_name} - {str(e)}")
            return np.full(len(df), default)

    # ======== Price Features ========
    def _calculate_normalized_close(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: zscore(d['close'], nan_policy='omit'), df)

    def _calculate_log_returns(self, df: pd.DataFrame) -> np.ndarray:
        close = df['close'].values.astype(np.float64)
        log_ret = np.zeros_like(close)
        log_ret[1:] = np.log(close[1:]/close[:-1])
        return log_ret

    def _calculate_zscore_50(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: (d['close'] - d['close'].rolling(50).mean()) / 
                     d['close'].rolling(50).std(), df)

    def _calculate_hilo_ratio(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: (d['high'] - d['low']) / (d['high'] + 1e-6), df)

    # ======== Momentum Features ========
    def _calculate_rsi(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.RSI(d['close'], timeperiod=14), df)

    def _calculate_macd(self, df: pd.DataFrame) -> np.ndarray:
        macd, _, _ = talib.MACD(df['close'])
        return self._safe_calculation(lambda _: macd, df)

    def _calculate_stoch(self, df: pd.DataFrame) -> np.ndarray:
        slowk, _ = talib.STOCH(df['high'], df['low'], df['close'])
        return self._safe_calculation(lambda _: slowk, df)

    def _calculate_cci(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.CCI(d['high'], d['low'], d['close'], timeperiod=20), df)

    def _calculate_adx(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.ADX(d['high'], d['low'], d['close'], timeperiod=14), df)

    # ======== Volatility Features ========
    def _calculate_atr(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.ATR(d['high'], d['low'], d['close'], timeperiod=self.atr_window), df)

    def _calculate_bb_width(self, df: pd.DataFrame) -> np.ndarray:
        upper, _, lower = talib.BBANDS(df['close'])
        return self._safe_calculation(
            lambda _: (upper - lower) / df['close'].rolling(20).mean(), df)

    def _calculate_keltner(self, df: pd.DataFrame) -> np.ndarray:
        atr = talib.ATR(df['high'], df['low'], df['close'], 20)
        ema = talib.EMA(df['close'], timeperiod=20)
        return self._safe_calculation(lambda _: (df['close'] - ema) / (2 * atr), df)

    def _calculate_chandelier(self, df: pd.DataFrame) -> np.ndarray:
        atr = talib.ATR(df['high'], df['low'], df['close'], 22)
        max_high = df['high'].rolling(22).max()
        return self._safe_calculation(lambda _: max_high - 3 * atr, df)

    def _calculate_vix(self, df: pd.DataFrame) -> np.ndarray:
        log_ret = np.log(df['close']/df['close'].shift(1))
        return self._safe_calculation(
            lambda _: log_ret.rolling(20).std() * np.sqrt(252) * 100, df)

    # ======== Volume Features ========
    def _calculate_obv(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.OBV(d['close'], d['volume']), df)

    def _calculate_vwap(self, df: pd.DataFrame) -> np.ndarray:
        tp = (df['high'] + df['low'] + df['close']) / 3
        return self._safe_calculation(
            lambda _: (tp * df['volume']).rolling(20).sum() / 
                      df['volume'].rolling(20).sum(), df)

    def _calculate_volume_osc(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: (d['volume'].rolling(5).mean() / 
                      d['volume'].rolling(20).mean() - 1) * 100, df)

    def _calculate_eom(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: ((d['high'] + d['low']) / 2 - 
                      (d['high'].shift(1) + d['low'].shift(1)) / 2) * 
                      (d['high'] - d['low']) / d['volume'], df)

    def _calculate_mfi(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.MFI(d['high'], d['low'], d['close'], d['volume'], 14), df)

    # ======== Pattern Features ========
    def _calculate_doji(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.CDLDOJI(d['open'], d['high'], d['low'], d['close']), df)

    def _calculate_engulfing(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.CDLENGULFING(d['open'], d['high'], d['low'], d['close']), df)

    def _calculate_hammer(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.CDLHAMMER(d['open'], d['high'], d['low'], d['close']), df)

    def _calculate_evening_star(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.CDLEVENINGSTAR(d['open'], d['high'], d['low'], d['close'], 0.5), df)

    def _calculate_piercing(self, df: pd.DataFrame) -> np.ndarray:
        return self._safe_calculation(
            lambda d: talib.CDLPIERCING(d['open'], d['high'], d['low'], d['close']), df)

    # ======== Core Processing ========
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all 25 features with robust error handling"""
        try:
            df_clean = df.copy()
            df_clean['volume'] = df_clean.get('volume', 1.0)
            df_clean = df_clean.ffill().bfill()
            
            features = {}
            for group_name, group in self.feature_groups.items():
                for feature_name, calculator in group:
                    key = f"{group_name}_{feature_name}" if group_name != 'price' else feature_name
                    if calculator is None:
                        # Handle direct column access
                        if feature_name in df_clean.columns:
                            features[key] = df_clean[feature_name].values.astype(np.float32)
                        else:
                            features[key] = np.full(len(df_clean), 0.0)
                    else:
                        features[key] = self._safe_calculation(calculator, df_clean)
            
            return self._create_output_dataframe(features, len(df_clean))
        
        except Exception as e:
            self.logger.error(f"Transform failed: {str(e)}")
            return self._create_fallback_output()

    def _create_output_dataframe(self, features: dict, data_length: int) -> pd.DataFrame:
        """Ensure exact 25-feature output shape"""
        feature_names = sorted(features.keys())[:25]  # Force 25 features
        output = pd.DataFrame(index=range(data_length))
        
        for name in feature_names:
            output[name] = features.get(name, 0.0)
        
        # Pad if needed (shouldn't occur with 25 features)
        if len(output.columns) < 25:
            for i in range(25 - len(output.columns)):
                output[f'pad_{i}'] = 0.0
                
        return output.iloc[-self.expected_shape[0]:].astype(np.float32)

    def _create_fallback_output(self) -> pd.DataFrame:
        return pd.DataFrame(
            0.0, 
            index=range(self.expected_shape[0]), 
            columns=[f"feature_{i}" for i in range(25)]
        ).astype(np.float32)
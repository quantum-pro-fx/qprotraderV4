import pandas as pd
import pandas_ta as ta
from src.config import Config

def generate_features(df):
    """Add technical indicators to the dataframe"""
    # Price transformations
    df['returns'] = df['close'].pct_change()
    
    # Moving averages
    df['ema_10'] = ta.ema(df['close'], length=10)
    df['ema_30'] = ta.ema(df['close'], length=30)
    
    # Oscillators
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # Volatility
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Drop NA values from indicator calculations
    return df.dropna()

def add_advanced_features(df):
    # Price action features
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    
    # Candlestick patterns
    df['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1).astype(int)
    
    # Volume spike
    mean_vol = df['volume'].rolling(20).mean()
    df['volume_spike'] = (df['volume'] > 2 * mean_vol).astype(int)
    
    return df
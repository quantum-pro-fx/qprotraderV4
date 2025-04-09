import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Optional, List
from core.data.processor import FeatureEngineer
from core.utils.logger import TradingLogger
import warnings

class MultiSymbolTradingEnv(gym.Env):
    def __init__(self, symbol_data: Dict[str, pd.DataFrame], initial_balance: float = 1e6):
        self.feature_engineer = FeatureEngineer()
        self.symbol_data = self._clean_and_validate_data(symbol_data)
        self.symbols = sorted(self.symbol_data.keys())
        
        # Initialize spaces after data validation
        self.action_space = self._init_action_space()
        self.observation_space = self._init_observation_space()
        
        self.reset()

    def _clean_and_validate_data(self, symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Thorough data cleaning and validation"""
        validated = {}
        for symbol, df in symbol_data.items():
            try:
                # Basic validation
                df = df.copy()
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns in {symbol}")
                    
                # Clean data
                df = df[required_cols].ffill().bfill()
                df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                
                # Validate no NaNs remain
                if df.isnull().any().any():
                    raise ValueError(f"NaN values remain in {symbol} after cleaning")
                    
                validated[symbol] = df
            except Exception as e:
                warnings.warn(f"Removing {symbol} due to: {str(e)}")
                
        if not validated:
            raise ValueError("No valid symbols after data cleaning")
        return validated

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Safe observation generation with validation"""
        market_obs = []
        for symbol in self.symbols:
            try:
                window = self.symbol_data[symbol].iloc[
                    max(0, self.current_step-self.feature_engineer.min_data_points):self.current_step
                ]
                features = self.feature_engineer.transform(window)
                
                # Validate features
                if features.isnull().any().any():
                    warnings.warn(f"NaN in features for {symbol} - using zeros")
                    features = features.fillna(0)
                    
                # Ensure we have the last observation
                if features.empty:
                    features = pd.DataFrame(
                        np.zeros((1, len(self.feature_engineer.feature_priority))),
                        columns=self.feature_engineer.feature_priority
                    )
                
                market_obs.append(features.iloc[-1].values.astype(np.float32))
            except Exception as e:
                warnings.warn(f"Observation failed for {symbol}: {str(e)}")
                market_obs.append(np.zeros(len(self.feature_engineer.feature_priority), dtype=np.float32))
        
        # Portfolio state
        try:
            portfolio_value = self._get_portfolio_value()
            exposure = sum(
                abs(pos['size'] * self._get_current_price(sym))
                for sym, pos in self.positions.items() if pos
            ) / max(1e-6, portfolio_value)
            
            peak = max(self._portfolio_values)
            drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
            
            portfolio_obs = np.array([
                portfolio_value,
                exposure,
                drawdown
            ], dtype=np.float32)
        except Exception:
            portfolio_obs = np.zeros(3, dtype=np.float32)
        
        return {
            "market": np.array(market_obs, dtype=np.float32),
            "portfolio": portfolio_obs
        }

    def reset(self, seed=None, options=None):
        """Safe reset with validation"""
        super().reset(seed=seed)
        
        self.current_step = self.feature_engineer.min_data_points
        self.balance = self.initial_balance
        self.positions = {symbol: None for symbol in self.symbols}
        self._portfolio_values = [self.initial_balance]
        
        # Validate initial observation
        obs, info = self._get_obs(), {}
        self._validate_observation(obs)
        return obs, info

    def _validate_observation(self, obs: Dict[str, np.ndarray]):
        """Explicit observation validation"""
        if np.isnan(obs['market']).any():
            raise ValueError("NaN detected in market observations after reset")
        if np.isnan(obs['portfolio']).any():
            raise ValueError("NaN detected in portfolio observations")
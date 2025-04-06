import gymnasium as gym
import numpy as np
from gymnasium import spaces
from core.data.processor import FeatureEngineer
from core.utils.logger import TradingLogger

class MultiSymbolTradingEnv(gym.Env):
    def __init__(self, symbol_data):
        self.symbol_data = symbol_data
        self.symbols = list(symbol_data.keys())
        self.feature_engineer = FeatureEngineer()
        
        # Verify data has enough rows
        min_length = self.feature_engineer.min_data_points
        for symbol, df in symbol_data.items():
            if len(df) < min_length:
                raise ValueError(f"{symbol} has only {len(df)} rows, need {min_length}")
        
        # Set observation space based on actual feature dimension
        sample_features = self._get_sample_features()
        self.feature_dim = sample_features.shape[1]
        
        # Action space (0=hold, 1=long, 2=short) for each symbol
        self.action_space = spaces.MultiDiscrete([3] * len(self.symbols))
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.symbols), self.feature_dim),
            dtype=np.float32
        )
        
        self.reset()

    def _get_sample_features(self):
        """Get features from first valid symbol"""
        for symbol, df in self.symbol_data.items():
            try:
                return self.feature_engineer.transform(df.iloc[:self.feature_engineer.min_data_points]).values
            except Exception:
                continue
        raise ValueError("No valid symbol data for feature extraction")

    def reset(self, seed=None, **kwargs):
        self.current_step = self.feature_engineer.min_data_points - 1
        self.positions = {symbol: 0 for symbol in self.symbols}
        return self._get_obs(), {}

    def step(self, actions):
        rewards = []
        for i, symbol in enumerate(self.symbols):
            reward = self._process_trade(symbol, actions[i])
            rewards.append(reward)
        
        self.current_step += 1
        done = self.current_step >= len(self.symbol_data[self.symbols[0]]) - 1
        return self._get_obs(), np.mean(rewards), done, False, {}

    def _process_trade(self, symbol, action):
        current_price = self.symbol_data[symbol].iloc[self.current_step]['close']
        prev_price = self.symbol_data[symbol].iloc[self.current_step-1]['close']
        
        # Calculate reward from previous action
        reward = (current_price - prev_price) / prev_price * self.positions[symbol]
        
        # Update position
        if action == 1:  # Long
            self.positions[symbol] = 1
        elif action == 2:  # Short
            self.positions[symbol] = -1
        else:  # Hold
            pass
            
        return reward

    def _get_obs(self):
        """Get current observation with feature validation"""
        obs = []
        for symbol in self.symbols:
            window = self.symbol_data[symbol].iloc[
                self.current_step-self.feature_engineer.min_data_points+1:self.current_step+1
            ]
            features = self.feature_engineer.transform(window).values[-1]
            obs.append(features)
        return np.array(obs)
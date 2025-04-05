import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.config import Config

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        
        # Action space: 0=hold, 1=long, 2=short
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self._get_features()),), dtype=np.float32)
        
        # Track trading state
        self.position = 0  # -1, 0, 1
        self.balance = Config.INITIAL_BALANCE
        self.equity = []

    def _get_features(self):
        return self.df.iloc[self.current_step][[
            'returns', 'ema_10', 'ema_30', 'rsi', 'atr'
        ]].values

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Calculate safe start range
        max_possible_start = len(self.df) - Config.MIN_STEPS_FOR_EPISODE
        
        # Ensure we have enough data
        if max_possible_start <= 0:
            raise ValueError(f"Not enough data. Need at least {Config.MIN_STEPS_FOR_EPISODE} steps, got {len(self.df)}")
        
        # Set starting point
        if Config.RANDOM_START and max_possible_start > 1:
            self.current_step = self.np_random.integers(0, max_possible_start)
        else:
            self.current_step = 0
        
        self.position = 0
        self.balance = Config.INITIAL_BALANCE
        info = {}
        return self._get_features(), info

    def step(self, action):
        # Calculate reward from previous action
        prev_value = self._calculate_portfolio_value()
        
        # Update position (0=hold, 1=long, 2=short)
        self.position = 0 if action == 0 else 1 if action == 1 else -1
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate new portfolio value
        current_value = self._calculate_portfolio_value()
        reward = current_value - prev_value
        
        # Check if done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # We don't use early stopping
        
        info = {
            'step': self.current_step,
            'position': self.position,
            'value': current_value
        }
        
        return self._get_features(), reward, terminated, truncated, info

    def _calculate_portfolio_value(self):
        if self.current_step == 0:
            return self.balance
        price = self.df.iloc[self.current_step]['close']
        prev_price = self.df.iloc[self.current_step-1]['close']
        position_value = self.position * (price - prev_price)
        return self.balance + position_value
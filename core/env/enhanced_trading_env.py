from core.data.enhanced_features_engine import EnhancedFeatureEngineer
from core.env.trading_env import InstitutionalTradingEnv
from core.execution.adaptive_execution_engine import AdaptiveExecutionEngine
import numpy as np
import logging
from gymnasium import spaces
from typing import Dict, Optional
import pandas as pd

class EnhancedTradingEnv(InstitutionalTradingEnv):
    """
    Enhanced trading environment with:
    - Advanced feature engineering
    - Improved regime detection
    - Adaptive execution parameters
    """
    
    def __init__(self, symbol_data: Dict[str, pd.DataFrame], 
                 initial_balance: float = 1000.00,
                 window_size: int = 5):
        """
        Initialize the enhanced trading environment
        
        Args:
            symbol_data: Dictionary of DataFrames with market data for each symbol
            initial_balance: Starting account balance
            window_size: Number of historical steps for observation window
        """
        # Initialize parent class
        super().__init__(
            symbol_data=symbol_data,
            initial_balance=initial_balance,
            window_size=window_size
        )
        
        # Override with enhanced components
        self.execution_engine = AdaptiveExecutionEngine()
        self.feature_engineer = EnhancedFeatureEngineer(
            min_data_points=100,
            expected_shape=(window_size, self.feature_engineer.get_feature_count())
        )
        
        # Update observation space to match new feature dimensions
        self.observation_space = self._init_observation_space()
        self.history = []  # Initialize observation history

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self.history.append(obs)  # Store observations
        return obs, reward, done, truncated, info

    def _init_observation_space(self):
        """Update observation space to match enhanced feature dimensions"""
        feature_count = self.feature_engineer.get_feature_count()
        return spaces.Dict({
            "market": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_symbols, self.window_size, feature_count),
                dtype=np.float32
            ),
            "portfolio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),  # [balance, exposure, drawdown, volatility]
                dtype=np.float32
            )
        })

    def _execute_trade(self, symbol: str, action: np.ndarray, execution_params: Optional[dict] = None):
        """
        Enhanced trade execution with adaptive parameters
        
        Args:
            symbol: Trading instrument symbol
            action: Array of [position_size, stop_loss_pct, take_profit_pct]
            execution_params: Optional pre-computed execution parameters
        """
        try:
            if execution_params is None:
                # Generate execution parameters if not provided
                market_data = {
                    'bid': self.symbol_data[symbol].iloc[self.current_step]['open'],
                    'ask': self.symbol_data[symbol].iloc[self.current_step]['close'],
                    'mid': (self.symbol_data[symbol].iloc[self.current_step]['high'] + 
                           self.symbol_data[symbol].iloc[self.current_step]['low'])/2,
                    'volume': self.symbol_data[symbol].iloc[self.current_step]['volume']
                }
                
                # Get current regime from enhanced features
                obs = self._get_observation()
                regime_feature = obs['market'][:, -3:]  # Use macro composite features
                regime = int(np.clip(np.mean(regime_feature), 0, 2))  # Ensure 0-2 range
                
                execution_params = self.execution_engine.get_execution_params(
                    symbol, 
                    market_data,
                    regime
                )
            
            # Call parent execution with computed parameters
            super()._execute_trade(symbol, action, execution_params)
            
        except Exception as e:
            logging.error(f"Enhanced trade execution failed for {symbol}: {str(e)}", exc_info=True)
            if symbol in self.positions:
                self._close_position(symbol)

    def _get_observation(self):
        """Generate observation with enhanced features"""
        feature_count = self.feature_engineer.get_feature_count()
        market_obs = np.zeros((self.n_symbols, self.window_size, feature_count), dtype=np.float32)
        
        for i, symbol in enumerate(self.symbols):
            try:
                window = self.symbol_data[symbol].iloc[
                    max(0, self.current_step-self.window_size+1):self.current_step+1
                ]
                features = self.feature_engineer.transform(window)
                
                # Ensure feature dimensions match
                if features.shape[1] != feature_count:
                    raise ValueError(f"Feature dimension mismatch for {symbol}: "
                                   f"expected {feature_count}, got {features.shape[1]}")
                
                market_obs[i] = features.values[-self.window_size:]
            except Exception as e:
                logging.warning(f"Error processing {symbol}: {str(e)}")
                # Fill with zeros if feature generation fails
                market_obs[i] = np.zeros((self.window_size, feature_count))
                
        # Portfolio state remains the same
        portfolio_value = self._get_portfolio_value()
        exposure = sum(
            abs(pos['size'] * self._get_current_price(sym))
            for sym, pos in self.positions.items() if pos
        ) / max(1e-6, portfolio_value)
        
        drawdown = (max(self.portfolio_history) - portfolio_value) / max(self.portfolio_history) if self.portfolio_history else 0
        volatility = np.std(self.returns[-20:]) * np.sqrt(252) if len(self.returns) >= 20 else 0.2
        
        return {
            "market": market_obs,
            "portfolio": np.array([
                portfolio_value,
                exposure,
                drawdown,
                volatility
            ], dtype=np.float32)
        }
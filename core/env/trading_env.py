import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Optional, List
from core.data.processor import InstitutionalFeatureEngineer
from core.utils.logger import TradingLogger
import warnings
from sklearn.impute import KNNImputer
import logging
from core.execution.adaptive_execution_engine import AdaptiveExecutionEngine

class InstitutionalTradingEnv(gym.Env):
    """Professional trading environment with improved feature handling"""
    
    def __init__(self, symbol_data: Dict[str, pd.DataFrame], 
                 initial_balance: float = 1000.00,
                 window_size: int = 5):
        
        self.n_features = InstitutionalFeatureEngineer().get_feature_count()
        self.feature_engineer = InstitutionalFeatureEngineer(
            min_data_points=100,
            expected_shape=(window_size, self.n_features)
        )
    
        self.symbol_data = self._preprocess_data(symbol_data)
        self.symbols = sorted(self.symbol_data.keys())
        self.n_symbols = len(self.symbols)
        self.initial_balance = initial_balance
        self.window_size = window_size
        
        # Trading parameters
        self.max_position_size = 0.2  # 20% per symbol
        self.max_sl_tp = 0.05  # 5% stop loss/take profit
        self.commission = 0.0002  # 2 basis points
        self.slippage = 0.0001  # 1 basis point
        
        # Initialize spaces
        self.action_space = self._init_action_space()
        self.observation_space = self._init_observation_space()
        self.execution_engine = AdaptiveExecutionEngine()

        # Initialize state variables
        self.positions = {}
        self.current_step = 0
        self.balance = initial_balance
        self.portfolio_history = []
        self.returns = []
        
        # Reset environment
        self.reset()

    def _preprocess_data(self, symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Enhanced preprocessing with feature validation"""
        processed = {}
        for symbol, df in symbol_data.items():
            # Validate OHLCV structure
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                warnings.warn(f"Skipping {symbol} - missing required columns")
                continue
                
            # Clean data
            df = df[required_cols].copy()
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()
            
            if df.isnull().any().any():
                warnings.warn(f"Skipping {symbol} - NaNs persist after cleaning")
                continue
                
            processed[symbol] = df
            
        return processed

    def _init_action_space(self):
        """Action space with position, SL, TP for each symbol"""
        return spaces.Box(
            low=np.array([-self.max_position_size, 0, 0] * self.n_symbols),
            high=np.array([self.max_position_size, self.max_sl_tp, self.max_sl_tp] * self.n_symbols),
            dtype=np.float32
        )

    def _init_observation_space(self):
        """Strict observation space definition"""
        return spaces.Dict({
            "market": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_symbols, self.window_size, self.n_features),
                dtype=np.float32
            ),
            "portfolio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),  # [balance, exposure, drawdown, volatility]
                dtype=np.float32
            )
        })

    def reset(self, seed=None, options=None):
        """Reset with proper initialization"""
        super().reset(seed=seed)
        
        self.current_step = self.feature_engineer.min_data_points
        self.balance = self.initial_balance
        self.positions = {symbol: None for symbol in self.symbols}
        self.portfolio_history = [self.initial_balance]
        self.returns = []
        
        return self._get_observation(), {}

    def step(self, action):
        """Institutional-grade step execution"""
        try:
            action = np.array(action, dtype=np.float32).reshape((self.n_symbols, 3))
            action = np.clip(
                action,
                [-self.max_position_size, 0, 0],
                [self.max_position_size, self.max_sl_tp, self.max_sl_tp]
            )
            
            obs = self._get_observation()
            market_features = obs['market'][:, -1, :]
            regime = self._detect_regime(market_features)
            
            for i, symbol in enumerate(self.symbols):
                if symbol not in self.symbol_data:
                    continue
                    
                current_data = self.symbol_data[symbol].iloc[self.current_step]
                market_data = {
                    'bid': current_data['low'],
                    'ask': current_data['high'],
                    'atr': np.mean(obs['market'][i, -5:, 2]),
                    'volume': current_data['volume'],
                    'close': current_data['close']
                }
                
                exec_params = self.execution_engine.get_execution_params(
                    symbol, 
                    market_data, 
                    regime
                )
                self._execute_trade(symbol, action[i], exec_params)
                
            self.current_step += 1
            portfolio_value = self._get_portfolio_value()
            reward = self._calculate_reward(portfolio_value)
            done = self._should_terminate(portfolio_value)
            
            return self._get_observation(), reward, done, False, {}
            
        except Exception as e:
            logging.error(f"Step failed: {str(e)}", exc_info=True)
            return self._get_observation(), 0, True, False, {}

    def _execute_trade(self, symbol: str, action: np.ndarray, execution_params: dict) -> None:
        """Professional trade execution with adaptive parameters"""
        try:
            # Input validation
            if symbol not in self.symbols:
                raise ValueError(f"Invalid symbol: {symbol}")
            if not isinstance(action, np.ndarray) or action.shape != (3,):
                raise ValueError(f"Invalid action shape: {action.shape}")
            if not isinstance(execution_params, dict):
                raise ValueError("execution_params must be a dictionary")
                    
            position_size, sl_pct, tp_pct = action
            current_data = self.symbol_data[symbol].iloc[self.current_step]
            
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in current_data for col in required_cols):
                missing = [col for col in required_cols if col not in current_data]
                raise ValueError(f"Missing columns for {symbol}: {missing}")
                
            # Calculate execution parameters
            price = current_data['close']
            daily_volume = current_data['volume']
            position_value = abs(position_size) * self.balance
            
            # Apply participation rate
            participation_rate = execution_params.get('participation_rate', 0.2)
            position_value *= participation_rate
            
            # Calculate slippage
            volume_ratio = position_value / (daily_volume + 1e-6)
            slippage_pct = 0.001 + 0.004 * volume_ratio
            
            # Determine execution price
            if execution_params['order_type'] == 'LIMIT':
                execution_price = price * (1 + np.sign(position_size) * 
                                        min(execution_params['price_tolerance'], slippage_pct))
            else:
                execution_price = price * (1 + np.sign(position_size) * slippage_pct)
            
            # Risk controls
            max_trade_value = daily_volume * 0.01
            if abs(position_value) > max_trade_value:
                position_value = np.sign(position_size) * max_trade_value
            
            # Apply hard limits
            position_size = np.clip(position_value / self.balance, 
                                  -self.max_position_size, 
                                  self.max_position_size)
            
            sl_pct = np.clip(sl_pct, 0.005, 0.05)
            tp_pct = np.clip(tp_pct, 0.005, 0.05)
            
            # Close existing position if direction changes
            current_pos = self.positions.get(symbol)
            if current_pos and np.sign(position_size) != np.sign(current_pos['size']):
                self._close_position(symbol)
            
            # Calculate commission
            commission = abs(position_value) * (self.commission + 0.0001 * volume_ratio)
            
            # Only open new position if above minimum threshold
            if abs(position_size) >= 0.001:
                self.positions[symbol] = {
                    'size': position_value / execution_price,
                    'entry_price': execution_price,
                    'stop_loss': execution_price * (1 - np.sign(position_size) * sl_pct),
                    'take_profit': execution_price * (1 + np.sign(position_size) * tp_pct),
                    'commission': commission,
                    'slippage': slippage_pct,
                    'timestamp': self.current_step
                }
                self.balance -= commission
                
        except Exception as e:
            logging.error(f"Trade execution failed for {symbol}: {str(e)}", exc_info=True)
            if symbol in self.positions:
                self._close_position(symbol)

    def _get_observation(self):
        """Generate validated observation"""
        market_obs = np.zeros((self.n_symbols, self.window_size, self.n_features), dtype=np.float32)
        
        for i, symbol in enumerate(self.symbols):
            try:
                window = self.symbol_data[symbol].iloc[
                    max(0, self.current_step-self.window_size+1):self.current_step+1
                ]
                features = self.feature_engineer.transform(window)
                market_obs[i] = features.values[-self.window_size:]
            except Exception as e:
                logging.warning(f"Error processing {symbol}: {str(e)}")
                
        # Portfolio state
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

    def _get_portfolio_value(self):
        """Mark-to-market valuation"""
        total = self.balance
        for symbol, pos in self.positions.items():
            if pos:
                total += pos['size'] * self._get_current_price(symbol)
                
        self.portfolio_history.append(total)
        if len(self.portfolio_history) > 1:
            self.returns.append((total - self.portfolio_history[-2]) / self.portfolio_history[-2])
            
        return total
    
    def _get_market_return(self) -> float:
        """Calculates the weighted market return across all symbols"""
        try:
            if len(self.symbol_data) == 0 or self.current_step < 1:
                return 0.0
            
            total_return = 0.0
            total_weight = 0.0
            
            for symbol in self.symbols:
                current_data = self.symbol_data[symbol].iloc[self.current_step]
                prev_data = self.symbol_data[symbol].iloc[self.current_step - 1]
                symbol_return = (current_data['close'] - prev_data['close']) / prev_data['close']
                
                position = self.positions.get(symbol)
                if position and position['size'] != 0:
                    weight = abs(position['size'])
                else:
                    weight = 1.0 / len(self.symbols)
                    
                total_return += symbol_return * weight
                total_weight += weight
                
            return total_return / (total_weight + 1e-6)
        
        except Exception as e:
            logging.error(f"Market return calculation failed: {str(e)}")
            return 0.0

    def _calculate_reward(self, portfolio_value: float) -> float:
        """Robust reward calculation"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        try:
            market_return = self._get_market_return()
            portfolio_return = (portfolio_value - self.portfolio_history[-2]) / (self.portfolio_history[-2] + 1e-6)
            directional_alignment = np.sign(portfolio_return * market_return) * min(abs(portfolio_return), 0.05)
        except:
            directional_alignment = 0.0

        risk_component = 0.0
        if len(self.returns) >= 5:
            valid_returns = np.array(self.returns[-20:])
            valid_returns = valid_returns[~np.isnan(valid_returns)]
            
            if len(valid_returns) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sharpe = np.mean(valid_returns) / (np.std(valid_returns) + 1e-6) * np.sqrt(252)
                    risk_component = np.clip(sharpe, -2, 2) / 2

        try:
            max_portfolio = max(self.portfolio_history)
            current_dd = (max_portfolio - portfolio_value) / (max_portfolio + 1e-6)
            drawdown_penalty = -min(current_dd ** 2, 0.25)
        except:
            drawdown_penalty = 0.0

        return (0.6 * directional_alignment + 0.3 * risk_component + 0.1 * drawdown_penalty)

    def _should_terminate(self, portfolio_value):
        """Professional termination conditions"""
        if self.current_step >= len(next(iter(self.symbol_data.values()))) - 2:
            return True
            
        if portfolio_value < self.initial_balance * 0.7:
            return True
            
        return False
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            price = self.symbol_data[symbol].iloc[self.current_step]['close']
            slippage = price * self.slippage * np.random.uniform(-1, 1)
            return price + slippage
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {str(e)}")
            return 0.0

    def _close_position(self, symbol: str):
        """Close position for a symbol and update balance"""
        if symbol in self.positions and self.positions[symbol]:
            position = self.positions[symbol]
            current_price = self._get_current_price(symbol)
            position_value = position['size'] * current_price
            self.balance += position_value - position['commission']
            self.positions[symbol] = None

    def _detect_regime(self, market_features: np.ndarray) -> int:
        """Simple 3-regime classifier (0=low, 1=normal, 2=high vol)"""
        volatility_score = np.mean(market_features[:, 2])
        return np.digitize(volatility_score, [0.3, 0.7]) - 1
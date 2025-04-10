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

class InstitutionalTradingEnv(gym.Env):
    """Professional trading environment with improved feature handling"""
    
    def __init__(self, symbol_data: Dict[str, pd.DataFrame], 
                 initial_balance: float = 1000.00,
                 window_size: int = 5):
        
        self.n_features = InstitutionalFeatureEngineer().get_feature_count()  # Dynamic
        
        self.feature_engineer = InstitutionalFeatureEngineer(
            min_data_points=100,
            expected_shape=(window_size, self.n_features)
        )
    
        self.symbol_data = self._preprocess_data(symbol_data)
        self.symbols = sorted(self.symbol_data.keys())
        self.n_symbols = len(self.symbols)
        self.initial_balance = initial_balance
        self.window_size = window_size
        
        # Trading parameters - MOVED BEFORE action_space initialization
        self.max_position_size = 0.2  # 20% per symbol
        self.max_sl_tp = 0.05  # 5% stop loss/take profit
        self.commission = 0.0002  # 2 basis points
        self.slippage = 0.0001  # 1 basis point
        
        # Initialize spaces with strict dimensionality
        self.action_space = self._init_action_space()
        self.observation_space = self._init_observation_space()
        
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
            
            # Forward fill then backfill remaining NaNs
            df = df.ffill().bfill()
            
            # Validate
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
        # Reshape and validate action
        action = action.reshape((self.n_symbols, 3))
        action = np.clip(action, 
                        [-self.max_position_size, 0, 0], 
                        [self.max_position_size, self.max_sl_tp, self.max_sl_tp])
        
        # Execute trades
        for i, symbol in enumerate(self.symbols):
            self._execute_trade(symbol, action[i])
            
        # Update state
        self.current_step += 1
        portfolio_value = self._get_portfolio_value()
        reward = self._calculate_reward(portfolio_value)
        done = self._should_terminate(portfolio_value)
        
        return self._get_observation(), reward, done, False, {}

    def _execute_trade(self, symbol: str, action: np.ndarray) -> None:
        """
        Executes trades with institutional-grade risk controls and volume checks
        
        Args:
            symbol: Trading symbol (e.g., 'EUR_USD')
            action: Array of [position_size, stop_loss_pct, take_profit_pct]
            
        Features:
            - Volume-adjusted position sizing
            - Slippage modeling
            - Commission accounting
            - Position validation
            - Comprehensive error handling
        """
        try:
            # Validate inputs
            if symbol not in self.symbols:
                raise ValueError(f"Invalid symbol: {symbol}")
            if not isinstance(action, np.ndarray) or action.shape != (3,):
                raise ValueError(f"Invalid action shape: {action.shape}")
                
            position_size, sl_pct, tp_pct = action
            current_data = self.symbol_data[symbol].iloc[self.current_step]
            
            # Get market data with validation
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in current_data for col in required_cols):
                missing = [col for col in required_cols if col not in current_data]
                raise ValueError(f"Missing columns for {symbol}: {missing}")
                
            price = current_data['close']
            daily_volume = current_data['volume']
            
            # ========== RISK CONTROLS ========== #
            # 1. Position sizing relative to volume
            max_trade_value = daily_volume * 0.01  # Max 1% of daily volume
            position_value = position_size * self.balance
            original_size = position_size
            
            if abs(position_value) > max_trade_value:
                position_size = np.sign(position_size) * max_trade_value / self.balance
                position_value = position_size * self.balance
                logging.debug(
                    f"Position adjusted from {original_size:.2%} to {position_size:.2%} "
                    f"due to volume constraints ({symbol})"
                )
            
            # 2. Hard limits
            position_size = np.clip(position_size, -self.max_position_size, self.max_position_size)
            sl_pct = np.clip(sl_pct, 0.005, 0.05)  # 0.5% to 5%
            tp_pct = np.clip(tp_pct, 0.005, 0.05)
            
            # ========== TRADE EXECUTION ========== #
            # Close existing position if direction change
            current_pos = self.positions[symbol]
            if current_pos and np.sign(position_size) != np.sign(current_pos['size']):
                self._close_position(symbol)
                current_pos = None
                
            # Calculate slippage (0.1-0.5% depending on volume)
            volume_ratio = abs(position_value) / (daily_volume + 1e-6)
            slippage_pct = 0.001 + 0.004 * volume_ratio  # 0.1% to 0.5%
            execution_price = price * (1 + np.sign(position_size) * slippage_pct)
            
            # Calculate commission (bid-ask spread + broker fee)
            commission = abs(position_value) * (self.commission + 0.0001 * volume_ratio)
            
            # Only open new position if above minimum threshold
            if abs(position_size) >= 0.001:  # At least 0.1% position
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
                logging.info(
                    f"Opened {position_size:.2%} position in {symbol} at {execution_price:.5f} "
                    f"(SL: {sl_pct:.2%}, TP: {tp_pct:.2%})"
                )
                
        except Exception as e:
            logging.error(f"Trade execution failed for {symbol}: {str(e)}", exc_info=True)
            # Fallback: Close position if something went wrong
            if symbol in self.positions and self.positions[symbol]:
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
            except Exception:
                logging.warning(f"Error processing {symbol} - zero filling")
                
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
                # Get current and previous close prices
                current_data = self.symbol_data[symbol].iloc[self.current_step]
                prev_data = self.symbol_data[symbol].iloc[self.current_step - 1]
                
                # Calculate symbol return
                symbol_return = (current_data['close'] - prev_data['close']) / prev_data['close']
                
                # Weight by position size if holding, else equal weight
                position = self.positions.get(symbol)
                if position and position['size'] != 0:
                    weight = abs(position['size'])
                else:
                    weight = 1.0 / len(self.symbols)
                    
                total_return += symbol_return * weight
                total_weight += weight
                
            return total_return / (total_weight + 1e-6)
        
        except Exception as e:
            self.logger.error(f"Market return calculation failed: {str(e)}")
            return 0.0

    def _calculate_reward(self, portfolio_value: float) -> float:
        """
        Robust reward calculation with complete NaN/inf protection
        and proper initialization handling
        """
        # 1. Basic validation
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # 2. Directional Component (60%)
        try:
            market_return = self._get_market_return()
            portfolio_return = (portfolio_value - self.portfolio_history[-2]) / (self.portfolio_history[-2] + 1e-6)
            directional_alignment = np.sign(portfolio_return * market_return) * min(abs(portfolio_return), 0.05)
        except:
            directional_alignment = 0.0

        # 3. Risk Component (30%) - Fully protected
        risk_component = 0.0
        if len(self.returns) >= 5:  # Minimum 5 periods for meaningful stats
            valid_returns = np.array(self.returns[-20:])
            valid_returns = valid_returns[~np.isnan(valid_returns)]
            
            if len(valid_returns) > 1:
                with warnings.catch_warnings():  # Suppress runtime warnings
                    warnings.simplefilter("ignore")
                    sharpe = np.mean(valid_returns) / (np.std(valid_returns) + 1e-6) * np.sqrt(252)
                    risk_component = np.clip(sharpe, -2, 2) / 2

        # 4. Drawdown Component (10%)
        try:
            max_portfolio = max(self.portfolio_history)
            current_dd = (max_portfolio - portfolio_value) / (max_portfolio + 1e-6)
            drawdown_penalty = -min(current_dd ** 2, 0.25)
        except:
            drawdown_penalty = 0.0

        return (
            0.6 * directional_alignment +
            0.3 * risk_component +
            0.1 * drawdown_penalty
        )

    def _should_terminate(self, portfolio_value):
        """Professional termination conditions"""
        # End of data
        if self.current_step >= len(next(iter(self.symbol_data.values()))) - 2:
            return True
            
        # Risk limits
        if portfolio_value < self.initial_balance * 0.7:  # 30% drawdown
            return True
            
        return False
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol with slippage simulation"""
        try:
            price = self.symbol_data[symbol].iloc[self.current_step]['close']
            slippage = price * self.slippage * np.random.uniform(-1, 1)
            return price + slippage
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {str(e)}")
            return 0.0

    def _close_position(self, symbol: str):
        """Close position for a symbol and update balance"""
        if self.positions[symbol]:
            price = self._get_current_price(symbol)
            position_value = self.positions[symbol]['size'] * price
            self.balance += position_value - (position_value * self.commission)
            self.positions[symbol] = None
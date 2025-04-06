from core.data.fetcher import DataFetcher
from core.env.multi_symbol_env import MultiSymbolTradingEnv
from core.agents.ppo_agent import PPOAgent
from core.agents.ensemble import EnsembleAgent
from core.agents.meta_learner import MetaLearner
from core.execution.oanda_executor import OandaExecutor
from core.utils.logger import TradingLogger
import pickle
import time
from typing import Dict
import numpy as np
from pathlib import Path
from datetime import datetime

class TradingSystem:
    def __init__(self, mode="train"):
        """
        Args:
            mode: 'train' or 'live'
        """
        self.mode = mode
        self.data_fetcher = DataFetcher()
        self.executor = OandaExecutor() if mode == "live" else None
        self.logger = TradingLogger()
        self.agent = None
        
    def run(self):
        """Main entry point for the trading system"""
        try:
            if self.mode == "train":
                self._train()
            elif self.mode == "live":
                self._live_trading()
        except Exception as e:
            self.logger.log_error(e, "system_run")
            raise

    def _train(self):
        """Train the trading agent"""
        # 1. Fetch and prepare data
        symbol_data = self.data_fetcher.fetch_multi_symbol_data()
        
        # 2. Initialize environment
        env = MultiSymbolTradingEnv(symbol_data)
        
        # 3. Train individual models
        self.logger.log_metric("training_start", 1)
        model1 = PPOAgent(env)
        model1.learn(total_timesteps=100000)
        
        model2 = PPOAgent(env, learning_rate=1e-3)
        model2.learn(total_timesteps=100000)
        
        # 4. Create ensemble
        ensemble = EnsembleAgent([model1, model2])
        
        # 5. Initialize meta-learner
        self.agent = MetaLearner({
            0: ensemble,  # Default regime
            1: model1,    # Regime 1 strategy
            2: model2     # Regime 2 strategy
        })
        
        # 6. Save trained models
        self._save_agent()
        self.logger.log_metric("training_complete", 1)

    def _live_trading(self):
        """Execute live trading loop"""
        try:
            if not hasattr(self, 'agent') or self.agent is None:
                self.logger.log_system_event("No agent loaded - attempting to load from disk")
                self.agent = self._load_agent()
        
            while True:
                try:
                    # 1. Get fresh market data
                    symbol_data = self.data_fetcher.fetch_multi_symbol_data(count=100)  # Last 100 periods
                    
                    # 2. Create environment with latest data
                    env = MultiSymbolTradingEnv(symbol_data)
                    
                    # 3. Get current observation
                    obs = env._get_obs()  # Get latest state
                    
                    # 4. Get agent's action
                    action = self.agent.predict(obs)
                    
                    # 5. Execute trades
                    if self.executor:
                        self._execute_trades(action, symbol_data)
                    
                    # 6. Wait for next interval
                    time.sleep(60 * 15)  # Wait 15 minutes for next M15 candle
                    
                except Exception as e:
                    self.logger.log_error(e, "live_trading_loop")
                    time.sleep(60)  # Wait before retrying
        except RuntimeError as e:
            self.logger.log_error(e, "live_trading_startup")
            self.logger.log_system_event("Switching to training mode")
            self.mode = "train"
            self.run()  # Fall back to training

    def _execute_trades(self, action: np.ndarray, symbol_data: Dict):
        """Execute trades with proper risk management"""
        for i, symbol in enumerate(symbol_data.keys()):
            if action[i] != 0:  # Only act on non-hold signals
                current_data = symbol_data[symbol].iloc[-1]
                price = current_data['close']
                
                # Determine action type
                trade_action = "BUY" if action[i] == 1 else "SELL"
                
                # Calculate dynamic stop loss based on volatility
                atr = current_data.get('atr_14', 20)  # Fallback to 20 pips
                stop_loss_pips = max(10, min(50, round(atr * 0.8)))  # 0.8 x ATR
                
                self.executor.execute_order(
                    symbol=symbol,
                    action=trade_action,
                    price=price,
                    stop_loss_pips=stop_loss_pips,
                    take_profit_pips=stop_loss_pips * 2  # 1:2 risk-reward
                )

    def _save_agent(self):
        """Save agent with timestamped version"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = model_dir / f"agent_{timestamp}.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump(self.agent, f)
        
        # Create symlink to latest
        latest = model_dir / "agent_latest.pkl"
        latest.unlink(missing_ok=True)
        latest.symlink_to(path.name)
        
        self.logger.log_system_event(f"Model saved to {path}")

    def _load_agent(self, path: str = "models/agent_latest.pkl"):
        """Load trained agent from disk with existence check"""
        try:
            # Create models directory if it doesn't exist
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            if not Path(path).exists():
                self.logger.log_error(FileNotFoundError(f"No trained model at {path}"), "model_loading")
                self.logger.log_system_event("Starting training since no model exists")
                self._train()  # Automatically train if no model exists
                return self.agent
                
            with open(path, 'rb') as f:
                agent = pickle.load(f)
            self.logger.log_metric("model_loaded", 1)
            return agent
            
        except Exception as e:
            self.logger.log_error(e, "model_loading")
            raise RuntimeError("Failed to load agent. Please train first.") from e
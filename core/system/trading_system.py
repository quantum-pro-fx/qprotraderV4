# core/system/trading_system.py
import torch
import numpy as np
import onnxruntime as ort
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
import time
import json

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from core.data.fetcher import DataFetcher
from core.env.trading_env import InstitutionalTradingEnv
from core.agents.ppo_agent import InstitutionalPPO
from core.agents.ensemble import EnsembleAgent
from core.agents.meta_learner import MetaLearner
from core.execution.oanda_executor import OandaExecutor
from core.system.model_persistence import ModelPersister
from core.utils.logger import TradingLogger

class TradingSystem(BaseAlgorithm):
    """Complete trading system with ONNX support and proper inheritance."""
    
    def __init__(self, mode: str = "train", model_dir: str = "models"):
        super().__init__(
            policy=None,
            env=None,
            learning_rate=0.0,
            device="auto"
        )
        
        self.mode = mode
        self.data_fetcher = DataFetcher()
        self.executor = OandaExecutor() if mode == "live" else None
        
        # Proper logger handling
        self._custom_logger = TradingLogger()  # Renamed to avoid conflict
        self.model_persister = ModelPersister(model_dir)
        self.agent = None
        
        # Trading parameters
        self.max_position_size = 0.2
        self.slippage = 0.0001
        self.commission = 0.0002

    @property
    def logger(self):
        """Provides access to both SB3 and custom loggers."""
        return self._custom_logger

    def run(self):
        """Main execution loop."""
        try:
            if self.mode == "train":
                self._train()
            elif self.mode == "live":
                self._live_trading()
        except Exception as e:
            self.logger.log_error(e, "system_run")
            raise

    def _train(self):
        """Training pipeline."""
        symbol_data = self.data_fetcher.fetch_multi_symbol_data()
        env = DummyVecEnv([lambda: InstitutionalTradingEnv(symbol_data)])
        
        # Train base models
        model1 = InstitutionalPPO(env)
        model1.learn(total_timesteps=100000)
        
        model2 = InstitutionalPPO(env, learning_rate=1e-3)
        model2.learn(total_timesteps=100000)
        
        # Initialize meta-learner
        self.agent = MetaLearner(
            agents={
                0: EnsembleAgent([model1, model2]),
                1: model1,
                2: model2
            },
            n_regimes=3
        )
        
        # Prime regime detector
        for obs in env.envs[0].history:
            self.agent.update_regime(obs['market'])
        
        # Save with metadata
        self._save_agent({
            "training_date": datetime.now().isoformat(),
            "symbols": list(symbol_data.keys()),
            "hyperparameters": {
                "max_position_size": self.max_position_size,
                "slippage": self.slippage,
                "commission": self.commission
            }
        })

    def _live_trading(self):
        """Live trading execution loop."""
        self.agent = self._load_agent()
        
        while True:
            try:
                symbol_data = self.data_fetcher.fetch_multi_symbol_data(count=100)
                env = InstitutionalTradingEnv(symbol_data)
                obs = env._get_obs()
                
                action = self.predict(obs)
                self._execute_trades(action, symbol_data)
                
                time.sleep(60 * 15)  # 15-minute intervals
                
            except Exception as e:
                self.logger.log_error(e, "live_trading_loop")
                time.sleep(60)

    def predict(self, observation: Dict, deterministic: bool = False):
        """
        Unified prediction interface for both ONNX and regular models.
        
        Args:
            observation: Dict with 'market' and 'portfolio' keys
            deterministic: Whether to use deterministic actions
            
        Returns:
            np.ndarray: Actions for each symbol
        """
        self._last_obs = observation  # Cache for debugging
        
        if isinstance(self.agent, ort.InferenceSession):
            ort_inputs = {
                'market': np.array(observation['market'], dtype=np.float32).reshape(1, -1),
                'portfolio': np.array(observation['portfolio'], dtype=np.float32).reshape(1, -1)
            }
            return self.agent.run(None, ort_inputs)[0][0]
        else:
            # Convert to tensors if not already
            obs_tensor = {
                k: torch.as_tensor(v, dtype=torch.float32) 
                if not isinstance(v, torch.Tensor) else v
                for k, v in observation.items()
            }
            action, _ = self.agent.predict(obs_tensor, deterministic)
            return np.clip(action, -self.max_position_size, self.max_position_size)

    def _execute_trades(self, action: np.ndarray, symbol_data: Dict):
        """Execute trades with risk controls."""
        for i, symbol in enumerate(symbol_data.keys()):
            if abs(action[i]) > 0.001:  # Minimum position threshold
                current_data = symbol_data[symbol].iloc[-1]
                price = current_data['close']
                atr = current_data.get('atr_14', 20)
                
                self.executor.execute_order(
                    symbol=symbol,
                    action="BUY" if action[i] > 0 else "SELL",
                    price=price,
                    stop_loss_pips=max(10, min(50, round(atr * 0.8)),
                    take_profit_pips=round(atr * 1.6)
                ))

    def _save_agent(self, metadata: Optional[Dict] = None):
        """Save agent in multiple formats."""
        saved_paths = self.model_persister.save(self.agent, metadata)
        self.logger.log_system_event(f"Models saved at: {saved_paths}")

    def _load_agent(self):
        """Load agent with fallback logic."""
        try:
            if self.mode == "live":
                try:
                    return self.model_persister.load(model_type='onnx')
                except Exception as e:
                    self.logger.log_error(e, "onnx_load_fallback")
                    
            agent = self.model_persister.load(model_type='pkl')
            if not hasattr(agent, 'predict'):
                raise AttributeError("Loaded model lacks predict() method")
            return agent
            
        except FileNotFoundError:
            self.logger.log_system_event("No trained model found - starting training")
            self._train()
            return self.agent

    # Required BaseAlgorithm abstract methods
    def learn(self, *args, **kwargs):
        raise NotImplementedError("Use _train() for training")
        
    def _setup_model(self):
        pass
        
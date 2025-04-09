import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict, Any
import warnings
import logging

class InstitutionalPPO(PPO):
    """Professional trading agent with institutional enhancements"""
    
    def __init__(self, env, **kwargs):
        # Updated net_arch format for SB3 v1.8.0+
        policy_kwargs = {
            "net_arch": dict(pi=[256, 128], vf=[256, 128]),  # Fixed format
            "activation_fn": nn.ReLU,
            "ortho_init": True,
            "features_extractor_class": InstitutionalFeatureExtractor,
            "features_extractor_kwargs": {
                "market_features_dim": 128,
                "portfolio_features_dim": 32,
                "combined_features_dim": 256
            }
        }
        
        # Institutional hyperparameters
        params = {
            "learning_rate": 2.5e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
            "policy_kwargs": policy_kwargs,
            "verbose": 1,
        }
        params.update(kwargs)
        
        super().__init__(
            policy="MultiInputPolicy",
            env=env,
            **params
        )
        
        # Enhanced risk management
        self.max_position_change = 0.05
        self.min_trade_volume_ratio = 0.001  # 0.1% of daily volume

    def predict(self, observation, deterministic=False, **kwargs):
        """Safe prediction with volume-adjusted risk controls"""
        try:
            # Validate observation
            if any(np.isnan(val).any() for val in observation.values()):
                return np.zeros(self.action_space.shape[0]), None
                
            action, state = super().predict(observation, deterministic, **kwargs)
            action = action.reshape((-1, 3))
            
            # Get current market data for volume checks
            market_data = observation['market']
            current_prices = market_data[:, -1, 0]  # Assuming close price is first feature
            
            # Apply volume-adjusted position limits
            for i in range(action.shape[0]):
                daily_volume = market_data[i, -1, -1]  # Assuming volume is last feature
                max_trade_value = daily_volume * self.min_trade_volume_ratio
                
                # Adjust position size based on available liquidity
                position_value = action[i, 0] * self.env.unwrapped.balance
                if abs(position_value) > max_trade_value:
                    action[i, 0] = np.sign(action[i, 0]) * max_trade_value / self.env.unwrapped.balance
                    logging.debug(f"Adjusted position for symbol {i} due to volume constraints")
                
                # Apply other risk controls
                action[i, 0] = np.clip(action[i, 0], -self.max_position_change, self.max_position_change)
                action[i, 1:] = np.clip(action[i, 1:], 0.005, 0.05)
            
            return action.flatten(), state
            
        except Exception as e:
            logging.warning(f"Prediction failed: {str(e)}", exc_info=True)
            return np.zeros(self.action_space.shape[0]), None


class InstitutionalFeatureExtractor(BaseFeaturesExtractor):
    """Advanced feature processing with NaN handling"""
    
    def __init__(self, observation_space: spaces.Dict, 
                 market_features_dim=128,
                 portfolio_features_dim=32,
                 combined_features_dim=256):
        
        super().__init__(observation_space, features_dim=combined_features_dim)
        
        # Calculate input dimensions
        market_shape = observation_space['market'].shape
        portfolio_shape = observation_space['portfolio'].shape
        
        # Market feature network with robust initialization
        self.market_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(market_shape), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, market_features_dim),
            nn.LayerNorm(market_features_dim)
        )
        
        # Portfolio feature network
        self.portfolio_net = nn.Sequential(
            nn.Linear(np.prod(portfolio_shape), 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, portfolio_features_dim),
            nn.LayerNorm(portfolio_features_dim)
        )
        
        # Combined feature processing
        self.combined_net = nn.Sequential(
            nn.Linear(market_features_dim + portfolio_features_dim, combined_features_dim),
            nn.LayerNorm(combined_features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Market features with NaN handling
        market_data = observations['market']
        market_data = torch.nan_to_num(market_data, nan=0.0, posinf=1e6, neginf=-1e6)
        market_features = self.market_net(market_data)
        
        # Portfolio features
        portfolio_data = observations['portfolio']
        portfolio_data = torch.nan_to_num(portfolio_data, nan=0.0)
        portfolio_features = self.portfolio_net(portfolio_data)
        
        # Combine features
        combined = torch.cat([market_features, portfolio_features], dim=1)
        return self.combined_net(combined)
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, SubprocVecEnv
from typing import Dict, Any, Optional, Callable
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import warnings
import logging

class InstitutionalPPO(PPO):
    """Professional trading agent with institutional enhancements"""
    
    def __init__(self, env, **kwargs):
        # Linear learning rate schedule - improved version
        def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
            """
            Linear learning rate schedule from initial_value to final_value
            :param initial_value: Initial learning rate
            :param final_value: Final learning rate (default: 0)
            """
            def func(progress_remaining: float) -> float:
                """Progress decreases from 1 (beginning) to 0 (end)"""
                return final_value + progress_remaining * (initial_value - final_value)
            return func
            
        # Institutional hyperparameters - improved values
        params = {
            "learning_rate": linear_schedule(3e-4, 1e-5),  # Goes from 3e-4 to 1e-5
            "n_steps": 2048,  # Better for trading (more frequent updates)
            "batch_size": 256,  # Kept your value
            "n_epochs": 10,  # Kept your value
            "gamma": 0.99,  # Standard
            "gae_lambda": 0.92,  # Smoother advantage estimation
            "clip_range": linear_schedule(0.2),  # Also decay clipping range
            "ent_coef": 0.02,  # Kept your value
            "max_grad_norm": 0.5,  # Kept your value
            "policy_kwargs": {
                "net_arch": dict(pi=[256, 128], vf=[256, 128]),  # Your architecture is good
                "activation_fn": nn.ReLU,
                "ortho_init": True,
                "features_extractor_class": InstitutionalFeatureExtractor,
                "features_extractor_kwargs": {
                    "market_features_dim": 256,
                    "portfolio_features_dim": 64,
                    "combined_features_dim": 512
                },
                "optimizer_kwargs": {  # Added explicit optimizer settings
                    "eps": 1e-5,
                    "weight_decay": 0.0
                }
            },
            "verbose": 1,
        }
        params.update(kwargs)
        
        super().__init__(
            policy="MultiInputPolicy",
            env=env,
            **params
        )

        # Risk management - good values
        self.max_position_change = 0.05  # 5% max change per step

    def predict(self, observation, deterministic=False, **kwargs):
        """Safe prediction with risk controls - improved version"""
        try:
            # Enhanced NaN/inf check
            if not isinstance(observation, dict) or \
               'market' not in observation or \
               'portfolio' not in observation:
                raise ValueError("Invalid observation format")
                
            if any(np.isnan(val).any() or np.isinf(val).any() 
                  for val in observation.values() if isinstance(val, np.ndarray)):
                return np.zeros(self.action_space.shape[0]), None
                
            action, state = super().predict(observation, deterministic, **kwargs)
            
            # Apply risk controls - your logic is good
            action = action.reshape((-1, 3))
            action[:, 0] = np.clip(action[:, 0], 
                                 -self.max_position_change, 
                                 self.max_position_change)
            action[:, 1:] = np.clip(action[:, 1:], 0.005, 0.05)
            
            return action.flatten(), state
            
        except Exception as e:
            logging.warning(f"Prediction failed: {str(e)}", exc_info=True)
            return np.zeros(self.action_space.shape[0]), None


class InstitutionalFeatureExtractor(BaseFeaturesExtractor):
    """Advanced feature processing for institutional trading - improved version"""
    
    def __init__(self, observation_space: spaces.Dict, 
                 market_features_dim=256,  # Changed default to match your PPO config
                 portfolio_features_dim=64,
                 combined_features_dim=512):
        
        # Input validation
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("Observation space must be a Dict")
            
        if 'market' not in observation_space.spaces or 'portfolio' not in observation_space.spaces:
            raise ValueError(
                f"Observation space must contain 'market' and 'portfolio' keys. "
                f"Found keys: {list(observation_space.spaces.keys())}"
            )

        super().__init__(observation_space, features_dim=combined_features_dim)
        
        # Market features processing - enhanced
        self.market_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(observation_space['market'].shape), 512),  # Larger intermediate
            nn.LayerNorm(512),
            nn.LeakyReLU(negative_slope=0.01),  # Better than ReLU for financial data
            nn.Dropout(0.1),  # Regularization
            nn.Linear(512, market_features_dim),
            nn.LayerNorm(market_features_dim)
        )
        
        # Portfolio features - enhanced
        self.portfolio_net = nn.Sequential(
            nn.Linear(np.prod(observation_space['portfolio'].shape), 128),  # Larger intermediate
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(128, portfolio_features_dim),
            nn.LayerNorm(portfolio_features_dim)
        )
        
        # Combined processing - enhanced
        self.combined_net = nn.Sequential(
            nn.Linear(market_features_dim + portfolio_features_dim, combined_features_dim),
            nn.LayerNorm(combined_features_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1)
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Input validation
        if 'market' not in observations or 'portfolio' not in observations:
            raise ValueError("Observations must contain 'market' and 'portfolio' keys")
            
        # Market features
        market_features = self.market_net(observations['market'].float())  # Ensure float
        
        # Portfolio features
        portfolio_features = self.portfolio_net(observations['portfolio'].float())
        
        # Combine with skip connection
        combined = torch.cat([market_features, portfolio_features], dim=1)
        return self.combined_net(combined)
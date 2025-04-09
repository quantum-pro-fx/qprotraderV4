# core/agents/institutional_ppo.py
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.preprocessing import preprocess_obs
from gymnasium import spaces
from typing import Dict
import warnings

class InstitutionalFeatureExtractor(BaseFeaturesExtractor):
    """Consistent feature extractor with proper dimensions"""
    
    def __init__(self, observation_space: spaces.Dict, 
                 market_features_dim=256,
                 portfolio_features_dim=64,
                 combined_features_dim=512):

        super().__init__(observation_space, features_dim=combined_features_dim)

        market_shape = observation_space['market'].shape
        
        # Market features
        self.market_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(market_shape), 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, market_features_dim),
            nn.LayerNorm(market_features_dim)
        )
        
        # Portfolio features
        portfolio_shape = observation_space['portfolio'].shape[0]
        self.portfolio_net = nn.Sequential(
            nn.Linear(portfolio_shape, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, portfolio_features_dim),
            nn.LayerNorm(portfolio_features_dim)
        )
        
        # Combined features
        self.combined_net = nn.Sequential(
            nn.Linear(market_features_dim + portfolio_features_dim, combined_features_dim),
            nn.LayerNorm(combined_features_dim),
            nn.LeakyReLU(0.01)
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        market_features = self.market_net(observations['market'].float())
        portfolio_features = self.portfolio_net(observations['portfolio'].float())
        combined = torch.cat([market_features, portfolio_features], dim=1)
        return self.combined_net(combined)

class TrainingMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.value_losses = []
        
    def _on_step(self) -> bool:
        if self.verbose >= 1 and "episode" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
            
        # Log training stats
        if self.n_calls % 100 == 0:
            for key, value in self.model.logger.name_to_value.items():
                if "loss" in key or "entropy" in key:
                    print(f"{key}: {value:.4f}")
        return True

class InstitutionalPPO(PPO):
    """Enhanced with proper initialization, risk controls, and training visibility"""
    
    def __init__(self, env, **kwargs):
        # Ensure verbosity is properly set
        verbose = kwargs.pop('verbose', 1)
        self.max_position_size = 0.2  # Define this in __init__ or pass via kwargs
        
        policy_kwargs = kwargs.pop('policy_kwargs', {})
        policy_kwargs.update({
            "features_extractor_class": InstitutionalFeatureExtractor,
            "features_extractor_kwargs": {
                "market_features_dim": 256,
                "portfolio_features_dim": 64,
                "combined_features_dim": 512
            },
            "net_arch": {
                "pi": [256, 128],  # Policy network
                "vf": [512, 256]    # Larger value network
            }
        })
        
        # Warn about potential hyperparameter conflicts
        if 'learning_rate' in kwargs:
            warnings.warn("Custom learning rate may affect training stability")
            
        super().__init__(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=20,
            gamma=0.99,
            gae_lambda=0.90,
            clip_range=0.15,
            ent_coef=0.05,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=verbose,  # Ensure verbosity is passed through
            **kwargs
        )
        
    def learn(self, total_timesteps, callback=None, **kwargs):
        """Enhanced learn method with better progress tracking"""
        # Create composite callback if none provided
        if callback is None:
            callbacks = [TrainingMonitorCallback(verbose=self.verbose)]
        elif isinstance(callback, list):
            callbacks = callback + [TrainingMonitorCallback(verbose=self.verbose)]
        else:
            callbacks = [callback, TrainingMonitorCallback(verbose=self.verbose)]
            
        # Ensure progress bar is shown
        kwargs['progress_bar'] = kwargs.get('progress_bar', True)
        
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            **kwargs
        )

    def predict(self, observation, deterministic=False):
        """Override predict to clip actions to risk limits"""
        observation = preprocess_obs(observation, self.observation_space)
        action, _ = super().predict(observation, deterministic)
        return np.clip(action, -self.max_position_size, self.max_position_size), _

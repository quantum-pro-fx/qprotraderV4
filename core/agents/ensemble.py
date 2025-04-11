# core/agents/ensemble.py
import numpy as np
from gymnasium import spaces
from typing import List
from core.utils.space_utils import are_spaces_equal

class EnsembleAgent:
    """Robust ensemble with comprehensive action space validation"""
    
    def __init__(self, models: List):
        if not models:
            raise ValueError("Cannot create ensemble with empty model list")
            
        # Validate all models have compatible action spaces
        self.action_space = models[0].action_space
        for i, model in enumerate(models[1:]):
            if not are_spaces_equal(model.action_space, self.action_space):
                raise ValueError(
                    f"Model {i+1} has incompatible action space:\n"
                    f"Expected: {self.action_space}\n"
                    f"Got: {model.action_space}"
                )
                
        self.models = models
        self.weights = np.ones(len(models)) / len(models)  # Equal weighting by default
    
    def predict(self, obs, deterministic=False):
        """Make prediction with optional model weighting"""
        predictions = []
        state = None  # For compatibility with SB3 API
        
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                # Handle custom prediction interfaces
                action = model(obs)
            predictions.append(action * weight)
        
        if isinstance(self.action_space, spaces.Discrete):
            # Weighted voting for discrete actions
            votes = np.stack(predictions)
            return np.array([np.argmax(np.bincount(votes[:,i], weights=self.weights))
                           for i in range(votes.shape[1])]), state
        else:
            # Weighted average for continuous actions
            return np.sum(predictions, axis=0), state
    
    def set_weights(self, weights: np.ndarray):
        """Update model weights (must sum to 1)"""
        if len(weights) != len(self.models):
            raise ValueError("Weights length must match number of models")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")
        self.weights = np.array(weights)
# core/agents/ensemble.py
import numpy as np
from gymnasium import spaces
from typing import List

class EnsembleAgent:
    """Robust ensemble with action space validation"""
    
    def __init__(self, models: List):
        if not models:
            raise ValueError("Empty model list")
            
        # Validate all models have compatible action spaces
        self.action_space = models[0].action_space
        for model in models[1:]:
            if not spaces.are_spaces_equal(model.action_space, self.action_space):
                raise ValueError("Incompatible action spaces in ensemble")
                
        self.models = models
    
    def predict(self, obs, deterministic=False):
        predictions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            predictions.append(action)
        
        if isinstance(self.action_space, spaces.Discrete):
            # Majority vote with random tie-break
            votes = np.array(predictions)
            return np.array([np.random.choice(np.where(votes[:,i] == votes[:,i].max())[0]) 
                            for i in range(votes.shape[1])]), None
        else:
            # Weighted average (could add model confidence weights)
            return np.mean(predictions, axis=0), None
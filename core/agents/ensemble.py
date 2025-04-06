from stable_baselines3 import PPO
import numpy as np
from gymnasium import spaces

class EnsembleAgent:
    def __init__(self, models):
        self.models = models  # List of trained PPO models
    
    def predict(self, obs, **kwargs):
        # Collect predictions from all models
        predictions = []
        for model in self.models:
            action, _ = model.predict(obs, **kwargs)
            predictions.append(action)
        
        # Majority voting for discrete actions
        if isinstance(self.models[0].action_space, spaces.Discrete):
            return np.array([
                np.argmax(np.bincount([pred[i] for pred in predictions]))
                for i in range(len(predictions[0]))
            ]), None
        
        # Mean for continuous actions
        return np.mean(predictions, axis=0), None
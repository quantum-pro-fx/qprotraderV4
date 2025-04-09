# core/agents/meta_learner.py
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from typing import Dict

class MetaLearner:
    """Enhanced with proper regime detection initialization"""
    
    def __init__(self, agents: Dict, n_regimes=3, window_size=100):
        self.agents = agents
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.regime_detector = MiniBatchKMeans(n_clusters=n_regimes)
        self.market_state_buffer = []
        self.current_regime = 0  # Default to first regime
        
    def update_regime(self, market_state: np.ndarray):
        """Update current market regime"""
        self.market_state_buffer.append(market_state)
        if len(self.market_state_buffer) > self.window_size:
            self.market_state_buffer.pop(0)
            
        # Online learning of regimes
        if len(self.market_state_buffer) >= self.n_regimes:
            self.regime_detector.partial_fit(self.market_state_buffer)
            self.current_regime = self.regime_detector.predict([market_state])[0]
    
    def predict(self, obs, market_state=None):
        if market_state is not None:
            self.update_regime(market_state)
        return self.agents.get(self.current_regime, self.agents[0]).predict(obs)
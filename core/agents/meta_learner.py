from sklearn.cluster import MiniBatchKMeans
from core.agents.ensemble import EnsembleAgent

class MetaLearner:
    def __init__(self, agents, n_regimes=3):
        self.agents = agents  # Dict of {regime_id: agent}
        self.regime_detector = MiniBatchKMeans(n_clusters=n_regimes)
        self.current_regime = None
    
    def update_regime(self, market_state):
        self.current_regime = self.regime_detector.predict([market_state])[0]
    
    def predict(self, obs, market_state=None):
        if market_state is not None:
            self.update_regime(market_state)
        return self.agents[self.current_regime].predict(obs)
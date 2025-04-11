import numpy as np
import pandas as pd
from core.data.processor import InstitutionalFeatureEngineer
from core.data.macro_data_loader import MacroDataLoader
from typing import Dict, List, Tuple
import logging

class EnhancedFeatureEngineer(InstitutionalFeatureEngineer):
    def __init__(self, macro_loader=None, **kwargs):
        """
        Enhanced feature engineer with macroeconomic regime features
        
        Args:
            macro_loader: MacroDataLoader instance for fetching economic indicators
            **kwargs: Passed to parent class constructor
        """
        super().__init__(**kwargs)
        self.macro_loader = macro_loader or MacroDataLoader()
        
        # Add macro regime features
        self.feature_groups['macro'] = [
            ('vix_regime', self._calculate_vix_regime),
            ('yield_regime', self._calculate_yield_regime),
            ('macro_composite', self._calculate_macro_composite)
        ]
        
        # Update expected feature count
        #self._update_feature_count()

    # def _update_feature_count(self):
    #     """Update the feature count after adding new feature groups"""
    #     self._feature_count = sum(len(group) for group in self.feature_groups.values())
    #     if hasattr(self, 'expected_shape'):
    #         self.expected_shape = (self.expected_shape[0], self._feature_count)

    # def get_feature_count(self) -> int:
    #     """
    #     Get the total number of features including macro features
    #     Overrides parent method to ensure correct count
    #     """
    #     return self._feature_count

    def _calculate_vix_regime(self, df: pd.DataFrame) -> np.ndarray:
        """
        3-state VIX regime classifier
        0 = low volatility (VIX < 15)
        1 = normal volatility (15 ≤ VIX ≤ 30) 
        2 = high volatility (VIX > 30)
        """
        try:
            current_vix = self.macro_loader.get_current_vix()
            regimes = np.select(
                [current_vix < 15, current_vix > 30],
                [0, 2],
                default=1
            )
            return np.full(len(df), regimes)
        except Exception as e:
            logging.warning(f"VIX regime calculation failed: {str(e)}")
            return np.ones(len(df))  # Default to normal regime

    def _calculate_yield_regime(self, df: pd.DataFrame) -> np.ndarray:
        """
        Yield curve regime detection
        0 = inverted (spread < -0.5)
        1 = normal (-0.5 ≤ spread ≤ 0.5)
        2 = steep (spread > 0.5)
        """
        try:
            spread = self.macro_loader.get_yield_spread()
            regimes = np.select(
                [spread < -0.5, spread > 0.5],
                [0, 2],
                default=1
            )
            return np.full(len(df), regimes)
        except Exception as e:
            logging.warning(f"Yield regime calculation failed: {str(e)}")
            return np.ones(len(df))  # Default to normal regime

    def _calculate_macro_composite(self, df: pd.DataFrame) -> np.ndarray:
        """
        Combined macro signal (average of VIX and yield regimes)
        Returns normalized value between 0 and 2
        """
        try:
            vix_regime = self._calculate_vix_regime(df)
            yield_regime = self._calculate_yield_regime(df)
            return (vix_regime + yield_regime) / 2
        except Exception as e:
            logging.warning(f"Macro composite calculation failed: {str(e)}")
            return np.ones(len(df))  # Default to neutral signal

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced transform with macro features
        Maintains consistent feature count even if macro data fails
        """
        # Get base features from parent
        features = super().transform(df)
        
        # Add macro features
        macro_features = {}
        for name, func in self.feature_groups['macro']:
            try:
                macro_features[name] = func(df)
            except Exception as e:
                logging.warning(f"Failed to calculate {name}: {str(e)}")
                macro_features[name] = np.ones(len(df))  # Fallback value
        
        # Combine all features ensuring consistent length
        for name, values in macro_features.items():
            features[name] = values[:len(df)]  # Ensure proper length
            
        return features
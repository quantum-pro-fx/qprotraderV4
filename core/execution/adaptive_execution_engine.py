from collections import deque
import numpy as np
import pandas as pd
import logging
import time
import requests
from typing import Dict

import numpy as np
from typing import Dict, Optional

class AdaptiveExecutionEngine:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional configuration overrides"""
        self.config = {
            'base_params': {
                'order_type': 'HYBRID',  # HYBRID = Limit orders with market fallback
                'base_participation': 0.2,
                'max_participation': 0.4,
                'min_participation': 0.05,
                'normal_atr': 0.02,  # Instrument-specific normal ATR value
                'base_time_horizons': [600, 300, 150],  # Regime-based horizons
                'price_tolerance_multiplier': 0.0015,
                'low_vol_tolerance': 0.0005,
                'volatility_threshold': 0.05,  # For extreme volatility override
                'liquidity_qty_threshold': 1000,  # Shares at top of book
                'size_sensitivity': 0.001  # For order size adjustments
            },
            **(config or {})
        }

    def get_execution_params(self, symbol: str, market_data: dict, regime: int, 
                           order_size_pct_adv: float = 0.0) -> dict:
        """
        Get dynamic execution parameters based on market conditions
        
        Args:
            symbol: Instrument identifier
            market_data: Dictionary containing market state (ATR, spread, top of book, etc.)
            regime: 0=low vol, 1=normal, 2=high vol
            order_size_pct_adv: Order size as percentage of average daily volume
        
        Returns:
            Dictionary of execution parameters
        """
        # Base parameters
        params = {
            'order_type': self.config['base_params']['order_type'],
            'time_horizon': self._get_time_horizon(regime, order_size_pct_adv),
            'participation_rate': self._get_participation_rate(market_data, regime),
            'price_tolerance': self._get_price_tolerance(market_data, regime),
            'dynamic_sizing': True
        }

        # Regime-specific adjustments
        if regime == 0:  # Low volatility
            params.update({
                'order_type': 'LIMIT',
                'price_tolerance': self.config['base_params']['low_vol_tolerance']
            })
        elif regime == 2:  # High volatility
            params.update({
                'order_type': 'MARKET',
                'participation_rate': min(
                    self.config['base_params']['max_participation'],
                    params['participation_rate'] * 1.5
                )
            })

        # Liquidity-based overrides
        if market_data.get('top_of_book_qty', 0) < self.config['base_params']['liquidity_qty_threshold']:
            params.update({
                'order_type': 'LIMIT',
                'participation_rate': max(
                    self.config['base_params']['min_participation'],
                    params['participation_rate'] * 0.5
                )
            })

        # Extreme volatility circuit breaker
        if market_data.get('atr_pct', 0) > self.config['base_params']['volatility_threshold']:
            params.update({
                'order_type': 'MARKET',
                'time_horizon': 60,
                'participation_rate': self.config['base_params']['min_participation'],
                'dynamic_sizing': False
            })

        return params

    def _get_time_horizon(self, regime: int, order_size_pct_adv: float) -> float:
        """Get execution time horizon with size adjustment"""
        base_time = self.config['base_params']['base_time_horizons'][regime]
        size_adjustment = 1 + (order_size_pct_adv / self.config['base_params']['size_sensitivity'])
        return base_time * size_adjustment

    def _get_participation_rate(self, market_data: dict, regime: int) -> float:
        """Dynamic participation rate based on volatility and regime"""
        normal_atr = self.config['base_params']['normal_atr']
        base_rate = (self.config['base_params']['base_participation'] - 
                    (0.1 * market_data.get('atr', 0) / normal_atr))
        
        regime_boost = 1 + (regime * 0.3)
        return np.clip(
            base_rate * regime_boost,
            self.config['base_params']['min_participation'],
            self.config['base_params']['max_participation']
        )

    def _get_price_tolerance(self, market_data: dict, regime: int) -> float:
        """Dynamic price tolerance based on volatility"""
        if regime == 0:  # Low volatility uses fixed tolerance
            return self.config['base_params']['low_vol_tolerance']
            
        return (self.config['base_params']['price_tolerance_multiplier'] * 
               (1 + market_data.get('atr', 0) / 0.01))
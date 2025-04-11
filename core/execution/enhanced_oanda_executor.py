from oandapyV20 import API
from oandapyV20.endpoints import orders
from config.oanda import ACCOUNT_ID, ACCESS_TOKEN
from core.execution.oanda_executor import OandaExecutor
import logging

class EnhancedOandaExecutor(OandaExecutor):
    def __init__(self, risk_pct=0.01):
        super().__init__(risk_pct)
        self.logger = logging.getLogger(__name__)
        
    def execute_adaptive_order(self, symbol: str, action: str, 
                             execution_params: dict, market_data: dict) -> dict:
        """
        Enhanced execution with adaptive parameters
        Args:
            execution_params: From AdaptiveExecutionEngine
            market_data: Current market conditions
        """
        try:
            # 1. Calculate dynamic position sizing
            units = self._calculate_adaptive_units(symbol, execution_params, market_data)
            
            # 2. Prepare order based on execution type
            order = {
                "order": {
                    "instrument": symbol,
                    "units": str(units) if action == "BUY" else f"-{units}",
                    "stopLossOnFill": self._get_stop_loss(market_data, execution_params),
                    "takeProfitOnFill": self._get_take_profit(market_data, execution_params)
                }
            }
            
            # 3. Set order type and price
            if execution_params['order_type'] == 'LIMIT':
                order["order"].update({
                    "type": "LIMIT",
                    "price": str(self._get_limit_price(market_data, execution_params)),
                    "timeInForce": "GTD",
                    "gtdTime": self._get_expiry_time(execution_params['time_horizon'])
                })
            else:  # MARKET or HYBRID
                order["order"]["type"] = "MARKET"
                
            # 4. Execute and log
            result = self.client.request(orders.OrderCreate(ACCOUNT_ID, order))
            self.log_execution(result, execution_params)
            return result
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {str(e)}")
            raise

    def _calculate_adaptive_units(self, symbol: str, exec_params: dict, market_data: dict) -> int:
        """Risk-adjusted position sizing with participation rate"""
        account = self.client.account.get(ACCOUNT_ID)
        balance = float(account['account']['balance'])
        
        # Base risk calculation
        pip_value = 10 if "JPY" not in symbol else 0.1
        stop_loss_pips = exec_params.get('stop_loss_pips', 20)
        risk_amount = balance * self.risk_pct * exec_params['participation_rate']
        
        # Volume-based adjustment
        max_units = market_data.get('volume', float('inf')) * 0.01  # 1% of daily volume
        calculated_units = int(risk_amount / (stop_loss_pips * pip_value))
        
        return min(calculated_units, max_units)

    def _get_limit_price(self, market_data: dict, exec_params: dict) -> float:
        """Calculate limit price with tolerance"""
        mid_price = (market_data['bid'] + market_data['ask'])/2
        if exec_params['order_type'] == 'LIMIT':
            return mid_price * (1 - exec_params['price_tolerance'])
        return mid_price

    def _get_stop_loss(self, market_data: dict, exec_params: dict) -> dict:
        """Dynamic stop loss calculation"""
        price = market_data['ask'] if exec_params['order_type'] == 'MARKET' else self._get_limit_price(market_data, exec_params)
        return {
            "price": str(price * (1 - exec_params.get('stop_loss_pct', 0.01))),
            "timeInForce": "GTC"
        }

    def _get_take_profit(self, market_data: dict, exec_params: dict) -> dict:
        """Dynamic take profit calculation"""
        price = market_data['bid'] if exec_params['order_type'] == 'MARKET' else self._get_limit_price(market_data, exec_params)
        return {
            "price": str(price * (1 + exec_params.get('take_profit_pct', 0.02))),
            "timeInForce": "GTC"
        }

    def _get_expiry_time(self, time_horizon: int) -> str:
        """Convert seconds to OANDA's expiry format"""
        from datetime import datetime, timedelta
        expiry = datetime.utcnow() + timedelta(seconds=time_horizon)
        return expiry.isoformat("T") + "Z"

    def log_execution(self, result: dict, exec_params: dict):
        """Record execution details"""
        self.logger.info(
            f"Executed {exec_params['order_type']} order: "
            f"Units={result['orderFillTransaction']['units']} "
            f"Price={result['orderFillTransaction'].get('price', 'market')} "
            f"SL={result['orderFillTransaction']['stopLossOrder'].get('price')} "
            f"TP={result['orderFillTransaction']['takeProfitOrder'].get('price')}"
        )
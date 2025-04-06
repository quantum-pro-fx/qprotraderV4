from oandapyV20 import API
from oandapyV20.endpoints import orders
from config.oanda import ACCOUNT_ID, ACCESS_TOKEN

class OandaExecutor:
    def __init__(self, risk_pct=0.01):
        self.client = API(access_token=ACCESS_TOKEN)
        self.risk_pct = risk_pct
    
    def execute_order(self, symbol, action, price, stop_loss_pips=20, take_profit_pips=40):
        """Execute order with risk management"""
        # Calculate position size based on account risk
        account = self.client.account.get(ACCOUNT_ID)
        balance = float(account['account']['balance'])
        risk_amount = balance * self.risk_pct
        pip_value = 10 if "JPY" not in symbol else 0.1
        units = int(risk_amount / (stop_loss_pips * pip_value))
        
        order = {
            "order": {
                "units": str(units) if action == "BUY" else f"-{units}",
                "instrument": symbol,
                "type": "MARKET",
                "stopLossOnFill": {
                    "price": str(round(price - stop_loss_pips*pip_value, 5))
                },
                "takeProfitOnFill": {
                    "price": str(round(price + take_profit_pips*pip_value, 5))
                }
            }
        }
        return self.client.request(orders.OrderCreate(ACCOUNT_ID, order))
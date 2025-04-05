from src.config import Config

class RiskManager:
    @staticmethod
    def calculate_position_size(balance, current_price, stop_loss_pips=20):
        risk_amount = balance * Config.RISK_PER_TRADE
        pip_value = 10 if "JPY" not in Config.SYMBOL else 0.1  # Simplified
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return round(position_size, 2)
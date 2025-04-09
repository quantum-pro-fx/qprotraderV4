from core.system.trading_system import TradingSystem

if __name__ == "__main__":
    system = TradingSystem(mode="train")
    system.run()
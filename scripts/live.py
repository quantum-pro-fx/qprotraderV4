from core.system import TradingSystem

if __name__ == "__main__":
    print("Starting live trading...")
    system = TradingSystem(mode="live")
    system.run()
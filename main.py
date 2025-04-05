from src.data_fetcher import fetch_historical_data
from src.feature_engine import generate_features
from src.trading_env import TradingEnv
from stable_baselines3 import PPO
from src.config import Config
import matplotlib.pyplot as plt

def plot_results(env):
    plt.figure(figsize=(12,6))
    plt.plot(env.df['close'], label='Price', alpha=0.5)
    
    # Mark trades
    longs = [i for i, x in enumerate(env.action_history) if x == 1]
    shorts = [i for i, x in enumerate(env.action_history) if x == 2]
    
    plt.scatter(longs, env.df.iloc[longs]['close'], marker='^', color='g', label='Long')
    plt.scatter(shorts, env.df.iloc[shorts]['close'], marker='v', color='r', label='Short')
    
    plt.legend()
    plt.show()

def main():
    # 1. Get and prepare data
    print("Fetching data...")
    df = fetch_historical_data(count=1000)
    df = generate_features(df)
    
    # 2. Create environment
    print("Creating environment...")
    env = TradingEnv(df)
    
    # 3. Initialize and train agent
    print("Training agent...")
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=20000)
    
    # 4. Save the model
    model.save("ppo_trading_agent")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
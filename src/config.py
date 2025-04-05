import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Oanda credentials
    ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')
    ACCESS_TOKEN = os.getenv('OANDA_API_KEY')
    
    # Trading parameters
    SYMBOL = 'EUR_USD'
    TIMEFRAME = 'M5'
    INITIAL_BALANCE = 10000
    RISK_PER_TRADE = 0.01  # 1% of balance
    
    # Feature parameters
    FEATURE_WINDOW = 50  # Lookback window for features

    RANDOM_START = True  # Whether to randomize starting point
    MIN_STEPS_FOR_EPISODE = 500  # Minimum steps needed for a valid episode
    EPISODE_LENGTH = 1000  # Target length for each training episode
    RANDOM_START = True
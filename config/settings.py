import os
from dotenv import load_dotenv

load_dotenv()

SYMBOLS = [
    "EUR_USD", 
    "GBP_USD", 
    "USD_JPY",
    "AUD_USD",
    "USD_CAD"
]

TIME_FRAME = "M15"  # Primary timeframe

# FRED credentials
FRED_API_KEY = os.getenv('FRED_API_KEY')
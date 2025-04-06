import os
from dotenv import load_dotenv

load_dotenv()

# Oanda credentials
ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')
ACCESS_TOKEN = os.getenv('OANDA_API_KEY')
ENVIRONMENT = 'practice'
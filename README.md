## Directory Overview

- **/config/**
  - Configuration files for the application including main settings and API credentials

- **/core/**
  - Main trading system components:
    - **agents/**: Reinforcement learning agents
    - **data/**: Data pipeline components
    - **env/**: Trading environments
    - **execution/**: Trade execution and risk management
    - **utils/**: Helper functions and utilities
  - `system.py`: Main system integration file

- **/scripts/**
  - Operational scripts for training, backtesting, and live trading

- Root files:
  - `requirements.txt`: Python dependencies
  - `README.md`: Project documentation


# First install the system dependencies
# On Mac:
brew install ta-lib

# On Linux (Debian/Ubuntu):
sudo apt-get install libta-lib-dev

# Then install the Python wrapper
pip install TA-Lib

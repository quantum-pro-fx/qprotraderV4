# core/utils/logger.py
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any
import sys

class TradingLogger:
    def __init__(self, name: str = "trading_system"):
        """
        Professional trading logger with:
        - Rotating file handler (daily)
        - Structured JSON logging
        - Console output
        - Error tracking
        
        Args:
            name: Logger name (appears in logs)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if needed
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 1. File Handler (rotates daily, keeps 7 days)
        file_handler = TimedRotatingFileHandler(
            filename=self.log_dir / "trading.log",
            when="midnight",
            backupCount=7,
            encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        
        # 2. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            "%(levelname)s | %(message)s"
        ))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Add our custom handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initial log message
        self.logger.info("TradingLogger initialized")

    def log_trade(self, 
                 symbol: str, 
                 action: str, 
                 price: float, 
                 quantity: float,
                 metadata: Dict[str, Any] = None):
        """
        Log structured trade data in JSON format
        
        Args:
            symbol: Trading instrument (e.g., "EUR_USD")
            action: "BUY", "SELL", or "HOLD"
            price: Execution price
            quantity: Trade size
            metadata: Additional trade context
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "TRADE",
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": quantity,
            **({} if metadata is None else metadata)
        }
        self.logger.info(json.dumps(log_entry))

    def log_metric(self, name: str, value: float):
        """
        Log performance metrics
        
        Args:
            name: Metric name (e.g., "sharpe_ratio")
            value: Numeric value
        """
        self.logger.info(f"METRIC {name}={value}")

    def log_error(self, error: Exception, context: str = None):
        """
        Log errors with stack traces
        
        Args:
            error: Exception object
            context: Where error occurred (e.g., "data_fetch")
        """
        self.logger.error(
            f"ERROR in {context}: {str(error)}", 
            exc_info=True
        )

    def log_system_event(self, message: str):
        """Log important system events"""
        self.logger.info(f"SYSTEM EVENT: {message}")
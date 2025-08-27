"""
Configuration management for the crypto trading bot.
Handles environment variables, JSON configs, and runtime settings.
"""

import os
import json
from typing import Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ExchangeConfig:
    """Exchange configuration settings"""
    name: str = "binance"
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    secret: str = os.getenv("BINANCE_SECRET", "")
    testnet: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    rate_limit: bool = True
    timeout: int = 30000

@dataclass
class TradingConfig:
    """Trading configuration settings"""
    mode: str = os.getenv("TRADING_MODE", "paper")  # paper or live
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "10000"))
    max_positions: int = int(os.getenv("MAX_POSITIONS", "5"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    max_drawdown: float = float(os.getenv("MAX_DRAWDOWN", "0.15"))
    stop_loss_percent: float = float(os.getenv("STOP_LOSS_PERCENT", "0.03"))
    take_profit_percent: float = float(os.getenv("TAKE_PROFIT_PERCENT", "0.06"))
    max_leverage: float = float(os.getenv("MAX_LEVERAGE", "3"))
    
@dataclass
class MLConfig:
    """Machine Learning configuration settings"""
    retrain_interval: int = int(os.getenv("ML_RETRAIN_INTERVAL", "24"))
    feature_lookback: int = int(os.getenv("FEATURE_LOOKBACK", "100"))
    confidence_threshold: float = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.6"))
    models_path: str = "ml/models/"
    
@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = os.getenv("DATABASE_URL", "sqlite:///trading_bot.db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
@dataclass
class NotificationConfig:
    """Notification configuration settings"""
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enable_notifications: bool = bool(telegram_token and telegram_chat_id)

class Config:
    """Main configuration class that loads and manages all settings"""
    
    def __init__(self):
        self.exchange = ExchangeConfig()
        self.trading = TradingConfig()
        self.ml = MLConfig()
        self.database = DatabaseConfig()
        self.notifications = NotificationConfig()
        
        # Load JSON configurations
        self.trading_pairs = self._load_json_config("config/trading_pairs.json")
        self.strategies = self._load_json_config("config/strategies.json")
        self.risk_rules = self._load_json_config("config/risk_management.json")
        
        # Runtime settings
        self.is_paper_trading = self.trading.mode == "paper"
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
    def _load_json_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: Config file {file_path} not found, using defaults")
                return {}
        except json.JSONDecodeError as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def get_active_trading_pairs(self) -> List[str]:
        """Get list of active trading pairs"""
        if "active_pairs" in self.trading_pairs:
            return self.trading_pairs["active_pairs"]
        return ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy"""
        return self.strategies.get(strategy_name, {})
    
    def get_timeframes(self) -> List[str]:
        """Get active timeframes for data collection"""
        return self.trading_pairs.get("timeframes", ["1m", "5m", "15m", "1h", "4h"])
    
    def update_runtime_config(self, key: str, value: Any) -> None:
        """Update runtime configuration"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            print(f"Warning: Unknown config key: {key}")
    
    def is_live_trading(self) -> bool:
        """Check if bot is in live trading mode"""
        return self.trading.mode == "live"
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate exchange credentials
        if not self.exchange.api_key or not self.exchange.secret:
            errors.append("Missing exchange API credentials")
        
        # Validate trading parameters
        if self.trading.risk_per_trade <= 0 or self.trading.risk_per_trade > 0.1:
            errors.append("Risk per trade must be between 0 and 0.1 (10%)")
            
        if self.trading.max_drawdown <= 0 or self.trading.max_drawdown > 0.5:
            errors.append("Max drawdown must be between 0 and 0.5 (50%)")
        
        # Validate ML settings
        if self.ml.confidence_threshold <= 0 or self.ml.confidence_threshold > 1:
            errors.append("ML confidence threshold must be between 0 and 1")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Global configuration instance
config = Config()

# Export commonly used settings
TRADING_PAIRS = config.get_active_trading_pairs()
TIMEFRAMES = config.get_timeframes()
IS_PAPER_TRADING = config.is_paper_trading
IS_LIVE_TRADING = config.is_live_trading()
DEBUG = config.debug
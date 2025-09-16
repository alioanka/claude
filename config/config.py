"""
Configuration management for the crypto trading bot.
Handles environment variables, JSON configs, and runtime settings.
"""

import os
import json
import yaml
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
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "100000"))
    max_positions: int = int(os.getenv("MAX_POSITIONS", "5"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    max_drawdown: float = float(os.getenv("MAX_DRAWDOWN", "0.15"))
    stop_loss_percent: float = float(os.getenv("STOP_LOSS_PERCENT", "0.03"))
    take_profit_percent: float = float(os.getenv("TAKE_PROFIT_PERCENT", "0.06"))
    max_leverage: float = float(os.getenv("MAX_LEVERAGE", "3"))
    
@dataclass
class MLConfig:
    """Machine Learning configuration settings"""
    enabled: bool = True
    retrain_interval: int = int(os.getenv("ML_RETRAIN_INTERVAL", "24"))
    retrain_interval_hours: int = 24
    feature_lookback: int = int(os.getenv("FEATURE_LOOKBACK", "100"))
    confidence_threshold: float = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.6"))
    models_path: str = "storage/models"
    min_samples_to_train: int = 500

 # ⬇️ ADD THIS BLOCK DIRECTLY BELOW MLConfig
@dataclass
class ThresholdsConfig:
    """Confidence thresholds by strategy family"""
    non_ml_confidence_threshold: float = float(os.getenv("NON_ML_CONFIDENCE_THRESHOLD", "0.45"))
    ml_confidence_threshold: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.60"))

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
 #   url: str = os.getenv("DATABASE_URL", "sqlite:///trading_bot.db")
    url: str = os.getenv("DATABASE_URL", "postgresql://trader:secure_password@postgres:5432/trading_bot")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
@dataclass
class NotificationConfig:
    """Notification configuration settings"""
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
#    enable_notifications: bool = bool(telegram_token and telegram_chat_id)
    enable_notifications: bool = False

    def __post_init__(self):
        self.enable_notifications = bool(self.telegram_token and self.telegram_chat_id)



class Config:
    """Main configuration class that loads and manages all settings"""
    
    def __init__(self):
        self.exchange = ExchangeConfig()
        self.trading = TradingConfig()
        self.ml = MLConfig()
        self.database = DatabaseConfig()
        self.notifications = NotificationConfig()
        
        # Load YAML configuration first (primary)
        yaml_config = self._load_yaml_config("config/config.yaml")
        
        # Load JSON configurations (fallback)
        self.trading_pairs = self._load_json_config("config/trading_pairs.json")
        self.strategies = self._load_json_config("config/strategies.json")
        self.risk_rules = self._load_json_config("config/risk_management.json")
        
        # Override with YAML config if available
        if yaml_config:
            # Update strategies config from YAML
            if 'strategies' in yaml_config:
                self.strategies.update(yaml_config['strategies'])
            
            # Update trading pairs from YAML
            if 'data' in yaml_config and 'symbols' in yaml_config['data']:
                self.trading_pairs['active_pairs'] = yaml_config['data']['symbols']
            
            # Store exchange configs from YAML
            if 'exchanges' in yaml_config:
                self.exchange_configs = yaml_config['exchanges']
            else:
                self.exchange_configs = {}
        
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
    
    def _load_yaml_config(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                print(f"Warning: Config file {file_path} not found, using defaults")
                return {}
        except yaml.YAMLError as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def get_active_trading_pairs(self) -> List[str]:
        """Get list of active trading pairs"""
        if "active_pairs" in self.trading_pairs:
            return self.trading_pairs["active_pairs"]
        return ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "LINKUSDT"]
    
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
    
    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """Get exchange configuration with environment variable substitution"""
        try:
            if not hasattr(self, 'exchange_configs'):
                return {}
            
            exchange_config = self.exchange_configs.get(exchange_name, {})
            if not exchange_config:
                return {}
            
            # Substitute environment variables
            config = {}
            for key, value in exchange_config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    # Extract environment variable name
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var, value)
                else:
                    config[key] = value
            
            return config
        except Exception as e:
            print(f"Error getting exchange config for {exchange_name}: {e}")
            return {}
    
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
# after assembling 'config'
config.thresholds = ThresholdsConfig()

# Export commonly used settings
TRADING_PAIRS = config.get_active_trading_pairs()
TIMEFRAMES = config.get_timeframes()
IS_PAPER_TRADING = config.is_paper_trading
IS_LIVE_TRADING = config.is_live_trading()
DEBUG = config.debug
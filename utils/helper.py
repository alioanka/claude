"""
Helper utilities for the trading bot system.
Provides common functions used across different modules.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration files and settings."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'trading': {
                'initial_capital': 100000,
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'max_drawdown': 0.15
            },
            'data': {
                'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD'],
                'timeframe': '1h',
                'lookback_days': 365
            },
            'ml': {
                'train_size': 0.8,
                'validation_size': 0.1,
                'test_size': 0.1,
                'random_state': 42
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by dot notation key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key: str, value: Any):
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

class DataValidator:
    """Validates data quality and consistency."""
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLCV data format and quality."""
        issues = []
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        if not issues:  # Only check data quality if columns exist
            # Check for negative values
            if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
                issues.append("Negative values found in OHLCV data")
            
            # Check high >= low
            if (df['high'] < df['low']).any():
                issues.append("High prices less than low prices")
            
            # Check OHLC relationships
            if (df['open'] > df['high']).any() or (df['open'] < df['low']).any():
                issues.append("Open prices outside high-low range")
            
            if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
                issues.append("Close prices outside high-low range")
            
            # Check for missing values
            if df.isnull().any().any():
                issues.append("Missing values found in data")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def check_data_completeness(df: pd.DataFrame, expected_frequency: str = '1H') -> float:
        """Check data completeness based on expected frequency."""
        if df.empty or not hasattr(df.index, 'freq'):
            return 0.0
        
        try:
            expected_points = len(pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq=expected_frequency
            ))
            actual_points = len(df)
            return actual_points / expected_points
        except Exception:
            return 0.0

class PerformanceCalculator:
    """Calculates various performance metrics."""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate percentage returns."""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
        return excess_returns / (returns.std() * np.sqrt(252))
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_volatility(returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, confidence * 100)
    
    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).mean()

class TechnicalIndicators:
    """Common technical indicators."""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

class TimeUtils:
    """Time-related utility functions."""
    
    @staticmethod
    def get_market_hours(timezone: str = 'US/Eastern') -> Tuple[int, int]:
        """Get market hours for a given timezone."""
        # Default to NYSE hours
        return 9, 16  # 9:30 AM to 4:00 PM
    
    @staticmethod
    def is_market_open(dt: datetime, timezone: str = 'US/Eastern') -> bool:
        """Check if market is open at given datetime."""
        # Simplified check - weekday and within market hours
        if dt.weekday() >= 5:  # Weekend
            return False
        
        hour = dt.hour
        market_open, market_close = TimeUtils.get_market_hours(timezone)
        return market_open <= hour < market_close
    
    @staticmethod
    def get_next_trading_day(dt: datetime) -> datetime:
        """Get next trading day (skip weekends)."""
        next_day = dt + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day

class FileManager:
    """File management utilities."""
    
    @staticmethod
    def ensure_directory_exists(path: str):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict, filepath: str):
        """Save data to JSON file."""
        FileManager.ensure_directory_exists(os.path.dirname(filepath))
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load data from JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return {}
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return {}
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str, format: str = 'parquet'):
        """Save DataFrame to file."""
        FileManager.ensure_directory_exists(os.path.dirname(filepath))
        
        if format.lower() == 'parquet':
            df.to_parquet(filepath)
        elif format.lower() == 'csv':
            df.to_csv(filepath)
        elif format.lower() == 'pickle':
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_dataframe(filepath: str, format: str = None) -> pd.DataFrame:
        """Load DataFrame from file."""
        if format is None:
            format = filepath.split('.')[-1]
        
        try:
            if format.lower() == 'parquet':
                return pd.read_parquet(filepath)
            elif format.lower() == 'csv':
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif format.lower() == 'pickle':
                return pd.read_pickle(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading DataFrame: {e}")
            return pd.DataFrame()

class NotificationManager:
    """Manages notifications and alerts."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enabled = self.config.get('notifications', {}).get('enabled', False)
    
    def send_alert(self, message: str, level: str = 'info'):
        """Send alert notification."""
        if not self.enabled:
            logger.info(f"Alert ({level}): {message}")
            return
        
        # In production, implement actual notification methods
        # (email, SMS, Slack, Discord, etc.)
        logger.info(f"ALERT ({level.upper()}): {message}")
    
    def send_trade_notification(self, trade_info: Dict):
        """Send trade execution notification."""
        message = f"Trade executed: {trade_info.get('action')} {trade_info.get('symbol')} at {trade_info.get('price')}"
        self.send_alert(message, 'trade')
    
    def send_error_notification(self, error: str):
        """Send error notification."""
        self.send_alert(f"System error: {error}", 'error')

# Convenience functions
def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        FileManager.ensure_directory_exists(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def format_currency(value: float, currency: str = 'USD') -> str:
    """Format currency value for display."""
    return f"{currency} {value:,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage for display."""
    return f"{value:.2%}"

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value."""
    return a / b if b != 0 else default

# Initialize global instances
config_manager = ConfigManager()
notification_manager = NotificationManager(config_manager.config)
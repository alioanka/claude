"""
Data package for trading bot.
Handles data storage, retrieval, and database operations.
"""

from .database import (
    DatabaseManager,
    Trade,
    Position,
    Signal,
    PerformanceMetrics,
    SystemLog,
    get_database,
    init_database,
    close_database
)

from .market_data import (
    MarketDataManager,
    YahooFinanceSource,
    CCXTSource,
    AlphaVantageSource,
    get_market_data,
    stream_market_data
)

from .data_processor import (
    DataProcessor,
    DataPipeline
)

__all__ = [
    'DatabaseManager',
    'Trade',
    'Position', 
    'MarketData',
    'Signal',
    'PerformanceMetrics',
    'SystemLog',
    'get_database',
    'init_database',
    'close_database',
    'MarketDataManager',
    'YahooFinanceSource',
    'CCXTSource', 
    'AlphaVantageSource',
    'get_market_data',
    'stream_market_data',
    'DataProcessor',
    'DataPipeline'
]
"""
Market data module for fetching and managing financial data.
Supports multiple data sources and real-time data streaming.
"""

import pandas as pd
import numpy as np
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import yfinance as yf
import ccxt
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import threading
from queue import Queue

from utils.helper import FileManager, ConfigManager, DataValidator

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for market data sources."""
    
    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, since: datetime = None, limit: int = None) -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get available symbols."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if data source is available."""
        pass

class YahooFinanceSource(DataSource):
    """Yahoo Finance data source implementation."""
    
    def __init__(self):
        self.name = "YahooFinance"
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', since: datetime = None, limit: int = None) -> pd.DataFrame:
        """Get OHLCV data from Yahoo Finance."""
        try:
            # Convert symbol format for Yahoo Finance
            yf_symbol = self._convert_symbol(symbol)
            
            # Convert timeframe
            period, interval = self._convert_timeframe(timeframe, since)
            
            # Fetch data
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            if 'adj close' in df.columns:
                df = df.drop('adj close', axis=1)
            
            # Limit data if specified
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            logger.info(f"Fetched {len(df)} rows for {symbol} from Yahoo Finance")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_symbols(self) -> List[str]:
        """Get popular symbols from Yahoo Finance."""
        # Return common symbols (in practice, you'd fetch this dynamically)
        return [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA',
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'BTC-USD', 'ETH-USD'
        ]
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is available."""
        try:
            test_ticker = yf.Ticker('AAPL')
            test_data = test_ticker.history(period='1d', interval='1h')
            return not test_data.empty
        except:
            return False
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format."""
        # Handle crypto symbols
        if '/' in symbol:
            base, quote = symbol.split('/')
            if quote.upper() == 'USD':
                return f"{base.upper()}-USD"
        return symbol.upper()
    
    def _convert_timeframe(self, timeframe: str, since: datetime = None) -> tuple:
        """Convert timeframe to Yahoo Finance format."""
        timeframe_map = {
            '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m',
            '60m': '60m', '90m': '90m', '1h': '1h', '1d': '1d', '5d': '5d',
            '1wk': '1wk', '1mo': '1mo', '3mo': '3mo'
        }
        
        interval = timeframe_map.get(timeframe, '1h')
        
        # Determine period based on since date
        if since:
            days_back = (datetime.now() - since).days
            if days_back <= 7:
                period = '7d'
            elif days_back <= 30:
                period = '1mo'
            elif days_back <= 90:
                period = '3mo'
            elif days_back <= 365:
                period = '1y'
            else:
                period = '5y'
        else:
            period = '1y'  # Default period
        
        return period, interval

class CCXTSource(DataSource):
    """CCXT-based cryptocurrency exchange data source."""
    
    def __init__(self, exchange_name: str = 'binance'):
        self.exchange_name = exchange_name
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize the CCXT exchange."""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'rateLimit': 1200,
                'enableRateLimit': True,
            })
            logger.info(f"Initialized {self.exchange_name} exchange")
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name}: {e}")
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', since: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """Get OHLCV data from cryptocurrency exchange."""
        if not self.exchange:
            return pd.DataFrame()
        
        try:
            # Convert since to timestamp
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since_ts, limit)
            
            if not ohlcv:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} rows for {symbol} from {self.exchange_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {self.exchange_name} for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_symbols(self) -> List[str]:
        """Get available trading symbols."""
        if not self.exchange:
            return []
        
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            logger.error(f"Error fetching symbols from {self.exchange_name}: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if exchange is available."""
        if not self.exchange:
            return False
        
        try:
            self.exchange.fetch_ticker('BTC/USDT')
            return True
        except:
            return False

class AlphaVantageSource(DataSource):
    """Alpha Vantage API data source."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', since: datetime = None, limit: int = None) -> pd.DataFrame:
        """Get OHLCV data from Alpha Vantage."""
        try:
            # Determine function based on timeframe
            if timeframe in ['1m', '5m', '15m', '30m', '60m']:
                function = 'TIME_SERIES_INTRADAY'
                interval = timeframe
            elif timeframe == '1d':
                function = 'TIME_SERIES_DAILY'
                interval = None
            else:
                logger.error(f"Unsupported timeframe for Alpha Vantage: {timeframe}")
                return pd.DataFrame()
            
            # Build request parameters
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            if interval:
                params['interval'] = interval
            
            # Make request
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Parse response
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return pd.DataFrame()
            
            # Find the time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                logger.error("No time series data found in Alpha Vantage response")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                row = {
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': float(values['5. volume'])
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Filter by date if specified
            if since:
                df = df[df.index >= since]
            
            # Limit data if specified
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            logger.info(f"Fetched {len(df)} rows for {symbol} from Alpha Vantage")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_symbols(self) -> List[str]:
        """Get available symbols (limited for Alpha Vantage free tier)."""
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage API is available."""
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': 'AAPL',
                'interval': '60min',
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            return 'Time Series' in str(data)
        except:
            return False

class MarketDataManager:
    """Main market data manager that handles multiple data sources."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.sources: Dict[str, DataSource] = {}
        self.primary_source = None
        self.file_manager = FileManager()
        self.validator = DataValidator()
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_dir = self.config.get('cache_dir', 'data/cache')
        
        # Initialize data sources
        self._initialize_sources()
        
        # Database for caching
        if self.cache_enabled:
            self._init_cache_db()
    
    def _initialize_sources(self):
        """Initialize available data sources."""
        # Yahoo Finance (free)
        yf_source = YahooFinanceSource()
        if yf_source.is_available():
            self.sources['yahoo'] = yf_source
            if not self.primary_source:
                self.primary_source = 'yahoo'
        
        # Alpha Vantage (if API key provided)
        if 'alphavantage_api_key' in self.config:
            av_source = AlphaVantageSource(self.config['alphavantage_api_key'])
            if av_source.is_available():
                self.sources['alphavantage'] = av_source
        
        # CCXT exchanges
        for exchange in self.config.get('exchanges', ['binance']):
            try:
                ccxt_source = CCXTSource(exchange)
                if ccxt_source.is_available():
                    self.sources[exchange] = ccxt_source
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange}: {e}")
        
        logger.info(f"Initialized data sources: {list(self.sources.keys())}")
    
    def _init_cache_db(self):
        """Initialize SQLite database for caching."""
        try:
            self.file_manager.ensure_directory_exists(self.cache_dir)
            self.cache_db_path = f"{self.cache_dir}/market_data_cache.db"
            
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    source TEXT,
                    timestamp TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, source, timestamp)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_source 
                ON ohlcv_cache(symbol, timeframe, source)
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Cache database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache database: {e}")
            self.cache_enabled = False
    
    def get_data(self, 
                 symbol: str, 
                 timeframe: str = '1h',
                 since: datetime = None,
                 limit: int = None,
                 source: str = None,
                 use_cache: bool = True) -> pd.DataFrame:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            since: Start date
            limit: Maximum number of records
            source: Preferred data source
            use_cache: Whether to use cached data
            
        Returns:
            OHLCV DataFrame
        """
        if not self.sources:
            logger.error("No data sources available")
            return pd.DataFrame()
        
        # Determine source to use
        source_name = source or self.primary_source or list(self.sources.keys())[0]
        
        if source_name not in self.sources:
            logger.warning(f"Source {source_name} not available, using {self.primary_source}")
            source_name = self.primary_source
        
        # Try to get from cache first
        if use_cache and self.cache_enabled:
            cached_data = self._get_from_cache(symbol, timeframe, source_name, since, limit)
            if not cached_data.empty:
                logger.info(f"Retrieved {len(cached_data)} rows from cache for {symbol}")
                return cached_data
        
        # Fetch from source
        data_source = self.sources[source_name]
        df = data_source.get_ohlcv(symbol, timeframe, since, limit)
        
        if df.empty:
            logger.warning(f"No data retrieved for {symbol}")
            return df
        
        # Validate data
        is_valid, issues = self.validator.validate_ohlcv_data(df)
        if not is_valid:
            logger.warning(f"Data validation issues for {symbol}: {issues}")
        
        # Cache the data
        if self.cache_enabled:
            self._cache_data(df, symbol, timeframe, source_name)
        
        return df
    
    def get_multiple_symbols(self, 
                           symbols: List[str],
                           timeframe: str = '1h',
                           since: datetime = None,
                           limit: int = None,
                           max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple symbols concurrently.
        
        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe
            since: Start date
            limit: Maximum number of records per symbol
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        data_dict = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.get_data, symbol, timeframe, since, limit): symbol
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result(timeout=30)
                    if not df.empty:
                        data_dict[symbol] = df
                    else:
                        logger.warning(f"No data retrieved for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
        
        logger.info(f"Retrieved data for {len(data_dict)}/{len(symbols)} symbols")
        return data_dict
    
    def _get_from_cache(self, symbol: str, timeframe: str, source: str, 
                       since: datetime = None, limit: int = None) -> pd.DataFrame:
        """Get data from cache database."""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            # Build query
            query = '''
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_cache
                WHERE symbol = ? AND timeframe = ? AND source = ?
            '''
            params = [symbol, timeframe, source]
            
            if since:
                query += ' AND timestamp >= ?'
                params.append(since.strftime('%Y-%m-%d %H:%M:%S'))
            
            query += ' ORDER BY timestamp'
            
            if limit:
                query += f' LIMIT {limit}'
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
            conn.close()
            
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return pd.DataFrame()
    
    def _cache_data(self, df: pd.DataFrame, symbol: str, timeframe: str, source: str):
        """Cache data to database."""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            # Prepare data for insertion
            cache_data = []
            for timestamp, row in df.iterrows():
                cache_data.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'source': source,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row.get('volume', 0)
                })
            
            # Insert data (ignore duplicates)
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR IGNORE INTO ohlcv_cache 
                (symbol, timeframe, source, timestamp, open, high, low, close, volume)
                VALUES (:symbol, :timeframe, :source, :timestamp, :open, :high, :low, :close, :volume)
            ''', cache_data)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Cached {len(cache_data)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def get_available_symbols(self, source: str = None) -> List[str]:
        """Get available symbols from a data source."""
        source_name = source or self.primary_source
        
        if source_name not in self.sources:
            return []
        
        return self.sources[source_name].get_symbols()
    
    def clear_cache(self, symbol: str = None, older_than_days: int = None):
        """Clear cached data."""
        if not self.cache_enabled:
            return
        
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('DELETE FROM ohlcv_cache WHERE symbol = ?', (symbol,))
            elif older_than_days:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                cursor.execute('DELETE FROM ohlcv_cache WHERE created_at < ?', 
                             (cutoff_date.strftime('%Y-%m-%d %H:%M:%S'),))
            else:
                cursor.execute('DELETE FROM ohlcv_cache')
            
            conn.commit()
            conn.close()
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all data sources."""
        health_status = {}
        
        for source_name, source in self.sources.items():
            try:
                health_status[source_name] = source.is_available()
            except Exception as e:
                logger.error(f"Health check failed for {source_name}: {e}")
                health_status[source_name] = False
        
        return health_status

class RealTimeDataStream:
    """Real-time market data streaming."""
    
    def __init__(self, symbols: List[str], callback: Callable, source: str = 'binance'):
        self.symbols = symbols
        self.callback = callback
        self.source = source
        self.running = False
        self.thread = None
        self.data_queue = Queue()
        
        # Initialize exchange for streaming
        if source in ccxt.exchanges:
            exchange_class = getattr(ccxt, source)
            self.exchange = exchange_class({'enableRateLimit': True})
        else:
            raise ValueError(f"Unsupported streaming source: {source}")
    
    def start(self):
        """Start real-time data streaming."""
        if self.running:
            logger.warning("Stream already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._stream_worker)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started real-time data stream for {self.symbols}")
    
    def stop(self):
        """Stop real-time data streaming."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Stopped real-time data stream")
    
    def _stream_worker(self):
        """Worker thread for streaming data."""
        while self.running:
            try:
                for symbol in self.symbols:
                    if not self.running:
                        break
                    
                    # Fetch latest ticker
                    ticker = self.exchange.fetch_ticker(symbol)
                    
                    # Create data point
                    data_point = {
                        'symbol': symbol,
                        'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000),
                        'price': ticker['last'],
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'volume': ticker['baseVolume']
                    }
                    
                    # Send to callback
                    self.callback(data_point)
                    
                    # Rate limiting
                    time.sleep(1)
                
                # Wait between cycles
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in streaming worker: {e}")
                time.sleep(10)  # Wait before retrying

# Convenience functions
def get_market_data(symbols: List[str], 
                   timeframe: str = '1h',
                   days_back: int = 30,
                   source: str = None,
                   config: Dict = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to get market data.
    
    Args:
        symbols: List of trading symbols
        timeframe: Data timeframe
        days_back: Number of days of historical data
        source: Preferred data source
        config: Configuration dictionary
        
    Returns:
        Dictionary of symbol -> DataFrame
    """
    manager = MarketDataManager(config)
    since = datetime.now() - timedelta(days=days_back)
    
    return manager.get_multiple_symbols(
        symbols=symbols,
        timeframe=timeframe,
        since=since,
        limit=days_back * 24 if timeframe == '1h' else None
    )

def stream_market_data(symbols: List[str], 
                      callback: Callable,
                      source: str = 'binance') -> RealTimeDataStream:
    """
    Convenience function to start real-time data streaming.
    
    Args:
        symbols: List of symbols to stream
        callback: Function to call with new data
        source: Data source for streaming
        
    Returns:
        RealTimeDataStream instance
    """
    stream = RealTimeDataStream(symbols, callback, source)
    stream.start()
    return stream
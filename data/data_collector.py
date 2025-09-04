"""
Data Collector - Real-time market data collection and management
Collects OHLCV, order book, trades, and other market data.
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import json
import websockets

from config.config import config, TRADING_PAIRS, TIMEFRAMES
from core.exchange_manager import ExchangeManager
from data.database import DatabaseManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MarketData:
    """Market data structure"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ohlcv: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.order_book: Optional[Dict] = None
        self.recent_trades: deque = deque(maxlen=100)
        self.ticker: Optional[Dict] = None
        self.funding_rate: Optional[float] = None
        self.last_update = datetime.utcnow()
    
    def update_ohlcv(self, timeframe: str, candle: List):
        """Update OHLCV data for a timeframe"""
        self.ohlcv[timeframe].append({
            'timestamp': candle[0],
            'open': candle[1],
            'high': candle[2],
            'low': candle[3],
            'close': candle[4],
            'volume': candle[5]
        })
        self.last_update = datetime.utcnow()
    
    def get_latest_candles(self, timeframe: str, count: int = 100) -> List[Dict]:
        """Get latest candles for a timeframe"""
        candles = list(self.ohlcv[timeframe])
        return candles[-count:] if len(candles) >= count else candles
    
    def get_current_price(self) -> Optional[float]:
        """Get current price from latest 1m candle or ticker"""
        if '1m' in self.ohlcv and self.ohlcv['1m']:
            return self.ohlcv['1m'][-1]['close']
        elif self.ticker:
            return self.ticker.get('last')
        return None

class DataCollector:
    """Main data collection and management class"""
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.db_manager = DatabaseManager(config.database.url)
        
        # Data storage
        self.market_data: Dict[str, MarketData] = {}
        self.websocket_connections: Dict[str, Any] = {}
        
        # Collection state
        self.is_collecting = False
        self.collection_tasks: List[asyncio.Task] = []
        
        # Configuration
        self.collection_interval = 60  # seconds
        self.websocket_enabled = True
        self.backup_rest_calls = True
        
        # Performance tracking
        self.data_stats = {
            'candles_collected': 0,
            'trades_collected': 0,
            'orderbook_updates': 0,
            'websocket_reconnects': 0,
            'last_collection_time': None
        }
    
    async def initialize(self):
        """Initialize data collector"""
        try:
            logger.info("📊 Initializing Data Collector...")
            
            # Initialize database
            await self.db_manager.initialize()
            
            # Initialize market data objects
            for symbol in TRADING_PAIRS:
                self.market_data[symbol] = MarketData(symbol)
            
            # Load historical data
            await self.load_initial_data()
            
            logger.info(f"✅ Data Collector initialized for {len(TRADING_PAIRS)} pairs")
            
        except Exception as e:
            logger.error(f"❌ Data Collector initialization failed: {e}")
            raise
    
    async def load_initial_data(self):
        """Load initial historical data for all pairs"""
        try:
            logger.info("📈 Loading initial historical data...")
            
            for symbol in TRADING_PAIRS:
                for timeframe in TIMEFRAMES:
                    try:
                        # Get last 500 candles
                        candles = await self.exchange_manager.get_ohlcv(
                            symbol, timeframe, limit=500
                        )
                        
                        if candles:
                            # Store in market data
                            for candle in candles:
                                self.market_data[symbol].update_ohlcv(timeframe, candle)
                            
                            # Store in database
                            await self.db_manager.store_ohlcv(symbol, timeframe, candles)
                            
                            logger.debug(f"📊 Loaded {len(candles)} {timeframe} candles for {symbol}")
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load data for {symbol} {timeframe}: {e}")
            
            logger.info("✅ Initial data loading completed")
            
        except Exception as e:
            logger.error(f"❌ Initial data loading failed: {e}")
    
    async def start_collection(self):
        """Start data collection processes"""
        try:
            if self.is_collecting:
                logger.warning("⚠️ Data collection already running")
                return
            
            self.is_collecting = True
            logger.info("🚀 Starting data collection...")
            
            # Start WebSocket connections if enabled
            if self.websocket_enabled:
                self.collection_tasks.append(
                    asyncio.create_task(self.start_websocket_collection())
                )
            
            # Start REST API data collection (backup)
            if self.backup_rest_calls:
                self.collection_tasks.append(
                    asyncio.create_task(self.start_rest_collection())
                )
            
            # Start order book collection
            self.collection_tasks.append(
                asyncio.create_task(self.collect_order_books())
            )
            
            # Start funding rate collection
            self.collection_tasks.append(
                asyncio.create_task(self.collect_funding_rates())
            )
            
            logger.info("✅ Data collection started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start data collection: {e}")
            self.is_collecting = False
            raise
    
    async def start_websocket_collection(self):
        """Start WebSocket data collection"""
        try:
            logger.info("🌐 Starting WebSocket data collection...")
            
            # Binance WebSocket stream URLs
            base_url = "wss://stream.binance.com:9443/ws/"
            
            # Create streams for each pair
            streams = []
            for symbol in TRADING_PAIRS:
                symbol_lower = symbol.lower()
                # Kline streams for different timeframes
                for timeframe in ['1m', '5m', '15m', '1h', '4h']:
                    streams.append(f"{symbol_lower}@kline_{timeframe}")
                
                # Trade stream
                streams.append(f"{symbol_lower}@trade")
                
                # Ticker stream
                streams.append(f"{symbol_lower}@ticker")
            
            # Combine streams
            stream_url = base_url + "/".join(streams)
            
            while self.is_collecting:
                try:
                    async with websockets.connect(stream_url) as websocket:
                        logger.info("✅ WebSocket connected")
                        
                        while self.is_collecting:
                            try:
                                message = await asyncio.wait_for(
                                    websocket.recv(), timeout=30
                                )
                                
                                data = json.loads(message)
                                await self.process_websocket_message(data)
                                
                            except asyncio.TimeoutError:
                                # Send ping to keep connection alive
                                await websocket.ping()
                            except json.JSONDecodeError as e:
                                logger.warning(f"⚠️ Invalid JSON in WebSocket message: {e}")
                            except Exception as e:
                                logger.error(f"❌ WebSocket message processing error: {e}")
                                break
                
                except Exception as e:
                    logger.error(f"❌ WebSocket connection error: {e}")
                    self.data_stats['websocket_reconnects'] += 1
                    
                    if self.is_collecting:
                        logger.info("🔄 Reconnecting WebSocket in 5 seconds...")
                        await asyncio.sleep(5)
        
        except Exception as e:
            logger.error(f"❌ WebSocket collection failed: {e}")
    
    async def process_websocket_message(self, data: Dict):
        """Process WebSocket message"""
        try:
            if 'stream' not in data or 'data' not in data:
                return
            
            stream = data['stream']
            msg_data = data['data']
            
            # Parse symbol from stream name
            symbol_part = stream.split('@')[0].upper()
            symbol = None
            
            # Find matching symbol (handle case differences)
            for trading_symbol in TRADING_PAIRS:
                if trading_symbol.lower().startswith(symbol_part.lower()):
                    symbol = trading_symbol
                    break
            
            if not symbol:
                return
            
            # Process different message types
            if '@kline' in stream:
                await self.process_kline_data(symbol, msg_data)
            elif '@trade' in stream:
                await self.process_trade_data(symbol, msg_data)
            elif '@ticker' in stream:
                await self.process_ticker_data(symbol, msg_data)
        
        except Exception as e:
            logger.error(f"❌ WebSocket message processing error: {e}")
    
    async def process_kline_data(self, symbol: str, data: Dict):
        """Process kline (candlestick) data"""
        try:
            kline = data.get('k', {})
            if not kline.get('x'):  # Only process closed candles
                return
            
            timeframe = kline.get('i')  # Interval
            candle = [
                int(kline.get('t', 0)),  # Open time
                float(kline.get('o', 0)),  # Open
                float(kline.get('h', 0)),  # High
                float(kline.get('l', 0)),  # Low
                float(kline.get('c', 0)),  # Close
                float(kline.get('v', 0)),  # Volume
            ]
            
            # Update market data
            self.market_data[symbol].update_ohlcv(timeframe, candle)
            
            # Store in database
            await self.db_manager.store_ohlcv(symbol, timeframe, [candle])
            
            self.data_stats['candles_collected'] += 1
            
        except Exception as e:
            logger.error(f"❌ Kline data processing error: {e}")
    
    async def process_trade_data(self, symbol: str, data: Dict):
        """Process individual trade data"""
        try:
            trade = {
                'id': data.get('t'),
                'price': float(data.get('p', 0)),
                'quantity': float(data.get('q', 0)),
                'time': int(data.get('T', 0)),
                'is_buyer_maker': data.get('m', False)
            }
            
            self.market_data[symbol].recent_trades.append(trade)
            self.data_stats['trades_collected'] += 1
            
        except Exception as e:
            logger.error(f"❌ Trade data processing error: {e}")
    
    async def process_ticker_data(self, symbol: str, data: Dict):
        """Process ticker data"""
        try:
            ticker = {
                'symbol': data.get('s'),
                'price_change': float(data.get('p', 0)),
                'price_change_percent': float(data.get('P', 0)),
                'last_price': float(data.get('c', 0)),
                'volume': float(data.get('v', 0)),
                'high': float(data.get('h', 0)),
                'low': float(data.get('l', 0))
            }
            
            self.market_data[symbol].ticker = ticker
            
        except Exception as e:
            logger.error(f"❌ Ticker data processing error: {e}")
    
    async def start_rest_collection(self):
        """Start REST API data collection (backup method)"""
        try:
            logger.info("🔄 Starting REST API data collection...")
            
            while self.is_collecting:
                try:
                    for symbol in TRADING_PAIRS:
                        # Get latest candles for primary timeframes
                        for timeframe in ['1m', '5m']:  # Most important timeframes
                            try:
                                candles = await self.exchange_manager.get_ohlcv(
                                    symbol, timeframe, limit=2
                                )
                                
                                if candles:
                                    # Only process the latest candle if it's complete
                                    latest_candle = candles[-1]
                                    self.market_data[symbol].update_ohlcv(timeframe, latest_candle)
                                
                            except Exception as e:
                                logger.debug(f"REST collection error for {symbol} {timeframe}: {e}")
                        
                        # Rate limiting
                        await asyncio.sleep(0.5)
                    
                    # Update collection time
                    self.data_stats['last_collection_time'] = datetime.utcnow()
                    
                    # Wait before next collection cycle
                    await asyncio.sleep(self.collection_interval)
                
                except Exception as e:
                    logger.error(f"❌ REST collection cycle error: {e}")
                    await asyncio.sleep(30)
        
        except Exception as e:
            logger.error(f"❌ REST collection failed: {e}")
    
    async def collect_order_books(self):
        """Collect order book data"""
        try:
            while self.is_collecting:
                try:
                    for symbol in TRADING_PAIRS:
                        try:
                            order_book = await self.exchange_manager.get_order_book(symbol, 20)
                            if order_book:
                                self.market_data[symbol].order_book = order_book
                                self.data_stats['orderbook_updates'] += 1
                        
                        except Exception as e:
                            logger.debug(f"Order book collection error for {symbol}: {e}")
                        
                        await asyncio.sleep(1)  # 1 second between symbols
                    
                    await asyncio.sleep(10)  # 10 seconds between cycles
                
                except Exception as e:
                    logger.error(f"❌ Order book collection error: {e}")
                    await asyncio.sleep(30)
        
        except Exception as e:
            logger.error(f"❌ Order book collection failed: {e}")
    
    async def collect_funding_rates(self):
        """Collect funding rates for futures"""
        try:
            # Fast capability check once
            ex = getattr(self.exchange_manager, "exchange", None)
            if not ex or not hasattr(ex, "fetch_funding_rate"):
                logger.info("Funding rate collection disabled: exchange has no fetch_funding_rate")
                return

            while self.is_collecting:
                try:
                    for symbol in TRADING_PAIRS:
                        try:
                            funding_rate = await self.exchange_manager.get_funding_rate(symbol)
                            if funding_rate is not None:
                                self.market_data[symbol].funding_rate = funding_rate
                        except Exception as e:
                            logger.debug(f"Funding rate collection error for {symbol}: {e}")

                    await asyncio.sleep(300)  # Update every 5 minutes
                except Exception as e:
                    logger.error(f"❌ Funding rate collection error: {e}")
                    await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"❌ Funding rate collection failed: {e}")

    
    async def get_latest_data(self, symbol: str, timeframe: str = '1m', 
                            count: int = 100) -> Optional[List[Dict]]:
        """Get latest market data for a symbol"""
        try:
            if symbol not in self.market_data:
                return None
            
            return self.market_data[symbol].get_latest_candles(timeframe, count)
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest data for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            if symbol not in self.market_data:
                return None
            
            return self.market_data[symbol].get_current_price()
            
        except Exception as e:
            logger.error(f"❌ Failed to get current price for {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol: str) -> Optional[Dict]:
        """Get latest order book for a symbol"""
        try:
            if symbol not in self.market_data:
                return None
            
            return self.market_data[symbol].order_book
            
        except Exception as e:
            logger.error(f"❌ Failed to get order book for {symbol}: {e}")
            return None
    
    async def get_recent_trades(self, symbol: str, count: int = 50) -> List[Dict]:
        """Get recent trades for a symbol"""
        try:
            if symbol not in self.market_data:
                return []
            
            trades = list(self.market_data[symbol].recent_trades)
            return trades[-count:] if len(trades) >= count else trades
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent trades for {symbol}: {e}")
            return []
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview"""
        try:
            overview = {
                'symbols_tracked': len(self.market_data),
                'active_collections': len(self.collection_tasks),
                'data_freshness': {},
                'collection_stats': self.data_stats.copy(),
                'last_update': datetime.utcnow().isoformat()
            }
            
            # Add data freshness for each symbol
            for symbol, market_data in self.market_data.items():
                time_since_update = (datetime.utcnow() - market_data.last_update).total_seconds()
                overview['data_freshness'][symbol] = {
                    'last_update': market_data.last_update.isoformat(),
                    'seconds_ago': time_since_update,
                    'is_fresh': time_since_update < 120  # Fresh if updated within 2 minutes
                }
            
            return overview
            
        except Exception as e:
            logger.error(f"❌ Failed to get market overview: {e}")
            return {}
    
    async def get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get detailed statistics for a symbol"""
        try:
            if symbol not in self.market_data:
                return {}
            
            market_data = self.market_data[symbol]
            
            # Get 24h statistics from 1m candles
            candles_24h = market_data.get_latest_candles('1m', 1440)  # 24 hours of 1m candles
            
            if not candles_24h:
                return {}
            
            prices = [c['close'] for c in candles_24h]
            volumes = [c['volume'] for c in candles_24h]
            
            current_price = prices[-1] if prices else 0
            price_24h_ago = prices[0] if prices else current_price
            
            stats = {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_24h': current_price - price_24h_ago,
                'price_change_24h_pct': ((current_price - price_24h_ago) / price_24h_ago * 100) if price_24h_ago > 0 else 0,
                'high_24h': max(prices) if prices else 0,
                'low_24h': min(prices) if prices else 0,
                'volume_24h': sum(volumes) if volumes else 0,
                'avg_volume': sum(volumes) / len(volumes) if volumes else 0,
                'volatility': self.calculate_volatility(prices),
                'candles_count': len(candles_24h),
                'last_update': market_data.last_update.isoformat(),
                'funding_rate': market_data.funding_rate
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics for {symbol}: {e}")
            return {}
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        try:
            if len(prices) < 2:
                return 0.0
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
            
            if not returns:
                return 0.0
            
            # Calculate standard deviation of returns
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = (variance ** 0.5) * 100  # Convert to percentage
            
            return volatility
            
        except Exception as e:
            logger.error(f"❌ Volatility calculation error: {e}")
            return 0.0
    
    async def get_correlation_matrix(self, timeframe: str = '1h', periods: int = 168) -> pd.DataFrame:
        """Get correlation matrix between trading pairs"""
        try:
            price_data = {}
            
            # Collect price data for all symbols
            for symbol in TRADING_PAIRS:
                candles = await self.get_latest_data(symbol, timeframe, periods)
                if candles and len(candles) >= periods:
                    prices = [c['close'] for c in candles[-periods:]]
                    price_data[symbol] = prices
            
            if len(price_data) < 2:
                return pd.DataFrame()
            
            # Create DataFrame and calculate correlation
            df = pd.DataFrame(price_data)
            correlation_matrix = df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"❌ Correlation calculation error: {e}")
            return pd.DataFrame()
    
    async def detect_market_regime(self) -> Dict[str, Any]:
        """Detect current market regime across all pairs"""
        try:
            regimes = {
                'trending_up': 0,
                'trending_down': 0,
                'ranging': 0,
                'high_volatility': 0,
                'low_volatility': 0,
                'overall_sentiment': 'neutral'
            }
            
            total_symbols = len(TRADING_PAIRS)
            volatility_threshold = 2.0  # 2% volatility threshold
            
            for symbol in TRADING_PAIRS:
                stats = await self.get_symbol_statistics(symbol)
                if not stats:
                    continue
                
                price_change_pct = stats.get('price_change_24h_pct', 0)
                volatility = stats.get('volatility', 0)
                
                # Trend detection
                if price_change_pct > 2:
                    regimes['trending_up'] += 1
                elif price_change_pct < -2:
                    regimes['trending_down'] += 1
                else:
                    regimes['ranging'] += 1
                
                # Volatility detection
                if volatility > volatility_threshold:
                    regimes['high_volatility'] += 1
                else:
                    regimes['low_volatility'] += 1
            
            # Convert to percentages
            if total_symbols > 0:
                for key in ['trending_up', 'trending_down', 'ranging', 'high_volatility', 'low_volatility']:
                    regimes[key] = (regimes[key] / total_symbols) * 100
            
            # Determine overall sentiment
            if regimes['trending_up'] > 50:
                regimes['overall_sentiment'] = 'bullish'
            elif regimes['trending_down'] > 50:
                regimes['overall_sentiment'] = 'bearish'
            elif regimes['ranging'] > 60:
                regimes['overall_sentiment'] = 'ranging'
            
            regimes['timestamp'] = datetime.utcnow().isoformat()
            return regimes
            
        except Exception as e:
            logger.error(f"❌ Market regime detection error: {e}")
            return {}
    
    async def export_data(self, symbol: str, timeframe: str, 
                         start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Export historical data as DataFrame"""
        try:
            data = await self.db_manager.get_historical_data(
                symbol, timeframe, start_date, end_date
            )
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Data export error: {e}")
            return pd.DataFrame()
    
    async def stop_collection(self):
        """Stop data collection"""
        try:
            logger.info("🛑 Stopping data collection...")
            
            self.is_collecting = False
            
            # Cancel all collection tasks
            for task in self.collection_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.collection_tasks.clear()
            
            # Close WebSocket connections
            for conn in self.websocket_connections.values():
                if hasattr(conn, 'close'):
                    await conn.close()
            self.websocket_connections.clear()
            
            # Close database connections
            await self.db_manager.close()
            
            logger.info("✅ Data collection stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping data collection: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on data collection"""
        try:
            health_status = {
                'is_collecting': self.is_collecting,
                'active_tasks': len([t for t in self.collection_tasks if not t.done()]),
                'websocket_status': 'connected' if self.websocket_connections else 'disconnected',
                'data_freshness': 'good',
                'collection_stats': self.data_stats.copy(),
                'issues': []
            }
            
            # Check data freshness
            stale_count = 0
            for symbol, market_data in self.market_data.items():
                time_since_update = (datetime.utcnow() - market_data.last_update).total_seconds()
                if time_since_update > 300:  # 5 minutes
                    stale_count += 1
            
            if stale_count > len(self.market_data) * 0.3:  # More than 30% stale
                health_status['data_freshness'] = 'stale'
                health_status['issues'].append(f"{stale_count} symbols have stale data")
            
            # Check collection statistics
            if self.data_stats['last_collection_time']:
                time_since_collection = (
                    datetime.utcnow() - self.data_stats['last_collection_time']
                ).total_seconds()
                
                if time_since_collection > 180:  # 3 minutes
                    health_status['issues'].append("No recent data collection activity")
            
            # Overall health score
            if not health_status['issues']:
                health_status['overall_health'] = 'healthy'
            elif len(health_status['issues']) == 1:
                health_status['overall_health'] = 'warning'
            else:
                health_status['overall_health'] = 'critical'
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {'overall_health': 'error', 'error': str(e)}
    
    async def get_top_movers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top price movers in the last 24 hours"""
        try:
            movers = []
            
            for symbol in TRADING_PAIRS:
                stats = await self.get_symbol_statistics(symbol)
                if stats and 'price_change_24h_pct' in stats:
                    movers.append({
                        'symbol': symbol,
                        'price_change_pct': stats['price_change_24h_pct'],
                        'current_price': stats['current_price'],
                        'volume_24h': stats['volume_24h'],
                        'volatility': stats['volatility']
                    })
            
            # Sort by absolute price change
            movers.sort(key=lambda x: abs(x['price_change_pct']), reverse=True)
            
            return movers[:limit]
            
        except Exception as e:
            logger.error(f"❌ Top movers calculation error: {e}")
            return []
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get detailed collection statistics"""
        return {
            'data_stats': self.data_stats.copy(),
            'symbols_tracked': len(self.market_data),
            'active_tasks': len(self.collection_tasks),
            'websocket_connections': len(self.websocket_connections),
            'is_collecting': self.is_collecting,
            'collection_interval': self.collection_interval,
            'websocket_enabled': self.websocket_enabled,
            'backup_rest_enabled': self.backup_rest_calls
        }
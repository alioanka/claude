"""
Exchange Manager - Handles all exchange interactions
Supports both paper trading and live trading modes.
"""

import ccxt.async_support as ccxt
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN

from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PaperTradingEngine:
    """Simulates trading for paper trading mode"""
    
    def __init__(self, initial_balance: float = 10000):
        self.balance = {
            'USDT': initial_balance,
            'total': initial_balance
        }
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[str, Dict] = {}
        self.order_id_counter = 1
        self.trade_history: List[Dict] = []
        
        # Paper trading settings - Fixed the slippage calculation
        self.slippage = 0.001  # 0.1% slippage
        self.commission = 0.001  # 0.1% commission
        
    async def create_order(self, symbol: str, side: str, amount: float, 
                          price: Optional[float] = None, order_type: str = 'market') -> Dict:
        """Create a paper trading order"""
        try:
            order_id = f"paper_{self.order_id_counter}"
            self.order_id_counter += 1
            
            # Get current market price
            current_price = await self.get_market_price(symbol)
            if not current_price:
                raise Exception(f"Cannot get market price for {symbol}")
            
            # Calculate execution price with slippage
            if order_type == 'market':
                if side == 'buy':
                    execution_price = current_price * (1 + self.slippage)
                else:
                    execution_price = current_price * (1 - self.slippage)
            else:
                execution_price = price or current_price
            
            # Calculate costs
            notional = amount * execution_price
            commission_cost = notional * self.commission
            total_cost = notional + commission_cost
            
            # Check balance for buy orders
            if side == 'buy' and total_cost > self.balance.get('USDT', 0):
                raise Exception("Insufficient balance for buy order")
            
            # Execute the order
            order = {
                'id': order_id,
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': execution_price,
                'cost': notional,
                'commission': commission_cost,
                'status': 'closed',
                'timestamp': datetime.utcnow().timestamp() * 1000,
                'type': order_type
            }
            
            # Update balances and positions
            await self.update_balances_and_positions(order)
            
            # Store order
            self.orders[order_id] = order
            self.trade_history.append(order)
            
            logger.info(f"Paper order executed: {symbol} {side} {amount} @ {execution_price:.4f}")
            
            return order
            
        except Exception as e:
            logger.error(f"Paper order failed: {e}")
            raise
    
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price from exchange; fallback to mock."""
        try:
            ex_symbol = self._to_exchange_symbol(symbol)
            async with self.rate_limiter:
                ticker = await self.exchange.fetch_ticker(ex_symbol)
                # prefer 'last', fallback to bid/ask mid
                price = ticker.get('last') or (
                    (ticker.get('bid') or 0) + (ticker.get('ask') or 0)
                ) / 2 or None
                if price and price > 0:
                    return float(price)
        except Exception as e:
            logger.debug(f"Price fetch failed for {symbol}: {e}")

        # Fallback mock (kept for resilience)
        mock_prices = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 3000.0,
            'BNBUSDT': 400.0,
            'SOLUSDT': 100.0,
            'XRPUSDT': 0.6
        }
        return mock_prices.get(symbol)

    
    async def update_balances_and_positions(self, order: Dict):
        """Update paper trading balances and positions"""
        symbol = order['symbol']
        side = order['side']
        amount = order['amount']
        price = order['price']
        cost = order['cost']
        commission = order['commission']
        
        base_asset = symbol.replace('USDT', '')
        
        if side == 'buy':
            # Decrease USDT balance
            self.balance['USDT'] -= (cost + commission)
            
            # Increase base asset balance
            if base_asset not in self.balance:
                self.balance[base_asset] = 0
            self.balance[base_asset] += amount
            
            # Update position
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_amount = pos['amount'] + amount
                avg_price = ((pos['amount'] * pos['price']) + (amount * price)) / total_amount
                pos['amount'] = total_amount
                pos['price'] = avg_price
            else:
                self.positions[symbol] = {
                    'amount': amount,
                    'price': price,
                    'side': 'long',
                    'timestamp': datetime.utcnow()
                }
        
        else:  # sell
            # Increase USDT balance
            self.balance['USDT'] += (cost - commission)
            
            # Decrease base asset balance
            if base_asset in self.balance:
                self.balance[base_asset] -= amount
                if self.balance[base_asset] <= 0:
                    del self.balance[base_asset]
            
            # Update or close position
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos['amount'] -= amount
                if pos['amount'] <= 0:
                    del self.positions[symbol]
        
        # Update total balance
        self.balance['total'] = self.balance['USDT']
        # Add value of other assets (simplified)
        for asset, amount in self.balance.items():
            if asset not in ['USDT', 'total']:
                price = await self.get_market_price(f"{asset}USDT")
                if price:
                    self.balance['total'] += amount * price

class ExchangeManager:
    """Manages exchange connections and trading operations"""
    
    def __init__(self):
        self.exchange: Optional[ccxt.Exchange] = None
        self.paper_engine: Optional[PaperTradingEngine] = None
        self.is_paper_trading = config.is_paper_trading
        self.exchange_info: Dict = {}
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
        
    async def initialize(self):
        """Initialize exchange connection"""
        try:
            logger.info(f"Initializing exchange connection (Mode: {config.trading.mode})...")
            
            if self.is_paper_trading:
                await self.initialize_paper_trading()
            else:
                await self.initialize_live_trading()
            
            # Load exchange info
            await self.load_exchange_info()
            
            logger.info("Exchange connection initialized successfully!")
            
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
            raise

    def _to_exchange_symbol(self, symbol: str) -> str:
        """Normalize 'BTCUSDT' -> 'BTC/USDT' for ccxt if needed."""
        if "/" in symbol:
            return symbol
        for q in ("USDT", "BUSD", "USD"):
            if symbol.endswith(q):
                base = symbol[:-len(q)]
                return f"{base}/{q}"
        return symbol


    async def initialize_paper_trading(self):
        """Initialize paper trading mode"""
        self.paper_engine = PaperTradingEngine(config.trading.initial_capital)
        
        # Still need exchange connection for market data
        self.exchange = ccxt.binance({
            'apiKey': config.exchange.api_key,
            'secret': config.exchange.secret,
            'sandbox': False,  # Use production for data
            'enableRateLimit': True,
            'timeout': config.exchange.timeout
        })

        # 🔧 Share exchange, limiter, and symbol normalizer with the paper engine
        self.paper_engine.exchange = self.exchange
        self.paper_engine.rate_limiter = self.rate_limiter
        self.paper_engine._to_exchange_symbol = self._to_exchange_symbol
        
        # Test connection
        await self.test_connection()

    
    async def initialize_live_trading(self):
        """Initialize live trading mode"""
        self.exchange = ccxt.binance({
            'apiKey': config.exchange.api_key,
            'secret': config.exchange.secret,
            'sandbox': config.exchange.testnet,
            'enableRateLimit': True,
            'timeout': config.exchange.timeout
        })
        
        # Test connection and permissions
        await self.test_connection()
        await self.verify_trading_permissions()
    
    async def test_connection(self):
        """Test exchange connection with proper async/await handling"""
        try:
            async with self.rate_limiter:
                # Load markets with proper async handling
                try:
                    markets = await self.exchange.load_markets()
                    logger.info(f"Markets loaded: {len(markets)} pairs")
                except Exception as load_error:
                    logger.warning(f"Markets loading failed: {load_error}")
                
                # Test connection differently for paper vs live trading
                if self.is_paper_trading:
                    logger.info("Paper trading mode connection successful")
                else:
                    # For live trading, test with a simple API call
                    try:
                        if hasattr(self.exchange, 'fetch_status'):
                            status = await self.exchange.fetch_status()
                            logger.info("Exchange connection successful")
                        else:
                            # Fallback: test with ticker data
                            ticker = await self.exchange.fetch_ticker('BTCUSDT')
                            logger.info(f"Exchange connection successful. BTC: ${ticker.get('last', 0):.2f}")
                    except Exception as api_error:
                        logger.warning(f"API test failed: {api_error}")
                        logger.info("Exchange connection established (limited API access)")
                        
        except Exception as e:
            logger.error(f"Exchange connection test failed: {e}")
            raise
    
    async def verify_trading_permissions(self):
        """Verify that API keys have trading permissions"""
        try:
            if self.is_paper_trading:
                logger.info("Paper trading - skipping permissions check")
                return
            
            # For live trading, check account info instead of creating test orders
            try:
                async with self.rate_limiter:
                    account = await self.exchange.fetch_balance()
                    logger.info("Trading permissions verified")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'permission' in error_msg or 'authorized' in error_msg or 'forbidden' in error_msg:
                    raise Exception(f"Trading permissions denied: {e}")
                else:
                    # Other errors might be temporary, so just warn
                    logger.warning(f"Could not verify trading permissions: {e}")
                    logger.info("Trading permissions check skipped due to API limitations")
                    
        except Exception as e:
            logger.error(f"Trading permissions verification failed: {e}")
            if not self.is_paper_trading:
                raise
    
    # core/exchange_manager.py
    async def load_exchange_info(self):
        """Load exchange information and trading rules"""
        try:
            # Properly await ccxt load
            markets = await self.exchange.load_markets()
            self.exchange_info = {}

            for symbol in config.get_active_trading_pairs():
                ex_symbol = self._to_exchange_symbol(symbol)   # <-- normalize BTCUSDT -> BTC/USDT

                if ex_symbol in markets:
                    market = markets[ex_symbol]
                    self.exchange_info[symbol] = {
                        'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0.001),
                        'min_cost': market.get('limits', {}).get('cost', {}).get('min', 10.0),
                        'price_precision': market.get('precision', {}).get('price', 2),
                        'amount_precision': market.get('precision', {}).get('amount', 6),
                        'tick_size': market.get('info', {}).get('tickSize', '0.01'),
                        'step_size': market.get('info', {}).get('stepSize', '0.001'),
                        'symbol_ccxt': ex_symbol
                    }
                else:
                    logger.debug(f"Pair {symbol} not found in exchange markets; looked for {ex_symbol}")

            logger.info(f"Loaded exchange info for {len(self.exchange_info)} trading pairs")

        except Exception as e:
            logger.error(f"Failed to load exchange info: {e}")

    
    async def create_order(self, symbol: str, side: str, amount: float, 
                          price: Optional[float] = None, order_type: str = 'market',
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Dict:
        """Create a trading order"""
        try:
            # Adjust amount and price according to exchange rules
            adjusted_amount = self.adjust_amount(symbol, amount)
            adjusted_price = self.adjust_price(symbol, price) if price else None
            
            if self.is_paper_trading:
                return await self.paper_engine.create_order(
                    symbol, side, adjusted_amount, adjusted_price, order_type
                )
            else:
                return await self.create_live_order(
                    symbol, side, adjusted_amount, adjusted_price, order_type,
                    stop_loss, take_profit
                )
                
        except Exception as e:
            logger.error(f"Order creation failed: {symbol} {side} {amount} - {e}")
            raise
    
    async def create_live_order(self, symbol: str, side: str, amount: float,
                               price: Optional[float], order_type: str,
                               stop_loss: Optional[float] = None,
                               take_profit: Optional[float] = None) -> Dict:
        """Create a live trading order"""
        try:
            async with self.rate_limiter:
                # Create main order
                if order_type == 'market':
                    order = await self.exchange.create_market_order(symbol, side, amount)
                else:
                    order = await self.exchange.create_limit_order(symbol, side, amount, price)
                
                # Create stop loss order if specified
                if stop_loss and order['status'] == 'closed':
                    try:
                        sl_side = 'sell' if side == 'buy' else 'buy'
                        await self.exchange.create_order(
                            symbol, 'stop_market', sl_side, amount, None, None, {
                                'stopPrice': stop_loss,
                                'timeInForce': 'GTC'
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create stop loss: {e}")
                
                # Create take profit order if specified
                if take_profit and order['status'] == 'closed':
                    try:
                        tp_side = 'sell' if side == 'buy' else 'buy'
                        await self.exchange.create_limit_order(
                            symbol, tp_side, amount, take_profit
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create take profit: {e}")
                
                logger.info(f"Live order created: {symbol} {side} {amount} @ {order.get('price', 'market')}")
                return order
                
        except Exception as e:
            logger.error(f"Live order creation failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        try:
            if self.is_paper_trading:
                if order_id in self.paper_engine.orders:
                    order = self.paper_engine.orders[order_id]
                    if order['status'] != 'closed':
                        order['status'] = 'canceled'
                        return True
                return False
            else:
                async with self.rate_limiter:
                    result = await self.exchange.cancel_order(order_id, symbol)
                    return result['status'] == 'canceled'
                    
        except Exception as e:
            logger.error(f"Order cancellation failed: {order_id} - {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        """Get order status"""
        try:
            if self.is_paper_trading:
                return self.paper_engine.orders.get(order_id)
            else:
                async with self.rate_limiter:
                    return await self.exchange.fetch_order(order_id, symbol)
                    
        except Exception as e:
            logger.error(f"Failed to get order status: {order_id} - {e}")
            return None
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        try:
            if self.is_paper_trading:
                return self.paper_engine.balance.copy()
            else:
                async with self.rate_limiter:
                    balance = await self.exchange.fetch_balance()
                    return {
                        'USDT': balance.get('USDT', {}).get('total', 0),
                        'total': balance.get('total', {}).get('USDT', 0)
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return {'USDT': 0, 'total': 0}
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            if self.is_paper_trading:
                positions = []
                for symbol, pos in self.paper_engine.positions.items():
                    positions.append({
                        'symbol': symbol,
                        'side': pos['side'],
                        'amount': pos['amount'],
                        'entry_price': pos['price'],
                        'timestamp': pos['timestamp'],
                        'unrealized_pnl': 0  # Would calculate based on current price
                    })
                return positions
            else:
                async with self.rate_limiter:
                    positions = await self.exchange.fetch_positions()
                    # Filter only positions with size > 0
                    active_positions = [
                        pos for pos in positions 
                        if float(pos.get('contracts', 0)) > 0
                    ]
                    return active_positions
                    
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol (normalized for ccxt in all modes)"""
        try:
            ex_symbol = self._to_exchange_symbol(symbol)
            async with self.rate_limiter:
                ticker = await self.exchange.fetch_ticker(ex_symbol)
                price = ticker.get('last') or (
                    ((ticker.get('bid') or 0) + (ticker.get('ask') or 0)) / 2 if ticker else 0
                )
                return float(price) if price else None
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            # Paper fallback via the paper engine mock as last resort
            if self.is_paper_trading and hasattr(self, "paper_engine"):
                try:
                    return await self.paper_engine.get_market_price(symbol)
                except Exception:
                    pass
            return None

    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Get order book data"""
        try:
            ex_symbol = self._to_exchange_symbol(symbol)
            async with self.rate_limiter:
                order_book = await self.exchange.fetch_order_book(ex_symbol, limit)
                return order_book

                
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return None
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        try:
            ex_symbol = self._to_exchange_symbol(symbol)
            async with self.rate_limiter:
                trades = await self.exchange.fetch_trades(ex_symbol, limit=limit)
                return trades

                
        except Exception as e:
            logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return []
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '1m',
                        since: Optional[int] = None, limit: int = 500) -> List[List]:
        """Get OHLCV candlestick data"""
        try:
            ex_symbol = self._to_exchange_symbol(symbol)
            async with self.rate_limiter:
                candles = await self.exchange.fetch_ohlcv(
                    ex_symbol, timeframe, since, limit
                )
                return candles
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {symbol}: {e}")
            return []

    
    def adjust_amount(self, symbol: str, amount: float) -> float:
        """Adjust order amount according to exchange rules"""
        try:
            if symbol not in self.exchange_info:
                return amount
            
            info = self.exchange_info[symbol]
            step_size = float(info.get('step_size', '0.001'))
            min_amount = info.get('min_amount', 0.001)
            
            # Round down to step size
            adjusted = float(Decimal(str(amount)).quantize(
                Decimal(str(step_size)), rounding=ROUND_DOWN
            ))
            
            # Ensure minimum amount
            return max(adjusted, min_amount)
            
        except Exception as e:
            logger.warning(f"Amount adjustment failed for {symbol}: {e}")
            return amount
    
    def adjust_price(self, symbol: str, price: float) -> float:
        """Adjust order price according to exchange rules"""
        try:
            if symbol not in self.exchange_info:
                return price
            
            info = self.exchange_info[symbol]
            tick_size = float(info.get('tick_size', '0.01'))
            
            # Round to tick size
            adjusted = float(Decimal(str(price)).quantize(
                Decimal(str(tick_size)), rounding=ROUND_DOWN
            ))
            
            return adjusted
            
        except Exception as e:
            logger.warning(f"Price adjustment failed for {symbol}: {e}")
            return price
    
    async def close_position(self, symbol: str) -> bool:
        """Close a position"""
        try:
            positions = await self.get_positions()
            
            for position in positions:
                if position['symbol'] == symbol:
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    amount = abs(float(position['amount']))
                    
                    order = await self.create_order(symbol, side, amount, order_type='market')
                    return order['status'] == 'closed'
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return False
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for a symbol"""
        try:
            if hasattr(self.exchange, 'fetch_trading_fees'):
                fees = await self.exchange.fetch_trading_fees()
                return fees.get(symbol, {'maker': 0.001, 'taker': 0.001})
            else:
                # Default Binance fees
                return {'maker': 0.001, 'taker': 0.001}
                
        except Exception as e:
            logger.warning(f"Failed to get trading fees for {symbol}: {e}")
            return {'maker': 0.001, 'taker': 0.001}
    
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for futures; returns None for spot or unsupported exchanges."""
        try:
            ex = getattr(self, "exchange", None)
            if not ex:
                return None

            # Quick feature test
            if not hasattr(ex, "fetch_funding_rate") or not callable(getattr(ex, "fetch_funding_rate")):
                return None

            # If your config has a market type, skip in spot mode
            market_type = getattr(getattr(config, "exchange", object), "market_type", "spot")
            if str(market_type).lower() == "spot":
                return None

            ex_symbol = self._to_exchange_symbol(symbol)
            async with self.rate_limiter:
                funding_rate = await ex.fetch_funding_rate(ex_symbol)
            # ccxt returns various shapes; be tolerant
            if isinstance(funding_rate, dict):
                return float(funding_rate.get("fundingRate") or funding_rate.get("fundingRate", 0.0) or 0.0)
            return None
        except Exception as e:
            logger.debug(f"Funding rate fetch skipped for {symbol}: {e}")
            return None


    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get overall market status"""
        try:
            status = {
                'exchange_status': 'online',
                'trading_enabled': True,
                'last_update': datetime.utcnow().isoformat()
            }
            
            if not self.is_paper_trading:
                # Check exchange status
                async with self.rate_limiter:
                    try:
                        await self.exchange.fetch_status()
                        status['exchange_status'] = 'online'
                    except:
                        status['exchange_status'] = 'offline'
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {'exchange_status': 'unknown', 'trading_enabled': False}
    
    async def close(self):
        """Close exchange connections"""
        try:
            if self.exchange and hasattr(self.exchange, 'close'):
                await self.exchange.close()
            
            logger.info("Exchange connections closed")
            
        except Exception as e:
            logger.error(f"Error closing exchange connections: {e}")
    
    def get_exchange_info(self, symbol: str) -> Dict[str, Any]:
        """Get exchange information for a symbol"""
        return self.exchange_info.get(symbol, {})
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is tradeable"""
        return symbol in self.exchange_info
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get detailed account information"""
        try:
            if self.is_paper_trading:
                return {
                    'account_type': 'paper',
                    'balance': self.paper_engine.balance,
                    'positions': len(self.paper_engine.positions),
                    'orders': len(self.paper_engine.orders)
                }
            else:
                async with self.rate_limiter:
                    account = await self.exchange.fetch_balance()
                    return {
                        'account_type': 'live',
                        'balance': account.get('total', {}),
                        'free': account.get('free', {}),
                        'used': account.get('used', {})
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {'account_type': 'unknown'}
    
    async def health_check(self) -> bool:
        """Perform health check on exchange connection"""
        try:
            # Test basic connectivity
            await self.get_current_price('BTCUSDT')
            return True
            
        except Exception as e:
            logger.error(f"Exchange health check failed: {e}")
            return False
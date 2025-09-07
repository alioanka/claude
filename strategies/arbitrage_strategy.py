"""
Arbitrage trading strategy for identifying and exploiting price differences
between exchanges or correlated trading pairs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass
import ccxt

from strategies.base_strategy import BaseStrategy, StrategySignal

from utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

@dataclass
class ArbitrageConfig:
    """Configuration for arbitrage strategy"""
    min_spread_threshold: float = 0.005  # 0.5% minimum spread
    max_spread_threshold: float = 0.05   # 5% maximum spread (might be error)
    correlation_threshold: float = 0.8   # Minimum correlation for pair trading
    lookback_period: int = 100           # Periods for correlation calculation
    z_score_threshold: float = 2.0       # Z-score threshold for mean reversion
    max_position_hold_time: int = 24     # Maximum hours to hold position
    transaction_cost: float = 0.002      # Total transaction costs (0.2%)

class ArbitrageStrategy(BaseStrategy):
    """
    Arbitrage strategy implementation supporting:
    1. Cross-exchange arbitrage
    2. Statistical arbitrage (pairs trading)
    3. Triangular arbitrage
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("arbitrage_strategy", config)
        self.arb_config = ArbitrageConfig(**config.get('arbitrage_params', {}))
        self.indicators = TechnicalIndicators()
        
        # Exchange connections for cross-exchange arbitrage
        self.exchanges = {}
        self.price_feeds = {}
        
        # Pairs for statistical arbitrage
        self.correlation_pairs = []
        self.spread_history = {}
        
        # Initialize exchanges
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            exchange_configs = self.config.get('exchanges', {})
            
            for exchange_name, config in exchange_configs.items():
                if config.get('enabled', False):
                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchanges[exchange_name] = exchange_class({
                        'apiKey': config.get('api_key', ''),
                        'secret': config.get('secret', ''),
                        'sandbox': config.get('sandbox', True),
                        'enableRateLimit': True
                    })
                    logger.info(f"Initialized exchange: {exchange_name}")
                    
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")

    def _to_strategy_signal(self, symbol: str, s: dict) -> StrategySignal:
        """Map internal dict signal → StrategySignal expected by the framework."""
        return StrategySignal(
            symbol=s.get('symbol', symbol),
            action='buy' if s.get('side') in ('buy', 'long') else
                   'sell' if s.get('side') in ('sell', 'short') else 'hold',
            confidence=float(s.get('confidence', 0.0)),
            entry_price=s.get('entry_price'),
            stop_loss=s.get('stop_loss'),
            take_profit=s.get('take_profit'),
            position_size=s.get('position_size'),
            reasoning=s.get('reasoning', ''),
            timeframe=s.get('timeframe', '1m'),
            strategy_name=self.name
        )

    
    async def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate arbitrage signals"""
        try:
            data = self._ensure_dataframe(data)

            signals = []
            
            # Cross-exchange arbitrage
            if len(self.exchanges) > 1:
                cross_exchange_signal = await self._check_cross_exchange_arbitrage(symbol)
                if cross_exchange_signal:
                    signals.append(cross_exchange_signal)
            
            # Statistical arbitrage (pairs trading)
            pairs_signal = await self._check_statistical_arbitrage(symbol, data)
            if pairs_signal:
                signals.append(pairs_signal)
            
            # Triangular arbitrage
            triangular_signal = await self._check_triangular_arbitrage(symbol)
            if triangular_signal:
                signals.append(triangular_signal)
            
            # Return highest confidence signal
            if signals:
                best_signal = max(signals, key=lambda s: s.get('confidence', 0))
                return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Arbitrage signal generation failed for {symbol}: {e}")
            return None

    def get_required_data(self) -> Dict[str, Any]:
        """
        Describe the minimal data requirements for this strategy.
        Called by the framework before analyze(). Keep this aligned with your config.
        """
        lookback = getattr(self.arb_config, "lookback_period", 100)
        return {
            "timeframe": "1m",           # or pull from self.config if you keep it there
            "lookback": max(lookback, 100),
            # if you drive symbols from config, expose that here:
            "symbols": self.config.get("symbols", []),
            "notes": "Pairs/stat-arb will request additional symbols internally"
        }

    async def analyze(self, symbol: str, market_data: Dict[str, Any]):
        """
        Adapter: consume raw market_data, produce a StrategySignal using normalized DF.
        """
        try:
            # Use the same normalizer everywhere
            df = self._ensure_dataframe(market_data, lookback=getattr(self, "lookback_period", 200))
            if df is None or df.empty or len(df) < getattr(self, "min_bars", 50):
                return self._no_signal(symbol, "Insufficient data")

            sig = await self.generate_signal(symbol, df)  # your internal dict-like signal
            if not sig:
                return self._no_signal(symbol, "No arb opportunity")

            action = sig.get("action") or sig.get("side")
            action = (str(action).lower() if action else "hold")
            if action in ("long", "buy"):
                action = "buy"
            elif action in ("short", "sell"):
                action = "sell"
            else:
                action = "hold"

            conf = float(sig.get("confidence", 0.0))
            entry = sig.get("entry_price") or sig.get("buy_price") or sig.get("sell_price")
            sl = sig.get("stop_loss")
            tp = sig.get("take_profit")

            return StrategySignal(
                symbol=symbol,
                action=action if conf >= self.min_confidence else "hold",
                confidence=conf,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                position_size=None,
                reasoning=sig.get("reasoning", "arbitrage"),
                timeframe="1m",
                strategy_name=self.name
            )
        except Exception as e:
            logger.error(f"[ArbitrageStrategy] analyze failed for {symbol}: {e}")
            return self._no_signal(symbol, f"error: {e}")



    async def _check_cross_exchange_arbitrage(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for arbitrage opportunities across exchanges"""
        try:
            prices = {}
            
            # Get current prices from all exchanges
            for exchange_name, exchange in self.exchanges.items():
                try:
                    ticker = await asyncio.wait_for(
                        asyncio.create_task(self._fetch_ticker(exchange, symbol)),
                        timeout=5.0
                    )
                    if ticker:
                        prices[exchange_name] = {
                            'bid': ticker['bid'],
                            'ask': ticker['ask'],
                            'timestamp': ticker['timestamp']
                        }
                except Exception as e:
                    logger.warning(f"Failed to get price from {exchange_name}: {e}")
                    continue
            
            if len(prices) < 2:
                return None
            
            # Find best arbitrage opportunity
            best_opportunity = None
            max_profit = 0
            
            for buy_exchange, buy_price in prices.items():
                for sell_exchange, sell_price in prices.items():
                    if buy_exchange == sell_exchange:
                        continue
                    
                    # Calculate potential profit
                    buy_cost = buy_price['ask']  # We buy at ask price
                    sell_revenue = sell_price['bid']  # We sell at bid price
                    
                    if buy_cost > 0 and sell_revenue > 0:
                        gross_profit = (sell_revenue - buy_cost) / buy_cost
                        net_profit = gross_profit - self.arb_config.transaction_cost
                        
                        if (net_profit > self.arb_config.min_spread_threshold and 
                            net_profit < self.arb_config.max_spread_threshold and
                            net_profit > max_profit):
                            
                            max_profit = net_profit
                            best_opportunity = {
                                'type': 'cross_exchange',
                                'buy_exchange': buy_exchange,
                                'sell_exchange': sell_exchange,
                                'buy_price': buy_cost,
                                'sell_price': sell_revenue,
                                'spread': gross_profit,
                                'net_profit': net_profit,
                                'confidence': min(net_profit * 20, 1.0)  # Scale to 0-1
                            }
            
            if best_opportunity:
                return {
                    'signal_type': 'arbitrage',
                    'symbol': symbol,
                    'confidence': best_opportunity['confidence'],
                    'metadata': best_opportunity,
                    'timestamp': datetime.utcnow()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Cross-exchange arbitrage check failed: {e}")
            return None
    
    async def _fetch_ticker(self, exchange, symbol: str) -> Optional[Dict]:
        """Fetch ticker data from exchange"""
        try:
            ticker = exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
            return None
    
    async def _check_statistical_arbitrage(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check for statistical arbitrage opportunities (pairs trading)"""
        try:
            # Find correlated pairs
            correlated_symbols = self._find_correlated_pairs(symbol, data)
            
            if not correlated_symbols:
                return None
            
            best_signal = None
            max_z_score = 0
            
            for corr_symbol, correlation in correlated_symbols:
                # Calculate spread and z-score
                signal = await self._calculate_pairs_signal(symbol, corr_symbol, data, correlation)
                
                if signal and abs(signal.get('z_score', 0)) > max_z_score:
                    max_z_score = abs(signal['z_score'])
                    best_signal = signal
            
            return best_signal
            
        except Exception as e:
            logger.error(f"Statistical arbitrage check failed: {e}")
            return None
    
    def _find_correlated_pairs(self, symbol: str, data: pd.DataFrame) -> List[Tuple[str, float]]:
        """Find symbols correlated with the target symbol"""
        # This would typically require data from multiple symbols
        # For now, return some common pairs
        
        common_pairs = {
            'BTCUSDT': ['ETHUSDT', 'BNBUSDT'],
            'ETHUSDT': ['BTCUSDT', 'ADAUSDT'],
            'BNBUSDT': ['BTCUSDT', 'ETHUSDT']
        }
        
        pairs = common_pairs.get(symbol, [])
        # Return with assumed correlation (in real implementation, calculate from data)
        return [(pair, 0.85) for pair in pairs]
    
    async def _calculate_pairs_signal(self, 
                                    symbol1: str, 
                                    symbol2: str, 
                                    data: pd.DataFrame, 
                                    correlation: float) -> Optional[Dict[str, Any]]:
        """Calculate pairs trading signal based on price spread"""
        try:
            # In a real implementation, you'd have data for both symbols
            # For now, simulate spread calculation
            
            if len(data) < self.arb_config.lookback_period:
                return None
            
            # Simulate spread calculation (price ratio)
            # In reality: spread = price1/price2 - historical_mean(price1/price2)
            recent_data = data.tail(self.arb_config.lookback_period)
            
            # Create synthetic spread for demonstration
            price_changes = recent_data['close'].pct_change().dropna()
            spread = np.cumsum(price_changes)  # Simplified spread proxy
            
            # Calculate z-score
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            if spread_std == 0:
                return None
            
            current_spread = spread.iloc[-1]
            z_score = (current_spread - spread_mean) / spread_std
            
            # Generate signal based on z-score
            if abs(z_score) > self.arb_config.z_score_threshold:
                signal_type = 'sell' if z_score > 0 else 'buy'  # Mean reversion
                confidence = min(abs(z_score) / 4.0, 1.0)  # Scale to 0-1
                
                return {
                    'signal_type': signal_type,
                    'symbol': symbol1,
                    'confidence': confidence,
                    'metadata': {
                        'type': 'statistical_arbitrage',
                        'paired_symbol': symbol2,
                        'z_score': z_score,
                        'spread': current_spread,
                        'correlation': correlation
                    },
                    'z_score': z_score,
                    'timestamp': datetime.utcnow()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Pairs signal calculation failed: {e}")
            return None
    
    async def _check_triangular_arbitrage(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for triangular arbitrage opportunities"""
        try:
            # Triangular arbitrage requires three currencies
            # Example: BTC/USD, ETH/USD, BTC/ETH
            
            if not symbol.endswith('USDT'):
                return None
            
            base_currency = symbol.replace('USDT', '')
            
            # Define triangular pairs
            triangular_pairs = [
                (f"{base_currency}USDT", "BTCUSDT", f"{base_currency}BTC"),
                (f"{base_currency}USDT", "ETHUSDT", f"{base_currency}ETH")
            ]
            
            best_opportunity = None
            max_profit = 0
            
            for pair1, pair2, pair3 in triangular_pairs:
                try:
                    # Get prices for all three pairs
                    prices = {}
                    for pair in [pair1, pair2, pair3]:
                        # Use first available exchange
                        exchange = list(self.exchanges.values())[0] if self.exchanges else None
                        if exchange:
                            ticker = await self._fetch_ticker(exchange, pair)
                            if ticker:
                                prices[pair] = ticker
                    
                    if len(prices) == 3:
                        profit = self._calculate_triangular_profit(prices, pair1, pair2, pair3)
                        
                        if profit > max_profit and profit > self.arb_config.min_spread_threshold:
                            max_profit = profit
                            best_opportunity = {
                                'type': 'triangular',
                                'pairs': [pair1, pair2, pair3],
                                'profit': profit,
                                'confidence': min(profit * 10, 1.0)
                            }
                            
                except Exception as e:
                    logger.warning(f"Triangular arbitrage calculation failed for {pair1}-{pair2}-{pair3}: {e}")
                    continue
            
            if best_opportunity:
                return {
                    'signal_type': 'arbitrage',
                    'symbol': symbol,
                    'confidence': best_opportunity['confidence'],
                    'metadata': best_opportunity,
                    'timestamp': datetime.utcnow()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Triangular arbitrage check failed: {e}")
            return None
    
    def _calculate_triangular_profit(self, 
                                   prices: Dict[str, Dict], 
                                   pair1: str, 
                                   pair2: str, 
                                   pair3: str) -> float:
        """Calculate profit from triangular arbitrage"""
        try:
            # Simplified triangular arbitrage calculation
            # Real implementation would consider all possible paths and fees
            
            p1_bid = prices[pair1]['bid']
            p1_ask = prices[pair1]['ask']
            p2_bid = prices[pair2]['bid'] 
            p2_ask = prices[pair2]['ask']
            p3_bid = prices[pair3]['bid']
            p3_ask = prices[pair3]['ask']
            
            # Calculate cross rate
            implied_rate = (p1_bid / p2_ask) * p2_bid / p3_ask
            market_rate = p3_bid
            
            # Calculate profit opportunity
            profit = (implied_rate - market_rate) / market_rate
            
            # Subtract transaction costs
            net_profit = profit - (self.arb_config.transaction_cost * 3)  # 3 trades
            
            return max(0, net_profit)
            
        except Exception as e:
            logger.error(f"Triangular profit calculation failed: {e}")
            return 0
    
    async def validate_arbitrage_opportunity(self, signal: Dict[str, Any]) -> bool:
        """Validate arbitrage opportunity before execution"""
        try:
            metadata = signal.get('metadata', {})
            arb_type = metadata.get('type')
            
            if arb_type == 'cross_exchange':
                return await self._validate_cross_exchange_opportunity(metadata)
            elif arb_type == 'statistical_arbitrage':
                return await self._validate_statistical_opportunity(metadata)
            elif arb_type == 'triangular':
                return await self._validate_triangular_opportunity(metadata)
            
            return False
            
        except Exception as e:
            logger.error(f"Arbitrage validation failed: {e}")
            return False
    
    async def _validate_cross_exchange_opportunity(self, metadata: Dict[str, Any]) -> bool:
        """Validate cross-exchange arbitrage opportunity"""
        try:
            buy_exchange = metadata['buy_exchange']
            sell_exchange = metadata['sell_exchange']
            symbol = metadata.get('symbol')
            
            # Refresh prices to ensure opportunity still exists
            current_prices = {}
            for exchange_name in [buy_exchange, sell_exchange]:
                if exchange_name in self.exchanges:
                    ticker = await self._fetch_ticker(self.exchanges[exchange_name], symbol)
                    if ticker:
                        current_prices[exchange_name] = ticker
            
            if len(current_prices) != 2:
                return False
            
            # Recalculate spread
            buy_price = current_prices[buy_exchange]['ask']
            sell_price = current_prices[sell_exchange]['bid']
            current_spread = (sell_price - buy_price) / buy_price
            net_profit = current_spread - self.arb_config.transaction_cost
            
            # Opportunity still exists if profit above threshold
            return net_profit > self.arb_config.min_spread_threshold
            
        except Exception as e:
            logger.error(f"Cross-exchange validation failed: {e}")
            return False
    
    async def _validate_statistical_opportunity(self, metadata: Dict[str, Any]) -> bool:
        """Validate statistical arbitrage opportunity"""
        try:
            z_score = abs(metadata.get('z_score', 0))
            correlation = metadata.get('correlation', 0)
            
            # Check if z-score still above threshold and correlation is strong
            return (z_score > self.arb_config.z_score_threshold and 
                   correlation > self.arb_config.correlation_threshold)
            
        except Exception as e:
            logger.error(f"Statistical arbitrage validation failed: {e}")
            return False
    
    async def _validate_triangular_opportunity(self, metadata: Dict[str, Any]) -> bool:
        """Validate triangular arbitrage opportunity"""
        try:
            pairs = metadata.get('pairs', [])
            
            if len(pairs) != 3:
                return False
            
            # Refresh all prices and recalculate
            exchange = list(self.exchanges.values())[0] if self.exchanges else None
            if not exchange:
                return False
            
            prices = {}
            for pair in pairs:
                ticker = await self._fetch_ticker(exchange, pair)
                if ticker:
                    prices[pair] = ticker
                else:
                    return False
            
            # Recalculate profit
            profit = self._calculate_triangular_profit(prices, pairs[0], pairs[1], pairs[2])
            return profit > self.arb_config.min_spread_threshold
            
        except Exception as e:
            logger.error(f"Triangular arbitrage validation failed: {e}")
            return False
    
    def calculate_arbitrage_position_size(self, 
                                        signal: Dict[str, Any], 
                                        portfolio_value: float,
                                        available_balance: Dict[str, float]) -> float:
        """Calculate optimal position size for arbitrage"""
        try:
            metadata = signal.get('metadata', {})
            confidence = signal.get('confidence', 0.5)
            
            # Base position size on confidence and available capital
            max_allocation = 0.2  # 20% max allocation to arbitrage
            base_size = portfolio_value * max_allocation * confidence
            
            # Check available balance constraints
            symbol = signal['symbol']
            base_currency = symbol.replace('USDT', '')
            
            if 'USDT' in available_balance:
                max_usdt_size = available_balance['USDT'] * 0.9  # Use 90% of available
                base_size = min(base_size, max_usdt_size)
            
            # Adjust for arbitrage type
            arb_type = metadata.get('type')
            if arb_type == 'triangular':
                # Triangular arbitrage requires more capital for multiple trades
                base_size *= 0.8
            elif arb_type == 'cross_exchange':
                # Cross-exchange requires balance on both exchanges
                base_size *= 0.7
            
            return max(base_size, portfolio_value * 0.001)  # Minimum 0.1% allocation
            
        except Exception as e:
            logger.error(f"Arbitrage position size calculation failed: {e}")
            return 0
    
    async def execute_arbitrage_strategy(self, signal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute arbitrage strategy"""
        try:
            metadata = signal.get('metadata', {})
            arb_type = metadata.get('type')
            
            if arb_type == 'cross_exchange':
                return await self._execute_cross_exchange_arbitrage(signal)
            elif arb_type == 'statistical_arbitrage':
                return await self._execute_statistical_arbitrage(signal)
            elif arb_type == 'triangular':
                return await self._execute_triangular_arbitrage(signal)
            
            return []
            
        except Exception as e:
            logger.error(f"Arbitrage execution failed: {e}")
            return []
    
    async def _execute_cross_exchange_arbitrage(self, signal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute cross-exchange arbitrage"""
        try:
            metadata = signal['metadata']
            symbol = signal['symbol']
            
            buy_exchange_name = metadata['buy_exchange']
            sell_exchange_name = metadata['sell_exchange']
            
            buy_exchange = self.exchanges.get(buy_exchange_name)
            sell_exchange = self.exchanges.get(sell_exchange_name)
            
            if not all([buy_exchange, sell_exchange]):
                logger.error("Required exchanges not available")
                return []
            
            # Execute simultaneous buy and sell
            trades = []
            
            # Note: In real implementation, these would be actual exchange orders
            # Here we simulate the trade execution
            
            position_size = 0.1  # This would come from position sizing
            
            trades.append({
                'exchange': buy_exchange_name,
                'symbol': symbol,
                'side': 'buy',
                'size': position_size,
                'price': metadata['buy_price'],
                'timestamp': datetime.utcnow(),
                'strategy': 'arbitrage_cross_exchange'
            })
            
            trades.append({
                'exchange': sell_exchange_name,
                'symbol': symbol,
                'side': 'sell',
                'size': position_size,
                'price': metadata['sell_price'],
                'timestamp': datetime.utcnow(),
                'strategy': 'arbitrage_cross_exchange'
            })
            
            logger.info(f"Cross-exchange arbitrage executed for {symbol}")
            return trades
            
        except Exception as e:
            logger.error(f"Cross-exchange arbitrage execution failed: {e}")
            return []
    
    async def _execute_statistical_arbitrage(self, signal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute statistical arbitrage (pairs trading)"""
        try:
            metadata = signal['metadata']
            symbol = signal['symbol']
            paired_symbol = metadata['paired_symbol']
            z_score = metadata['z_score']
            
            trades = []
            position_size = 0.1  # This would come from position sizing
            
            # If z-score is positive, sell the expensive asset and buy the cheap one
            if z_score > 0:
                # Spread is above mean - sell symbol, buy paired_symbol
                trades.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'size': position_size,
                    'strategy': 'arbitrage_statistical',
                    'metadata': {'pair': paired_symbol, 'z_score': z_score}
                })
                trades.append({
                    'symbol': paired_symbol,
                    'side': 'buy',
                    'size': position_size,
                    'strategy': 'arbitrage_statistical',
                    'metadata': {'pair': symbol, 'z_score': z_score}
                })
            else:
                # Spread is below mean - buy symbol, sell paired_symbol
                trades.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'size': position_size,
                    'strategy': 'arbitrage_statistical',
                    'metadata': {'pair': paired_symbol, 'z_score': z_score}
                })
                trades.append({
                    'symbol': paired_symbol,
                    'side': 'sell',
                    'size': position_size,
                    'strategy': 'arbitrage_statistical',
                    'metadata': {'pair': symbol, 'z_score': z_score}
                })
            
            logger.info(f"Statistical arbitrage executed for {symbol}/{paired_symbol}")
            return trades
            
        except Exception as e:
            logger.error(f"Statistical arbitrage execution failed: {e}")
            return []
    
    async def _execute_triangular_arbitrage(self, signal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute triangular arbitrage"""
        try:
            metadata = signal['metadata']
            pairs = metadata['pairs']
            
            # Execute sequence of three trades
            trades = []
            position_size = 0.1
            
            # This is a simplified implementation
            # Real triangular arbitrage requires precise timing and size calculation
            
            for i, pair in enumerate(pairs):
                side = 'buy' if i % 2 == 0 else 'sell'
                trades.append({
                    'symbol': pair,
                    'side': side,
                    'size': position_size,
                    'strategy': 'arbitrage_triangular',
                    'sequence': i + 1,
                    'metadata': {'total_pairs': len(pairs)}
                })
            
            logger.info(f"Triangular arbitrage executed for {pairs}")
            return trades
            
        except Exception as e:
            logger.error(f"Triangular arbitrage execution failed: {e}")
            return []
    
    def should_close_arbitrage_position(self, position: Dict[str, Any]) -> bool:
        """Determine if arbitrage position should be closed"""
        try:
            # Check maximum hold time
            if 'opened_at' in position:
                hold_time = datetime.utcnow() - position['opened_at']
                if hold_time.total_seconds() / 3600 > self.arb_config.max_position_hold_time:
                    return True
            
            # Check if spread has converged (for statistical arbitrage)
            if position.get('strategy') == 'arbitrage_statistical':
                # In real implementation, recalculate z-score
                # Close if z-score has returned to normal range
                return abs(position.get('z_score', 0)) < 1.0
            
            # For cross-exchange and triangular, close immediately after execution
            if position.get('strategy') in ['arbitrage_cross_exchange', 'arbitrage_triangular']:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Position close check failed: {e}")
            return False
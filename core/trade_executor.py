"""
Trade Executor - Handles order execution and trade management
Manages order placement, execution algorithms, and trade lifecycle.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time

from config.config import config
from data.database import DatabaseManager, Trade
from utils.logger import setup_logger, TradingLogger

logger = setup_logger(__name__)
trade_logger = TradingLogger("trade_executor")

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"

class ExecutionAlgorithm(Enum):
    IMMEDIATE = "immediate"
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price
    ICEBERG = "iceberg"
    SMART = "smart"  # Intelligent execution

@dataclass
class TradeSignal:
    """Trade signal from strategies"""
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    confidence: float
    strategy_name: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    urgency: str = "normal"  # 'low', 'normal', 'high', 'urgent'
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SMART

@dataclass
class ExecutionResult:
    """Trade execution result"""
    success: bool
    order_id: Optional[str] = None
    executed_amount: float = 0.0
    executed_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    partial_fill: bool = False

@dataclass
class ActiveOrder:
    """Active order tracking"""
    order_id: str
    symbol: str
    side: str
    amount: float
    price: Optional[float]
    order_type: OrderType
    timestamp: datetime
    strategy: str
    status: str = "pending"
    filled_amount: float = 0.0
    average_price: float = 0.0

class TradeExecutor:
    """Main trade execution engine"""
    
    def __init__(self, exchange_manager, portfolio_manager, risk_manager):
        self.exchange_manager = exchange_manager
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.db_manager = DatabaseManager(config.database.url)
        
        # Order tracking
        self.active_orders: Dict[str, ActiveOrder] = {}
        self.pending_trades: List[TradeSignal] = []
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        
        # Execution settings
        self.max_slippage_pct = 0.5  # 0.5% maximum slippage
        self.max_execution_time = 30  # 30 seconds max execution time
        self.retry_attempts = 3
        self.min_order_size = 10.0  # Minimum $10 order size
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_slippage': 0.0,
            'avg_execution_time': 0.0,
            'total_commission': 0.0
        }
        
        # Execution algorithms
        self.algorithms = {
            ExecutionAlgorithm.IMMEDIATE: self.execute_immediate,
            ExecutionAlgorithm.TWAP: self.execute_twap,
            ExecutionAlgorithm.VWAP: self.execute_vwap,
            ExecutionAlgorithm.ICEBERG: self.execute_iceberg,
            ExecutionAlgorithm.SMART: self.execute_smart
        }
        
        # Market impact estimation
        self.impact_thresholds = {
            'low': 1000,     # $1k - low impact
            'medium': 5000,  # $5k - medium impact
            'high': 20000,   # $20k - high impact
        }
        
        # Start execution worker
        self.execution_task = None
        self.is_running = False
    
    async def initialize(self):
        """Initialize trade executor"""
        try:
            logger.info("⚡ Initializing Trade Executor...")
            
            # Initialize database
            await self.db_manager.initialize()
            
            # Start execution worker
            self.is_running = True
            self.execution_task = asyncio.create_task(self.execution_worker())
            
            logger.info("✅ Trade Executor initialized")
            
        except Exception as e:
            logger.error(f"❌ Trade Executor initialization failed: {e}")
            raise
    
    async def execute_signal(self, signal: TradeSignal) -> ExecutionResult:
        """Execute a trading signal"""
        start_time = time.time()
        
        try:
            trade_logger.log_trade_entry(
                signal.symbol, signal.side, signal.amount, 
                signal.entry_price or 0, signal.strategy_name
            )
            
            # Pre-execution validation
            validation_result = await self.validate_signal(signal)
            if not validation_result['valid']:
                return ExecutionResult(
                    success=False,
                    error_message=validation_result['reason']
                )
            
            # Get current market conditions
            market_conditions = await self.analyze_market_conditions(signal.symbol)
            
            # Select execution algorithm
            algorithm = self.select_execution_algorithm(signal, market_conditions)
            
            # Execute trade using selected algorithm
            result = await self.algorithms[algorithm](signal, market_conditions)
            
            # Post-execution processing
            if result.success:
                await self.post_execution_processing(signal, result)
            
            # Update statistics
            execution_time = time.time() - start_time
            self.update_execution_stats(result, execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Signal execution failed for {signal.symbol}: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def validate_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """Validate trading signal before execution"""
        try:
            # Check symbol validity
            if not await self.exchange_manager.validate_symbol(signal.symbol):
                return {'valid': False, 'reason': f'Invalid symbol: {signal.symbol}'}
            
            # Check minimum order size
            current_price = await self.exchange_manager.get_current_price(signal.symbol)

            # Paper-mode fallback: use the signal's entry_price if price feed is momentarily unavailable
            try:
                trading_mode = getattr(getattr(config, 'trading', None), 'mode', 'paper')
            except Exception:
                trading_mode = 'paper'

            if (not current_price) and str(trading_mode).lower() == 'paper':
                if getattr(signal, 'entry_price', None):
                    current_price = float(signal.entry_price)
                    logger.debug(f"ℹ️ Using signal.entry_price as paper fallback for {signal.symbol}: {current_price}")

            if not current_price:
                return {'valid': False, 'reason': f'Cannot get price for {signal.symbol}'}

            
            order_value = signal.amount * current_price
            if order_value < self.min_order_size:
                return {'valid': False, 'reason': f'Order value ${order_value:.2f} below minimum ${self.min_order_size}'}
            
            # Check available balance
            available_balance = self.portfolio_manager.get_available_balance()
            if order_value > available_balance:
                return {'valid': False, 'reason': f'Insufficient balance: ${available_balance:.2f} < ${order_value:.2f}'}
            
            # Risk management validation
            risk_check = await self.risk_manager.validate_signal(signal)
            if not risk_check:
                return {'valid': False, 'reason': 'Risk management rejection'}
            
            return {'valid': True, 'reason': 'Signal validated'}
            
        except Exception as e:
            logger.error(f"❌ Signal validation error: {e}")
            return {'valid': False, 'reason': f'Validation error: {e}'}
    
    async def analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze current market conditions for optimal execution"""
        try:
            conditions = {
                'volatility': 'normal',
                'spread': 'normal',
                'volume': 'normal',
                'market_impact': 'low',
                'optimal_algorithm': ExecutionAlgorithm.IMMEDIATE
            }
            
            # Get order book
            order_book = await self.exchange_manager.get_order_book(symbol, 20)
            if order_book:
                # Calculate spread
                best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
                best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
                
                if best_bid > 0 and best_ask > 0:
                    spread_pct = ((best_ask - best_bid) / best_ask) * 100
                    
                    if spread_pct > 0.2:
                        conditions['spread'] = 'wide'
                    elif spread_pct < 0.05:
                        conditions['spread'] = 'tight'
                
                # Analyze order book depth
                bid_depth = sum([bid[1] for bid in order_book['bids'][:5]])
                ask_depth = sum([ask[1] for ask in order_book['asks'][:5]])
                
                if bid_depth + ask_depth < 10:  # Low depth
                    conditions['volume'] = 'low'
                elif bid_depth + ask_depth > 100:  # High depth
                    conditions['volume'] = 'high'
            
            # Get recent trades
            recent_trades = await self.exchange_manager.get_recent_trades(symbol, 20)
            if recent_trades:
                # Calculate price volatility from recent trades
                prices = [float(trade['price']) for trade in recent_trades[-10:]]
                if len(prices) > 1:
                    price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                    avg_volatility = sum(price_changes) / len(price_changes) * 100
                    
                    if avg_volatility > 0.5:
                        conditions['volatility'] = 'high'
                    elif avg_volatility < 0.1:
                        conditions['volatility'] = 'low'
            
            return conditions
            
        except Exception as e:
            logger.error(f"❌ Market conditions analysis failed: {e}")
            return {
                'volatility': 'unknown',
                'spread': 'unknown',
                'volume': 'unknown',
                'market_impact': 'unknown',
                'optimal_algorithm': ExecutionAlgorithm.IMMEDIATE
            }
    
    def select_execution_algorithm(self, signal: TradeSignal, market_conditions: Dict[str, Any]) -> ExecutionAlgorithm:
        """Select optimal execution algorithm"""
        try:
            # If algorithm specified in signal, use it
            if signal.execution_algorithm != ExecutionAlgorithm.SMART:
                return signal.execution_algorithm
            
            # Calculate order value for impact estimation
            current_price = market_conditions.get('current_price', signal.entry_price or 0)
            order_value = signal.amount * current_price
            
            # High urgency always uses immediate execution
            if signal.urgency == 'urgent':
                return ExecutionAlgorithm.IMMEDIATE
            
            # Large orders in low volume should use TWAP or Iceberg
            if order_value > self.impact_thresholds['medium'] and market_conditions['volume'] == 'low':
                return ExecutionAlgorithm.TWAP
            
            # Very large orders should use Iceberg
            if order_value > self.impact_thresholds['high']:
                return ExecutionAlgorithm.ICEBERG
            
            # High volatility markets benefit from VWAP
            if market_conditions['volatility'] == 'high' and order_value > self.impact_thresholds['low']:
                return ExecutionAlgorithm.VWAP
            
            # Default to immediate for small orders
            return ExecutionAlgorithm.IMMEDIATE
            
        except Exception as e:
            logger.error(f"❌ Algorithm selection error: {e}")
            return ExecutionAlgorithm.IMMEDIATE
    
    async def execute_immediate(self, signal: TradeSignal, market_conditions: Dict[str, Any]) -> ExecutionResult:
        """Immediate market execution"""
        try:
            logger.debug(f"⚡ Executing immediate order: {signal.symbol} {signal.side}")
            
            # Place market order
            order = await self.exchange_manager.create_order(
                symbol=signal.symbol,
                side=signal.side,
                amount=signal.amount,
                order_type='market'
            )
            
            if order and order.get('status') == 'closed':
                # Calculate slippage
                executed_price = float(order.get('price', 0))
                expected_price = signal.entry_price or executed_price
                slippage = abs(executed_price - expected_price) / expected_price * 100
                
                return ExecutionResult(
                    success=True,
                    order_id=order.get('id'),
                    executed_amount=float(order.get('amount', 0)),
                    executed_price=executed_price,
                    commission=float(order.get('fee', {}).get('cost', 0)),
                    slippage=slippage
                )
            else:
                return ExecutionResult(
                    success=False,
                    error_message="Order not filled immediately"
                )
                
        except Exception as e:
            logger.error(f"❌ Immediate execution failed: {e}")
            return ExecutionResult(success=False, error_message=str(e))
    
    async def execute_twap(self, signal: TradeSignal, market_conditions: Dict[str, Any]) -> ExecutionResult:
        """Time Weighted Average Price execution"""
        try:
            logger.debug(f"⏰ Executing TWAP order: {signal.symbol} {signal.side}")
            
            # Split order into time-based chunks
            execution_duration = 5 * 60  # 5 minutes
            num_chunks = 10
            chunk_size = signal.amount / num_chunks
            interval = execution_duration / num_chunks
            
            total_executed = 0.0
            total_cost = 0.0
            total_commission = 0.0
            
            for i in range(num_chunks):
                try:
                    # Place limit order near current price
                    current_price = await self.exchange_manager.get_current_price(signal.symbol)
                    if not current_price:
                        break
                    
                    # Adjust price slightly to improve fill probability
                    if signal.side == 'buy':
                        limit_price = current_price * 1.001  # 0.1% above market
                    else:
                        limit_price = current_price * 0.999  # 0.1% below market
                    
                    order = await self.exchange_manager.create_order(
                        symbol=signal.symbol,
                        side=signal.side,
                        amount=chunk_size,
                        price=limit_price,
                        order_type='limit'
                    )
                    
                    if order:
                        # Wait for fill or timeout
                        filled = await self.wait_for_fill(order['id'], signal.symbol, 30)
                        
                        if filled:
                            total_executed += filled['amount']
                            total_cost += filled['cost']
                            total_commission += filled.get('commission', 0)
                    
                    # Wait for next interval
                    if i < num_chunks - 1:
                        await asyncio.sleep(interval)
                        
                except Exception as e:
                    logger.warning(f"⚠️ TWAP chunk {i+1} failed: {e}")
                    continue
            
            if total_executed > 0:
                avg_price = total_cost / total_executed
                slippage = abs(avg_price - (signal.entry_price or avg_price)) / (signal.entry_price or avg_price) * 100
                
                return ExecutionResult(
                    success=True,
                    executed_amount=total_executed,
                    executed_price=avg_price,
                    commission=total_commission,
                    slippage=slippage,
                    partial_fill=total_executed < signal.amount * 0.9
                )
            else:
                return ExecutionResult(
                    success=False,
                    error_message="TWAP execution failed - no fills"
                )
                
        except Exception as e:
            logger.error(f"❌ TWAP execution failed: {e}")
            return ExecutionResult(success=False, error_message=str(e))
    
    async def execute_vwap(self, signal: TradeSignal, market_conditions: Dict[str, Any]) -> ExecutionResult:
        """Volume Weighted Average Price execution"""
        try:
            logger.debug(f"📊 Executing VWAP order: {signal.symbol} {signal.side}")
            
            # Get historical volume pattern (simplified)
            volume_profile = await self.get_volume_profile(signal.symbol)
            
            # Split order based on volume distribution
            num_chunks = len(volume_profile)
            total_executed = 0.0
            total_cost = 0.0
            total_commission = 0.0
            
            for i, volume_weight in enumerate(volume_profile):
                try:
                    chunk_size = signal.amount * volume_weight
                    
                    if chunk_size < 0.001:  # Skip very small chunks
                        continue
                    
                    # Execute chunk
                    order = await self.exchange_manager.create_order(
                        symbol=signal.symbol,
                        side=signal.side,
                        amount=chunk_size,
                        order_type='market'
                    )
                    
                    if order and order.get('status') == 'closed':
                        total_executed += float(order.get('amount', 0))
                        total_cost += float(order.get('cost', 0))
                        total_commission += float(order.get('fee', {}).get('cost', 0))
                    
                    # Small delay between chunks
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"⚠️ VWAP chunk {i+1} failed: {e}")
                    continue
            
            if total_executed > 0:
                avg_price = total_cost / total_executed
                slippage = abs(avg_price - (signal.entry_price or avg_price)) / (signal.entry_price or avg_price) * 100
                
                return ExecutionResult(
                    success=True,
                    executed_amount=total_executed,
                    executed_price=avg_price,
                    commission=total_commission,
                    slippage=slippage
                )
            else:
                return ExecutionResult(
                    success=False,
                    error_message="VWAP execution failed"
                )
                
        except Exception as e:
            logger.error(f"❌ VWAP execution failed: {e}")
            return ExecutionResult(success=False, error_message=str(e))
    
    async def execute_iceberg(self, signal: TradeSignal, market_conditions: Dict[str, Any]) -> ExecutionResult:
        """Iceberg order execution (hide large order size)"""
        try:
            logger.debug(f"🧊 Executing Iceberg order: {signal.symbol} {signal.side}")
            
            # Split large order into smaller visible chunks
            visible_size = min(signal.amount * 0.1, signal.amount / 10)  # 10% or 1/10th
            remaining_amount = signal.amount
            
            total_executed = 0.0
            total_cost = 0.0
            total_commission = 0.0
            
            while remaining_amount > 0.001:  # Continue until fully executed
                try:
                    chunk_size = min(visible_size, remaining_amount)
                    
                    # Get current best price
                    order_book = await self.exchange_manager.get_order_book(signal.symbol, 5)
                    if not order_book:
                        break
                    
                    if signal.side == 'buy':
                        limit_price = order_book['asks'][0][0] * 0.9995  # Slightly below best ask
                    else:
                        limit_price = order_book['bids'][0][0] * 1.0005  # Slightly above best bid
                    
                    # Place chunk order
                    order = await self.exchange_manager.create_order(
                        symbol=signal.symbol,
                        side=signal.side,
                        amount=chunk_size,
                        price=limit_price,
                        order_type='limit'
                    )
                    
                    if order:
                        # Wait for fill
                        filled = await self.wait_for_fill(order['id'], signal.symbol, 60)
                        
                        if filled:
                            total_executed += filled['amount']
                            total_cost += filled['cost']
                            total_commission += filled.get('commission', 0)
                            remaining_amount -= filled['amount']
                        else:
                            # Cancel unfilled order and try again
                            await self.exchange_manager.cancel_order(order['id'], signal.symbol)
                            await asyncio.sleep(5)  # Wait before retry
                    
                except Exception as e:
                    logger.warning(f"⚠️ Iceberg chunk failed: {e}")
                    await asyncio.sleep(10)  # Wait before retry
                    continue
            
            if total_executed > 0:
                avg_price = total_cost / total_executed
                slippage = abs(avg_price - (signal.entry_price or avg_price)) / (signal.entry_price or avg_price) * 100
                
                return ExecutionResult(
                    success=True,
                    executed_amount=total_executed,
                    executed_price=avg_price,
                    commission=total_commission,
                    slippage=slippage,
                    partial_fill=total_executed < signal.amount * 0.95
                )
            else:
                return ExecutionResult(
                    success=False,
                    error_message="Iceberg execution failed"
                )
                
        except Exception as e:
            logger.error(f"❌ Iceberg execution failed: {e}")
            return ExecutionResult(success=False, error_message=str(e))
    
    async def execute_smart(self, signal: TradeSignal, market_conditions: Dict[str, Any]) -> ExecutionResult:
        """Smart execution - dynamically chooses best method"""
        try:
            # This is handled by select_execution_algorithm
            # If we reach here, fall back to immediate execution
            return await self.execute_immediate(signal, market_conditions)
            
        except Exception as e:
            logger.error(f"❌ Smart execution failed: {e}")
            return ExecutionResult(success=False, error_message=str(e))
    
    async def get_volume_profile(self, symbol: str) -> List[float]:
        """Get volume distribution profile (simplified)"""
        try:
            # In a real implementation, this would analyze historical volume patterns
            # For now, return a simplified profile
            profile = [0.05, 0.08, 0.12, 0.15, 0.20, 0.20, 0.15, 0.05]  # 8 time slots
            return profile
            
        except Exception as e:
            logger.error(f"❌ Volume profile calculation failed: {e}")
            return [1.0]  # Single chunk fallback
    
    async def wait_for_fill(self, order_id: str, symbol: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for order to be filled"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                order_status = await self.exchange_manager.get_order_status(order_id, symbol)
                
                if not order_status:
                    await asyncio.sleep(1)
                    continue
                
                if order_status.get('status') == 'closed':
                    return {
                        'amount': float(order_status.get('amount', 0)),
                        'cost': float(order_status.get('cost', 0)),
                        'commission': float(order_status.get('fee', {}).get('cost', 0))
                    }
                elif order_status.get('status') == 'canceled':
                    return None
                
                await asyncio.sleep(2)  # Check every 2 seconds
            
            return None  # Timeout
            
        except Exception as e:
            logger.error(f"❌ Wait for fill error: {e}")
            return None
    
    async def post_execution_processing(self, signal: TradeSignal, result: ExecutionResult):
        """Post-execution processing and record keeping"""
        try:
            # Trade (DB model uses size/price/fee, and side expects 'buy'/'sell')
            trade_data = {
                "symbol": signal.symbol,
                "side": "buy" if signal.side == "buy" else "sell",
                "size": float(result.executed_amount),
                "price": float(result.executed_price),
                "fee": float(getattr(result, "commission", 0.0)),
                "timestamp": datetime.utcnow(),
                "exchange_order_id": result.order_id or "",
                "status": "filled" if result.success else "failed",
                "strategy": signal.strategy_name or "unknown",
                "notes": json.dumps(getattr(signal, "features", None) or {}),
            }
            await self.db_manager.save_trade(trade_data)

            # Portfolio position (store strategy explicitly)
            await self.portfolio_manager.add_position(
                symbol=signal.symbol,
                side="long" if signal.side == "buy" else "short",
                amount=float(result.executed_amount),
                entry_price=float(result.executed_price),
                strategy=signal.strategy_name or "unknown",
            )

            # Human-readable trade log
            trade_logger.log_trade_entry(
                signal.symbol, signal.side, result.executed_amount,
                result.executed_price, signal.strategy_name
            )
            logger.info(
                f"✅ Trade executed: {signal.symbol} {signal.side} "
                f"{result.executed_amount:.6f} @ ${result.executed_price:.4f}"
            )

        except Exception as e:
            logger.error(f"❌ Post-execution processing failed: {e}")
   
    def update_execution_stats(self, result: ExecutionResult, execution_time: float):
        """Update execution statistics"""
        try:
            self.execution_stats['total_executions'] += 1
            
            if result.success:
                self.execution_stats['successful_executions'] += 1
                self.execution_stats['total_slippage'] += result.slippage
                self.execution_stats['total_commission'] += result.commission
            else:
                self.execution_stats['failed_executions'] += 1
            
            # Update average execution time
            current_avg = self.execution_stats['avg_execution_time']
            total_count = self.execution_stats['total_executions']
            
            self.execution_stats['avg_execution_time'] = (
                (current_avg * (total_count - 1) + execution_time) / total_count
            )
            
        except Exception as e:
            logger.error(f"❌ Stats update failed: {e}")
    
    async def close_position(self, symbol: str) -> ExecutionResult:
        """Close an existing position"""
        try:
            # Get position details
            position = await self.portfolio_manager.get_position(symbol)
            if not position:
                return ExecutionResult(
                    success=False,
                    error_message=f"No position found for {symbol}"
                )
            
            # Create close signal
            close_side = 'sell' if position['side'] == 'long' else 'buy'
            
            close_signal = TradeSignal(
                symbol=symbol,
                side=close_side,
                amount=position['amount'],
                confidence=1.0,
                strategy_name="position_close",
                urgency="high"
            )
            
            # Execute close
            result = await self.execute_immediate(close_signal, {})
            
            if result.success:
                # Remove from portfolio
                removed_position = await self.portfolio_manager.remove_position(
                    symbol, result.executed_price
                )

                # PATCH ⬇ Persist exit trade if DB exposes a saver (no-breaking)
                try:
                    trade_data = {
                        "symbol": symbol,
                        "side": close_side,
                        "size": float(result.executed_amount),
                        "price": float(result.executed_price),
                        "fee": float(getattr(result, 'commission', 0.0)),
                        "timestamp": datetime.utcnow(),
                        "exchange_order_id": result.order_id or "",
                        "status": "exit",
                        "strategy": (getattr(removed_position, "strategy", None) or "position_close"),
                        "notes": json.dumps({"reason": "manual/SL/TP", "pnl": getattr(removed_position, "realized_pnl", None)})
                    }
                    if hasattr(self.db_manager, 'save_trade'):
                        await self.db_manager.save_trade(trade_data)
                    elif hasattr(self.db_manager, 'store_trade'):
                        await self.db_manager.store_trade(trade_data)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to persist exit trade for {symbol}: {e}")
                # /PATCH
                
                if removed_position:
                    trade_logger.log_trade_exit(
                        symbol, close_side, result.executed_amount,
                        result.executed_price, removed_position.unrealized_pnl,
                        removed_position.unrealized_pnl_pct
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Position close failed for {symbol}: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    async def execution_worker(self):
        """Background worker for processing execution queue"""
        try:
            while self.is_running:
                try:
                    # Get next signal from queue (with timeout)
                    signal = await asyncio.wait_for(
                        self.execution_queue.get(), timeout=1.0
                    )
                    
                    # Execute signal
                    result = await self.execute_signal(signal)
                    
                    # Mark task as done
                    self.execution_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"❌ Execution worker error: {e}")
                    await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Execution worker failed: {e}")
    
    async def queue_signal(self, signal: TradeSignal):
        """Add signal to execution queue"""
        try:
            await self.execution_queue.put(signal)
            logger.debug(f"📝 Signal queued: {signal.symbol} {signal.side}")
            
        except Exception as e:
            logger.error(f"❌ Failed to queue signal: {e}")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        stats = self.execution_stats.copy()
        
        # Calculate derived metrics
        if stats['total_executions'] > 0:
            stats['success_rate'] = (stats['successful_executions'] / stats['total_executions']) * 100
            
        if stats['successful_executions'] > 0:
            stats['avg_slippage'] = stats['total_slippage'] / stats['successful_executions']
            stats['avg_commission'] = stats['total_commission'] / stats['successful_executions']
        else:
            stats['avg_slippage'] = 0.0
            stats['avg_commission'] = 0.0
        
        stats['active_orders'] = len(self.active_orders)
        stats['queue_size'] = self.execution_queue.qsize()
        
        return stats
    
    async def cancel_all_orders(self):
        """Cancel all active orders"""
        try:
            cancelled_count = 0
            
            for order_id, order in self.active_orders.items():
                try:
                    success = await self.exchange_manager.cancel_order(order_id, order.symbol)
                    if success:
                        cancelled_count += 1
                        logger.info(f"🚫 Cancelled order: {order_id}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cancel order {order_id}: {e}")
            
            self.active_orders.clear()
            logger.info(f"🚫 Cancelled {cancelled_count} orders")
            
        except Exception as e:
            logger.error(f"❌ Cancel all orders failed: {e}")
    
    async def shutdown(self):
        """Shutdown trade executor"""
        try:
            logger.info("🛑 Shutting down Trade Executor...")
            
            # Stop execution worker
            self.is_running = False
            if self.execution_task:
                self.execution_task.cancel()
                try:
                    await self.execution_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel all pending orders
            await self.cancel_all_orders()
            
            # Process remaining queue items
            remaining_items = []
            while not self.execution_queue.empty():
                try:
                    item = self.execution_queue.get_nowait()
                    remaining_items.append(item)
                    self.execution_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            if remaining_items:
                logger.info(f"⚠️ {len(remaining_items)} signals were not processed")
            
            # Close database connection
            await self.db_manager.close()
            
            logger.info("✅ Trade Executor shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Trade Executor shutdown error: {e}")
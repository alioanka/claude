"""
Portfolio Manager - Manages portfolio state, positions, and balance tracking
Handles position tracking, PnL calculation, and portfolio optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
import pandas as pd

from config.config import config, TRADING_PAIRS
from data.database import DatabaseManager, Position, Trade
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class PortfolioPosition:
    """Portfolio position structure"""
    symbol: str
    side: str  # 'long' or 'short'
    amount: float
    entry_price: float
    current_price: float
    timestamp: datetime
    strategy: str = "unknown"
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    entry_value: float = field(init=False)
    current_value: float = field(init=False)
    
    def __post_init__(self):
        self.entry_value = self.amount * self.entry_price
        self.update_current_value()
    
    def update_current_value(self):
        """Update current value and PnL"""
        self.current_value = self.amount * self.current_price
        
        if self.side == 'long':
            self.unrealized_pnl = self.current_value - self.entry_value
        else:  # short
            self.unrealized_pnl = self.entry_value - self.current_value
        
        if self.entry_value > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl / self.entry_value) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'amount': self.amount,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'entry_value': self.entry_value,
            'current_value': self.current_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'timestamp': self.timestamp.isoformat(),
            'strategy': self.strategy
        }

@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot for tracking"""
    timestamp: datetime
    total_balance: float
    available_balance: float
    used_balance: float
    unrealized_pnl: float
    daily_pnl: float
    total_trades: int
    active_positions: int
    portfolio_value: float

class PortfolioManager:
    """Main portfolio management class"""
    
    def __init__(self, exchange_manager, initial_capital: float):
        self.exchange_manager = exchange_manager
        self.db_manager = DatabaseManager(config.database.url)
        
        # Portfolio state
        self.initial_capital = initial_capital
        self.total_balance = initial_capital
        self.available_balance = initial_capital
        self.used_balance = 0.0
        self.unrealized_pnl = 0.0
        
        # Positions tracking
        self.positions: Dict[str, PortfolioPosition] = {}
        self.position_history: List[PortfolioPosition] = []
        
        # Performance tracking
        self.daily_start_balance = initial_capital
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_capital
        
        # Portfolio metrics
        self.performance_history: List[PortfolioSnapshot] = []
        self.last_update = datetime.utcnow()
        self.last_daily_reset = datetime.utcnow().date()
        
        # Risk tracking
        self.position_limits = {
            'max_position_size': config.trading.initial_capital * 0.10,  # 10% max per position
            'max_total_exposure': config.trading.initial_capital * 0.80,  # 80% max exposure
            'max_correlation': 0.7,  # 70% max correlation between positions
        }
    
    async def initialize(self):
        """Initialize portfolio manager"""
        try:
            logger.info("💼 Initializing Portfolio Manager...")
            
            # Initialize database connection
            await self.db_manager.initialize()
            
            # Load existing positions if any
            await self.load_existing_positions()
            
            # Load performance history
            await self.load_performance_history()
            
            # Set daily start balance
            await self.update_daily_metrics()
            
            logger.info(f"✅ Portfolio Manager initialized - Balance: ${self.total_balance:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Portfolio Manager initialization failed: {e}")
            raise
    
    async def load_existing_positions(self):
        """Load existing positions from database"""
        try:
            positions_data = await self.db_manager.get_active_positions()
            
            for pos_data in positions_data:
                # Get current market price
                current_price = await self.exchange_manager.get_current_price(pos_data['symbol'])
                if current_price:
                    position = PortfolioPosition(
                        symbol=pos_data['symbol'],
                        side=pos_data['side'],
                        amount=pos_data['amount'],
                        entry_price=pos_data['entry_price'],
                        current_price=current_price,
                        timestamp=datetime.fromtimestamp(pos_data['timestamp'] / 1000),
                        strategy=pos_data.get('strategy', 'unknown')
                    )
                    
                    self.positions[pos_data['symbol']] = position
                    
                    logger.info(f"📊 Loaded position: {position.symbol} {position.side}")
            
            # Update portfolio metrics
            await self.calculate_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing positions: {e}")

    async def get_total_value(self) -> float:
        """
        Return current portfolio total value in quote (USDT) based on our own book.
        We deliberately avoid trusting external paper-engine totals.
        """
        try:
            # Ensure unrealized is up to date
            await self.update_position_prices()
        except Exception:
            pass
        return float(self.total_balance + self.unrealized_pnl)


    async def get_portfolio_status(self) -> dict:
        """
        Dashboard payload derived from our own records.
        """
        try:
            await self.update_position_prices()
        except Exception:
            pass

        total_value = float(self.total_balance + self.unrealized_pnl)
        open_positions = int(len(self.positions))
        cash = float(self.available_balance)

        return {
            "total_value": total_value,
            "open_positions": open_positions,
            "cash": cash,
            "timestamp": datetime.utcnow().isoformat(),
        }



    async def load_performance_history(self, days_back: int = 30):
        """Load portfolio performance history"""
        try:
            history_data = await self.db_manager.get_portfolio_history(days_back)
            
            for data in history_data:
                snapshot = PortfolioSnapshot(
                    timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
                    total_balance=data['total_balance'],
                    available_balance=data['available_balance'],
                    used_balance=data['used_balance'],
                    unrealized_pnl=data['unrealized_pnl'],
                    daily_pnl=data['daily_pnl'],
                    total_trades=data['total_trades'],
                    active_positions=data['active_positions'],
                    portfolio_value=data['total_balance'] + data['unrealized_pnl']
                )
                self.performance_history.append(snapshot)
            
            logger.info(f"📈 Loaded {len(self.performance_history)} portfolio snapshots")
            
        except Exception as e:
            logger.error(f"❌ Failed to load performance history: {e}")
    
    async def add_position(self, symbol: str, side: str, amount: float,
                          entry_price: float, strategy: str = "unknown") -> bool:
        """Add new position to portfolio"""
        try:
            if symbol in self.positions:
                logger.warning(f"⚠️ Position already exists for {symbol}")
                return False

            current_price = await self.exchange_manager.get_current_price(symbol)
            if not current_price:
                logger.error(f"❌ Cannot get current price for {symbol}")
                return False

            # in-memory
            position = PortfolioPosition(
                symbol=symbol,
                side=side,
                amount=amount,
                entry_price=entry_price,
                current_price=current_price,
                timestamp=datetime.utcnow(),
                strategy=strategy
            )
            self.positions[symbol] = position

            # balances
            notional = amount * entry_price
            self.used_balance += notional
            self.available_balance -= notional

            # persist to DB (dict path maps amount -> size and stores strategy)
            pos_data = {
                "symbol": symbol,
                "side": side,
                "amount": float(amount),
                "entry_price": float(entry_price),
                "current_price": float(current_price),
                "strategy": strategy,
            }
            await self.db_manager.store_position(pos_data)

            await self.calculate_portfolio_metrics()
            logger.info(f"➕ Position added: {symbol} {side} {amount:.6f} @ ${entry_price:.4f} [{strategy}]")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to add position for {symbol}: {e}")
            return False

    
    async def remove_position(self, symbol: str, exit_price: float) -> Optional[PortfolioPosition]:
        """Remove position from portfolio"""
        try:
            if symbol not in self.positions:
                logger.warning(f"⚠️ No position found for {symbol}")
                return None
            
            position = self.positions[symbol]
            
            # Calculate final PnL
            position.current_price = exit_price
            position.update_current_value()
            
            # Update balances
            position_value = position.amount * position.entry_price
            self.used_balance -= position_value
            self.available_balance += position_value + position.unrealized_pnl
            
            # Update total balance with realized PnL
            self.total_balance += position.unrealized_pnl
            self.total_pnl += position.unrealized_pnl
            
            # Update trade statistics
            self.total_trades += 1
            if position.unrealized_pnl > 0:
                self.winning_trades += 1
            
            # Add to history
            self.position_history.append(position)
            
            # Remove from active positions
            removed_position = self.positions.pop(symbol)

            # PATCH ⬇ Best-effort persist closure without breaking existing DB layer
            try:
                if hasattr(self.db_manager, 'close_position'):
                    await self.db_manager.close_position(
                        symbol=symbol,
                        exit_price=float(exit_price),
                        realized_pnl=float(position.unrealized_pnl),
                        closed_at=datetime.utcnow()
                    )
                elif hasattr(self.db_manager, 'mark_position_closed'):
                    await self.db_manager.mark_position_closed(
                        symbol=symbol,
                        exit_price=float(exit_price),
                        realized_pnl=float(position.unrealized_pnl)
                    )
                elif hasattr(self.db_manager, 'update_position'):
                    await self.db_manager.update_position(
                        symbol,
                        {
                            'is_open': False,
                            'exit_price': float(exit_price),
                            'pnl': float(position.unrealized_pnl),
                            'closed_at': datetime.utcnow()
                        }
                    )
            except Exception as _db_close_err:
                logger.warning(f"⚠️ Could not mark {symbol} closed in DB: {_db_close_err}")
            # /PATCH
            
            # Update metrics
            await self.calculate_portfolio_metrics()
            
            logger.info(f"➖ Position removed: {symbol} PnL: ${position.unrealized_pnl:.2f}")
            return removed_position
            
        except Exception as e:
            logger.error(f"❌ Failed to remove position for {symbol}: {e}")
            return None
    
    async def update_position_prices(self):
        """Update current prices for all positions"""
        try:
            # iterate over a stable view
            items = list(self.positions.items())
            for symbol, position in items:
                current_price = await self.exchange_manager.get_current_price(symbol)
                if current_price:
                    position.current_price = float(current_price)
                    position.update_current_value()

            # Update portfolio metrics once at the end
            await self.calculate_portfolio_metrics()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update position prices: {e}")
            return False


    async def calculate_portfolio_metrics(self):
        """Calculate current portfolio metrics"""
        try:
            # Calculate total unrealized PnL
            self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Calculate portfolio value
            portfolio_value = self.total_balance + self.unrealized_pnl
            
            # Update peak balance for drawdown calculation
            if portfolio_value > self.peak_balance:
                self.peak_balance = portfolio_value
            
            # Calculate current drawdown
            current_drawdown = (self.peak_balance - portfolio_value) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Calculate daily PnL
            self.daily_pnl = portfolio_value - self.daily_start_balance
            
            self.last_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate portfolio metrics: {e}")
    
    async def update_daily_metrics(self):
        """Update daily metrics and reset if new day"""
        try:
            current_date = datetime.utcnow().date()
            
            if current_date > self.last_daily_reset:
                # New day - reset daily metrics
                portfolio_value = self.total_balance + self.unrealized_pnl
                self.daily_start_balance = portfolio_value
                self.daily_pnl = 0.0
                self.last_daily_reset = current_date
                
                logger.info(f"📅 Daily metrics reset - Starting balance: ${self.daily_start_balance:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update daily metrics: {e}")
    
    async def update(self):
        """Main update method - call this regularly"""
        try:
            # Update position prices
            await self.update_position_prices()
            
            # Update daily metrics
            await self.update_daily_metrics()
            
            # Store portfolio snapshot
            await self.store_portfolio_snapshot()
            
        except Exception as e:
            logger.error(f"❌ Portfolio update failed: {e}")
    
    async def store_portfolio_snapshot(self):
        """Store current portfolio snapshot"""
        try:
            portfolio_value = self.total_balance + self.unrealized_pnl
            
            snapshot_data = {
                'total_balance': self.total_balance,
                'available_balance': self.available_balance,
                'used_balance': self.used_balance,
                'unrealized_pnl': self.unrealized_pnl,
                'daily_pnl': self.daily_pnl,
                'total_trades': self.total_trades,
                'active_positions': len(self.positions)
            }
            
            await self.db_manager.store_portfolio_snapshot(snapshot_data)
            
            # Add to history
            snapshot = PortfolioSnapshot(
                timestamp=datetime.utcnow(),
                total_balance=self.total_balance,
                available_balance=self.available_balance,
                used_balance=self.used_balance,
                unrealized_pnl=self.unrealized_pnl,
                daily_pnl=self.daily_pnl,
                total_trades=self.total_trades,
                active_positions=len(self.positions),
                portfolio_value=portfolio_value
            )
            
            self.performance_history.append(snapshot)
            
            # Keep only last 1000 snapshots in memory
            if len(self.performance_history) > 1000:
                self.performance_history.pop(0)
            
        except Exception as e:
            logger.error(f"❌ Failed to store portfolio snapshot: {e}")
    
    # Query methods
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions"""
        return [pos.to_dict() for pos in self.positions.values()]
    
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get specific position"""
        if symbol in self.positions:
            return self.positions[symbol].to_dict()
        return None
    
    async def has_position(self, symbol: str) -> bool:
        """Check if position exists"""
        return symbol in self.positions
    
    def get_available_balance(self) -> float:
        """Get available balance for new trades"""
        return self.available_balance
    
    def get_total_balance(self) -> float:
        """Get total balance including unrealized PnL"""
        return self.total_balance + self.unrealized_pnl
    
    def get_position_count(self) -> int:
        """Get number of active positions"""
        return len(self.positions)
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        return self.total_balance + self.unrealized_pnl
    
    def get_daily_pnl(self) -> float:
        """Get daily PnL"""
        return self.daily_pnl
    
    def get_daily_pnl_pct(self) -> float:
        """Get daily PnL percentage"""
        if self.daily_start_balance > 0:
            return (self.daily_pnl / self.daily_start_balance) * 100
        return 0.0
    
    def get_total_return(self) -> float:
        """Get total return since inception"""
        current_value = self.get_portfolio_value()
        return ((current_value - self.initial_capital) / self.initial_capital) * 100
    
    def get_win_rate(self) -> float:
        """Get win rate percentage"""
        if self.total_trades > 0:
            return (self.winning_trades / self.total_trades) * 100
        return 0.0
    
    def get_sharpe_ratio(self, days: int = 30) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.performance_history) < 2:
                return 0.0
            
            # Get recent returns
            recent_snapshots = self.performance_history[-days:] if len(self.performance_history) >= days else self.performance_history
            
            if len(recent_snapshots) < 2:
                return 0.0
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(recent_snapshots)):
                prev_value = recent_snapshots[i-1].portfolio_value
                curr_value = recent_snapshots[i].portfolio_value
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if not returns:
                return 0.0
            
            # Calculate Sharpe ratio
            mean_return = sum(returns) / len(returns)
            
            if len(returns) > 1:
                variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
                std_dev = variance ** 0.5
                
                if std_dev > 0:
                    # Annualized Sharpe ratio (assuming 365 trading days)
                    sharpe = (mean_return * 365) / (std_dev * (365 ** 0.5))
                    return sharpe
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def get_max_drawdown_pct(self) -> float:
        """Get maximum drawdown percentage"""
        return self.max_drawdown * 100
    
    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics"""
        portfolio_value = self.get_portfolio_value()
        
        return {
            'total_balance': self.total_balance,
            'available_balance': self.available_balance,
            'used_balance': self.used_balance,
            'portfolio_value': portfolio_value,
            'unrealized_pnl': self.unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.get_daily_pnl_pct(),
            'total_return_pct': self.get_total_return(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate_pct': self.get_win_rate(),
            'sharpe_ratio': self.get_sharpe_ratio(),
            'max_drawdown_pct': self.get_max_drawdown_pct(),
            'active_positions': len(self.positions),
            'last_update': self.last_update.isoformat()
        }
    
    async def get_position_exposure(self) -> Dict[str, float]:
        """Get exposure by position"""
        exposure = {}
        total_portfolio = self.get_portfolio_value()
        
        for symbol, position in self.positions.items():
            position_value = abs(position.current_value)
            exposure_pct = (position_value / total_portfolio) * 100 if total_portfolio > 0 else 0
            exposure[symbol] = exposure_pct
        
        return exposure
    
    async def get_correlation_risk(self) -> float:
        """Calculate correlation risk between positions"""
        try:
            if len(self.positions) < 2:
                return 0.0
            
            # This would require historical price correlation analysis
            # For now, return a simplified risk metric based on position count
            position_count = len(self.positions)
            if position_count > 5:
                return min(position_count * 10, 100)  # Higher risk with more positions
            
            return position_count * 15
            
        except Exception as e:
            logger.error(f"❌ Correlation risk calculation failed: {e}")
            return 0.0
    
    async def check_risk_limits(self) -> List[str]:
        """Check portfolio risk limits"""
        warnings = []
        
        try:
            portfolio_value = self.get_portfolio_value()
            
            # Check individual position sizes
            for symbol, position in self.positions.items():
                position_value = abs(position.current_value)
                position_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
                
                if position_value > self.position_limits['max_position_size']:
                    warnings.append(f"{symbol} position size exceeds limit: ${position_value:.2f}")
                
                if position_pct > 15:  # 15% warning threshold
                    warnings.append(f"{symbol} represents {position_pct:.1f}% of portfolio")
            
            # Check total exposure
            total_exposure = self.used_balance
            exposure_pct = (total_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            if exposure_pct > 85:  # 85% warning threshold
                warnings.append(f"High portfolio exposure: {exposure_pct:.1f}%")
            
            # Check drawdown
            drawdown_pct = self.get_max_drawdown_pct()
            if drawdown_pct > 10:  # 10% warning threshold
                warnings.append(f"High drawdown: {drawdown_pct:.1f}%")
            
            return warnings
            
        except Exception as e:
            logger.error(f"❌ Risk limits check failed: {e}")
            return ["Risk check failed"]
    
    async def optimize_portfolio(self) -> Dict[str, Any]:
        """Portfolio optimization suggestions"""
        try:
            suggestions = {
                'rebalance': False,
                'reduce_exposure': [],
                'diversify': False,
                'actions': []
            }
            
            # Check if rebalancing is needed
            exposure = await self.get_position_exposure()
            
            for symbol, exp_pct in exposure.items():
                if exp_pct > 20:  # If any position > 20%
                    suggestions['reduce_exposure'].append(symbol)
                    suggestions['actions'].append(f"Consider reducing {symbol} position (currently {exp_pct:.1f}%)")
            
            # Check diversification
            if len(self.positions) < 3 and self.get_portfolio_value() > self.initial_capital * 0.5:
                suggestions['diversify'] = True
                suggestions['actions'].append("Consider diversifying into more positions")
            
            # Check if rebalancing is beneficial
            if len(suggestions['reduce_exposure']) > 0:
                suggestions['rebalance'] = True
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Portfolio optimization failed: {e}")
            return {'error': str(e)}
    
    def export_performance_data(self) -> pd.DataFrame:
        """Export performance data as DataFrame"""
        try:
            data = []
            for snapshot in self.performance_history:
                data.append({
                    'timestamp': snapshot.timestamp,
                    'total_balance': snapshot.total_balance,
                    'portfolio_value': snapshot.portfolio_value,
                    'daily_pnl': snapshot.daily_pnl,
                    'unrealized_pnl': snapshot.unrealized_pnl,
                    'active_positions': snapshot.active_positions,
                    'total_trades': snapshot.total_trades
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"❌ Performance data export failed: {e}")
            return pd.DataFrame()
    
    async def close(self):
        """Close portfolio manager and save final state"""
        try:
            # Store final portfolio snapshot
            await self.store_portfolio_snapshot()
            
            # Close database connection
            await self.db_manager.close()
            
            logger.info("💼 Portfolio Manager closed")
            
        except Exception as e:
            logger.error(f"❌ Portfolio Manager close failed: {e}")
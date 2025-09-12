"""
Database models and manager for the crypto trading bot.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, String, Float, DateTime, Integer, Text, Boolean, 
    ForeignKey, create_engine, Index, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

logger = logging.getLogger(__name__)

Base = declarative_base()

class MarketData(Base):
    """Market data model for OHLCV candlestick data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False) 
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}', close={self.close})>"
    
    @property
    def ohlcv(self):
        """Return OHLCV data as tuple"""
        return (self.open, self.high, self.low, self.close, self.volume)
        
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Position(Base):
    """Position model for tracking open positions"""
    __tablename__ = 'positions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'long' or 'short'
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    strategy = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime)
    is_open = Column(Boolean, default=True)
    pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    
    # Relationship to trades
    trades = relationship("Trade", back_populates="position")
    
    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', side='{self.side}', size={self.size}, pnl={self.pnl})>"
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current PnL"""
        if self.side == 'long':
            pnl = (current_price - self.entry_price) * self.size
        else:  # short
            pnl = (self.entry_price - current_price) * self.size
        
        self.pnl = pnl
        self.pnl_percentage = (pnl / (self.entry_price * self.size)) * 100
        self.current_price = current_price
        return pnl
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'strategy': self.strategy,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'is_open': self.is_open,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage
        }

class Trade(Base):
    """Trade model for recording individual trades"""
    __tablename__ = 'trades'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    position_id = Column(String, ForeignKey('positions.id'), nullable=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    size = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    exchange_order_id = Column(String(100), unique=True)
    status = Column(String(20), default='pending')  # pending, filled, cancelled, failed
    strategy = Column(String(50))
    notes = Column(Text)
    
    # Relationship to position
    position = relationship("Position", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', side='{self.side}', size={self.size}, price={self.price})>"
    
    @property
    def total_value(self):
        """Calculate total trade value including fees"""
        return (self.size * self.price) + self.fee
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'price': self.price,
            'fee': self.fee,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'exchange_order_id': self.exchange_order_id,
            'status': self.status,
            'strategy': self.strategy,
            'notes': self.notes,
            'total_value': self.total_value
        }

class Signal(Base):
    """Signal model for storing trading signals"""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    signal_type = Column(String(10), nullable=False)  # 'buy', 'sell', 'hold'
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    executed = Column(Boolean, default=False)
    signal_metadata = Column(Text)  # JSON string for additional data
    
    def __repr__(self):
        return f"<Signal(symbol='{self.symbol}', type='{self.signal_type}', confidence={self.confidence})>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'strategy': self.strategy,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'price': self.price,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'executed': self.executed,
            'signal_metadata': self.signal_metadata
        }

class DatabaseManager:
    """Database manager with PostgreSQL fallback to SQLite"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        
        # Determine database type and handle missing dependencies
        self.is_postgres = 'postgresql' in database_url.lower()
        
        if self.is_postgres:
            try:
                import psycopg2
                logger.info("PostgreSQL adapter available")
            except ImportError:
                logger.warning("PostgreSQL adapter (psycopg2) not available, falling back to SQLite")
                self.database_url = "sqlite:///storage/trading_bot.db"
                self.is_postgres = False
        
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            logger.info(f"Initializing database: {'PostgreSQL' if self.is_postgres else 'SQLite'}")
            
            # Create engine based on database type
            if self.is_postgres:
                self.engine = create_engine(
                    self.database_url,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    echo=False
                )
            else:
                # SQLite setup - ensure storage directory exists
                import os
                os.makedirs('storage', exist_ok=True)
                
                self.engine = create_engine(
                    self.database_url,
                    connect_args={"check_same_thread": False},
                    echo=False
                )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            # Test connection
            await self._test_connection()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            
            # If PostgreSQL fails, try SQLite fallback
            if self.is_postgres:
                logger.info("Attempting SQLite fallback...")
                self.database_url = "sqlite:///storage/trading_bot.db"
                self.is_postgres = False
                await self.initialize()  # Retry with SQLite
            else:
                raise
    
    async def _test_connection(self):
        """Test database connection with SQLAlchemy 2.0 compatibility"""
        session = None
        try:
            session = self.get_session()
            
            # Use text() wrapper for raw SQL queries (SQLAlchemy 2.0+ requirement)
            session.execute(text("SELECT 1"))
            session.close()
            logger.info("Database connection test successful")
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            if session:
                session.close()
            raise
    
    def get_session(self):
        """Get database session"""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        return self.session_factory()

    # Add these methods to your existing database.py DatabaseManager class

    async def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions from database"""
        session = None
        try:
            session = self.get_session()
            
            positions = session.query(Position).filter(
                Position.is_open == True
            ).all()
            
            result = []
            for pos in positions:
                result.append({
                    'id': pos.id,
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'amount': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'strategy': pos.strategy,
                    'timestamp': int(pos.created_at.timestamp() * 1000) if pos.created_at else 0,
                    'pnl': pos.pnl,
                    'pnl_percentage': pos.pnl_percentage
                })
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get active positions: {e}")
            if session:
                session.close()
            return []

    async def get_portfolio_history(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get portfolio history from database"""
        session = None
        try:
            session = self.get_session()
            
            from_date = datetime.utcnow() - timedelta(days=days_back)
            
            # For now, return empty list since we don't have portfolio_snapshots table yet
            # In a full implementation, you'd query a portfolio_snapshots table
            result = []
            
            # Create some mock historical data based on positions if needed
            positions = session.query(Position).filter(
                Position.created_at >= from_date
            ).all()
            
            for pos in positions:
                result.append({
                    'timestamp': int(pos.created_at.timestamp() * 1000) if pos.created_at else 0,
                    'total_balance': 10000.0,  # Mock data
                    'available_balance': 8000.0,
                    'used_balance': 2000.0,
                    'unrealized_pnl': pos.pnl or 0.0,
                    'daily_pnl': 0.0,
                    'total_trades': 0,
                    'active_positions': 1
                })
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get portfolio history: {e}")
            if session:
                session.close()
            return []


    # --- add below inside DatabaseManager (near other getters) ---
    def get_recent_trades(self, limit: int = 50) -> list[dict]:
        """Return the most recent trades (newest first) as plain dicts."""
        session = None
        try:
            session = self.get_session()
            rows = (
                session.query(Trade)
                .order_by(Trade.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [t.to_dict() for t in rows]
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []
        finally:
            if session:
                session.close()


    async def store_position(self, position) -> bool:
        """Store position in database"""
        session = None
        try:
            session = self.get_session()
            
            # Create Position object if it's a dict
            if isinstance(position, dict):
                size_val = position.get('size')
                if size_val is None:
                    size_val = position.get('amount')  # backward compat
                if size_val is None:
                    raise ValueError("Position requires 'size' or 'amount'")

                db_position = Position(
                    symbol=position['symbol'],
                    side=position['side'],
                    size=float(size_val),
                    entry_price=float(position['entry_price']),
                    current_price=float(position.get('current_price', position['entry_price'])),
                    strategy=position.get('strategy', 'unknown'),
                    pnl=float(position.get('pnl', 0.0)),
                    pnl_percentage=float(position.get('pnl_percentage', 0.0))
                )

            else:
                db_position = position
            
            session.add(db_position)
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store position: {e}")
            if session:
                session.rollback()
                session.close()
            return False

    async def store_portfolio_snapshot(self, snapshot_data: Dict[str, Any]) -> bool:
        """Store portfolio snapshot (placeholder implementation)"""
        try:
            # For now, just log the snapshot data
            # In a full implementation, you'd store this in a portfolio_snapshots table
            logger.debug(f"Portfolio snapshot: {snapshot_data}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store portfolio snapshot: {e}")
            return False

    async def save_market_data(self, data: List[Dict[str, Any]]) -> bool:
        """Save market data to database"""
        session = None
        try:
            session = self.get_session()
            
            for candle in data:
                market_data = MarketData(
                    symbol=candle['symbol'],
                    timeframe=candle['timeframe'],
                    timestamp=datetime.fromtimestamp(candle['timestamp'] / 1000, tz=timezone.utc),
                    open=float(candle['open']),
                    high=float(candle['high']),
                    low=float(candle['low']),
                    close=float(candle['close']),
                    volume=float(candle['volume'])
                )
                session.merge(market_data)  # Use merge to handle duplicates
            
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save market data: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    # Store OHLCV data method for data collector
    async def store_ohlcv(self, symbol: str, timeframe: str, candles: List[List]) -> bool:
        """Store OHLCV data for data collector compatibility"""
        try:
            data = []
            for candle in candles:
                data.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': candle[0],  # timestamp
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                })
            
            return await self.save_market_data(data)
            
        except Exception as e:
            logger.error(f"Failed to store OHLCV data: {e}")
            return False
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get historical market data"""
        session = None
        try:
            session = self.get_session()
            
            data = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe,
                MarketData.timestamp >= start_date,
                MarketData.timestamp <= end_date
            ).order_by(MarketData.timestamp).all()
            
            result = [
                {
                    'timestamp': int(item.timestamp.timestamp() * 1000),
                    'open': item.open,
                    'high': item.high,
                    'low': item.low,
                    'close': item.close,
                    'volume': item.volume
                }
                for item in data
            ]
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            if session:
                session.close()
            return []
    
    async def close(self):
        """Close database connections"""
        try:
            if self.engine:
                self.engine.dispose()
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    async def get_latest_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[MarketData]:
        """Get latest market data for a symbol"""
        session = None
        try:
            session = self.get_session()
            
            data = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe
            ).order_by(MarketData.timestamp.desc()).limit(limit).all()
            
            session.close()
            return data
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            if session:
                session.close()
            return []
    
    async def save_position(self, position_data: Dict[str, Any]) -> bool:
        """Save position to database"""
        session = None
        try:
            session = self.get_session()
            
            position = Position(**position_data)
            session.add(position)
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        session = None
        try:
            session = self.get_session()
            
            positions = session.query(Position).filter(
                Position.is_open == True
            ).all()
            
            session.close()
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            if session:
                session.close()
            return []
    
    async def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Save trade to database"""
        session = None
        try:
            session = self.get_session()
            
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    async def save_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Save trading signal to database"""
        session = None
        try:
            session = self.get_session()
            
            signal = Signal(**signal_data)
            session.add(signal)
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    async def get_signals(self, symbol: str = None, executed: bool = None, limit: int = 100) -> List[Signal]:
        """Get trading signals with optional filters"""
        session = None
        try:
            session = self.get_session()
            
            query = session.query(Signal)
            
            if symbol:
                query = query.filter(Signal.symbol == symbol)
            if executed is not None:
                query = query.filter(Signal.executed == executed)
                
            signals = query.order_by(Signal.timestamp.desc()).limit(limit).all()
            
            session.close()
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            if session:
                session.close()
            return []
    
    async def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """Update position with new data"""
        session = None
        try:
            session = self.get_session()
            
            position = session.query(Position).filter(Position.id == position_id).first()
            if position:
                for key, value in updates.items():
                    setattr(position, key, value)
                position.updated_at = datetime.utcnow()
                session.commit()
                session.close()
                return True
            
            session.close()
            return False
            
        except Exception as e:
            logger.error(f"Failed to update position: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    async def close_position(self, position_id: str, closing_price: float) -> bool:
        """Close a position"""
        session = None
        try:
            session = self.get_session()
            
            position = session.query(Position).filter(Position.id == position_id).first()
            if position:
                position.calculate_pnl(closing_price)
                position.is_open = False
                position.closed_at = datetime.utcnow()
                position.updated_at = datetime.utcnow()
                session.commit()
                session.close()
                return True
            
            session.close()
            return False
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    async def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics"""
        session = None
        try:
            session = self.get_session()
            
            from_date = datetime.utcnow() - timedelta(days=days)
            
            # Get closed positions
            closed_positions = session.query(Position).filter(
                Position.is_open == False,
                Position.closed_at >= from_date
            ).all()
            
            # Calculate stats
            total_trades = len(closed_positions)
            winning_trades = len([p for p in closed_positions if p.pnl > 0])
            losing_trades = total_trades - winning_trades
            
            total_pnl = sum(p.pnl for p in closed_positions)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = sum(p.pnl for p in closed_positions if p.pnl > 0) / winning_trades if winning_trades > 0 else 0
            avg_loss = sum(p.pnl for p in closed_positions if p.pnl < 0) / losing_trades if losing_trades > 0 else 0
            
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
            }
            
            session.close()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            if session:
                session.close()
            return {}
    
    async def cleanup_old_data(self, days: int = 90):
        """Clean up old market data"""
        session = None
        try:
            session = self.get_session()
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old market data
            deleted = session.query(MarketData).filter(
                MarketData.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            session.close()
            
            logger.info(f"Cleaned up {deleted} old market data records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            if session:
                session.rollback()
                session.close()
            return False
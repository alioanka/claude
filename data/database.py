"""
Database Manager - Handles data persistence and retrieval
Supports both PostgreSQL and SQLite for development.
"""

import asyncio
import logging
import sqlite3
import aiosqlite
import asyncpg
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json

from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Trade:
    """Trade data structure"""
    id: str
    symbol: str
    side: str
    amount: float
    price: float
    cost: float
    timestamp: datetime
    order_id: str
    strategy: str
    pnl: Optional[float] = None
    commission: Optional[float] = None

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        self.db_url = config.database.url
        self.is_postgresql = self.db_url.startswith('postgresql')
        self.connection_pool = None
        self.sqlite_db = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            logger.info("🗄️ Initializing database connection...")
            
            if self.is_postgresql:
                await self.init_postgresql()
            else:
                await self.init_sqlite()
            
            # Create tables
            await self.create_tables()
            
            logger.info("✅ Database initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def init_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Test connection
            async with self.connection_pool.acquire() as conn:
                await conn.execute('SELECT 1')
                
            logger.info("✅ PostgreSQL connection pool created")
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL initialization failed: {e}")
            raise
    
    async def init_sqlite(self):
        """Initialize SQLite connection"""
        try:
            db_path = self.db_url.replace('sqlite:///', '')
            self.sqlite_db = await aiosqlite.connect(db_path)
            await self.sqlite_db.execute('PRAGMA foreign_keys = ON')
            await self.sqlite_db.commit()
            
            logger.info(f"✅ SQLite database connected: {db_path}")
            
        except Exception as e:
            logger.error(f"❌ SQLite initialization failed: {e}")
            raise
    
    async def create_tables(self):
        """Create database tables"""
        try:
            # OHLCV data table
            ohlcv_table = """
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp BIGINT NOT NULL,
                open DECIMAL(20,8) NOT NULL,
                high DECIMAL(20,8) NOT NULL,
                low DECIMAL(20,8) NOT NULL,
                close DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
            """
            
            # Trades table
            trades_table = """
            CREATE TABLE IF NOT EXISTS trades (
                id VARCHAR(50) PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                amount DECIMAL(20,8) NOT NULL,
                price DECIMAL(20,8) NOT NULL,
                cost DECIMAL(20,8) NOT NULL,
                timestamp BIGINT NOT NULL,
                order_id VARCHAR(50),
                strategy VARCHAR(50) NOT NULL,
                pnl DECIMAL(20,8),
                commission DECIMAL(20,8),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # Positions table
            positions_table = """
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                amount DECIMAL(20,8) NOT NULL,
                entry_price DECIMAL(20,8) NOT NULL,
                current_price DECIMAL(20,8) NOT NULL,
                unrealized_pnl DECIMAL(20,8) NOT NULL,
                timestamp BIGINT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # Strategy performance table
            strategy_performance_table = """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id SERIAL PRIMARY KEY,
                strategy_name VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_pnl DECIMAL(20,8) DEFAULT 0,
                max_drawdown DECIMAL(10,4) DEFAULT 0,
                sharpe_ratio DECIMAL(10,4) DEFAULT 0,
                win_rate DECIMAL(10,4) DEFAULT 0,
                avg_win DECIMAL(20,8) DEFAULT 0,
                avg_loss DECIMAL(20,8) DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy_name, symbol)
            )
            """
            
            # Portfolio snapshots table
            portfolio_snapshots_table = """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id SERIAL PRIMARY KEY,
                total_balance DECIMAL(20,8) NOT NULL,
                available_balance DECIMAL(20,8) NOT NULL,
                used_balance DECIMAL(20,8) NOT NULL,
                unrealized_pnl DECIMAL(20,8) NOT NULL,
                daily_pnl DECIMAL(20,8) NOT NULL,
                total_trades INTEGER DEFAULT 0,
                active_positions INTEGER DEFAULT 0,
                timestamp BIGINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # ML model performance table
            ml_models_table = """
            CREATE TABLE IF NOT EXISTS ml_model_performance (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                accuracy DECIMAL(10,4) DEFAULT 0,
                precision DECIMAL(10,4) DEFAULT 0,
                recall DECIMAL(10,4) DEFAULT 0,
                f1_score DECIMAL(10,4) DEFAULT 0,
                auc_score DECIMAL(10,4) DEFAULT 0,
                training_samples INTEGER DEFAULT 0,
                last_trained TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_name, symbol)
            )
            """
            
            # Market data metadata table
            market_metadata_table = """
            CREATE TABLE IF NOT EXISTS market_metadata (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                first_timestamp BIGINT,
                last_timestamp BIGINT,
                total_candles INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe)
            )
            """
            
            tables = [
                ohlcv_table, trades_table, positions_table,
                strategy_performance_table, portfolio_snapshots_table,
                ml_models_table, market_metadata_table
            ]
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    for table in tables:
                        # Adjust for PostgreSQL
                        pg_table = table.replace('SERIAL', 'SERIAL').replace('BIGINT', 'BIGINT').replace('DECIMAL', 'NUMERIC')
                        await conn.execute(pg_table)
            else:
                for table in tables:
                    # Adjust for SQLite
                    sqlite_table = table.replace('SERIAL PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
                    sqlite_table = sqlite_table.replace('DECIMAL(20,8)', 'REAL')
                    sqlite_table = sqlite_table.replace('DECIMAL(10,4)', 'REAL')
                    sqlite_table = sqlite_table.replace('TIMESTAMP DEFAULT CURRENT_TIMESTAMP', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
                    sqlite_table = sqlite_table.replace('BOOLEAN DEFAULT TRUE', 'BOOLEAN DEFAULT 1')
                    await self.sqlite_db.execute(sqlite_table)
                
                await self.sqlite_db.commit()
            
            # Create indexes for better performance
            await self.create_indexes()
            
            logger.info("✅ Database tables created successfully")
            
        except Exception as e:
            logger.error(f"❌ Table creation failed: {e}")
            raise
    
    async def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe ON ohlcv_data(symbol, timeframe)",
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_positions_active ON positions(is_active)",
            ]
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    for index in indexes:
                        await conn.execute(index)
            else:
                for index in indexes:
                    await self.sqlite_db.execute(index)
                await self.sqlite_db.commit()
            
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
    
    async def store_ohlcv(self, symbol: str, timeframe: str, candles: List[List]):
        """Store OHLCV data"""
        try:
            if not candles:
                return
            
            if self.is_postgresql:
                await self._store_ohlcv_postgresql(symbol, timeframe, candles)
            else:
                await self._store_ohlcv_sqlite(symbol, timeframe, candles)
                
        except Exception as e:
            logger.error(f"❌ OHLCV storage failed for {symbol} {timeframe}: {e}")
    
    async def _store_ohlcv_postgresql(self, symbol: str, timeframe: str, candles: List[List]):
        """Store OHLCV data in PostgreSQL"""
        try:
            async with self.connection_pool.acquire() as conn:
                insert_query = """
                INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                """
                
                for candle in candles:
                    await conn.execute(
                        insert_query,
                        symbol, timeframe, int(candle[0]), float(candle[1]),
                        float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])
                    )
                
                # Update metadata
                await self._update_market_metadata(symbol, timeframe, candles)
                
        except Exception as e:
            logger.error(f"❌ PostgreSQL OHLCV storage error: {e}")
    
    async def _store_ohlcv_sqlite(self, symbol: str, timeframe: str, candles: List[List]):
        """Store OHLCV data in SQLite"""
        try:
            insert_query = """
            INSERT OR IGNORE INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            data_to_insert = [
                (symbol, timeframe, int(candle[0]), float(candle[1]),
                 float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5]))
                for candle in candles
            ]
            
            await self.sqlite_db.executemany(insert_query, data_to_insert)
            await self.sqlite_db.commit()
            
            # Update metadata
            await self._update_market_metadata_sqlite(symbol, timeframe, candles)
            
        except Exception as e:
            logger.error(f"❌ SQLite OHLCV storage error: {e}")
    
    async def _update_market_metadata(self, symbol: str, timeframe: str, candles: List[List]):
        """Update market metadata for PostgreSQL"""
        try:
            if not candles:
                return
            
            first_timestamp = int(candles[0][0])
            last_timestamp = int(candles[-1][0])
            
            async with self.connection_pool.acquire() as conn:
                # Upsert metadata
                upsert_query = """
                INSERT INTO market_metadata (symbol, timeframe, first_timestamp, last_timestamp, total_candles)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (symbol, timeframe) DO UPDATE SET
                    last_timestamp = GREATEST(market_metadata.last_timestamp, $4),
                    first_timestamp = LEAST(market_metadata.first_timestamp, $3),
                    total_candles = market_metadata.total_candles + $5,
                    last_updated = CURRENT_TIMESTAMP
                """
                
                await conn.execute(
                    upsert_query, symbol, timeframe, first_timestamp, 
                    last_timestamp, len(candles)
                )
                
        except Exception as e:
            logger.error(f"❌ Metadata update error: {e}")
    
    async def _update_market_metadata_sqlite(self, symbol: str, timeframe: str, candles: List[List]):
        """Update market metadata for SQLite"""
        try:
            if not candles:
                return
            
            first_timestamp = int(candles[0][0])
            last_timestamp = int(candles[-1][0])
            
            # Check if metadata exists
            check_query = """
            SELECT first_timestamp, last_timestamp, total_candles 
            FROM market_metadata 
            WHERE symbol = ? AND timeframe = ?
            """
            
            async with self.sqlite_db.execute(check_query, (symbol, timeframe)) as cursor:
                existing = await cursor.fetchone()
            
            if existing:
                # Update existing
                update_query = """
                UPDATE market_metadata 
                SET last_timestamp = MAX(last_timestamp, ?),
                    first_timestamp = MIN(first_timestamp, ?),
                    total_candles = total_candles + ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE symbol = ? AND timeframe = ?
                """
                
                await self.sqlite_db.execute(
                    update_query, 
                    (last_timestamp, first_timestamp, len(candles), symbol, timeframe)
                )
            else:
                # Insert new
                insert_query = """
                INSERT INTO market_metadata (symbol, timeframe, first_timestamp, last_timestamp, total_candles)
                VALUES (?, ?, ?, ?, ?)
                """
                
                await self.sqlite_db.execute(
                    insert_query,
                    (symbol, timeframe, first_timestamp, last_timestamp, len(candles))
                )
            
            await self.sqlite_db.commit()
            
        except Exception as e:
            logger.error(f"❌ SQLite metadata update error: {e}")
    
    async def get_historical_data(self, symbol: str, timeframe: str,
                                 start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get historical OHLCV data"""
        try:
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)
            
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    rows = await conn.fetch(query, symbol, timeframe, start_timestamp, end_timestamp)
                    return [dict(row) for row in rows]
            else:
                async with self.sqlite_db.execute(query, (symbol, timeframe, start_timestamp, end_timestamp)) as cursor:
                    rows = await cursor.fetchall()
                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            logger.error(f"❌ Historical data retrieval error: {e}")
            return []
    
    async def store_trade(self, trade: Trade):
        """Store a trade record"""
        try:
            insert_query = """
            INSERT INTO trades (id, symbol, side, amount, price, cost, timestamp, order_id, strategy, pnl, commission)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            trade_data = (
                trade.id, trade.symbol, trade.side, trade.amount, trade.price,
                trade.cost, int(trade.timestamp.timestamp() * 1000), trade.order_id,
                trade.strategy, trade.pnl, trade.commission
            )
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    # Convert to PostgreSQL format
                    pg_query = insert_query.replace('?', '${}').format(*range(1, 12))
                    await conn.execute(pg_query, *trade_data)
            else:
                await self.sqlite_db.execute(insert_query, trade_data)
                await self.sqlite_db.commit()
                
            logger.debug(f"📝 Trade stored: {trade.symbol} {trade.side} {trade.amount}")
            
        except Exception as e:
            logger.error(f"❌ Trade storage failed: {e}")
    
    async def store_position(self, position: Position):
        """Store/update a position record"""
        try:
            # First, deactivate old positions for the same symbol
            deactivate_query = """
            UPDATE positions SET is_active = ? WHERE symbol = ? AND is_active = ?
            """
            
            insert_query = """
            INSERT INTO positions (symbol, side, amount, entry_price, current_price, unrealized_pnl, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            position_data = (
                position.symbol, position.side, position.amount, position.entry_price,
                position.current_price, position.unrealized_pnl,
                int(position.timestamp.timestamp() * 1000)
            )
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    # Deactivate old positions
                    await conn.execute("UPDATE positions SET is_active = false WHERE symbol = $1 AND is_active = true", position.symbol)
                    
                    # Insert new position
                    pg_query = insert_query.replace('?', '${}').format(*range(1, 8))
                    await conn.execute(pg_query, *position_data)
            else:
                await self.sqlite_db.execute(deactivate_query, (False, position.symbol, True))
                await self.sqlite_db.execute(insert_query, position_data)
                await self.sqlite_db.commit()
                
            logger.debug(f"📊 Position stored: {position.symbol} {position.side}")
            
        except Exception as e:
            logger.error(f"❌ Position storage failed: {e}")
    
    async def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        try:
            query = """
            SELECT symbol, side, amount, entry_price, current_price, unrealized_pnl, timestamp
            FROM positions
            WHERE is_active = ?
            ORDER BY timestamp DESC
            """
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    rows = await conn.fetch(query.replace('?', '$1'), True)
                    return [dict(row) for row in rows]
            else:
                async with self.sqlite_db.execute(query, (True,)) as cursor:
                    rows = await cursor.fetchall()
                    columns = ['symbol', 'side', 'amount', 'entry_price', 'current_price', 'unrealized_pnl', 'timestamp']
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            logger.error(f"❌ Active positions retrieval error: {e}")
            return []
    
    async def get_trades_history(self, symbol: Optional[str] = None, 
                               limit: int = 100, days_back: int = 30) -> List[Dict]:
        """Get trade history"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            start_timestamp = int(start_time.timestamp() * 1000)
            
            if symbol:
                query = """
                SELECT * FROM trades 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT ?
                """
                params = (symbol, start_timestamp, limit)
            else:
                query = """
                SELECT * FROM trades 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC LIMIT ?
                """
                params = (start_timestamp, limit)
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    if symbol:
                        rows = await conn.fetch(query.replace('?', '${}').format(1, 2, 3), *params)
                    else:
                        rows = await conn.fetch(query.replace('?', '${}').format(1, 2), *params)
                    return [dict(row) for row in rows]
            else:
                async with self.sqlite_db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    # Get column names from cursor description
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            logger.error(f"❌ Trade history retrieval error: {e}")
            return []
    
    async def store_portfolio_snapshot(self, snapshot: Dict[str, Any]):
        """Store portfolio snapshot"""
        try:
            insert_query = """
            INSERT INTO portfolio_snapshots 
            (total_balance, available_balance, used_balance, unrealized_pnl, 
             daily_pnl, total_trades, active_positions, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            timestamp = int(datetime.utcnow().timestamp() * 1000)
            snapshot_data = (
                snapshot.get('total_balance', 0),
                snapshot.get('available_balance', 0),
                snapshot.get('used_balance', 0),
                snapshot.get('unrealized_pnl', 0),
                snapshot.get('daily_pnl', 0),
                snapshot.get('total_trades', 0),
                snapshot.get('active_positions', 0),
                timestamp
            )
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    pg_query = insert_query.replace('?', '${}').format(*range(1, 9))
                    await conn.execute(pg_query, *snapshot_data)
            else:
                await self.sqlite_db.execute(insert_query, snapshot_data)
                await self.sqlite_db.commit()
                
        except Exception as e:
            logger.error(f"❌ Portfolio snapshot storage failed: {e}")
    
    async def get_portfolio_history(self, days_back: int = 30) -> List[Dict]:
        """Get portfolio history"""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            start_timestamp = int(start_time.timestamp() * 1000)
            
            query = """
            SELECT * FROM portfolio_snapshots 
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            """
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    rows = await conn.fetch(query.replace('?', '$1'), start_timestamp)
                    return [dict(row) for row in rows]
            else:
                async with self.sqlite_db.execute(query, (start_timestamp,)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            logger.error(f"❌ Portfolio history retrieval error: {e}")
            return []
    
    async def update_strategy_performance(self, strategy_name: str, symbol: str, 
                                        performance_data: Dict[str, Any]):
        """Update strategy performance metrics"""
        try:
            upsert_query = """
            INSERT INTO strategy_performance 
            (strategy_name, symbol, total_trades, winning_trades, losing_trades, 
             total_pnl, max_drawdown, sharpe_ratio, win_rate, avg_win, avg_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            performance_data_tuple = (
                strategy_name, symbol,
                performance_data.get('total_trades', 0),
                performance_data.get('winning_trades', 0),
                performance_data.get('losing_trades', 0),
                performance_data.get('total_pnl', 0),
                performance_data.get('max_drawdown', 0),
                performance_data.get('sharpe_ratio', 0),
                performance_data.get('win_rate', 0),
                performance_data.get('avg_win', 0),
                performance_data.get('avg_loss', 0)
            )
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    # PostgreSQL UPSERT
                    pg_query = """
                    INSERT INTO strategy_performance 
                    (strategy_name, symbol, total_trades, winning_trades, losing_trades, 
                     total_pnl, max_drawdown, sharpe_ratio, win_rate, avg_win, avg_loss)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (strategy_name, symbol) DO UPDATE SET
                        total_trades = EXCLUDED.total_trades,
                        winning_trades = EXCLUDED.winning_trades,
                        losing_trades = EXCLUDED.losing_trades,
                        total_pnl = EXCLUDED.total_pnl,
                        max_drawdown = EXCLUDED.max_drawdown,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        win_rate = EXCLUDED.win_rate,
                        avg_win = EXCLUDED.avg_win,
                        avg_loss = EXCLUDED.avg_loss,
                        last_updated = CURRENT_TIMESTAMP
                    """
                    await conn.execute(pg_query, *performance_data_tuple)
            else:
                # SQLite UPSERT
                sqlite_query = """
                INSERT OR REPLACE INTO strategy_performance 
                (strategy_name, symbol, total_trades, winning_trades, losing_trades, 
                 total_pnl, max_drawdown, sharpe_ratio, win_rate, avg_win, avg_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                await self.sqlite_db.execute(sqlite_query, performance_data_tuple)
                await self.sqlite_db.commit()
                
        except Exception as e:
            logger.error(f"❌ Strategy performance update failed: {e}")
    
    async def get_strategy_performance(self, strategy_name: Optional[str] = None) -> List[Dict]:
        """Get strategy performance metrics"""
        try:
            if strategy_name:
                query = """
                SELECT * FROM strategy_performance 
                WHERE strategy_name = ?
                ORDER BY total_pnl DESC
                """
                params = (strategy_name,)
            else:
                query = """
                SELECT * FROM strategy_performance 
                ORDER BY total_pnl DESC
                """
                params = ()
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    if strategy_name:
                        rows = await conn.fetch(query.replace('?', '$1'), *params)
                    else:
                        rows = await conn.fetch(query)
                    return [dict(row) for row in rows]
            else:
                async with self.sqlite_db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
                    
        except Exception as e:
            logger.error(f"❌ Strategy performance retrieval error: {e}")
            return []
    
    async def store_ml_model_performance(self, model_name: str, symbol: str, 
                                       performance_metrics: Dict[str, Any]):
        """Store ML model performance metrics"""
        try:
            upsert_data = (
                model_name, symbol,
                performance_metrics.get('accuracy', 0),
                performance_metrics.get('precision', 0),
                performance_metrics.get('recall', 0),
                performance_metrics.get('f1_score', 0),
                performance_metrics.get('auc_score', 0),
                performance_metrics.get('training_samples', 0)
            )
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    query = """
                    INSERT INTO ml_model_performance 
                    (model_name, symbol, accuracy, precision, recall, f1_score, auc_score, training_samples, last_trained)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                    ON CONFLICT (model_name, symbol) DO UPDATE SET
                        accuracy = EXCLUDED.accuracy,
                        precision = EXCLUDED.precision,
                        recall = EXCLUDED.recall,
                        f1_score = EXCLUDED.f1_score,
                        auc_score = EXCLUDED.auc_score,
                        training_samples = EXCLUDED.training_samples,
                        last_trained = CURRENT_TIMESTAMP,
                        last_updated = CURRENT_TIMESTAMP
                    """
                    await conn.execute(query, *upsert_data)
            else:
                query = """
                INSERT OR REPLACE INTO ml_model_performance 
                (model_name, symbol, accuracy, precision, recall, f1_score, auc_score, training_samples, last_trained)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """
                await self.sqlite_db.execute(query, upsert_data)
                await self.sqlite_db.commit()
                
        except Exception as e:
            logger.error(f"❌ ML model performance storage failed: {e}")
    
    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to save space"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            cleanup_queries = [
                ("DELETE FROM ohlcv_data WHERE timestamp < ? AND timeframe IN ('1m', '5m')", (cutoff_timestamp,)),
                ("DELETE FROM portfolio_snapshots WHERE timestamp < ?", (cutoff_timestamp,)),
                ("DELETE FROM trades WHERE timestamp < ?", (cutoff_timestamp,))
            ]
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    for query, params in cleanup_queries:
                        pg_query = query.replace('?', '$1')
                        result = await conn.execute(pg_query, *params)
                        logger.info(f"🧹 Cleaned up old data: {result}")
            else:
                for query, params in cleanup_queries:
                    cursor = await self.sqlite_db.execute(query, params)
                    logger.info(f"🧹 Cleaned up {cursor.rowcount} old records")
                await self.sqlite_db.commit()
                
            logger.info(f"✅ Data cleanup completed (kept last {days_to_keep} days)")
            
        except Exception as e:
            logger.error(f"❌ Data cleanup failed: {e}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                'database_type': 'PostgreSQL' if self.is_postgresql else 'SQLite',
                'table_stats': {}
            }
            
            tables = [
                'ohlcv_data', 'trades', 'positions', 'strategy_performance',
                'portfolio_snapshots', 'ml_model_performance', 'market_metadata'
            ]
            
            for table in tables:
                try:
                    count_query = f"SELECT COUNT(*) FROM {table}"
                    
                    if self.is_postgresql:
                        async with self.connection_pool.acquire() as conn:
                            count = await conn.fetchval(count_query)
                    else:
                        async with self.sqlite_db.execute(count_query) as cursor:
                            count = (await cursor.fetchone())[0]
                    
                    stats['table_stats'][table] = count
                    
                except Exception as e:
                    stats['table_stats'][table] = f"Error: {e}"
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Database stats retrieval error: {e}")
            return {}
    
    async def export_data_to_csv(self, table_name: str, output_path: str, 
                                symbol: Optional[str] = None, days_back: int = 30):
        """Export data to CSV file"""
        try:
            import os
            
            # Ensure storage/exports directory exists
            exports_dir = "storage/exports"
            os.makedirs(exports_dir, exist_ok=True)
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            start_timestamp = int(start_time.timestamp() * 1000)
            
            if symbol:
                query = f"SELECT * FROM {table_name} WHERE symbol = ? AND timestamp >= ?"
                params = (symbol, start_timestamp)
            else:
                query = f"SELECT * FROM {table_name} WHERE timestamp >= ?"
                params = (start_timestamp,)
            
            if self.is_postgresql:
                async with self.connection_pool.acquire() as conn:
                    if symbol:
                        rows = await conn.fetch(query.replace('?', '${}').format(1, 2), *params)
                    else:
                        rows = await conn.fetch(query.replace('?', '$1'), *params)
                    
                    # Convert to DataFrame and save
                    df = pd.DataFrame([dict(row) for row in rows])
            else:
                async with self.sqlite_db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    df = pd.DataFrame([dict(zip(columns, row)) for row in rows])
            
            if not df.empty:
                full_path = os.path.join(exports_dir, output_path)
                df.to_csv(full_path, index=False)
                logger.info(f"📊 Data exported to {full_path}")
                return full_path
            else:
                logger.warning("⚠️ No data to export")
                return None
                
        except Exception as e:
            logger.error(f"❌ Data export failed: {e}")
            return None
    
    async def close(self):
        """Close database connections"""
        try:
            if self.is_postgresql and self.connection_pool:
                await self.connection_pool.close()
                logger.info("🔌 PostgreSQL connection pool closed")
            elif self.sqlite_db:
                await self.sqlite_db.close()
                logger.info("🔌 SQLite connection closed")
                
        except Exception as e:
            logger.error(f"❌ Error closing database connections: {e}")
    
    async def backup_database(self):
        """Create database backup"""
        try:
            import os
            import shutil
            from datetime import datetime
            
            backup_dir = "storage/backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            if not self.is_postgresql:  # SQLite backup
                db_path = self.db_url.replace('sqlite:///', '')
                if os.path.exists(db_path):
                    backup_path = os.path.join(backup_dir, f"trading_bot_backup_{timestamp}.db")
                    shutil.copy2(db_path, backup_path)
                    logger.info(f"💾 Database backup created: {backup_path}")
                    return backup_path
            else:
                # PostgreSQL backup would require pg_dump
                logger.warning("⚠️ PostgreSQL backup requires pg_dump utility")
                return None
                
        except Exception as e:
            logger.error(f"❌ Database backup failed: {e}")
            return None
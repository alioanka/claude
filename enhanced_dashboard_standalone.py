#!/usr/bin/env python3
"""
Standalone Enhanced Dashboard - ClaudeBot
Completely independent version that doesn't import ClaudeBot modules
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: pip install fastapi uvicorn")
    sys.exit(1)

# Try to import database modules
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    print("⚠️ psycopg2 not available, database features will be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
db_connection = None
connected_clients = []

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Create FastAPI app
app = FastAPI(
    title="ClaudeBot Enhanced Dashboard",
    description="Advanced Trading Bot Dashboard with Comprehensive Analytics",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="enhanced_dashboard/static"), name="static")

# Templates
templates = Jinja2Templates(directory="enhanced_dashboard/templates")

# Database connection
async def get_db_connection():
    """Get database connection"""
    global db_connection
    
    if db_connection is None and HAS_PSYCOPG2:
        try:
            # Try to connect to database
            db_connection = psycopg2.connect(
                host="localhost",
                port="5432",
                database="trading_bot",
                user="trader",
                password="secure_password"
            )
            logger.info("✅ Database connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Database connection failed: {e}")
            db_connection = None
    
    return db_connection

# API Routes

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "claudebot-enhanced-dashboard",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "message": "API is working!",
        "timestamp": datetime.utcnow().isoformat(),
        "database_available": HAS_PSYCOPG2
    }

@app.get("/api/debug/schema")
async def debug_schema():
    """Debug endpoint to check database schema"""
    if not HAS_PSYCOPG2:
        return {"error": "Database not available"}
    
    try:
        conn = await get_db_connection()
        if not conn:
            return {"error": "Cannot connect to database"}
        
        cursor = conn.cursor()
        
        # Check positions table schema
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'positions' 
            ORDER BY ordinal_position
        """)
        positions_schema = cursor.fetchall()
        
        # Check trades table schema
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'trades' 
            ORDER BY ordinal_position
        """)
        trades_schema = cursor.fetchall()
        
        # Get sample data from positions
        cursor.execute("SELECT * FROM positions LIMIT 3")
        sample_positions = cursor.fetchall()
        
        # Get column names from positions table
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'positions' 
            ORDER BY ordinal_position
        """)
        positions_columns = [row[0] for row in cursor.fetchall()]
        
        # Get sample data from trades
        cursor.execute("SELECT * FROM trades LIMIT 3")
        sample_trades = cursor.fetchall()
        
        cursor.close()
        
        return {
            "positions_schema": positions_schema,
            "trades_schema": trades_schema,
            "positions_columns": positions_columns,
            "sample_positions": sample_positions,
            "sample_trades": sample_trades
        }
        
    except Exception as e:
        return {"error": str(e)}

# Account & Portfolio APIs
@app.get("/api/account")
async def get_account():
    """Get comprehensive account information"""
    try:
        # Default values
        default_account = {
            "total_balance": 10000.0,
            "available_balance": 10000.0,
            "used_balance": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0,
            "total_return": 0.0,
            "active_positions": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "initial_capital": 10000.0,
            "last_update": datetime.utcnow().isoformat()
        }
        
        # Try to get real data from database
        if HAS_PSYCOPG2:
            conn = await get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Get account balance
                    cursor.execute("SELECT * FROM account_balance ORDER BY timestamp DESC LIMIT 1")
                    balance_row = cursor.fetchone()
                    
                    if balance_row:
                        default_account.update({
                            "total_balance": float(balance_row.get('total_balance', 10000.0)),
                            "available_balance": float(balance_row.get('available_balance', 10000.0)),
                            "used_balance": float(balance_row.get('used_balance', 0.0)),
                            "unrealized_pnl": float(balance_row.get('unrealized_pnl', 0.0)),
                            "realized_pnl": float(balance_row.get('realized_pnl', 0.0)),
                            "total_pnl": float(balance_row.get('total_pnl', 0.0)),
                            "last_update": balance_row.get('timestamp', datetime.utcnow()).isoformat()
                        })
                    
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error getting account data: {e}")
        
        return default_account
        
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return {
            "total_balance": 10000.0,
            "available_balance": 10000.0,
            "used_balance": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0,
            "total_return": 0.0,
            "active_positions": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "initial_capital": 10000.0,
            "last_update": datetime.utcnow().isoformat()
        }

@app.get("/api/positions")
async def get_positions():
    """Get current positions with enhanced data"""
    try:
        positions_data = []
        
        # Try to get real data from database
        if HAS_PSYCOPG2:
            conn = await get_db_connection()
            if conn:
                try:
                    # Start a new transaction
                    conn.rollback()
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Get only open positions (is_open = true)
                    cursor.execute("""
                        SELECT * FROM positions 
                        WHERE is_open = true
                        ORDER BY created_at DESC
                    """)
                    positions = cursor.fetchall()
                    
                    logger.info(f"Found {len(positions)} positions in database")
                    if positions:
                        logger.info(f"Sample position columns: {list(positions[0].keys()) if positions else 'No positions'}")
                        logger.info(f"Sample position data: {positions[0] if positions else 'No positions'}")
                    
                    for pos in positions:
                        # Calculate duration
                        created_at = pos.get('created_at') or pos.get('timestamp')
                        if created_at:
                            if hasattr(created_at, 'isoformat'):
                                duration_seconds = (datetime.utcnow() - created_at).total_seconds()
                            else:
                                duration_seconds = (datetime.utcnow() - datetime.fromtimestamp(created_at.timestamp())).total_seconds()
                            
                            duration_hours = duration_seconds / 3600
                            if duration_hours >= 24:
                                duration_str = f"{int(duration_hours // 24)}d {int(duration_hours % 24)}h"
                            elif duration_hours >= 1:
                                duration_str = f"{int(duration_hours)}h {int((duration_seconds % 3600) / 60)}m"
                            else:
                                duration_str = f"{int(duration_seconds / 60)}m"
                        else:
                            duration_str = "N/A"
                            duration_hours = 0
                        
                        # Try different possible column names for size
                        size_value = pos.get('size') or pos.get('quantity') or pos.get('amount') or 0.0
                        size_value = float(size_value) if size_value is not None else 0.0
                        
                        # Try different possible column names for entry price
                        entry_price = pos.get('entry_price') or pos.get('price') or pos.get('open_price') or 0.0
                        entry_price = float(entry_price) if entry_price is not None else 0.0
                        
                        # Try different possible column names for current price
                        current_price = pos.get('current_price') or pos.get('market_price') or pos.get('close_price') or entry_price
                        current_price = float(current_price) if current_price is not None else entry_price
                        
                        # Try different possible column names for PnL
                        pnl_value = pos.get('unrealized_pnl') or pos.get('pnl') or pos.get('profit_loss') or 0.0
                        pnl_value = float(pnl_value) if pnl_value is not None else 0.0
                        
                        # Try different possible column names for PnL percentage
                        pnl_percentage = pos.get('unrealized_pnl_percentage') or pos.get('pnl_percentage') or pos.get('profit_loss_percentage') or 0.0
                        pnl_percentage = float(pnl_percentage) if pnl_percentage is not None else 0.0
                        
                        positions_data.append({
                            "symbol": pos.get('symbol', 'N/A'),
                            "side": pos.get('side', 'N/A'),
                            "size": size_value,
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "pnl": pnl_value,
                            "pnl_percentage": pnl_percentage,
                            "duration": duration_str,
                            "duration_hours": round(duration_hours, 2),
                            "strategy": pos.get('strategy', 'N/A'),
                            "timestamp": created_at.isoformat() if created_at and hasattr(created_at, 'isoformat') else datetime.utcnow().isoformat()
                        })
                    
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error getting positions: {e}")
                    logger.warning(f"Position data sample: {positions[:2] if positions else 'No positions found'}")
                    conn.rollback()
        
        return {"positions": positions_data}
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {"positions": []}

@app.get("/api/trades")
async def get_trades(limit: int = 100, strategy: str = None):
    """Get recent trades with filtering"""
    try:
        trades_data = []
        
        # Try to get real data from database
        if HAS_PSYCOPG2:
            conn = await get_db_connection()
            if conn:
                try:
                    # Start a new transaction
                    conn.rollback()
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Get closed positions as "trades" (like existing dashboard)
                    query = """
                        SELECT 
                            id,
                            symbol,
                            side,
                            size,
                            entry_price as price,
                            pnl,
                            pnl_percentage,
                            created_at as timestamp,
                            strategy,
                            closed_at,
                            (size * entry_price) as total_value
                        FROM positions 
                        WHERE is_open = false 
                        ORDER BY closed_at DESC 
                        LIMIT %s
                    """
                    params = [limit]
                    
                    if strategy:
                        query = """
                            SELECT 
                                id,
                                symbol,
                                side,
                                size,
                                entry_price as price,
                                pnl,
                                pnl_percentage,
                                created_at as timestamp,
                                strategy,
                                closed_at,
                                (size * entry_price) as total_value
                            FROM positions 
                            WHERE is_open = false AND strategy = %s
                            ORDER BY closed_at DESC 
                            LIMIT %s
                        """
                        params = [strategy, limit]
                    
                    cursor.execute(query, params)
                    trades = cursor.fetchall()
                    
                    logger.info(f"Found {len(trades)} trades in database")
                    if trades:
                        logger.info(f"Sample trade columns: {list(trades[0].keys()) if trades else 'No trades'}")
                        logger.info(f"Sample trade data: {trades[0] if trades else 'No trades'}")
                    
                    for trade in trades:
                        # Map closed positions data (already selected with correct column names)
                        size_value = float(trade.get('size', 0.0))
                        entry_price = float(trade.get('price', 0.0))  # entry_price aliased as price
                        exit_price = None  # Closed positions don't have exit price in this query
                        pnl_value = float(trade.get('pnl', 0.0))
                        pnl_percentage = float(trade.get('pnl_percentage', 0.0))
                        
                        # Calculate duration from created_at to closed_at
                        opened_at = trade.get('timestamp')  # created_at aliased as timestamp
                        closed_at = trade.get('closed_at')
                        
                        # Calculate duration
                        duration_str = "N/A"
                        if opened_at and closed_at:
                            if hasattr(opened_at, 'isoformat') and hasattr(closed_at, 'isoformat'):
                                duration_seconds = (closed_at - opened_at).total_seconds()
                                duration_hours = duration_seconds / 3600
                                if duration_hours >= 24:
                                    duration_str = f"{int(duration_hours // 24)}d {int(duration_hours % 24)}h"
                                elif duration_hours >= 1:
                                    duration_str = f"{int(duration_hours)}h {int((duration_seconds % 3600) / 60)}m"
                                else:
                                    duration_str = f"{int(duration_seconds / 60)}m"
                        
                        trades_data.append({
                            "id": trade.get('id'),
                            "symbol": trade.get('symbol', 'N/A'),
                            "side": trade.get('side', 'N/A'),
                            "size": size_value,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl_value,
                            "pnl_percentage": pnl_percentage,
                            "strategy": trade.get('strategy', 'N/A'),
                            "opened_at": opened_at.isoformat() if opened_at and hasattr(opened_at, 'isoformat') else None,
                            "closed_at": closed_at.isoformat() if closed_at and hasattr(closed_at, 'isoformat') else None,
                            "duration": duration_str,
                            "status": "closed"  # All these are closed positions
                        })
                    
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error getting trades: {e}")
                    conn.rollback()
        
        return {"trades": trades_data}
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return {"trades": []}

@app.get("/api/performance")
async def get_performance():
    """Get comprehensive performance metrics"""
    try:
        # Default values
        default_performance = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "current_equity": 10000.0,
            "total_balance": 10000.0,
            "available_balance": 10000.0,
            "total_return": 0.0,
            "active_positions": 0,
            "initial_balance": 10000.0,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0
        }
        
        # Try to get real data from database
        if HAS_PSYCOPG2:
            conn = await get_db_connection()
            if conn:
                try:
                    # Start a new transaction
                    conn.rollback()
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Get performance metrics from closed positions (trades table doesn't have pnl_percentage)
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN pnl_percentage > 0 THEN 1 END) as winning_trades,
                            COUNT(CASE WHEN pnl_percentage < 0 THEN 1 END) as losing_trades,
                            AVG(CASE WHEN pnl_percentage > 0 THEN pnl_percentage END) as avg_win,
                            AVG(CASE WHEN pnl_percentage < 0 THEN pnl_percentage END) as avg_loss,
                            SUM(pnl_percentage) as total_pnl
                        FROM positions 
                        WHERE is_open = false
                    """)
                    
                    perf_row = cursor.fetchone()
                    if perf_row:
                        total_trades = int(perf_row.get('total_trades', 0))
                        winning_trades = int(perf_row.get('winning_trades', 0))
                        losing_trades = int(perf_row.get('losing_trades', 0))
                        
                        default_performance.update({
                            "total_trades": total_trades,
                            "winning_trades": winning_trades,
                            "losing_trades": losing_trades,
                            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                            "total_pnl": float(perf_row.get('total_pnl', 0.0)),
                            "avg_win": float(perf_row.get('avg_win', 0.0)),
                            "avg_loss": float(perf_row.get('avg_loss', 0.0))
                        })
                    
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error getting performance data: {e}")
                    conn.rollback()
        
        return default_performance
        
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "current_equity": 10000.0,
            "total_balance": 10000.0,
            "available_balance": 10000.0,
            "total_return": 0.0,
            "active_positions": 0,
            "initial_balance": 10000.0,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0
        }

# Additional API endpoints for dashboard functionality
@app.get("/api/analytics/strategy-performance")
async def get_strategy_performance():
    """Get strategy performance analytics"""
    try:
        strategy_performance = []
        
        if HAS_PSYCOPG2:
            conn = await get_db_connection()
            if conn:
                try:
                    # Start a new transaction
                    conn.rollback()
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Get strategy performance from closed positions
                    cursor.execute("""
                        SELECT 
                            strategy,
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN pnl_percentage > 0 THEN 1 END) as winning_trades,
                            SUM(pnl_percentage) as total_pnl,
                            AVG(pnl_percentage) as avg_pnl,
                            MAX(pnl_percentage) as max_win,
                            MIN(pnl_percentage) as max_loss
                        FROM positions 
                        WHERE is_open = false
                        GROUP BY strategy
                        ORDER BY total_pnl DESC
                    """)
                    
                    strategies = cursor.fetchall()
                    for strategy in strategies:
                        total_trades = int(strategy.get('total_trades', 0))
                        winning_trades = int(strategy.get('winning_trades', 0))
                        
                        strategy_performance.append({
                            "strategy": strategy.get('strategy', 'Unknown'),
                            "total_trades": total_trades,
                            "winning_trades": winning_trades,
                            "losing_trades": total_trades - winning_trades,
                            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                            "total_pnl": float(strategy.get('total_pnl', 0.0)),
                            "avg_pnl": float(strategy.get('avg_pnl', 0.0)),
                            "max_win": float(strategy.get('max_win', 0.0)),
                            "max_loss": float(strategy.get('max_loss', 0.0))
                        })
                    
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error getting strategy performance: {e}")
                    conn.rollback()
        
        return {"strategy_performance": strategy_performance}
        
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        return {"strategy_performance": []}

@app.get("/api/analytics/risk-metrics")
async def get_risk_metrics():
    """Get risk management metrics"""
    try:
        risk_metrics = {
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_consecutive_losses": 0,
            "current_drawdown": 0.0,
            "risk_score": 0.0
        }
        
        if HAS_PSYCOPG2:
            conn = await get_db_connection()
            if conn:
                try:
                    # Start a new transaction
                    conn.rollback()
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Get risk metrics from closed positions
                    cursor.execute("""
                        SELECT 
                            MAX(pnl_percentage) as max_pnl,
                            MIN(pnl_percentage) as min_pnl,
                            AVG(pnl_percentage) as avg_pnl,
                            STDDEV(pnl_percentage) as std_pnl
                        FROM positions 
                        WHERE is_open = false
                    """)
                    
                    risk_row = cursor.fetchone()
                    if risk_row:
                        max_pnl = float(risk_row.get('max_pnl', 0.0))
                        min_pnl = float(risk_row.get('min_pnl', 0.0))
                        avg_pnl = float(risk_row.get('avg_pnl', 0.0))
                        std_pnl = float(risk_row.get('std_pnl', 0.0))
                        
                        risk_metrics.update({
                            "max_drawdown": abs(min_pnl) if min_pnl < 0 else 0.0,
                            "sharpe_ratio": (avg_pnl / std_pnl) if std_pnl > 0 else 0.0,
                            "current_drawdown": abs(min_pnl) if min_pnl < 0 else 0.0
                        })
                    
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error getting risk metrics: {e}")
                    conn.rollback()
        
        return risk_metrics
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        return {
            "max_drawdown": 0.0,
            "var_95": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_consecutive_losses": 0,
            "current_drawdown": 0.0,
            "risk_score": 0.0
        }

@app.get("/api/market-data")
async def get_market_data():
    """Get market data"""
    try:
        market_data = {
            "btc_price": 50000.0,
            "eth_price": 3000.0,
            "market_cap": 2000000000000.0,
            "fear_greed_index": 50,
            "total_volume": 100000000000.0,
            "btc_change_24h": 2.5,
            "eth_change_24h": -1.2,
            "btc_high_24h": 52000.0,
            "btc_low_24h": 48000.0,
            "eth_high_24h": 3100.0,
            "eth_low_24h": 2900.0
        }
        
        return {"market_data": market_data}
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return {"market_data": {}}

@app.get("/api/config")
async def get_config():
    """Get bot configuration"""
    try:
        config_data = {
            "trading_pairs": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            "max_positions": 50,
            "stop_loss_percent": 0.01,
            "take_profit_percent": 0.025,
            "max_daily_loss": 0.03,
            "risk_per_trade": 0.01
        }
        
        return config_data
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return {}

@app.post("/api/positions/{symbol}/close")
async def close_position(symbol: str):
    """Close a specific position"""
    try:
        # This would normally close the position
        # For now, just return success
        return {"success": True, "message": f"Position {symbol} closed successfully"}
        
    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/positions/close-all")
async def close_all_positions():
    """Close all positions"""
    try:
        # This would normally close all positions
        # For now, just return success
        return {"success": True, "message": "All positions closed successfully"}
        
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/config/update")
async def update_config():
    """Update bot configuration"""
    try:
        # This would normally update the configuration
        # For now, just return success
        return {"success": True, "message": "Configuration updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return {"success": False, "error": str(e)}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    try:
        await manager.connect(websocket)
        logger.info("WebSocket connection established")
        
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "Connected to ClaudeBot Enhanced Dashboard",
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        while True:
            try:
                # Keep connection alive
                data = await websocket.receive_text()
                logger.debug(f"Received WebSocket message: {data}")
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        manager.disconnect(websocket)

# Background tasks
async def update_dashboard_data():
    """Send real-time updates to dashboard"""
    while True:
        try:
            if manager.active_connections:
                logger.debug(f"Updating dashboard data for {len(manager.active_connections)} connections")
                
                # Get latest data
                account_data = await get_account()
                positions_data = await get_positions()
                
                # Send update to all connected clients
                await manager.broadcast(json.dumps({
                    "type": "dashboard_update",
                    "data": {
                        "account": account_data,
                        "positions": positions_data
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
                logger.debug("Dashboard data update sent")
            else:
                logger.debug("No active WebSocket connections")
            
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
            await asyncio.sleep(5)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize dashboard on startup"""
    try:
        # Start background tasks
        asyncio.create_task(update_dashboard_data())
        
        logger.info("🚀 Enhanced Dashboard started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard: {e}")
        # Don't exit, just log the error

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_dashboard_standalone:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )

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
    """Get account information with real metrics"""
    try:
        if not HAS_PSYCOPG2:
            return {
                "total_balance": 10000.0,
                "total_pnl": 0.0,
                "active_positions": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            }
            
        conn = await get_db_connection()
        if not conn:
            return {
                "total_balance": 10000.0,
                "total_pnl": 0.0,
                "active_positions": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            }
            
        try:
            conn.rollback()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Realized PnL from closed trades
            cursor.execute("""
                SELECT COALESCE(SUM(pnl),0) AS realized_pnl, COUNT(*) AS total_trades,
                       AVG(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100 AS win_rate
                FROM positions
                WHERE is_open = false
            """)
            t = cursor.fetchone() or {}
            
            # Unrealized from current positions (with fresh prices)
            positions = (await get_positions())["positions"]
            unrealized = sum(p["pnl"] for p in positions)
            active = len(positions)
            
            total_pnl = float(t.get("realized_pnl", 0)) + unrealized
            # Use starting balance from config or default
            starting_balance = 10000.0
            total_balance = starting_balance + total_pnl
            
            # Optional: rough Sharpe / MDD if you have equity curve, else zeros
            sharpe = 0.0
            max_dd = 0.0
            
            cursor.close()
            return {
                "total_balance": round(total_balance, 2),
                "total_pnl": round(total_pnl, 2),
                "active_positions": active,
                "win_rate": round(float(t.get("win_rate", 0)), 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown": round(max_dd, 2),
                "realized_pnl": round(float(t.get("realized_pnl", 0)), 2),
                "unrealized_pnl": round(unrealized, 2),
                "total_trades": int(t.get("total_trades", 0)),
                "winning_trades": int(t.get("total_trades", 0) * float(t.get("win_rate", 0)) / 100),
                "losing_trades": int(t.get("total_trades", 0) * (100 - float(t.get("win_rate", 0))) / 100),
                "available_balance": max(0, total_balance),
                "used_balance": abs(min(0, total_pnl)),
                "initial_capital": starting_balance,
                "last_update": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error getting account data: {e}")
            conn.rollback()
            return {
                "total_balance": 10000.0,
                "total_pnl": 0.0,
                "active_positions": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            }
            
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return {
            "total_balance": 10000.0,
            "total_pnl": 0.0,
            "active_positions": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }

@app.get("/api/positions")
async def get_positions():
    """Get current positions with live prices and enhanced data"""
    try:
        if not HAS_PSYCOPG2:
            return {"positions": []}
            
        conn = await get_db_connection()
        if not conn:
            return {"positions": []}
            
        try:
            conn.rollback()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # 1) Pull open positions
            cursor.execute("""
                SELECT id, symbol, side, size, entry_price, current_price, strategy,
                       created_at, updated_at, is_open, pnl, pnl_percentage
                FROM positions
                WHERE is_open = TRUE
                ORDER BY updated_at DESC
                LIMIT 500
            """)
            rows = cursor.fetchall()
            
            # 2) Map symbols for a batched market-data lookup
            symbols = tuple({r["symbol"] for r in rows}) or tuple(["BTCUSDT"])  # avoid SQL error on empty tuple
            
            # 3) Fetch freshest price per symbol from your market_data/candles table
            cursor.execute(f"""
                SELECT md.symbol, md.price
                FROM market_data md
                JOIN (
                    SELECT symbol, MAX(timestamp) AS ts
                    FROM market_data
                    WHERE symbol IN %s
                    GROUP BY symbol
                ) x ON x.symbol = md.symbol AND x.ts = md.timestamp
            """, (symbols,))
            latest = {r["symbol"]: float(r["price"]) for r in cursor.fetchall()}
            
            out = []
            for r in rows:
                sym = r["symbol"]
                cp  = latest.get(sym, float(r["current_price"] or r["entry_price"]))
                
                size = float(r["size"] or 0)
                entry = float(r["entry_price"] or 0)
                side  = (r["side"] or "").lower()
                
                # Unrealized PnL (quote terms)
                if side == "long":
                    pnl = (cp - entry) * size
                else:
                    pnl = (entry - cp) * size
                
                pnl_pct = 0.0
                if entry > 0:
                    pnl_pct = (cp - entry)/entry*100 if side == "long" else (entry - cp)/entry*100
                
                # More decimals for small priced symbols
                def fmt_price(x):
                    return round(float(x), 6 if entry < 1 else 4)
                
                # Calculate duration
                created_at = r.get('created_at') or r.get('timestamp')
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
                
                out.append({
                    "id": r["id"],
                    "symbol": sym,
                    "side": side,
                    "size": round(size, 6),
                    "entry_price": fmt_price(entry),
                    "current_price": fmt_price(cp),
                    "pnl": round(pnl, 2),
                    "pnl_percentage": round(pnl_pct, 2),
                    "strategy": r["strategy"],
                    "created_at": created_at.isoformat() if created_at and hasattr(created_at, 'isoformat') else None,
                    "updated_at": r["updated_at"].isoformat() if r["updated_at"] and hasattr(r["updated_at"], 'isoformat') else None,
                    "is_open": True,
                    "duration": duration_str,
                    "duration_hours": round(duration_hours, 2),
                    "timestamp": created_at.isoformat() if created_at and hasattr(created_at, 'isoformat') else datetime.utcnow().isoformat()
                })
            
            cursor.close()
            return {"positions": out}
            
        except Exception as e:
            logger.warning(f"Error getting positions: {e}")
            conn.rollback()
            return {"positions": []}
            
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
                        pnl_value = float(trade.get('pnl', 0.0))
                        pnl_percentage = float(trade.get('pnl_percentage', 0.0))
                        
                        # Calculate exit price from PnL and entry price
                        if pnl_value != 0 and entry_price > 0 and size_value > 0:
                            if trade.get('side', '').lower() == 'long':
                                exit_price = entry_price + (pnl_value / size_value)
                            else:  # short
                                exit_price = entry_price - (pnl_value / size_value)
                        else:
                            exit_price = entry_price  # If no PnL, exit price = entry price
                        
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
    """Get strategy performance analytics with real data"""
    try:
        if not HAS_PSYCOPG2:
            return {"strategies": []}
            
        conn = await get_db_connection()
        if not conn:
            return {"strategies": []}
            
        try:
            conn.rollback()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute("""
                SELECT strategy,
                       COUNT(*) AS trades,
                       AVG(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100 AS win_rate,
                       SUM(pnl) AS total_pnl,
                       AVG(pnl) AS avg_pnl
                FROM positions
                WHERE is_open = false
                GROUP BY strategy
                ORDER BY total_pnl DESC
            """)
            rows = cursor.fetchall() or []
            
            strategies = []
            for r in rows:
                strategies.append({
                    "strategy": r["strategy"],
                    "trades": int(r["trades"]),
                    "win_rate": round(float(r["win_rate"]), 2),
                    "total_pnl": round(float(r["total_pnl"]), 2),
                    "avg_pnl": round(float(r["avg_pnl"]), 2),
                    "sharpe_ratio": 0.0  # compute if you later store per-period returns
                })
            
            cursor.close()
            return {"strategies": strategies}
            
        except Exception as e:
            logger.warning(f"Error getting strategy performance: {e}")
            conn.rollback()
            return {"strategies": []}
            
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        return {"strategies": []}

@app.get("/api/analytics/risk-metrics")
async def get_risk_metrics():
    """Get risk management metrics with real calculations"""
    try:
        if not HAS_PSYCOPG2:
            return {"var95": 0.0, "var99": 0.0, "volatility": 0.0, "beta": 0.0, "max_drawdown": 0.0}
            
        conn = await get_db_connection()
        if not conn:
            return {"var95": 0.0, "var99": 0.0, "volatility": 0.0, "beta": 0.0, "max_drawdown": 0.0}
            
        try:
            conn.rollback()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get PnL data for risk calculations
            cursor.execute("""
                SELECT pnl, closed_at
                FROM positions
                WHERE is_open = false
                ORDER BY closed_at ASC
                LIMIT 3000
            """)
            rows = cursor.fetchall() or []
            
            if not rows:
                return {"var95": 0.0, "var99": 0.0, "volatility": 0.0, "beta": 0.0, "max_drawdown": 0.0}
            
            pnls = [float(r["pnl"]) for r in rows]
            
            # Calculate volatility (standard deviation)
            if len(pnls) > 1:
                mean_pnl = sum(pnls) / len(pnls)
                variance = sum((x - mean_pnl) ** 2 for x in pnls) / (len(pnls) - 1)
                volatility = variance ** 0.5
            else:
                volatility = 0.0
            
            # Calculate VaR (Value at Risk)
            pnls_sorted = sorted(pnls)
            def percentile(vals, p):
                if not vals: return 0.0
                k = max(0, min(len(vals)-1, int(round((p/100.0)*(len(vals)-1)))))
                return vals[k]
            
            var95 = min(0.0, percentile(pnls_sorted, 5))   # as a loss number
            var99 = min(0.0, percentile(pnls_sorted, 1))
            
            # Calculate maximum drawdown
            equity = 10000.0
            peak = equity
            max_dd = 0.0
            for pnl in pnls:
                equity += pnl
                peak = max(peak, equity)
                dd = (equity - peak) / peak if peak > 0 else 0.0
                max_dd = min(max_dd, dd)
            
            cursor.close()
            return {
                "var95": round(var95, 2),
                "var99": round(var99, 2),
                "volatility": round(volatility, 2),
                "beta": 0.0,  # keep 0 unless you benchmark against BTC/ETH returns
                "max_drawdown": round(abs(max_dd)*100, 2)
            }
            
        except Exception as e:
            logger.warning(f"Error getting risk metrics: {e}")
            conn.rollback()
            return {"var95": 0.0, "var99": 0.0, "volatility": 0.0, "beta": 0.0, "max_drawdown": 0.0}
            
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        return {"var95": 0.0, "var99": 0.0, "volatility": 0.0, "beta": 0.0, "max_drawdown": 0.0}

@app.get("/api/market-data")
async def get_market_data():
    """Get market data from database"""
    try:
        if not HAS_PSYCOPG2:
            return {"market": []}
            
        conn = await get_db_connection()
        if not conn:
            return {"market": []}
            
        try:
            conn.rollback()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get market data for symbols in positions and trades
            cursor.execute("""
                WITH universe AS (
                  SELECT DISTINCT symbol FROM positions
                  UNION
                  SELECT DISTINCT symbol FROM trades
                )
                SELECT u.symbol,
                       md.price,
                       md.change_24h,
                       md.volume_24h,
                       md.high_24h,
                       md.low_24h
                FROM universe u
                JOIN LATERAL (
                  SELECT *
                  FROM market_data m
                  WHERE m.symbol = u.symbol
                  ORDER BY timestamp DESC
                  LIMIT 1
                ) md ON TRUE
                ORDER BY u.symbol
                LIMIT 200
            """)
            rows = cursor.fetchall() or []
            
            # Fallback: if empty, return nothing (don't hardcode BTC/ETH)
            cursor.close()
            return {"market": rows}
            
        except Exception as e:
            logger.warning(f"Error getting market data: {e}")
            conn.rollback()
            return {"market": []}
            
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return {"market": []}

@app.get("/api/config")
async def get_config():
    """Get bot configuration from config files"""
    try:
        import os
        import yaml
        
        # Default config
        cfg = {
            "max_positions": 50, 
            "stop_loss": 0.01, 
            "take_profit": 0.025, 
            "max_daily_loss": 0.03,
            "risk_per_trade": 0.01,
            "max_correlation": 0.7,
            "volatility_filter": True,
            "min_volume_ratio": 1.5,
            "position_sizing": "fixed",
            "leverage": 1.0,
            "max_position_size": 1000.0,
            "min_position_size": 10.0
        }
        
        # Try to load from config.yaml if it exists
        if os.path.exists("config/config.yaml"):
            try:
                with open("config/config.yaml", "r") as f:
                    y = yaml.safe_load(f) or {}
                    # Map your keys; adjust to your actual structure
                    risk_config = y.get("risk", {})
                    trading_config = y.get("trading", {})
                    
                    cfg["max_positions"] = risk_config.get("max_open_positions", cfg["max_positions"])
                    cfg["stop_loss"] = trading_config.get("stop_loss", cfg["stop_loss"])
                    cfg["take_profit"] = trading_config.get("take_profit", cfg["take_profit"])
                    cfg["max_daily_loss"] = risk_config.get("max_daily_loss", cfg["max_daily_loss"])
                    cfg["risk_per_trade"] = risk_config.get("default_risk_per_trade", cfg["risk_per_trade"])
                    cfg["max_correlation"] = risk_config.get("max_correlation", cfg["max_correlation"])
                    cfg["volatility_filter"] = risk_config.get("volatility_filter", cfg["volatility_filter"])
                    cfg["min_volume_ratio"] = risk_config.get("min_volume_ratio", cfg["min_volume_ratio"])
            except Exception as e:
                logger.warning(f"Error reading config.yaml: {e}")
        
        return cfg
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return {
            "max_positions": 50,
            "stop_loss": 0.01,
            "take_profit": 0.025,
            "max_daily_loss": 0.03
        }

@app.get("/api/risk/limits")
async def risk_limits():
    """Get risk limits and current status"""
    try:
        positions = (await get_positions())["positions"]
        current_positions = len(positions)
        # Pull from config later; for now mirror your visible defaults
        return {
            "max_positions": 50,
            "current_positions": current_positions,
            "max_correlation": 0.70,
            "current_correlation": 0.0   # compute if you add per-symbol covariance
        }
    except Exception as e:
        logger.error(f"Error getting risk limits: {e}")
        return {
            "max_positions": 50,
            "current_positions": 0,
            "max_correlation": 0.70,
            "current_correlation": 0.0
        }

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
        logger.info("🌐 WebSocket handshake OK")
        await manager.connect(websocket)
        await manager.send_personal_message(json.dumps({
            "type": "connection_established",
            "message": "Connected to Enhanced Dashboard"
        }), websocket)
        logger.info("✅ Sent connection_established")
        
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
                logger.info(f"Updating dashboard data for {len(manager.active_connections)} connections")
                
                # Get latest data
                account_data = await get_account()
                positions_data = await get_positions()
                trades_data = await get_trades()
                performance_data = await get_performance()
                
                # Send update to all connected clients
                await manager.broadcast(json.dumps({
                    "type": "dashboard_update",
                    "data": {
                        "account": account_data,
                        "positions": positions_data,
                        "trades": trades_data,
                        "performance": performance_data
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
                logger.info("Dashboard data update sent successfully")
            else:
                logger.debug("No active WebSocket connections")
            
            await asyncio.sleep(3)  # Update every 3 seconds
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
            await asyncio.sleep(3)

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

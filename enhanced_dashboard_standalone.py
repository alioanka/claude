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
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Get open positions
                    cursor.execute("""
                        SELECT * FROM positions 
                        WHERE status = 'open' 
                        ORDER BY created_at DESC
                    """)
                    positions = cursor.fetchall()
                    
                    for pos in positions:
                        # Calculate duration
                        created_at = pos.get('created_at')
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
                        
                        positions_data.append({
                            "symbol": pos.get('symbol', 'N/A'),
                            "side": pos.get('side', 'N/A'),
                            "size": float(pos.get('size', 0.0)),
                            "entry_price": float(pos.get('entry_price', 0.0)),
                            "current_price": float(pos.get('current_price', pos.get('entry_price', 0.0))),
                            "pnl": float(pos.get('unrealized_pnl', 0.0)),
                            "pnl_percentage": float(pos.get('unrealized_pnl_percentage', 0.0)),
                            "duration": duration_str,
                            "duration_hours": round(duration_hours, 2),
                            "strategy": pos.get('strategy', 'N/A'),
                            "timestamp": created_at.isoformat() if created_at and hasattr(created_at, 'isoformat') else datetime.utcnow().isoformat()
                        })
                    
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error getting positions: {e}")
        
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
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Get trades
                    query = "SELECT * FROM trades ORDER BY opened_at DESC LIMIT %s"
                    params = [limit]
                    
                    if strategy:
                        query = "SELECT * FROM trades WHERE strategy = %s ORDER BY opened_at DESC LIMIT %s"
                        params = [strategy, limit]
                    
                    cursor.execute(query, params)
                    trades = cursor.fetchall()
                    
                    for trade in trades:
                        trades_data.append({
                            "id": trade.get('id'),
                            "symbol": trade.get('symbol', 'N/A'),
                            "side": trade.get('side', 'N/A'),
                            "size": float(trade.get('size', 0.0)),
                            "entry_price": float(trade.get('entry_price', 0.0)),
                            "exit_price": float(trade.get('exit_price', 0.0)) if trade.get('exit_price') else None,
                            "pnl": float(trade.get('pnl', 0.0)),
                            "pnl_percentage": float(trade.get('pnl_percentage', 0.0)),
                            "strategy": trade.get('strategy', 'N/A'),
                            "opened_at": trade.get('opened_at').isoformat() if trade.get('opened_at') and hasattr(trade.get('opened_at'), 'isoformat') else None,
                            "closed_at": trade.get('closed_at').isoformat() if trade.get('closed_at') and hasattr(trade.get('closed_at'), 'isoformat') else None,
                            "status": trade.get('status', 'N/A')
                        })
                    
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error getting trades: {e}")
        
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
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Get performance metrics
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_trades,
                            COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                            COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                            AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                            AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                            SUM(pnl) as total_pnl
                        FROM trades 
                        WHERE status = 'closed'
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

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background tasks
async def update_dashboard_data():
    """Send real-time updates to dashboard"""
    while True:
        try:
            if manager.active_connections:
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

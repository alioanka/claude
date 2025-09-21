#!/usr/bin/env python3
"""
Simple Enhanced Dashboard - ClaudeBot
Compatible version that works with existing setup
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
    from pydantic import BaseModel
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: pip install fastapi uvicorn")
    sys.exit(1)

# Import ClaudeBot modules
try:
    from config.config import config
    from data.database import DatabaseManager
    from core.portfolio_manager import PortfolioManager
    from core.exchange_manager import ExchangeManager
    from core.trade_executor import TradeExecutor
    from utils.logger import setup_logger
except ImportError as e:
    print(f"❌ Error importing ClaudeBot modules: {e}")
    print("Make sure you're running from the ClaudeBot directory")
    sys.exit(1)

# Configure logging
logger = setup_logger(__name__)

# Pydantic models
class TradingConfig(BaseModel):
    trading_mode: str
    risk_level: str
    max_positions: int
    max_daily_loss: float
    max_position_size: float

class PositionCloseRequest(BaseModel):
    symbol: str
    reason: str = "manual"

# Global variables
db_manager = None
portfolio_manager = None
exchange_manager = None
trade_executor = None
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

# Initialize components
async def initialize_components():
    """Initialize all dashboard components"""
    global db_manager, portfolio_manager, exchange_manager, trade_executor
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(config.database.url)
        await db_manager.initialize()
        
        # Initialize exchange manager
        exchange_manager = ExchangeManager()
        await exchange_manager.initialize()
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager(exchange_manager, config.trading.initial_capital)
        await portfolio_manager.initialize()
        
        # Initialize trade executor
        trade_executor = TradeExecutor(exchange_manager, portfolio_manager, db_manager)
        await trade_executor.initialize()
        
        logger.info("✅ All dashboard components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize dashboard components: {e}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize dashboard on startup"""
    try:
        await initialize_components()
        
        # Start background tasks
        asyncio.create_task(update_dashboard_data())
        asyncio.create_task(update_market_data())
        
        logger.info("🚀 Enhanced Dashboard started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard: {e}")
        # Don't exit, just log the error

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
        if not portfolio_manager:
            return {
                "total_balance": 0,
                "available_balance": 0,
                "used_balance": 0,
                "unrealized_pnl": 0,
                "realized_pnl": 0,
                "total_pnl": 0,
                "daily_pnl": 0,
                "weekly_pnl": 0,
                "monthly_pnl": 0,
                "total_return": 0,
                "active_positions": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "initial_capital": config.trading.initial_capital,
                "last_update": datetime.utcnow().isoformat()
            }
        
        # Get portfolio status
        portfolio_status = await portfolio_manager.get_portfolio_status()
        
        return portfolio_status
        
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return {
            "total_balance": config.trading.initial_capital,
            "available_balance": config.trading.initial_capital,
            "used_balance": 0,
            "unrealized_pnl": 0,
            "realized_pnl": 0,
            "total_pnl": 0,
            "daily_pnl": 0,
            "weekly_pnl": 0,
            "monthly_pnl": 0,
            "total_return": 0,
            "active_positions": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "initial_capital": config.trading.initial_capital,
            "last_update": datetime.utcnow().isoformat()
        }

@app.get("/api/positions")
async def get_positions():
    """Get current positions with enhanced data"""
    try:
        if not portfolio_manager:
            return {"positions": []}
        
        # Get positions from portfolio manager
        positions = await portfolio_manager.get_positions()
        
        # Convert to dictionary format
        positions_data = []
        for symbol, position in positions.items():
            pos_dict = position.to_dict()
            positions_data.append(pos_dict)
        
        return {"positions": positions_data}
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {"positions": []}

@app.get("/api/trades")
async def get_trades(limit: int = 100, strategy: str = None):
    """Get recent trades with filtering"""
    try:
        if not db_manager:
            return {"trades": []}
        
        # Get trades from database
        trades = await db_manager.get_trades(limit=limit)
        
        # Filter by strategy if specified
        if strategy:
            trades = [t for t in trades if t.get('strategy') == strategy]
        
        # Convert to JSON-serializable format
        for trade in trades:
            if 'opened_at' in trade and trade['opened_at']:
                trade['opened_at'] = trade['opened_at'].isoformat()
            if 'closed_at' in trade and trade['closed_at']:
                trade['closed_at'] = trade['closed_at'].isoformat()
        
        return {"trades": trades}
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return {"trades": []}

@app.get("/api/performance")
async def get_performance():
    """Get comprehensive performance metrics"""
    try:
        if not portfolio_manager:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "current_equity": config.trading.initial_capital,
                "total_balance": config.trading.initial_capital,
                "available_balance": config.trading.initial_capital,
                "total_return": 0,
                "active_positions": 0,
                "initial_balance": config.trading.initial_capital,
                "daily_pnl": 0,
                "weekly_pnl": 0,
                "monthly_pnl": 0
            }
        
        # Get performance data
        performance = await portfolio_manager.get_performance_metrics()
        
        return performance
        
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "current_equity": config.trading.initial_capital,
            "total_balance": config.trading.initial_capital,
            "available_balance": config.trading.initial_capital,
            "total_return": 0,
            "active_positions": 0,
            "initial_balance": config.trading.initial_capital,
            "daily_pnl": 0,
            "weekly_pnl": 0,
            "monthly_pnl": 0
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

async def update_market_data():
    """Update market data periodically"""
    while True:
        try:
            if exchange_manager:
                # Update market data for all trading pairs
                for symbol in config.trading_pairs[:10]:
                    try:
                        await exchange_manager.get_ticker(symbol)
                    except Exception as e:
                        logger.warning(f"Failed to update market data for {symbol}: {e}")
            
            await asyncio.sleep(60)  # Update every minute
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_dashboard_simple:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )

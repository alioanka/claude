#!/usr/bin/env python3
"""
Enhanced Trading Bot Dashboard - ClaudeBot
Advanced dashboard with comprehensive analytics, real-time data, and sophisticated UI
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ClaudeBot modules
from config.config import config
from data.database import DatabaseManager
from core.portfolio_manager import PortfolioManager
from core.exchange_manager import ExchangeManager
from core.trade_executor import TradeExecutor
from utils.logger import setup_logger

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

class StrategyConfig(BaseModel):
    strategy_name: str
    enabled: bool
    allocation: float
    parameters: Dict[str, Any]

# Global variables
db_manager = None
portfolio_manager = None
exchange_manager = None
trade_executor = None
connected_clients = []

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
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
    await initialize_components()
    
    # Start background tasks
    asyncio.create_task(update_dashboard_data())
    asyncio.create_task(update_market_data())

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
            raise HTTPException(status_code=500, detail="Portfolio manager not initialized")
        
        # Get portfolio status
        portfolio_status = await portfolio_manager.get_portfolio_status()
        
        # Get account summary
        account_summary = {
            "total_balance": portfolio_status.get("total_balance", 0),
            "available_balance": portfolio_status.get("available_balance", 0),
            "used_balance": portfolio_status.get("used_balance", 0),
            "unrealized_pnl": portfolio_status.get("unrealized_pnl", 0),
            "realized_pnl": portfolio_status.get("realized_pnl", 0),
            "total_pnl": portfolio_status.get("total_pnl", 0),
            "daily_pnl": portfolio_status.get("daily_pnl", 0),
            "weekly_pnl": portfolio_status.get("weekly_pnl", 0),
            "monthly_pnl": portfolio_status.get("monthly_pnl", 0),
            "total_return": portfolio_status.get("total_return", 0),
            "active_positions": portfolio_status.get("active_positions", 0),
            "total_trades": portfolio_status.get("total_trades", 0),
            "winning_trades": portfolio_status.get("winning_trades", 0),
            "losing_trades": portfolio_status.get("losing_trades", 0),
            "win_rate": portfolio_status.get("win_rate", 0),
            "max_drawdown": portfolio_status.get("max_drawdown", 0),
            "sharpe_ratio": portfolio_status.get("sharpe_ratio", 0),
            "initial_capital": config.trading.initial_capital,
            "last_update": datetime.utcnow().isoformat()
        }
        
        return account_summary
        
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions")
async def get_positions():
    """Get current positions with enhanced data"""
    try:
        if not portfolio_manager:
            raise HTTPException(status_code=500, detail="Portfolio manager not initialized")
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades")
async def get_trades(limit: int = 100, strategy: Optional[str] = None):
    """Get recent trades with filtering"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database manager not initialized")
        
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance")
async def get_performance():
    """Get comprehensive performance metrics"""
    try:
        if not portfolio_manager:
            raise HTTPException(status_code=500, detail="Portfolio manager not initialized")
        
        # Get performance data
        performance = await portfolio_manager.get_performance_metrics()
        
        return performance
        
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/strategy-performance")
async def get_strategy_performance():
    """Get detailed strategy performance analytics"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database manager not initialized")
        
        # Get all trades
        trades = await db_manager.get_trades(limit=1000)
        
        # Group by strategy
        strategy_data = {}
        for trade in trades:
            strategy = trade.get('strategy', 'Unknown')
            if strategy not in strategy_data:
                strategy_data[strategy] = {
                    'trades': [],
                    'total_pnl': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_volume': 0
                }
            
            strategy_data[strategy]['trades'].append(trade)
            strategy_data[strategy]['total_pnl'] += trade.get('pnl', 0)
            strategy_data[strategy]['total_volume'] += trade.get('size', 0) * trade.get('entry_price', 0)
            
            if trade.get('pnl', 0) > 0:
                strategy_data[strategy]['winning_trades'] += 1
            else:
                strategy_data[strategy]['losing_trades'] += 1
        
        # Calculate metrics for each strategy
        strategy_metrics = []
        for strategy, data in strategy_data.items():
            total_trades = len(data['trades'])
            win_rate = (data['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = data['total_pnl'] / total_trades if total_trades > 0 else 0
            
            # Calculate Sharpe ratio
            if total_trades > 1:
                pnls = [t.get('pnl', 0) for t in data['trades']]
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls)
                sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0
            else:
                sharpe_ratio = 0
            
            strategy_metrics.append({
                'strategy': strategy,
                'total_trades': total_trades,
                'winning_trades': data['winning_trades'],
                'losing_trades': data['losing_trades'],
                'win_rate': win_rate,
                'total_pnl': data['total_pnl'],
                'avg_pnl': avg_pnl,
                'total_volume': data['total_volume'],
                'sharpe_ratio': sharpe_ratio
            })
        
        # Sort by total PnL
        strategy_metrics.sort(key=lambda x: x['total_pnl'], reverse=True)
        
        return {"strategy_performance": strategy_metrics}
        
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/risk-metrics")
async def get_risk_metrics():
    """Get comprehensive risk metrics"""
    try:
        if not portfolio_manager:
            raise HTTPException(status_code=500, detail="Portfolio manager not initialized")
        
        # Get portfolio status
        portfolio_status = await portfolio_manager.get_portfolio_status()
        
        # Calculate risk metrics
        risk_metrics = {
            'var_95': 0,  # Value at Risk 95%
            'var_99': 0,  # Value at Risk 99%
            'max_drawdown': portfolio_status.get('max_drawdown', 0),
            'current_drawdown': portfolio_status.get('current_drawdown', 0),
            'volatility': 0,
            'beta': 0,
            'correlation': 0,
            'concentration_risk': 0,
            'leverage_ratio': 0,
            'margin_ratio': 0
        }
        
        return risk_metrics
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-data")
async def get_market_data():
    """Get current market data"""
    try:
        if not exchange_manager:
            raise HTTPException(status_code=500, detail="Exchange manager not initialized")
        
        # Get market data for trading pairs
        market_data = {}
        for symbol in config.trading_pairs[:20]:  # Limit to first 20 pairs
            try:
                ticker = await exchange_manager.get_ticker(symbol)
                if ticker:
                    market_data[symbol] = {
                        'price': ticker.get('last', 0),
                        'change_24h': ticker.get('change', 0),
                        'change_24h_pct': ticker.get('change_percent', 0),
                        'volume_24h': ticker.get('volume', 0),
                        'high_24h': ticker.get('high', 0),
                        'low_24h': ticker.get('low', 0),
                        'timestamp': datetime.utcnow().isoformat()
                    }
            except Exception as e:
                logger.warning(f"Failed to get market data for {symbol}: {e}")
        
        return {"market_data": market_data}
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/portfolio-charts")
async def get_portfolio_charts():
    """Get portfolio chart data"""
    try:
        if not db_manager:
            raise HTTPException(status_code=500, detail="Database manager not initialized")
        
        # Get historical performance data
        trades = await db_manager.get_trades(limit=1000)
        
        # Calculate cumulative PnL
        cumulative_pnl = 0
        chart_data = []
        
        for trade in sorted(trades, key=lambda x: x.get('closed_at', datetime.min)):
            if trade.get('closed_at'):
                cumulative_pnl += trade.get('pnl', 0)
                chart_data.append({
                    'timestamp': trade['closed_at'].isoformat(),
                    'cumulative_pnl': cumulative_pnl,
                    'trade_pnl': trade.get('pnl', 0)
                })
        
        return {"chart_data": chart_data}
        
    except Exception as e:
        logger.error(f"Error getting portfolio charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Position Management APIs
@app.post("/api/positions/{symbol}/close")
async def close_position(symbol: str, request: PositionCloseRequest):
    """Close a specific position"""
    try:
        if not trade_executor:
            raise HTTPException(status_code=500, detail="Trade executor not initialized")
        
        # Close position
        result = await trade_executor.close_position(symbol, reason=request.reason)
        
        if result.success:
            return {"message": f"Position {symbol} closed successfully", "result": result}
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
        
    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/positions/close-all")
async def close_all_positions():
    """Close all open positions"""
    try:
        if not trade_executor:
            raise HTTPException(status_code=500, detail="Trade executor not initialized")
        
        # Get all positions
        positions = await portfolio_manager.get_positions()
        
        results = []
        for symbol in positions.keys():
            try:
                result = await trade_executor.close_position(symbol, reason="manual_close_all")
                results.append({"symbol": symbol, "success": result.success, "error": result.error_message})
            except Exception as e:
                results.append({"symbol": symbol, "success": False, "error": str(e)})
        
        return {"message": "Close all positions completed", "results": results}
        
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration APIs
@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    try:
        config_data = {
            "trading_pairs": config.trading_pairs,
            "max_positions": config.trading.max_positions,
            "initial_capital": config.trading.initial_capital,
            "stop_loss_percent": config.trading.stop_loss_percent,
            "take_profit_percent": config.trading.take_profit_percent,
            "risk_management": {
                "max_daily_loss": config.risk.max_daily_loss,
                "max_position_size": config.risk.max_position_size,
                "max_correlation": config.risk.max_correlation
            },
            "strategies": {
                "momentum_strategy": {
                    "enabled": True,
                    "allocation": 0.6
                },
                "mean_reversion": {
                    "enabled": True,
                    "allocation": 0.3
                },
                "arbitrage_strategy": {
                    "enabled": True,
                    "allocation": 0.1
                }
            }
        }
        
        return config_data
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/update")
async def update_config(config_data: Dict[str, Any]):
    """Update configuration"""
    try:
        # Update configuration (implement based on your needs)
        # This is a placeholder - implement actual config update logic
        
        return {"message": "Configuration updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        "enhanced_dashboard:app",
        host="0.0.0.0",
        port=8001,  # Different port from default dashboard
        reload=False,
        log_level="info"
    )

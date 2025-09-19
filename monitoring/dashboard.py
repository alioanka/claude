"""
Real-time dashboard for monitoring trading bot performance and metrics.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
from data.database import DatabaseManager, Trade, Position
from core.portfolio_manager import PortfolioManager
#from fastapi import APIRouter

#router = APIRouter()
logger = logging.getLogger(__name__)

class DashboardManager:
    """Manages the web dashboard for monitoring bot performance"""
    
    def __init__(self, database_manager: DatabaseManager, portfolio_manager: PortfolioManager, trade_executor=None, bot=None):
        self.db = database_manager
        self.portfolio_manager = portfolio_manager
        self.trade_executor = trade_executor
        self.bot = bot
        self.app = FastAPI(title="Crypto Trading Bot Dashboard")
        self.active_connections: List[WebSocket] = []
        
        # Performance optimization: Cache for analytics data
        self.analytics_cache = {}
        self.cache_timeout = 30  # seconds
        self.last_cache_update = 0
        
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect(websocket)
            try:
                while True:
                    # Send real-time updates every 3 seconds for better responsiveness
                    data = await self.get_real_time_data()
                    await self.broadcast(data)
                    await asyncio.sleep(3)
            except WebSocketDisconnect:
                self.disconnect(websocket)
        
        @self.app.get("/api/dashboard", response_class=HTMLResponse)
        async def get_dashboard():
            """Get dashboard HTML"""
            return await self.get_dashboard_html()
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            return await self.get_dashboard_html()

        
        @self.app.get("/api/positions")
        async def get_positions():
            """Get current positions"""
            return await self.get_positions_data()
        
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "time": datetime.utcnow().isoformat()}

        
        @self.app.get("/api/performance")
        async def get_performance():
            """Get performance metrics"""
            return await self.get_performance_data()
        
        @self.app.get("/api/trades")
        async def get_recent_trades():
            """Get recent trades"""
            return await self.get_trades_data()
        
        @self.app.get("/api/signals")
        async def get_recent_signals():
            """Get recent signals"""
            return await self.get_signals_data()
        
        @self.app.get("/api/rejections")
        async def get_rejections():
            """Get recent signal/order rejections"""
            return await self.get_rejections_data()
        
        @self.app.get("/api/pnl-by-pairs")
        async def get_pnl_by_pairs():
            """Get PnL statistics grouped by trading pairs"""
            return await self.get_pnl_by_pairs_data()
        
        @self.app.get("/api/pnl-by-strategies")
        async def get_pnl_by_strategies():
            """Get PnL statistics grouped by strategies"""
            return await self.get_pnl_by_strategies_data()
        
        @self.app.get("/api/trade-statistics")
        async def get_trade_statistics():
            """Get comprehensive trade statistics"""
            return await self.get_trade_statistics_data()
        
        @self.app.get("/api/performance-charts")
        async def get_performance_charts():
            """Get performance chart data"""
            return await self.get_performance_charts_data()

        @self.app.post("/api/positions/{symbol}/close")
        async def api_close_position(symbol: str):
            """
            Close a single position via TradeExecutor if wired; otherwise
            return a clear error without breaking anything else.
            """
            try:
                if not getattr(self, "trade_executor", None):
                    return {"ok": False, "error": "trade_executor not wired to DashboardManager"}
                
                # 🔥 CRITICAL FIX: Refresh prices before closing to avoid "Position size has changed" error
                await self.portfolio_manager.update_position_prices()
                
                res = await self.trade_executor.close_position(symbol)
                return {"ok": bool(getattr(res, "success", False)), "error": getattr(res, "error_message", None)}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @self.app.post("/api/positions/close_all")
        async def api_close_all_positions():
            """
            Emergency exit: prefer bot.close_all_positions() if available,
            otherwise fall back to iterating positions with trade_executor.
            """
            try:
                if getattr(self, "bot", None) and hasattr(self.bot, "close_all_positions"):
                    await self.bot.close_all_positions()
                    return {"ok": True}

                if not getattr(self, "trade_executor", None):
                    return {"ok": False, "error": "trade_executor not wired to DashboardManager"}

                # 🔥 CRITICAL FIX: Refresh prices before closing all positions
                await self.portfolio_manager.update_position_prices()
                
                positions = await self.portfolio_manager.get_positions()
                for p in positions:
                    await self.trade_executor.close_position(p["symbol"])
                return {"ok": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}


    
    async def connect(self, websocket: WebSocket):
        """Accept websocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New dashboard connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove websocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Dashboard connection closed. Total: {len(self.active_connections)}")
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(data))
            except Exception as e:
                logger.warning(f"Failed to send data to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def get_real_time_data(self) -> Dict[str, Any]:
        """Get real-time data for dashboard"""
        try:
            positions = await self.get_positions_data()
            performance = await self.get_performance_data()
            portfolio = await self.portfolio_manager.get_portfolio_status()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'positions': positions,
                'performance': performance,
                'portfolio': portfolio,
                'status': 'running'
            }
        except Exception as e:
            logger.error(f"Failed to get real-time data: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'status': 'error'
            }
    
    async def get_positions_data(self) -> List[Dict[str, Any]]:
        """Get current positions with live PnL from PortfolioManager (authoritative)."""
        try:
            # ensure PM has fresh prices
            await self.portfolio_manager.update_position_prices()
            live_positions = getattr(self.portfolio_manager, "positions", {}) or {}

            out: List[Dict[str, Any]] = []
            for sym, pos in live_positions.items():
                # compute values from the in-memory object:
                item = {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "size": round(float(pos.amount), 6),     # dashboard expects 'size'
                    "entry_price": float(pos.entry_price),
                    "current_price": float(pos.current_price),
                    "pnl": round(float(pos.unrealized_pnl), 2),
                    "pnl_percentage": round(float(pos.unrealized_pnl_pct), 2),
                    "strategy": getattr(pos, "strategy", "unknown")
                }
                out.append(item)
            return out
        except Exception as e:
            logger.error(f"Failed to get positions data: {e}")
            return []


    
    async def get_performance_data(self) -> Dict[str, Any]:
        """Get performance metrics (DB if present, enriched with PortfolioManager stats)."""
        try:
            db_stats = await self.db.get_performance_stats() if hasattr(self.db, "get_performance_stats") else {}
            pm_stats = self.portfolio_manager.get_portfolio_statistics()

            # Prefer in-memory truth for dynamic metrics; fall back to DB if missing
            return {
                'portfolio_value': await self.portfolio_manager.get_total_value(),
                'daily_pnl': pm_stats.get('daily_pnl', 0.0),
                'total_pnl': getattr(self.portfolio_manager, 'total_pnl', 0.0),
                'total_trades': pm_stats.get('total_trades', db_stats.get('total_trades', 0)),
                'win_rate': pm_stats.get('win_rate_pct', 0.0),
                'last_updated': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get performance data: {e}")
            return {}

    
    # NORMALIZE KEYS FOR FRONTEND: add "total" alias
    async def get_trades_data(self, limit: int = 50) -> dict:
        # Since trades table is empty, get closed positions instead
        try:
            session = self.db.get_session()
            from sqlalchemy import text
            
            # Get recent closed positions as "trades"
            result = session.execute(text("""
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
                LIMIT :limit
            """), {"limit": limit})
            
            rows = result.fetchall()
            session.close()
            
            normalized = []
            for row in rows:
                # Convert row to dict format expected by frontend
                trade_data = {
                    "id": row[0],
                    "symbol": row[1],
                    "side": row[2],
                    "size": float(row[3]),
                    "price": float(row[4]),
                    "pnl": float(row[5]) if row[5] else 0.0,
                    "pnl_percentage": float(row[6]) if row[6] else 0.0,
                    "timestamp": row[7].isoformat() if row[7] else None,
                    "strategy": row[8] or "unknown",
                    "closed_at": row[9].isoformat() if row[9] else None,
                    "total_value": float(row[10]) if row[10] else 0.0,
                    "total": float(row[10]) if row[10] else 0.0  # Frontend expects "total"
                }
                normalized.append(trade_data)
            
            return {"trades": normalized}
            
        except Exception as e:
            logger.error(f"Failed to get trades data: {e}")
            # Fallback to original method
            rows = self.db.get_recent_trades(limit=limit)
            normalized = []
            for t in rows:
                total = t.get("total_value")
                if total is None:
                    sz = t.get("size") or 0
                    px = t.get("price") or 0
                    fee = t.get("fee") or 0
                    total = sz * px + fee
                normalized.append({
                    **t,
                    "total": float(total),
                })
            return {"trades": normalized}


        
        async def get_signals_data(self, limit: int = 50) -> List[Dict[str, Any]]:
            """Get recent signals data"""
            try:
                signals = await self.db.get_signals(limit=limit)
                return [signal.to_dict() for signal in signals]
            except Exception as e:
                logger.error(f"Failed to get signals data: {e}")
                return []
        
    # monitoring/dashboard.py
    async def get_rejections_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Return recent rejection events:
        1) If DatabaseManager exposes get_rejections(), use it.
        2) Else parse logs/trading_bot.log for lines starting with 'REJECTION | '.
        """
        try:
            # 1) DB path (if implemented)
            if hasattr(self.db, "get_rejections"):
                rows = await self.db.get_rejections(limit=limit)
                out = []
                for r in rows:
                    if hasattr(r, "to_dict"):
                        out.append(r.to_dict())
                    else:
                        out.append({
                            "timestamp": getattr(r, "timestamp", None),
                            "strategy": getattr(r, "strategy", None),
                            "symbol": getattr(r, "symbol", None),
                            "reason": getattr(r, "reason", None),
                            "confidence": float(getattr(r, "confidence", 0.0)),
                        })
                return out[:limit]

            # monitoring/dashboard.py (inside the log-fallback branch of get_rejections_data)

            # --- LOG FALLBACK with rotation awareness ---
            import os, re, glob
            from datetime import datetime, timezone

            log_files = sorted(
                glob.glob(os.path.join("logs", "trading_bot.log*")),
                key=lambda p: os.path.getmtime(p),
                reverse=True
            )
            if not log_files:
                return []

            pattern = re.compile(
                r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*REJECTION \| strategy=(?P<strategy>\S+) symbol=(?P<symbol>\S+) reason=(?P<reason>.+?) conf=(?P<conf>[\d\.]+)"
            )

            items = []
            for path in log_files:
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            m = pattern.search(line)
                            if not m:
                                continue
                            d = m.groupdict()
                            try:
                                dt = datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)
                                ts_iso = dt.isoformat()
                                ts_epoch = int(dt.timestamp() * 1000)
                            except Exception:
                                ts_iso, ts_epoch = d["ts"], None
                            items.append({
                                "timestamp": ts_iso,
                                "ts_epoch": ts_epoch,
                                "strategy": d["strategy"],
                                "symbol": d["symbol"],
                                "reason": d["reason"].strip(),
                                "confidence": float(d["conf"]),
                            })
                            if len(items) >= limit:
                                break
                except Exception as _e:
                    logger.warning(f"could not read {path}: {_e}")
                if len(items) >= limit:
                    break

            items.sort(key=lambda x: x.get("ts_epoch") or x["timestamp"], reverse=True)
            return items[:limit]



        except Exception as e:
            logger.error(f"Failed to retrieve rejections: {e}")
            return []

    
    async def get_daily_pnl(self) -> float:
        """Calculate daily PnL"""
        try:
            session = self.db.get_session()
            today = datetime.utcnow().date()
            
            positions = session.query(Position).filter(
                Position.closed_at >= today,
                Position.is_open == False
            ).all()
            
            daily_pnl = sum(pos.pnl for pos in positions)
            session.close()
            return daily_pnl
            
        except Exception as e:
            logger.error(f"Failed to calculate daily PnL: {e}")
            return 0.0
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid"""
        import time
        if cache_key not in self.analytics_cache:
            return False
        return (time.time() - self.analytics_cache[cache_key]['timestamp']) < self.cache_timeout
    
    def _get_cached_data(self, cache_key: str) -> Any:
        """Get cached data if valid"""
        if self._is_cache_valid(cache_key):
            return self.analytics_cache[cache_key]['data']
        return None
    
    def _set_cached_data(self, cache_key: str, data: Any) -> None:
        """Set cached data with timestamp"""
        import time
        self.analytics_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }

    async def get_pnl_by_pairs_data(self) -> Dict[str, Any]:
        """Get PnL statistics grouped by trading pairs"""
        cache_key = 'pnl_by_pairs'
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            session = self.db.get_session()
            from sqlalchemy import text
            
            # Get PnL data grouped by symbol
            result = session.execute(text("""
                SELECT 
                    symbol,
                    COUNT(*) as trade_count,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as win_count,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as loss_count,
                    MAX(pnl) as max_pnl,
                    MIN(pnl) as min_pnl
                FROM positions 
                WHERE is_open = false AND pnl IS NOT NULL
                GROUP BY symbol
                ORDER BY total_pnl DESC
            """))
            
            rows = result.fetchall()
            session.close()
            
            data = []
            for row in rows:
                win_rate = (row[4] / row[1] * 100) if row[1] > 0 else 0
                data.append({
                    "symbol": row[0],
                    "trade_count": row[1],
                    "total_pnl": float(row[2]) if row[2] else 0.0,
                    "avg_pnl": float(row[3]) if row[3] else 0.0,
                    "win_count": row[4],
                    "loss_count": row[5],
                    "win_rate": round(win_rate, 2),
                    "max_pnl": float(row[6]) if row[6] else 0.0,
                    "min_pnl": float(row[7]) if row[7] else 0.0
                })
            
            result_data = {"pairs": data}
            self._set_cached_data(cache_key, result_data)
            return result_data
            
        except Exception as e:
            logger.error(f"Failed to get PnL by pairs data: {e}")
            return {"pairs": []}
    
    async def get_pnl_by_strategies_data(self) -> Dict[str, Any]:
        """Get PnL statistics grouped by strategies"""
        cache_key = 'pnl_by_strategies'
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            session = self.db.get_session()
            from sqlalchemy import text
            
            # Get PnL data grouped by strategy
            result = session.execute(text("""
                SELECT 
                    strategy,
                    COUNT(*) as trade_count,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as win_count,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as loss_count,
                    MAX(pnl) as max_pnl,
                    MIN(pnl) as min_pnl
                FROM positions 
                WHERE is_open = false AND pnl IS NOT NULL AND strategy IS NOT NULL
                GROUP BY strategy
                ORDER BY total_pnl DESC
            """))
            
            rows = result.fetchall()
            session.close()
            
            data = []
            for row in rows:
                win_rate = (row[4] / row[1] * 100) if row[1] > 0 else 0
                data.append({
                    "strategy": row[0],
                    "trade_count": row[1],
                    "total_pnl": float(row[2]) if row[2] else 0.0,
                    "avg_pnl": float(row[3]) if row[3] else 0.0,
                    "win_count": row[4],
                    "loss_count": row[5],
                    "win_rate": round(win_rate, 2),
                    "max_pnl": float(row[6]) if row[6] else 0.0,
                    "min_pnl": float(row[7]) if row[7] else 0.0
                })
            
            result_data = {"strategies": data}
            self._set_cached_data(cache_key, result_data)
            return result_data
            
        except Exception as e:
            logger.error(f"Failed to get PnL by strategies data: {e}")
            return {"strategies": []}
    
    async def get_trade_statistics_data(self) -> Dict[str, Any]:
        """Get comprehensive trade statistics"""
        cache_key = 'trade_statistics'
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            session = self.db.get_session()
            from sqlalchemy import text
            
            # Get overall statistics
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as win_count,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as loss_count,
                    MAX(pnl) as max_pnl,
                    MIN(pnl) as min_pnl,
                    STDDEV(pnl) as pnl_stddev
                FROM positions 
                WHERE is_open = false AND pnl IS NOT NULL
            """))
            
            row = result.fetchone()
            session.close()
            
            if row and row[0] > 0:
                win_rate = (row[3] / row[0] * 100) if row[0] > 0 else 0
                return {
                    "total_trades": row[0],
                    "total_pnl": float(row[1]) if row[1] else 0.0,
                    "avg_pnl": float(row[2]) if row[2] else 0.0,
                    "win_count": row[3],
                    "loss_count": row[4],
                    "win_rate": round(win_rate, 2),
                    "max_pnl": float(row[5]) if row[5] else 0.0,
                    "min_pnl": float(row[6]) if row[6] else 0.0,
                    "pnl_stddev": float(row[7]) if row[7] else 0.0,
                    "profit_factor": abs(row[1] / abs(row[4])) if row[4] != 0 else 0
                }
            else:
                result_data = {
                    "total_trades": 0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "win_count": 0,
                    "loss_count": 0,
                    "win_rate": 0.0,
                    "max_pnl": 0.0,
                    "min_pnl": 0.0,
                    "pnl_stddev": 0.0,
                    "profit_factor": 0.0
                }
                self._set_cached_data(cache_key, result_data)
                return result_data
                
        except Exception as e:
            logger.error(f"Failed to get trade statistics: {e}")
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": 0.0,
                "max_pnl": 0.0,
                "min_pnl": 0.0,
                "pnl_stddev": 0.0,
                "profit_factor": 0.0
            }
    
    async def get_performance_charts_data(self) -> Dict[str, Any]:
        """Get performance chart data"""
        cache_key = 'performance_charts'
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            session = self.db.get_session()
            from sqlalchemy import text
            
            # Get daily PnL data for the last 30 days
            result = session.execute(text("""
                SELECT 
                    DATE(closed_at) as trade_date,
                    COUNT(*) as trade_count,
                    SUM(pnl) as daily_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as daily_wins
                FROM positions 
                WHERE is_open = false 
                AND closed_at >= CURRENT_DATE - INTERVAL '30 days'
                AND pnl IS NOT NULL
                GROUP BY DATE(closed_at)
                ORDER BY trade_date
            """))
            
            rows = result.fetchall()
            session.close()
            
            daily_data = []
            cumulative_pnl = 0.0
            
            for row in rows:
                cumulative_pnl += float(row[2]) if row[2] else 0.0
                daily_data.append({
                    "date": row[0].isoformat() if row[0] else None,
                    "trade_count": row[1],
                    "daily_pnl": float(row[2]) if row[2] else 0.0,
                    "daily_wins": row[3],
                    "cumulative_pnl": cumulative_pnl
                })
            
            result_data = {
                "daily_performance": daily_data,
                "total_days": len(daily_data)
            }
            self._set_cached_data(cache_key, result_data)
            return result_data
            
        except Exception as e:
            logger.error(f"Failed to get performance charts data: {e}")
            return {"daily_performance": [], "total_days": 0}
    
    async def get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Trading Bot Dashboard</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
                .container { max-width: 1200px; margin: 0 auto; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .metric-card { background: #2a2a2a; padding: 20px; border-radius: 8px; border: 1px solid #333; }
                .metric-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
                .metric-label { color: #ccc; margin-top: 5px; }
                .chart-container { background: #2a2a2a; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .status-running { color: #4CAF50; }
                .status-error { color: #f44336; }
                table { width: 100%; border-collapse: collapse; background: #2a2a2a; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #333; }
                th { background: #333; }
                .positive { color: #4CAF50; }
                .negative { color: #f44336; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Crypto Trading Bot Dashboard</h1>
                <div style="margin: 10px 0 20px;">
                  <button id="btnCloseAll" onclick="closeAllPositions()" style="padding:8px 12px; background:#f44336; color:#fff; border:none; border-radius:6px; cursor:pointer;">
                    Emergency Close All
                  </button>
                </div>


                <div id="status" class="status-running">Status: Running</div>
                
                <div class="metrics" id="metrics">
                    <!-- Metrics will be populated by JavaScript -->
                </div>
                


                <div class="chart-container">
                    <h3>Portfolio Value</h3>
                    <div style="margin-bottom:12px;">
                      <label for="tf">Timeframe:</label>
                      <select id="tf">
                        <option value="1m">1m</option>
                        <option value="5m" selected>5m</option>
                        <option value="15m">15m</option>
                        <option value="1h">1h</option>
                        <option value="1d">1d</option>
                        <option value="7d">7d</option>
                        <option value="all">All</option>
                      </select>
                    </div>
                    <canvas id="portfolioChart"></canvas>
                </div>

                
                <div class="chart-container">
                    <h3>Recent Positions</h3>
                    <table id="positionsTable">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Size</th>
                                <th>Entry Price</th>
                                <th>Current Price</th>
                                <th>PnL</th>
                                <th>PnL %</th>
                                <th>Duration</th>
                                <th>Strategy</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="positionsBody">
                            <!-- Positions will be populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
                
                <div class="chart-container">
                    <h3>Recent Trades</h3>
                    <table id="tradesTable">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Size</th>
                                <th>Price</th>
                                <th>Total</th>
                                <th>PnL</th>
                                <th>PnL %</th>
                                <th>Strategy</th>
                            </tr>
                        </thead>
                        <tbody id="tradesBody">
                            <!-- Trades will be populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
                <div class="chart-container">
                    <h3>Recent Rejections</h3>
                    <table id="rejectionsTable">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Strategy</th>
                                <th>Symbol</th>
                                <th>Reason</th>
                            </tr>
                        </thead>
                        <tbody id="rejectionsBody">
                            <!-- Rejections will be populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
                
                <!-- Enhanced Analytics Section -->
                <div class="chart-container">
                    <h3>📊 Trade Statistics</h3>
                    <div id="tradeStats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
                        <!-- Statistics will be populated by JavaScript -->
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                    <div class="chart-container">
                        <h3>🥧 PnL by Trading Pairs</h3>
                        <canvas id="pnlPairsChart" width="400" height="300"></canvas>
                    </div>
                    <div class="chart-container">
                        <h3>🎯 PnL by Strategies</h3>
                        <canvas id="pnlStrategiesChart" width="400" height="300"></canvas>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>📈 Daily Performance</h3>
                    <canvas id="dailyPerformanceChart" width="800" height="400"></canvas>
                </div>
                
                <div class="chart-container">
                    <h3>📋 Pairs Performance Details</h3>
                    <table id="pairsPerformanceTable">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Trades</th>
                                <th>Total PnL</th>
                                <th>Avg PnL</th>
                                <th>Win Rate</th>
                                <th>Max PnL</th>
                                <th>Min PnL</th>
                            </tr>
                        </thead>
                        <tbody id="pairsPerformanceBody">
                            <!-- Pairs performance will be populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
                
                <div class="chart-container">
                    <h3>🎯 Strategies Performance Details</h3>
                    <table id="strategiesPerformanceTable">
                        <thead>
                            <tr>
                                <th>Strategy</th>
                                <th>Trades</th>
                                <th>Total PnL</th>
                                <th>Avg PnL</th>
                                <th>Win Rate</th>
                                <th>Max PnL</th>
                                <th>Min PnL</th>
                            </tr>
                        </thead>
                        <tbody id="strategiesPerformanceBody">
                            <!-- Strategies performance will be populated by JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
            
            <script>
                const wsProto = (location.protocol === 'https:') ? 'wss' : 'ws';
                const ws = new WebSocket(`${wsProto}://${window.location.host}/ws`);

                let portfolioChart;
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                ws.onopen = function(event) {
                    console.log('Dashboard connected');
                };
                
                ws.onclose = function(event) {
                    console.log('Dashboard disconnected');
                    document.getElementById('status').className = 'status-error';
                    document.getElementById('status').textContent = 'Status: Disconnected';
                };




                
                function updateDashboard(data) {
                    if (data.error) {
                        document.getElementById('status').className = 'status-error';
                        document.getElementById('status').textContent = 'Status: Error - ' + data.error;
                        return;
                    }
                    
                    updateMetrics(data.performance, data.portfolio);
                    updatePositions(data.positions);
                    updateChart(data.portfolio);
                }
                
                function updateMetrics(performance, portfolio) {
                    const metricsHtml = `
                        <div class="metric-card">
                            <div class="metric-value">${(portfolio.total_value || 0).toFixed(2)}</div>
                            <div class="metric-label">Portfolio Value</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value ${(performance.daily_pnl || 0) >= 0 ? 'positive' : 'negative'}">
                                ${(performance.daily_pnl || 0).toFixed(2)}
                            </div>
                            <div class="metric-label">Daily PnL</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(performance.win_rate || 0).toFixed(1)}%</div>
                            <div class="metric-label">Win Rate</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${performance.total_trades || 0}</div>
                            <div class="metric-label">Total Trades</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(portfolio.open_positions || 0)}</div>
                            <div class="metric-label">Open Positions</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value ${(performance.total_pnl || 0) >= 0 ? 'positive' : 'negative'}">
                                ${(performance.total_pnl || 0).toFixed(2)}
                            </div>
                            <div class="metric-label">Total PnL</div>
                        </div>
                    `;
                    document.getElementById('metrics').innerHTML = metricsHtml;
                }
                
                function updatePositions(positions) {
                    const tbody = document.getElementById('positionsBody');
                    tbody.innerHTML = positions.map(pos => `
                        <tr>
                            <td>${pos.symbol}</td>
                            <td>${pos.side}</td>
                            <td>${pos.size}</td>
                            <td>${pos.entry_price.toFixed(4)}</td>
                            <td>${(pos.current_price || pos.entry_price).toFixed(4)}</td>
                            <td class="${pos.pnl >= 0 ? 'positive' : 'negative'}">${pos.pnl.toFixed(2)}</td>
                            <td class="${pos.pnl_percentage >= 0 ? 'positive' : 'negative'}">${pos.pnl_percentage.toFixed(2)}%</td>
                            <td title="${pos.duration_hours} hours">${pos.duration || 'N/A'}</td>
                            <td>${pos.strategy}</td>
                            <td><button onclick="closePosition('${pos.symbol}')" style="padding:6px 10px; background:#ff9800; color:#fff; border:none; border-radius:6px; cursor:pointer;">Close</button></td>
                        </tr>
                    `).join('');
                }
                



                // --- timeframe controls (non-breaking) ---
                // --- timeframe controls (fixed bucketing) ---
                let rawPoints = []; // [{ts: number, val: number}]

                // View windows per TF (how much history to show)
                const TF_WINDOW = {
                '1m':  5 * 60_000,      // show last 5 minutes
                '5m':  30 * 60_000,     // show last 30 minutes
                '15m': 2 * 60 * 60_000, // last 2 hours
                '1h':  12 * 60 * 60_000,// last 12 hours
                '1d':  3 * 24 * 60 * 60_000, // last 3 days
                '7d':  7 * 24 * 60 * 60_000, // last 7 days
                'all': Number.POSITIVE_INFINITY
                };

                // Bucket sizes per TF (should be ≪ window to avoid 1–2 labels)
                const TF_BUCKET = {
                '1m':  5_000,        // 5s buckets
                '5m':  15_000,       // 15s buckets
                '15m': 60_000,       // 1m buckets
                '1h':  5 * 60_000,   // 5m buckets
                '1d':  30 * 60_000,  // 30m buckets
                '7d':  2 * 60 * 60_000, // 2h buckets
                'all': 0             // no bucketing for 'all'
                };

                function resampleAndRender(tf) {
                const now = Date.now();
                const windowMs = TF_WINDOW[tf] ?? 30 * 60_000;  // default 30m
                const bucketMs = TF_BUCKET[tf] ?? 15_000;       // default 15s

                // 1) filter to timeframe
                let filtered = (tf === 'all')
                    ? rawPoints.slice()
                    : rawPoints.filter(p => (now - p.ts) <= windowMs);

                // 2) bucketize: keep the LAST point in each time bucket
                let out;
                if (bucketMs > 0 && filtered.length > 1) {
                    const map = new Map();
                    for (const p of filtered) {
                    const k = Math.floor(p.ts / bucketMs);
                    // keep the latest sample in each bucket
                    const prev = map.get(k);
                    if (!prev || p.ts > prev.ts) map.set(k, p);
                    }
                    out = Array.from(map.values()).sort((a,b) => a.ts - b.ts);
                } else {
                    out = filtered.slice();
                }

                // 3) ensure ≥2 points for a visible line
                if (out.length === 1) {
                    out = [{ ts: out[0].ts - Math.max(5_000, bucketMs || 5_000), val: out[0].val }, out[0]];
                } else if (out.length === 0 && rawPoints.length) {
                    const last = rawPoints[rawPoints.length - 1];
                    out = [{ ts: last.ts - Math.max(5_000, bucketMs || 5_000), val: last.val }, last];
                }

                // 4) apply to chart
                const labels = out.map(p => new Date(p.ts).toLocaleTimeString());
                const data = out.map(p => p.val);
                portfolioChart.data.labels = labels;
                portfolioChart.data.datasets[0].data = data;

                // 5) sensible y-range padding
                if (data.length) {
                    const min = Math.min(...data), max = Math.max(...data);
                    const pad = Math.max(1, (max - min) * 0.002);
                    portfolioChart.options.scales.y.suggestedMin = min - pad;
                    portfolioChart.options.scales.y.suggestedMax = max + pad;
                }

                portfolioChart.update();
                }




                // apply on timeframe change
                document.addEventListener('change', (e) => {
                    if (e.target && e.target.id === 'tf') {
                        const v = e.target.value;
                        dataWindowMs = (v === 'all') ? Infinity : (TF_BUCKETS[v] || 5*60e3);
                        resampleAndRender(v);
                    }
                });


                function updateChart(portfolio) {
                    // init chart once (unchanged look)
                    if (!portfolioChart) {
                        const ctx = document.getElementById('portfolioChart').getContext('2d');
                        portfolioChart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: [],
                                datasets: [{
                                    label: 'Portfolio Value',
                                    data: [],
                                    borderColor: '#4CAF50',
                                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                    tension: 0.1
                                }]
                            },
                            options: {
                                responsive: true,
                                plugins: { legend: { labels: { color: '#fff' } } },
                                scales: {
                                    x: { ticks: { color: '#ccc' } },
                                    y: { ticks: { color: '#ccc' } }
                                }
                            }
                        });
                    }

                    // guard: if backend gave nothing, don’t add zeros
                    if (!portfolio || typeof portfolio !== 'object') return;

                    // take timestamp + value from payload; harden to numbers
                    const tsIso = portfolio.timestamp || new Date().toISOString();
                    const ts = Date.parse(tsIso);
                    // prefer total_value, fallback to nested performance if ever present
                    const val = Number(
                        (portfolio.total_value != null ? portfolio.total_value : (portfolio.performance && portfolio.performance.portfolio_value))
                    );

                    if (!Number.isFinite(ts) || !Number.isFinite(val)) {
                        // bad data → skip without breaking the rest of the dashboard
                        return;
                    }

                    // push to raw buffer, keep a reasonable cap (e.g., last 10k points)
                    rawPoints.push({ ts, val });
                    if (rawPoints.length > 10000) rawPoints.shift();


                    // draw according to selected timeframe
                    const sel = document.getElementById('tf');
                    resampleAndRender(sel ? sel.value : 'all');
        
                }



                async function closePosition(symbol) {
                  try {
                    const res = await fetch(`/api/positions/${encodeURIComponent(symbol)}/close`, { method: 'POST' });
                    const j = await res.json();
                    if (j.ok) {
                      alert(`Close requested for ${symbol}`);
                    } else {
                      alert(`Close failed: ${j.error || 'unknown error'}`);
                    }
                  } catch (e) {
                    alert('Close failed: ' + e.message);
                  }
                }

                async function closeAllPositions() {
                  try {
                    const res = await fetch('/api/positions/close_all', { method: 'POST' });
                    const j = await res.json();
                    if (j.ok) {
                      alert('Emergency close requested for all positions');
                    } else {
                      alert(`Close all failed: ${j.error || 'unknown error'}`);
                    }
                  } catch (e) {
                    alert('Close all failed: ' + e.message);
                  }
                }


                async function refreshTrades() {
                  try {
                    const res = await fetch('/api/trades');
                    const payload = await res.json();
                    const trades = (payload && (payload.trades || payload)) || [];
                    
                    // Sort trades by closed_at (newest first)
                    trades.sort((a, b) => {
                      const dateA = new Date(a.closed_at || a.timestamp || 0);
                      const dateB = new Date(b.closed_at || b.timestamp || 0);
                      return dateB - dateA; // Newest first
                    });
                    
                    const tbody = document.getElementById('tradesBody');
                    tbody.innerHTML = trades.map(t => {
                      // Use closed_at for display, fallback to timestamp
                      const ts = new Date(t.closed_at || t.timestamp || t.time || Date.now()).toLocaleString();
                      const side = (t.side || '').toUpperCase();
                      const size = Number(t.size || t.amount || 0);
                      const price = Number(t.price || 0);
                      const total = size * price;
                      const pnl = Number(t.pnl || 0);
                      const pnlPct = Number(t.pnl_percentage || 0);
                      
                      // Color coding for PnL
                      const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                      const pnlSign = pnl >= 0 ? '+' : '';
                      
                      return `
                        <tr>
                          <td>${ts}</td>
                          <td>${t.symbol || ''}</td>
                          <td>${side}</td>
                          <td>${size.toFixed(6)}</td>
                          <td>${price.toFixed(4)}</td>
                          <td>${total.toFixed(2)}</td>
                          <td class="${pnlClass}">${pnlSign}${pnl.toFixed(2)}</td>
                          <td class="${pnlClass}">${pnlSign}${pnlPct.toFixed(2)}%</td>
                          <td>${t.strategy || ''}</td>
                        </tr>`;
                    }).join('');
                  } catch (e) {
                    console.warn('Trades refresh failed', e);
                  }
                }

                // poll trades every 10s + once at start
                setInterval(refreshTrades, 10000);
                refreshTrades();


                async function refreshRejections() {
                try {
                    const res = await fetch('/api/rejections');
                    const rows = await res.json();
                    updateRejections(rows || []);
                } catch (e) {
                    console.warn('Rejections refresh failed', e);
                }
                }

                function updateRejections(rows) {
                const tbody = document.getElementById('rejectionsBody');
                if (!tbody) return;
                tbody.innerHTML = rows.map(r => `
                    <tr>
                    <td>${new Date(r.timestamp || Date.now()).toLocaleString()}</td>
                    <td>${r.strategy ?? ''}</td>
                    <td>${r.symbol ?? ''}</td>
                    <td>${r.reason ?? ''}</td>
                    </tr>
                `).join('');
                }


                // poll rejections every 10s
                setInterval(refreshRejections, 10000);
                // also load immediately at page open
                refreshRejections();

                // Enhanced Analytics Functions
                let pnlPairsChart, pnlStrategiesChart, dailyPerformanceChart;
                
                async function loadEnhancedAnalytics() {
                    try {
                        await Promise.all([
                            loadTradeStatistics(),
                            loadPnLByPairs(),
                            loadPnLByStrategies(),
                            loadPerformanceCharts()
                        ]);
                    } catch (e) {
                        console.warn('Enhanced analytics load failed', e);
                    }
                }
                
                async function loadTradeStatistics() {
                    try {
                        const res = await fetch('/api/trade-statistics');
                        const stats = await res.json();
                        updateTradeStatistics(stats);
                    } catch (e) {
                        console.warn('Trade statistics load failed', e);
                    }
                }
                
                function updateTradeStatistics(stats) {
                    const statsHtml = `
                        <div class="metric-card">
                            <div class="metric-value">${stats.total_trades || 0}</div>
                            <div class="metric-label">Total Trades</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value ${(stats.total_pnl || 0) >= 0 ? 'positive' : 'negative'}">
                                $${(stats.total_pnl || 0).toFixed(2)}
                            </div>
                            <div class="metric-label">Total PnL</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(stats.win_rate || 0).toFixed(1)}%</div>
                            <div class="metric-label">Win Rate</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(stats.avg_pnl || 0).toFixed(2)}</div>
                            <div class="metric-label">Avg PnL</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(stats.profit_factor || 0).toFixed(2)}</div>
                            <div class="metric-label">Profit Factor</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(stats.pnl_stddev || 0).toFixed(2)}</div>
                            <div class="metric-label">PnL Std Dev</div>
                        </div>
                    `;
                    document.getElementById('tradeStats').innerHTML = statsHtml;
                }
                
                async function loadPnLByPairs() {
                    try {
                        const res = await fetch('/api/pnl-by-pairs');
                        const data = await res.json();
                        updatePnLPairsChart(data.pairs || []);
                        updatePairsPerformanceTable(data.pairs || []);
                    } catch (e) {
                        console.warn('PnL by pairs load failed', e);
                    }
                }
                
                function updatePnLPairsChart(pairs) {
                    const ctx = document.getElementById('pnlPairsChart').getContext('2d');
                    
                    if (pnlPairsChart) {
                        pnlPairsChart.destroy();
                    }
                    
                    const labels = pairs.slice(0, 10).map(p => p.symbol);
                    const data = pairs.slice(0, 10).map(p => p.total_pnl);
                    const colors = data.map(pnl => pnl >= 0 ? '#4CAF50' : '#f44336');
                    
                    pnlPairsChart = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: labels,
                            datasets: [{
                                data: data,
                                backgroundColor: colors,
                                borderWidth: 2,
                                borderColor: '#333'
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                legend: {
                                    position: 'bottom',
                                    labels: {
                                        color: '#fff'
                                    }
                                }
                            }
                        }
                    });
                }
                
                function updatePairsPerformanceTable(pairs) {
                    const tbody = document.getElementById('pairsPerformanceBody');
                    tbody.innerHTML = pairs.map(pair => `
                        <tr>
                            <td>${pair.symbol}</td>
                            <td>${pair.trade_count}</td>
                            <td class="${pair.total_pnl >= 0 ? 'positive' : 'negative'}">
                                $${pair.total_pnl.toFixed(2)}
                            </td>
                            <td class="${pair.avg_pnl >= 0 ? 'positive' : 'negative'}">
                                $${pair.avg_pnl.toFixed(2)}
                            </td>
                            <td>${pair.win_rate.toFixed(1)}%</td>
                            <td class="positive">$${pair.max_pnl.toFixed(2)}</td>
                            <td class="negative">$${pair.min_pnl.toFixed(2)}</td>
                        </tr>
                    `).join('');
                }
                
                async function loadPnLByStrategies() {
                    try {
                        const res = await fetch('/api/pnl-by-strategies');
                        const data = await res.json();
                        updatePnLStrategiesChart(data.strategies || []);
                        updateStrategiesPerformanceTable(data.strategies || []);
                    } catch (e) {
                        console.warn('PnL by strategies load failed', e);
                    }
                }
                
                function updatePnLStrategiesChart(strategies) {
                    const ctx = document.getElementById('pnlStrategiesChart').getContext('2d');
                    
                    if (pnlStrategiesChart) {
                        pnlStrategiesChart.destroy();
                    }
                    
                    const labels = strategies.map(s => s.strategy);
                    const data = strategies.map(s => s.total_pnl);
                    const colors = data.map(pnl => pnl >= 0 ? '#4CAF50' : '#f44336');
                    
                    pnlStrategiesChart = new Chart(ctx, {
                        type: 'pie',
                        data: {
                            labels: labels,
                            datasets: [{
                                data: data,
                                backgroundColor: colors,
                                borderWidth: 2,
                                borderColor: '#333'
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                legend: {
                                    position: 'bottom',
                                    labels: {
                                        color: '#fff'
                                    }
                                }
                            }
                        }
                    });
                }
                
                function updateStrategiesPerformanceTable(strategies) {
                    const tbody = document.getElementById('strategiesPerformanceBody');
                    tbody.innerHTML = strategies.map(strategy => `
                        <tr>
                            <td>${strategy.strategy}</td>
                            <td>${strategy.trade_count}</td>
                            <td class="${strategy.total_pnl >= 0 ? 'positive' : 'negative'}">
                                $${strategy.total_pnl.toFixed(2)}
                            </td>
                            <td class="${strategy.avg_pnl >= 0 ? 'positive' : 'negative'}">
                                $${strategy.avg_pnl.toFixed(2)}
                            </td>
                            <td>${strategy.win_rate.toFixed(1)}%</td>
                            <td class="positive">$${strategy.max_pnl.toFixed(2)}</td>
                            <td class="negative">$${strategy.min_pnl.toFixed(2)}</td>
                        </tr>
                    `).join('');
                }
                
                async function loadPerformanceCharts() {
                    try {
                        const res = await fetch('/api/performance-charts');
                        const data = await res.json();
                        updateDailyPerformanceChart(data.daily_performance || []);
                    } catch (e) {
                        console.warn('Performance charts load failed', e);
                    }
                }
                
                function updateDailyPerformanceChart(dailyData) {
                    const ctx = document.getElementById('dailyPerformanceChart').getContext('2d');
                    
                    if (dailyPerformanceChart) {
                        dailyPerformanceChart.destroy();
                    }
                    
                    const labels = dailyData.map(d => new Date(d.date).toLocaleDateString());
                    const pnlData = dailyData.map(d => d.daily_pnl);
                    const cumulativeData = dailyData.map(d => d.cumulative_pnl);
                    
                    dailyPerformanceChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [
                                {
                                    label: 'Daily PnL',
                                    data: pnlData,
                                    borderColor: '#4CAF50',
                                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                    tension: 0.1
                                },
                                {
                                    label: 'Cumulative PnL',
                                    data: cumulativeData,
                                    borderColor: '#2196F3',
                                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                                    tension: 0.1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    ticks: {
                                        color: '#fff'
                                    },
                                    grid: {
                                        color: '#333'
                                    }
                                },
                                x: {
                                    ticks: {
                                        color: '#fff'
                                    },
                                    grid: {
                                        color: '#333'
                                    }
                                }
                            },
                            plugins: {
                                legend: {
                                    labels: {
                                        color: '#fff'
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Load enhanced analytics on page load (with delay to not impact initial load)
                setTimeout(loadEnhancedAnalytics, 2000); // Delay 2 seconds after page load
                setInterval(loadEnhancedAnalytics, 30000); // Refresh every 30 seconds

                // Consider calling refreshRejections() inside ws.onmessage if you want WS-driven refresh too
            </script>
        </body>
        </html>
        """
    
    async def start_dashboard(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the dashboard server (async, non-blocking)"""
        import uvicorn
        from uvicorn import Config, Server
        logger.info(f"Starting dashboard on {host}:{port}")
        config = Config(app=self.app, host=host, port=port, log_level="info", loop="asyncio")
        server = Server(config)
        await server.serve()

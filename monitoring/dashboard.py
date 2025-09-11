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


logger = logging.getLogger(__name__)

class DashboardManager:
    """Manages the web dashboard for monitoring bot performance"""
    
    def __init__(self, database_manager: DatabaseManager, portfolio_manager: PortfolioManager):
        self.db = database_manager
        self.portfolio_manager = portfolio_manager
        self.app = FastAPI(title="Crypto Trading Bot Dashboard")
        self.active_connections: List[WebSocket] = []
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect(websocket)
            try:
                while True:
                    # Send real-time updates every 5 seconds
                    data = await self.get_real_time_data()
                    await self.broadcast(data)
                    await asyncio.sleep(5)
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
        """Get current positions data"""
        try:
            positions = await self.db.get_open_positions()
            return [pos.to_dict() for pos in positions]
        except Exception as e:
            logger.error(f"Failed to get positions data: {e}")
            return []
    
    async def get_performance_data(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            stats = await self.db.get_performance_stats()
            
            # Add additional metrics
            portfolio_value = await self.portfolio_manager.get_total_value()
            daily_pnl = await self.get_daily_pnl()
            
            return {
                **stats,
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'last_updated': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get performance data: {e}")
            return {}
    
    async def get_trades_data(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades data"""
        try:
            session = self.db.get_session()
            
            trades = session.query(Trade).order_by(
                Trade.timestamp.desc()
            ).limit(limit).all()
            
            session.close()
            return [trade.to_dict() for trade in trades]
        except Exception as e:
            logger.error(f"Failed to get trades data: {e}")
            return []
    
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

            import os, re, datetime
            from datetime import datetime, timezone

            path = os.path.join("logs", "trading_bot.log")
            if not os.path.exists(path):
                return []

            pattern = re.compile(
                r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*REJECTION \| strategy=(?P<strategy>\S+) symbol=(?P<symbol>\S+) reason=(?P<reason>.+?) conf=(?P<conf>[\d\.]+)"
            )

            items = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = pattern.search(line)
                    if not m:
                        continue
                    d = m.groupdict()

                    # Convert "YYYY-MM-DD HH:MM:SS,mmm" -> ISO8601 + epoch ms
                    try:
                        dt = datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)
                        ts_iso = dt.isoformat()            # e.g. "2025-09-07T20:19:03.074+00:00"
                        ts_epoch = int(dt.timestamp() * 1000)
                    except Exception:
                        # Fallback to raw string if parsing ever fails
                        ts_iso = d["ts"]
                        ts_epoch = None

                    items.append({
                        "timestamp": ts_iso,               # browser-friendly
                        "ts_epoch": ts_epoch,              # number for charts/sorting
                        "strategy": d["strategy"],
                        "symbol": d["symbol"],
                        "reason": d["reason"].strip(),
                        "confidence": float(d["conf"]),
                    })

            # newest first by epoch if available
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
                                <th>Strategy</th>
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
                            <td>${pos.strategy}</td>
                        </tr>
                    `).join('');
                }
                



                // --- timeframe controls (non-breaking) ---
                let rawPoints = [];            // [{ts: number, val: number}]
                let dataWindowMs = 5 * 60 * 1000; // default 5m
                const TF_BUCKETS = { '1m':60e3, '5m':5*60e3, '15m':15*60e3, '1h':60*60e3, '1d':24*60*60e3, '7d':7*24*60*60e3 };

                function resampleAndRender() {
                    if (!portfolioChart) return;

                    const tfSel = document.getElementById('tf');
                    const tf = tfSel ? tfSel.value : '5m';
                    const bucket = (tf === 'all') ? 0 : (TF_BUCKETS[tf] || 5*60e3);

                    // window filter
                    const now = Date.now();
                    const windowMs = (tf === 'all') ? Infinity : dataWindowMs;
                    const filtered = rawPoints.filter(p => (tf === 'all') || (now - p.ts <= windowMs));

                    // simple bucketed dedup: keep last point per bucket
                    let out = filtered;
                    if (bucket > 0) {
                        const map = new Map();
                        for (const p of filtered) {
                            const k = Math.floor(p.ts / bucket);
                            map.set(k, p); // last one in bucket wins
                        }
                        out = Array.from(map.values()).sort((a,b)=>a.ts-b.ts);
                    }

                    // render
                    portfolioChart.data.labels = out.map(p => Object.assign(new Date(p.ts).toLocaleTimeString(), { _ts: p.ts }));
                    portfolioChart.data.datasets[0].data = out.map(p => p.val);
                    portfolioChart.update('none');
                }

                // apply on timeframe change
                document.addEventListener('change', (e) => {
                    if (e.target && e.target.id === 'tf') {
                        const v = e.target.value;
                        dataWindowMs = (v === 'all') ? Infinity : (TF_BUCKETS[v] || 5*60e3);
                        resampleAndRender();
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
                    resampleAndRender();
                }


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

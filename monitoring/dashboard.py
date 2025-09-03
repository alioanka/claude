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
from data.database import DatabaseManager
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
            </div>
            
            <script>
                const ws = new WebSocket('ws://localhost:8000/ws');
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
                
                function updateChart(portfolio) {
                    // Simple portfolio chart implementation
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
                                plugins: {
                                    legend: { labels: { color: '#fff' } }
                                },
                                scales: {
                                    x: { ticks: { color: '#ccc' } },
                                    y: { ticks: { color: '#ccc' } }
                                }
                            }
                        });
                    }
                    
                    // Add new data point
                    const now = new Date().toLocaleTimeString();
                    portfolioChart.data.labels.push(now);
                    portfolioChart.data.datasets[0].data.push(portfolio.total_value || 0);
                    
                    // Keep only last 50 points
                    if (portfolioChart.data.labels.length > 50) {
                        portfolioChart.data.labels.shift();
                        portfolioChart.data.datasets[0].data.shift();
                    }
                    
                    portfolioChart.update();
                }
                
                // Initialize dashboard
                window.onload = function() {
                    console.log('Dashboard loaded');
                };
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

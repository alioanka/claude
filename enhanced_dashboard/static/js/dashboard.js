/**
 * ClaudeBot Enhanced Dashboard JavaScript
 * Advanced trading dashboard with real-time updates and interactive charts
 */

// Global variables
let ws = null;
let charts = {};
let currentData = {};
let updateInterval = null;
let isConnected = false;

// WebSocket status helper - matches actual DOM structure
function updateConnectionStatus(state) {
    const root = document.getElementById('connectionStatus');
    if (!root) return;
  
    const wrap = root.querySelector('.status-indicator');
    if (!wrap) return;
  
    const icon = wrap.querySelector('i');
    const text = wrap.querySelector('span');
  
    const map = {
      connected:    { icon: 'fas fa-circle text-success',    text: 'Connected' },
      disconnected: { icon: 'fas fa-circle text-secondary',  text: 'Disconnected' },
      error:        { icon: 'fas fa-circle text-danger',     text: 'Error' },
      connecting:   { icon: 'fas fa-circle text-warning',    text: 'Connecting...' }
    };
    const m = map[state] || map.connecting;
  
    if (icon) icon.className = m.icon;
    if (text) text.textContent = m.text;
  }
  

// Risk data normalization helper
function normalizeRisk(r) {
  if (!r) return {};
  return {
    var95:  r.var95  ?? r.var_95  ?? 0,
    var99:  r.var99  ?? r.var_99  ?? 0,
    volatility: r.volatility ?? r.vol ?? 0,
    beta: r.beta ?? 0,
    max_drawdown: r.max_drawdown ?? r.maxDrawdown ?? 0,
  };
}

// Price formatting helper - adaptive precision
function formatPrice(x) {
    if (x == null || isNaN(x)) return '-';
    const v = Number(x);
    if (v >= 1000) return v.toFixed(2);
    if (v >= 1)    return v.toFixed(4);
    if (v >= 0.01) return v.toFixed(6);
    return v.toFixed(8); // ultra small caps
}

function computePnl(pos) {
    const entry = Number(pos.entry_price ?? 0);
    const curr  = Number(pos.current_price ?? entry);
    const size  = Number(pos.size ?? 0);
    const side  = (pos.side || '').toLowerCase();

    const diff = side === 'short' ? (entry - curr) : (curr - entry);
    const pnl  = diff * size;
    const pct  = entry ? (diff / entry) * 100 : 0;
    return { pnl, pct };
}

// Chart.js default configuration
Chart.defaults.font.family = "'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#5a5c69';

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

/**
 * Initialize the dashboard
 */
async function initializeDashboard() {
    try {
        console.log('Initializing dashboard...');
        console.log('DOM loaded, starting initialization...');
        
        showLoading(true);
        
        // Initialize WebSocket connection
        console.log('Initializing WebSocket...');
        updateConnectionStatus('connecting');
        initializeWebSocket();
        
        // Load initial data
        console.log('Loading initial data...');
        await loadInitialData();
        
        // Load risk data
        console.log('Loading risk data...');
        await loadRisk();
        
        // Initialize charts
        console.log('Initializing charts...');
        initializeCharts();
        
        // Start periodic updates
        console.log('Starting periodic updates...');
        startPeriodicUpdates();
        
        // Start price updates every 5 seconds
        startPriceUpdates();
        
        // Set up event listeners
        console.log('Setting up event listeners...');
        setupEventListeners();
        
        // Check connection status after a delay
        setTimeout(() => {
            if (!isConnected) {
                console.log('WebSocket not connected, trying to reconnect...');
                updateConnectionStatus('connecting', 'Reconnecting...');
                initializeWebSocket();
            }
        }, 3000);
        
        // Fallback: Load data even without WebSocket after 10 seconds
        setTimeout(() => {
            if (!isConnected) {
                console.log('WebSocket still not connected, loading data via API...');
                updateConnectionStatus('disconnected', 'Using API fallback');
                loadInitialData();
            }
        }, 10000);
        
        showLoading(false);
        showToast('Dashboard initialized successfully', 'success');
        console.log('Dashboard initialization complete');
        
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        showToast('Error initializing dashboard', 'error');
        showLoading(false);
    }
}

/**
 * Initialize WebSocket connection
 */
function initializeWebSocket() {
    try {
      const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
      const wsUrl = `${wsProtocol}://${location.host}/ws`;
      // IMPORTANT: assign to the global, not a new const
      window.ws = new WebSocket(wsUrl);
  
      ws.onopen = () => {
        window.isConnected = true;
        updateConnectionStatus('connected');
      };
  
      ws.onclose = () => {
        window.isConnected = false;
        updateConnectionStatus('disconnected');
        setTimeout(initializeWebSocket, 2000);
      };
  
      ws.onerror = () => {
        window.isConnected = false;
        updateConnectionStatus('error');
      };
  
      ws.onmessage = (ev) => {
        // Your backend sends plain JSON text; parse then route
        try {
          const msg = JSON.parse(ev.data);
          // accept both {positions:[...]} and {type:'dashboard_update', data:{...}}
          if (msg?.positions || Array.isArray(msg)) {
            updateDashboardData(msg);
          } else if (msg?.type === 'dashboard_update') {
            updateDashboardData(msg.data || {});
          }
        } catch (e) {
          // if server ever broadcasts a simple list, still try to render
          if (Array.isArray(ev.data)) updateDashboardData(ev.data);
        }
      };
    } catch (e) {
      window.isConnected = false;
      updateConnectionStatus('error');
    }
  }
  

function setStatus(status) {
    const el = document.getElementById('connectionStatus');
    if (!el) return;
    el.textContent = status;
    el.className = status === "Connected" ? "badge bg-success" : 
                   status === "Disconnected" ? "badge bg-danger" : 
                   "badge bg-warning";
}

/**
 * Handle WebSocket messages
 */
function handleWebSocketMessage(data) {
    console.log('Handling WebSocket message:', data);
    
    switch(data.type) {
        case 'connection_established':
            console.log('WebSocket connection established:', data.message);
            updateConnectionStatus('connected', data.message);
            break;
        case 'dashboard_update':
            console.log('Processing dashboard update...');
            updateDashboardData(data.data);
            break;
        default:
            console.log('Unknown message type:', data.type);
    }
}


/**
 * Load initial data
 */
async function loadInitialData() {
    try {
        console.log('Loading initial data...');
        
        const [accountData, positionsData, tradesData, performanceData] = await Promise.all([
            fetchData('/api/account'),
            fetchData('/api/positions'),
            fetchData('/api/trades'),
            fetchData('/api/performance')
        ]);
        
        console.log('API responses:', {
            account: accountData,
            positions: positionsData,
            trades: tradesData,
            performance: performanceData
        });
        
        currentData = {
            account: accountData?.account || accountData || {},
            positions: Array.isArray(positionsData) ? positionsData : (positionsData?.positions || []),
            trades: Array.isArray(tradesData) ? tradesData : (tradesData?.trades || []),
            performance: performanceData?.performance || performanceData || {}
        };
        
        console.log('Current data set:', currentData);
        updateDashboardData(currentData);
        
    } catch (error) {
        console.error('Error loading initial data:', error);
        throw error;
    }
}

/**
 * Fetch data from API
 */
async function fetchData(endpoint) {
    try {
        console.log(`Fetching data from: ${endpoint}`);
        const response = await fetch(endpoint);
        console.log(`Response status: ${response.status} for ${endpoint}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log(`Data received from ${endpoint}:`, data);
        return data;
    } catch (error) {
        console.error(`Error fetching data from ${endpoint}:`, error);
        return null;
    }
}

/**
 * Update dashboard with new data
 */
function updateDashboardData(payload) {
    if (!payload) return;
  
    // Positions
    if (Array.isArray(payload)) {
      updatePositionsTable(payload);
    } else if (payload.positions) {
      updatePositionsTable(payload.positions);
    }
  
    // Account/Performance
    if (payload.account) {
      updateAccountMetrics(payload.account);
    } else if (payload.performance) {
      // if you prefer to derive Overview from /api/performance
      updateAccountMetrics({
        total_balance: payload.performance.portfolio_value,
        total_pnl: payload.performance.total_pnl,
        active_positions: (payload.positions || []).length,
        win_rate: payload.performance.win_rate
      });
    }
  
    // Trades (if included in WS payloads)
    if (payload.trades) {
      updateTradesTable(payload.trades);
    }
  }
  

/**
 * Update account metrics
 */
function updateAccountMetrics(account) {
    console.log('Updating account metrics with data:', account);
    
    // Update key metrics
    updateElement('totalBalance', formatCurrency(account.total_balance || 0));
    updateElement('totalPnL', formatCurrency(account.total_pnl || 0), account.total_pnl >= 0 ? 'positive' : 'negative');
    updateElement('activePositions', account.active_positions || 0);
    updateElement('winRate', formatPercentage(account.win_rate || 0));
    
    // Update performance metrics
    updateElement('sharpeRatio', (account.sharpe_ratio || 0).toFixed(2));
    updateElement('maxDrawdown', formatPercentage(account.max_drawdown || 0));
    
    console.log('Account metrics updated');
}

/**
 * Update positions table
 */
function updatePositionsTable(positions) {
    console.log('Updating positions table with data:', positions);
    
    const tbody = document.getElementById('positionsTableBody');
    if (!tbody) {
        console.warn('Positions table body not found');
        return;
    }
    
    tbody.innerHTML = '';
    
    if (positions.length === 0) {
        console.log('No positions to display');
        tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No active positions</td></tr>';
        return;
    }
    
    for (const p of positions) {
        const { pnl, pct } = computePnl(p);
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${p.symbol}</td>
            <td class="${(p.side||'').toLowerCase()==='long'?'text-success':'text-danger'}">${(p.side||'').toUpperCase()}</td>
            <td>${formatPrice(p.size)}</td>
            <td>$${formatPrice(p.entry_price)}</td>
            <td>$${formatPrice(p.current_price)}</td>
            <td class="${pnl>=0?'positive':'negative'}">${formatCurrency(pnl)}</td>
            <td class="${pnl>=0?'positive':'negative'}">${pct.toFixed(2)}%</td>
            <td>${p.duration_human ?? '-'}</td>
            <td>${p.strategy ?? '-'}</td>
            <td><!-- actions if any --></td>
        `;
        tbody.appendChild(row);
    }
}

/**
 * Update trades table
 */
function updateTradesTable(trades) {
    console.log('Updating trades table with data:', trades);
    
    const tbody = document.getElementById('tradesTableBody');
    if (!tbody) {
        console.warn('Trades table body not found');
        return;
    }
    
    tbody.innerHTML = '';
    
    if (trades.length === 0) {
        console.log('No trades to display');
        tbody.innerHTML = '<tr><td colspan="11" class="text-center text-muted">No recent trades</td></tr>';
        return;
    }
    
    trades.forEach(trade => {
        const row = document.createElement('tr');
        const pnl = trade.pnl || 0;
        const pnlPercent = trade.pnl_percentage || 0;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        
        // Calculate duration
        let duration = 'N/A';
        if (trade.opened_at && trade.closed_at) {
            const opened = new Date(trade.opened_at);
            const closed = new Date(trade.closed_at);
            const diffMs = closed - opened;
            const diffMins = Math.floor(diffMs / (1000 * 60));
            const diffHours = Math.floor(diffMins / 60);
            const remainingMins = diffMins % 60;
            duration = diffHours > 0 ? `${diffHours}h ${remainingMins}m` : `${diffMins}m`;
        }
        
        row.innerHTML = `
            <td><strong>${trade.symbol}</strong></td>
            <td><span class="badge bg-${trade.side === 'long' ? 'success' : 'danger'}">${trade.side.toUpperCase()}</span></td>
            <td>${formatNumber(trade.size || 0, 6)}</td>
            <td>${formatCurrency(trade.entry_price || 0)}</td>
            <td>${formatCurrency(trade.exit_price || 0)}</td>
            <td class="${pnlClass}">${formatCurrency(pnl)}</td>
            <td class="${pnlClass}">${formatPercentage(pnlPercent)}</td>
            <td>${duration}</td>
            <td>${formatDateTime(trade.opened_at)}</td>
            <td>${formatDateTime(trade.closed_at)}</td>
            <td><span class="badge bg-info">${trade.strategy || 'Unknown'}</span></td>
        `;
        
        tbody.appendChild(row);
    });
}

/**
 * Update performance metrics
 */
function updatePerformanceMetrics(performance) {
    console.log('Updating performance metrics with data:', performance);
    
    // Update risk metrics
    updateElement('var95', formatCurrency(performance.var_95 || 0));
    updateElement('var99', formatCurrency(performance.var_99 || 0));
    updateElement('maxDrawdown', formatPercentage(performance.max_drawdown || 0));
    updateElement('volatility', formatPercentage(performance.volatility || 0));
    updateElement('beta', (performance.beta || 0).toFixed(2));
    
    console.log('Performance metrics updated');
}

/**
 * Initialize all charts
 */
function initializeCharts() {
    console.log('Initializing charts...');
    try {
        initializePortfolioChart();
        console.log('Portfolio chart initialized');
        
        initializeStrategyChart();
        console.log('Strategy chart initialized');
        
        initializePnlDistributionChart();
        console.log('PnL distribution chart initialized');
        
        initializeDrawdownChart();
        console.log('Drawdown chart initialized');
        
        console.log('All charts initialized successfully');
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
}

/**
 * Initialize portfolio performance chart
 */
function initializePortfolioChart() {
    const ctx = document.getElementById('portfolioChart');
    if (!ctx) return;
    
    charts.portfolio = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#4e73df',
                backgroundColor: 'rgba(78, 115, 223, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#4e73df',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: '#4e73df',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#5a5c69'
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        color: '#5a5c69',
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

/**
 * Initialize strategy distribution chart
 */
function initializeStrategyChart() {
    const ctx = document.getElementById('strategyChart');
    if (!ctx) return;
    
    charts.strategy = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Momentum', 'Mean Reversion', 'Arbitrage'],
            datasets: [{
                data: [60, 30, 10],
                backgroundColor: [
                    '#4e73df',
                    '#1cc88a',
                    '#f6c23e'
                ],
                borderColor: '#ffffff',
                borderWidth: 2,
                hoverBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Initialize P&L distribution chart
 */
function initializePnlDistributionChart() {
    const container = document.getElementById('pnlDistributionChart');
    if (!container) return;
    
    charts.pnlDistribution = ApexCharts.exec('pnlDistributionChart', 'updateOptions', {
        series: [{
            name: 'P&L Distribution',
            data: []
        }],
        chart: {
            type: 'histogram',
            height: 300,
            toolbar: {
                show: false
            }
        },
        plotOptions: {
            bar: {
                horizontal: false,
                columnWidth: '70%'
            }
        },
        dataLabels: {
            enabled: false
        },
        xaxis: {
            title: {
                text: 'P&L ($)'
            }
        },
        yaxis: {
            title: {
                text: 'Frequency'
            }
        },
        colors: ['#4e73df'],
        fill: {
            type: 'gradient',
            gradient: {
                shade: 'light',
                type: 'vertical',
                shadeIntensity: 0.5,
                gradientToColors: ['#1cc88a'],
                inverseColors: false,
                opacityFrom: 0.8,
                opacityTo: 0.3
            }
        }
    }) || new ApexCharts(container, {
        series: [{
            name: 'P&L Distribution',
            data: []
        }],
        chart: {
            type: 'bar',
            height: 300,
            toolbar: {
                show: false
            }
        },
        plotOptions: {
            bar: {
                horizontal: false,
                columnWidth: '70%'
            }
        },
        dataLabels: {
            enabled: false
        },
        xaxis: {
            title: {
                text: 'P&L ($)'
            }
        },
        yaxis: {
            title: {
                text: 'Frequency'
            }
        },
        colors: ['#4e73df'],
        fill: {
            type: 'gradient',
            gradient: {
                shade: 'light',
                type: 'vertical',
                shadeIntensity: 0.5,
                gradientToColors: ['#1cc88a'],
                inverseColors: false,
                opacityFrom: 0.8,
                opacityTo: 0.3
            }
        }
    });
    
    charts.pnlDistribution.render();
}

/**
 * Initialize drawdown chart
 */
function initializeDrawdownChart() {
    const container = document.getElementById('drawdownChart');
    if (!container) return;
    
    charts.drawdown = new ApexCharts(container, {
        series: [{
            name: 'Drawdown',
            data: []
        }],
        chart: {
            type: 'area',
            height: 300,
            toolbar: {
                show: false
            }
        },
        dataLabels: {
            enabled: false
        },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        fill: {
            type: 'gradient',
            gradient: {
                shade: 'light',
                type: 'vertical',
                shadeIntensity: 0.5,
                gradientToColors: ['#e74a3b'],
                inverseColors: false,
                opacityFrom: 0.8,
                opacityTo: 0.3
            }
        },
        colors: ['#e74a3b'],
        xaxis: {
            type: 'datetime',
            title: {
                text: 'Time'
            }
        },
        yaxis: {
            title: {
                text: 'Drawdown (%)'
            },
            labels: {
                formatter: function(value) {
                    return value.toFixed(2) + '%';
                }
            }
        }
    });
    
    charts.drawdown.render();
}

/**
 * Update portfolio chart
 */
function updatePortfolioChart(timeframe = '7d') {
    // This would typically fetch historical data based on timeframe
    // For now, we'll simulate with current data
    if (charts.portfolio && currentData.account) {
        const now = new Date();
        const timeLabel = now.toLocaleTimeString();
        
        charts.portfolio.data.labels.push(timeLabel);
        charts.portfolio.data.datasets[0].data.push(currentData.account.total_balance || 0);
        
        // Keep only last 20 data points
        if (charts.portfolio.data.labels.length > 20) {
            charts.portfolio.data.labels.shift();
            charts.portfolio.data.datasets[0].data.shift();
        }
        
        charts.portfolio.update('none');
    }
}

/**
 * Show/hide loading overlay
 */
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
        console.log(`Loading overlay ${show ? 'shown' : 'hidden'}`);
    } else {
        console.warn('Loading overlay element not found');
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    console.log(`Showing toast: ${message} (${type})`);
    
    const toast = document.getElementById('toast');
    const toastBody = document.getElementById('toastBody');
    const toastHeader = toast ? toast.querySelector('.toast-header i') : null;
    
    if (toastBody) {
        toastBody.textContent = message;
    } else {
        console.warn('Toast body not found');
    }
    
    // Update icon based on type
    if (toastHeader) {
        toastHeader.className = `fas fa-${type === 'success' ? 'check-circle text-success' : 
                                          type === 'error' ? 'exclamation-circle text-danger' : 
                                          type === 'warning' ? 'exclamation-triangle text-warning' : 
                                          'info-circle text-primary'} me-2`;
    } else {
        console.warn('Toast header not found');
    }
    
    // Show toast
    if (toast) {
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    } else {
        console.warn('Toast element not found');
    }
}

/**
 * Show section
 */
function showSection(sectionName) {
    console.log(`Showing section: ${sectionName}`);
    
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Remove active class from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Show selected section
    const targetSection = document.getElementById(`${sectionName}-section`);
    if (targetSection) {
        targetSection.style.display = 'block';
        targetSection.classList.add('animate__animated', 'animate__fadeIn');
        console.log(`Section ${sectionName} displayed`);
    } else {
        console.warn(`Section ${sectionName} not found`);
    }
    
    // Add active class to selected nav link
    event.target.classList.add('active');
    console.log(`Nav link for ${sectionName} activated`);
    
    // Load section-specific data
    switch(sectionName) {
        case 'overview':
            loadOverviewData();
            break;
        case 'positions':
            loadPositionsData();
            break;
        case 'trades':
            loadTradesData();
            break;
        case 'analytics':
            loadAnalyticsData();
            break;
        case 'risk':
            loadRiskData();
            break;
        case 'market':
            loadMarketData();
            break;
        case 'settings':
            loadSettingsData();
            break;
    }
}

/**
 * Load overview data
 */
async function loadOverviewData() {
    try {
        const [perf, strat] = await Promise.all([
            fetchData('/api/performance'),                        // KPIs
            fetchData('/api/analytics/strategy-performance')      // table
        ]);

        // KPIs (fallbacks so the page never shows "undefined")
        updateElement('totalBalance', formatCurrency(perf?.total_balance ?? 0));
        updateElement('totalPnl',     formatCurrency(perf?.total_pnl ?? 0),
            (perf?.total_pnl ?? 0) >= 0 ? 'positive' : 'negative');
        updateElement('activePositions', String(perf?.open_positions ?? 0));
        updateElement('winRate', `${(perf?.win_rate ?? 0).toFixed(1)}%`);

        // Analytics mini cards (Overview)
        updateElement('sharpeValue',      (perf?.sharpe_ratio ?? 0).toFixed(2));
        updateElement('maxDrawdownValue', `${((perf?.max_drawdown ?? 0) * 100).toFixed(2)}%`);
        updateElement('volatilityValue',  `${((perf?.volatility ?? 0) * 100).toFixed(2)}%`);
        updateElement('betaValue',        (perf?.beta ?? 0).toFixed(2));

        // Strategy table
        updateStrategyTable(strat?.strategy_performance ?? []);
    } catch (err) {
        console.error('Overview load failed', err);
    }
}

/**
 * Update strategy performance table
 */
function updateStrategyTable(strategies) {
    console.log('Updating strategy table with data:', strategies);
    
    const tbody = document.getElementById('strategyTableBody');
    if (!tbody) {
        console.warn('Strategy table body not found');
        return;
    }
    
    tbody.innerHTML = '';
    
    if (strategies.length === 0) {
        console.log('No strategies to display');
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No strategy data available</td></tr>';
        return;
    }
    
    strategies.forEach(strategy => {
        const row = document.createElement('tr');
        const statusClass = strategy.total_pnl >= 0 ? 'success' : 'danger';
        
        row.innerHTML = `
            <td><strong>${strategy.strategy}</strong></td>
            <td>${strategy.total_trades}</td>
            <td><span class="badge bg-${strategy.win_rate >= 50 ? 'success' : 'danger'}">${strategy.win_rate.toFixed(1)}%</span></td>
            <td class="${strategy.total_pnl >= 0 ? 'positive' : 'negative'}">${formatCurrency(strategy.total_pnl)}</td>
            <td class="${strategy.avg_pnl >= 0 ? 'positive' : 'negative'}">${formatCurrency(strategy.avg_pnl)}</td>
            <td>${strategy.sharpe_ratio.toFixed(2)}</td>
            <td><span class="badge bg-${statusClass}">${strategy.total_pnl >= 0 ? 'Profitable' : 'Loss'}</span></td>
        `;
        
        tbody.appendChild(row);
    });
}

/**
 * Load positions data
 */
async function loadPositionsData() {
    try {
        const res = await fetchData('/api/positions');
        const list = res?.positions ?? [];    // <- single level
        updatePositionsTable(list);
    } catch (e) {
        console.error('Positions load failed', e);
    }
}

/**
 * Load trades data
 */
async function loadTradesData() {
    try {
        console.log('Loading trades data...');
        
        const tradesData = await fetchData('/api/trades');
        console.log('Trades data loaded:', tradesData);
        
        if (tradesData) {
            updateTradesTable(tradesData.trades || []);
        } else {
            console.log('No trades data available');
        }
    } catch (error) {
        console.error('Error loading trades data:', error);
    }
}

/**
 * Load analytics data
 */
async function loadAnalyticsData() {
    try {
        const [perf, risk] = await Promise.all([
            fetchData('/api/performance'),
            fetchData('/api/analytics/risk-metrics')
        ]);

        // Cards on Analytics page
        updateElement('sharpeValue',      (perf?.sharpe_ratio ?? 0).toFixed(2));
        updateElement('maxDrawdownValue', `${((perf?.max_drawdown ?? 0) * 100).toFixed(2)}%`);
        updateElement('volatilityValue',  `${((perf?.volatility ?? 0) * 100).toFixed(2)}%`);
        updateElement('betaValue',        (perf?.beta ?? 0).toFixed(2));

        // If you have charts, set their series here from perf/risk
        // (left as-is if your charts already read these globals)
    } catch (e) {
        console.error('Analytics load failed', e);
    }
}

/**
 * Update risk metrics
 */
function updateRiskMetrics(riskData) {
    console.log('Updating risk metrics with data:', riskData);
    
    updateElement('var95', formatCurrency(riskData.var_95 || 0));
    updateElement('var99', formatCurrency(riskData.var_99 || 0));
    updateElement('maxCorrelation', (riskData.max_correlation || 0).toFixed(2));
    updateElement('currentCorrelation', (riskData.current_correlation || 0).toFixed(2));
    
    console.log('Risk metrics updated');
}

/**
 * Load risk data
 */
async function loadRiskData() {
    try {
        const risk = await fetchData('/api/analytics/risk-metrics');

        updateElement('var95', formatCurrency(risk?.var_95 ?? 0));
        updateElement('var99', formatCurrency(risk?.var_99 ?? 0));

        updateElement('maxPositionsValue', String(risk?.max_positions ?? 50));
        updateElement('currentPositionsValue', String(risk?.current_positions ?? 0));

        updateElement('maxCorrelationValue', (risk?.max_correlation ?? 0.7).toFixed(2));
        updateElement('currentCorrelationValue', (risk?.current_correlation ?? 0).toFixed(2));
    } catch (e) {
        console.error('Risk load failed', e);
    }
}

/**
 * Load market data
 */
async function loadMarketData() {
    try {
        const res = await fetchData('/api/market-data');
        const rows = Array.isArray(res) ? res : (res?.market_data ?? []);
        updateMarketTable(rows);
    } catch (e) {
        console.error('Market data load failed', e);
    }
}

/**
 * Update market table
 */
function updateMarketTable(marketData) {
    console.log('Updating market table with data:', marketData);
    
    const tbody = document.getElementById('marketTableBody');
    if (!tbody) {
        console.warn('Market table body not found');
        return;
    }
    
    tbody.innerHTML = '';
    
        // Handle both old format (object with BTC/ETH keys) and new format (array of market data)
        let marketRows = [];
        
        if (Array.isArray(marketData)) {
            // New format: array of market data objects
            marketRows = marketData.map(data => ({
                symbol: data.symbol || 'Unknown',
                price: data.price || 0,
                change: data.change_24h || 0,
                volume: data.volume_24h || 0,
                high: data.high_24h || 0,
                low: data.low_24h || 0
            }));
        } else {
            // Old format: object with BTC/ETH keys
            marketRows = [
                {
                    symbol: 'BTC',
                    price: marketData.btc_price || 0,
                    change: marketData.btc_change_24h || 0,
                    volume: marketData.btc_volume_24h || 0,
                    high: marketData.btc_high_24h || 0,
                    low: marketData.btc_low_24h || 0
                },
                {
                    symbol: 'ETH',
                    price: marketData.eth_price || 0,
                    change: marketData.eth_change_24h || 0,
                    volume: marketData.eth_volume_24h || 0,
                    high: marketData.eth_high_24h || 0,
                    low: marketData.eth_low_24h || 0
                }
            ];
        }
        
    for (const m of marketData) {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${m.symbol}</td>
          <td>$${formatPrice(m.price)}</td>
          <td class="${(m.change_24h ?? 0) >= 0 ? 'positive':'negative'}">
             ${(Number(m.change_24h ?? 0)).toFixed(2)}%
          </td>
          <td>${Number(m.volume_24h ?? 0).toLocaleString()}</td>
          <td>$${formatPrice(m.high_24h)}</td>
          <td>$${formatPrice(m.low_24h)}</td>
          <td>
            <button class="btn btn-sm btn-outline-primary" onclick="analyzeSymbol('${m.symbol}')">Analyze</button>
          </td>
        `;
        tbody.appendChild(row);
    }
}

/**
 * Load settings data
 */
async function loadSettingsData() {
    try {
        const c = await fetchData('/api/config');
        document.getElementById('maxPositionsInput').value = c?.max_positions ?? 50;
        document.getElementById('stopLossInput').value    = ((c?.stop_loss_percent ?? 0) * 100).toFixed(2);
        document.getElementById('takeProfitInput').value  = ((c?.take_profit_percent ?? 0) * 100).toFixed(2);
        document.getElementById('maxDailyLossInput').value= ((c?.max_daily_loss ?? 0) * 100).toFixed(2);
    } catch (e) {
        console.error('Settings load failed', e);
    }
}

/**
 * Analyze symbol function for market data buttons
 */
function analyzeSymbol(symbol) {
    console.log(`Analyzing symbol: ${symbol}`);
    // Add your analysis logic here
    showToast(`Analyzing ${symbol}...`, 'info');
}

/**
 * Update settings form
 */
function updateSettingsForm(config) {
    console.log('Updating settings form with data:', config);
    
    updateElement('maxPositionsInput', config.max_positions || 50);
    updateElement('stopLossInput', config.stop_loss_percent || 1.0);
    updateElement('takeProfitInput', config.take_profit_percent || 2.5);
    updateElement('maxDailyLossInput', config.max_daily_loss || 3.0);
    updateElement('riskPerTradeInput', config.risk_per_trade || 1.0);
    updateElement('maxCorrelationInput', config.max_correlation || 0.7);
    updateElement('volatilityFilterInput', config.volatility_filter ? 'checked' : '');
    updateElement('minVolumeRatioInput', config.min_volume_ratio || 1.5);
    updateElement('positionSizingInput', config.position_sizing || 'fixed');
    updateElement('leverageInput', config.leverage || 1.0);
    updateElement('maxPositionSizeInput', config.max_position_size || 1000.0);
    updateElement('minPositionSizeInput', config.min_position_size || 10.0);
    
    console.log('Settings form updated');
}

/**
 * Start periodic updates
 */
function startPeriodicUpdates() {
    console.log('Starting periodic updates (every 30 seconds)...');
    updateInterval = setInterval(async () => {
        try {
            console.log('Running periodic update...');
            await loadInitialData();
            updateAccountMetrics(currentData.account);
            updatePositionsTable(currentData.positions);
            updateTradesTable(currentData.trades);


        } catch (error) {
            console.error('Error in periodic update:', error);
        }
    }, 30000); // Update every 30 seconds
    console.log('Periodic updates started');
}

/**
 * Start price updates for positions
 */
let priceUpdateTimer = null;
function startPriceUpdates() {
    if (priceUpdateTimer) clearInterval(priceUpdateTimer);
    priceUpdateTimer = setInterval(async () => {
        try {
            const res = await fetchData('/api/positions');
            const list = res?.positions ?? [];
            updatePositionsTable(list);
        } catch (_) {}
    }, 5000);
}

/**
 * Update position prices
 */
async function updatePositionPrices() {
    try {
        const positionsData = await fetchData('/api/positions');
        if (positionsData) {
            const positions = Array.isArray(positionsData) ? positionsData : (positionsData?.positions || []);
            updatePositionsTable(positions);
        }
    } catch (error) {
        console.error('Error updating position prices:', error);
    }
}

/**
 * Load risk data
 */
async function loadRisk() {
    try {
        console.log('Loading risk data...');
        const [metrics, limits] = await Promise.all([
            fetchData('/api/analytics/risk-metrics'),
            fetchData('/api/risk/limits')
        ]);
        
        if (limits) {
            updateElement('maxPositions', limits.max_positions);
            updateElement('currentPositions', limits.current_positions);
            updateElement('maxCorrelation', limits.max_correlation.toFixed(2));
            updateElement('currentCorrelation', (limits.current_correlation || 0).toFixed(2));
        }
        
        if (metrics) {
            const normalizedMetrics = normalizeRisk(metrics);
            updateElement('var95', formatCurrency(normalizedMetrics.var95));
            updateElement('var99', formatCurrency(normalizedMetrics.var99));
            updateElement('maxDrawdown', formatPercentage(normalizedMetrics.max_drawdown));
            updateElement('volatility', formatPercentage(normalizedMetrics.volatility));
            updateElement('beta', normalizedMetrics.beta.toFixed(2));
        }
        
        console.log('Risk data loaded successfully');
    } catch (error) {
        console.error('Error loading risk data:', error);
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    console.log('Setting up event listeners...');
    // Add any additional event listeners here
    console.log('Event listeners setup complete');
}

/**
 * Close position
 */
async function closePosition(symbol) {
    console.log(`Closing position: ${symbol}`);
    
    if (confirm(`Are you sure you want to close position ${symbol}?`)) {
        try {
            console.log(`User confirmed closing position: ${symbol}`);
            showLoading(true);
            
            const response = await fetch(`/api/positions/${symbol}/close`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symbol: symbol,
                    reason: 'manual'
                })
            });
            
            if (response.ok) {
                console.log(`Position ${symbol} closed successfully`);
                showToast(`Position ${symbol} closed successfully`, 'success');
                await loadPositionsData();
            } else {
                const error = await response.json();
                console.error(`Error closing position: ${error.detail || 'Unknown error'}`);
                showToast(`Error closing position: ${error.detail || 'Unknown error'}`, 'error');
            }
            
        } catch (error) {
            console.error('Error closing position:', error);
            showToast('Error closing position', 'error');
        } finally {
            showLoading(false);
        }
    } else {
        console.log(`User cancelled closing position: ${symbol}`);
    }
}

/**
 * Close all positions
 */
async function closeAllPositions() {
    console.log('Closing all positions...');
    
    if (confirm('Are you sure you want to close ALL positions? This action cannot be undone.')) {
        try {
            console.log('User confirmed closing all positions');
            showLoading(true);
            
            const response = await fetch('/api/positions/close-all', {
                method: 'POST'
            });
            
            console.log('Close all positions response:', response.status);
            
            if (response.ok) {
                const result = await response.json();
                console.log('Close all positions result:', result);
                showToast(`Close all positions completed. ${result.results?.length || 0} positions processed.`, 'success');
                await loadPositionsData();
            } else {
                const error = await response.json();
                console.error(`Error closing all positions: ${error.detail || 'Unknown error'}`);
                showToast(`Error closing all positions: ${error.detail || 'Unknown error'}`, 'error');
            }
            
        } catch (error) {
            console.error('Error closing all positions:', error);
            showToast('Error closing all positions', 'error');
        } finally {
            showLoading(false);
        }
    } else {
        console.log('User cancelled closing all positions');
    }
}

/**
 * Refresh all data
 */
async function refreshAllData() {
    try {
        console.log('Refreshing all data...');
        showLoading(true);
        await loadInitialData();
        updateAccountMetrics(currentData.account);
        updatePositionsTable(currentData.positions);
        updateTradesTable(currentData.trades);
        showToast('All data refreshed', 'success');
        console.log('All data refreshed successfully');
    } catch (error) {
        console.error('Error refreshing data:', error);
        showToast('Error refreshing data', 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Refresh positions
 */
async function refreshPositions() {
    console.log('Refreshing positions...');
    await loadPositionsData();
    showToast('Positions refreshed', 'info');
    console.log('Positions refreshed successfully');
}

/**
 * Refresh trades
 */
async function refreshTrades() {
    console.log('Refreshing trades...');
    await loadTradesData();
    showToast('Trades refreshed', 'info');
    console.log('Trades refreshed successfully');
}

/**
 * Refresh market data
 */
async function refreshMarketData() {
    console.log('Refreshing market data...');
    await loadMarketData();
    showToast('Market data refreshed', 'info');
    console.log('Market data refreshed successfully');
}

/**
 * Filter trades by strategy
 */
function filterTrades() {
    const strategy = document.getElementById('strategyFilter').value;
    console.log('Filtering trades by strategy:', strategy);
    // Implement filtering logic here
    console.log('Trades filtered by strategy:', strategy);
}

/**
 * Analyze symbol
 */
function analyzeSymbol(symbol) {
    console.log(`Analyzing symbol: ${symbol}`);
    showToast(`Analyzing ${symbol}...`, 'info');
    // Implement symbol analysis here
    console.log(`Symbol analysis completed for: ${symbol}`);
}

/**
 * Export data
 */
function exportData() {
    console.log('Exporting data...');
    showToast('Exporting data...', 'info');
    // Implement data export here
    console.log('Data export completed');
}

/**
 * Show system info
 */
function showSystemInfo() {
    console.log('Showing system info...');
    
    const info = {
        'Dashboard Version': '2.0.0',
        'Connection Status': isConnected ? 'Connected' : 'Disconnected',
        'Last Update': new Date().toLocaleString(),
        'Browser': navigator.userAgent,
        'Screen Resolution': `${screen.width}x${screen.height}`
    };
    
    console.log('System info:', info);
    
    let message = 'System Information:\n\n';
    for (const [key, value] of Object.entries(info)) {
        message += `${key}: ${value}\n`;
    }
    
    console.log('System info message:', message);
    alert(message);
    console.log('System info displayed');
}

/**
 * Save trading configuration
 */
async function saveTradingConfig() {
    try {
        console.log('Saving trading configuration...');
        
        const config = {
            max_positions: parseInt(document.getElementById('maxPositionsInput').value),
            stop_loss_percent: parseFloat(document.getElementById('stopLossInput').value) / 100,
            take_profit_percent: parseFloat(document.getElementById('takeProfitInput').value) / 100,
            max_daily_loss: parseFloat(document.getElementById('maxDailyLossInput').value) / 100
        };
        
        console.log('Trading config:', config);
        
        const response = await fetch('/api/config/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            console.log('Configuration saved successfully');
            showToast('Configuration saved successfully', 'success');
        } else {
            console.error('Error saving configuration:', response.status);
            showToast('Error saving configuration', 'error');
        }
        
    } catch (error) {
        console.error('Error saving configuration:', error);
        showToast('Error saving configuration', 'error');
    }
}

/**
 * Utility functions
 */
function updateElement(id, value, className = null) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
        if (className) {
            element.className = className;
        }
        console.log(`Updated element ${id} with value: ${value}`);
    } else {
        console.warn(`Element with id ${id} not found`);
    }
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

/**
 * Format price with appropriate precision
 */
function formatPrice(v) {
    const x = Number(v || 0);
    return x < 1 ? x.toFixed(6) : x.toFixed(4);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

function formatNumber(value, decimals = 2) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZone: 'UTC'
    });
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
    
    if (ws) {
        ws.close();
    }
});

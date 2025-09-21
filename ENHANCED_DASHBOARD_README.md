# 🚀 ClaudeBot Enhanced Dashboard

A sophisticated, real-time trading dashboard for the ClaudeBot trading system with advanced analytics, comprehensive monitoring, and professional UI.

## ✨ Features

### 🎯 **Core Features**
- **Real-time Updates**: WebSocket-based live data streaming
- **Multi-Strategy Support**: Momentum, Mean Reversion, Arbitrage
- **Advanced Analytics**: Performance metrics, risk analysis, strategy comparison
- **Professional UI**: Modern, responsive design with dark/light themes
- **Interactive Charts**: Chart.js and ApexCharts integration
- **Position Management**: Real-time position tracking and management
- **Trade History**: Comprehensive trade analysis and reporting

### 📊 **Analytics & Monitoring**
- **Portfolio Performance**: Real-time P&L tracking and visualization
- **Strategy Performance**: Individual strategy analysis and comparison
- **Risk Metrics**: VaR, drawdown analysis, correlation monitoring
- **Market Data**: Live market data with technical indicators
- **Performance Charts**: Interactive charts with multiple timeframes

### 🔧 **Technical Features**
- **FastAPI Backend**: High-performance async API
- **WebSocket Support**: Real-time bidirectional communication
- **Database Integration**: PostgreSQL with connection pooling
- **Redis Caching**: Optional Redis integration for performance
- **Docker Support**: Complete containerization setup
- **Nginx Proxy**: Production-ready reverse proxy configuration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (HTML/CSS/JS) │◄──►│   (FastAPI)     │◄──►│   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   WebSocket     │              │
         └──────────────►│   (Real-time)   │◄─────────────┘
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   Redis Cache   │
                        │   (Optional)    │
                        └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis (optional)
- Docker & Docker Compose (optional)

### Installation

#### Option 1: Direct Installation
```bash
# Clone the repository
git clone <repository-url>
cd ClaudeBot

# Install dependencies
pip install -r enhanced_dashboard_requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://claudebot:password@localhost:5432/claudebot_db"
export DASHBOARD_PORT=8001

# Run the dashboard
python run_enhanced_dashboard.py
```

#### Option 2: Docker Compose
```bash
# Start the enhanced dashboard
docker-compose -f enhanced_dashboard_docker.yml up -d

# Check status
docker-compose -f enhanced_dashboard_docker.yml ps

# View logs
docker-compose -f enhanced_dashboard_docker.yml logs -f enhanced-dashboard
```

### Access the Dashboard
- **URL**: http://localhost:8001
- **API Docs**: http://localhost:8001/api/docs
- **Health Check**: http://localhost:8001/api/health

## 📁 Project Structure

```
enhanced_dashboard/
├── templates/
│   └── dashboard.html          # Main HTML template
├── static/
│   ├── css/
│   │   └── dashboard.css       # Custom styles
│   ├── js/
│   │   └── dashboard.js        # Frontend JavaScript
│   └── images/                 # Static images
├── nginx.conf                  # Nginx configuration
└── ...

enhanced_dashboard.py           # Main FastAPI application
enhanced_dashboard_config.py    # Configuration settings
run_enhanced_dashboard.py       # Dashboard runner script
enhanced_dashboard_requirements.txt  # Python dependencies
enhanced_dashboard_docker.yml   # Docker Compose configuration
enhanced_dashboard.Dockerfile   # Docker image definition
```

## ⚙️ Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@host:port/database

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_password

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8001
ENVIRONMENT=production

# Security
SECRET_KEY=your_secret_key
CORS_ORIGINS=["http://localhost:3000"]
```

### Configuration Files
- `enhanced_dashboard_config.py`: Main configuration
- `enhanced_dashboard_docker.yml`: Docker services
- `enhanced_dashboard/nginx.conf`: Nginx proxy settings

## 🔌 API Endpoints

### Core Endpoints
- `GET /` - Main dashboard
- `GET /api/health` - Health check
- `GET /api/account` - Account information
- `GET /api/positions` - Current positions
- `GET /api/trades` - Trade history
- `GET /api/performance` - Performance metrics

### Analytics Endpoints
- `GET /api/analytics/strategy-performance` - Strategy analysis
- `GET /api/analytics/risk-metrics` - Risk metrics
- `GET /api/analytics/portfolio-charts` - Chart data

### Management Endpoints
- `POST /api/positions/{symbol}/close` - Close position
- `POST /api/positions/close-all` - Close all positions
- `GET /api/config` - Get configuration
- `POST /api/config/update` - Update configuration

### WebSocket
- `WS /ws` - Real-time updates

## 📊 Dashboard Sections

### 1. **Overview**
- Key metrics cards (Balance, P&L, Positions, Win Rate)
- Portfolio performance chart
- Strategy distribution pie chart
- Strategy performance table

### 2. **Positions**
- Current positions table
- Real-time P&L updates
- Position management actions
- Duration tracking

### 3. **Trades**
- Recent trades table
- Strategy filtering
- P&L analysis
- Trade duration metrics

### 4. **Analytics**
- P&L distribution charts
- Drawdown analysis
- Performance metrics
- Risk analysis

### 5. **Risk Management**
- Value at Risk (VaR)
- Position limits
- Correlation analysis
- Risk alerts

### 6. **Market Data**
- Live market prices
- 24h change tracking
- Volume analysis
- Technical indicators

### 7. **Settings**
- Trading configuration
- Strategy settings
- Risk parameters
- System information

## 🎨 UI Features

### Design System
- **Color Scheme**: Professional blue/green palette
- **Typography**: Nunito font family
- **Components**: Bootstrap 5 with custom styling
- **Icons**: Font Awesome 6
- **Charts**: Chart.js + ApexCharts

### Responsive Design
- Mobile-first approach
- Breakpoints: xs(576px), sm(768px), md(992px), lg(1200px), xl(1400px)
- Touch-friendly interface
- Optimized for tablets and phones

### Animations
- Smooth transitions
- Loading states
- Chart animations
- Hover effects

## 🔧 Development

### Local Development
```bash
# Install development dependencies
pip install -r enhanced_dashboard_requirements.txt

# Run in development mode
export ENVIRONMENT=development
python run_enhanced_dashboard.py
```

### Code Quality
```bash
# Format code
black enhanced_dashboard.py run_enhanced_dashboard.py

# Lint code
flake8 enhanced_dashboard.py run_enhanced_dashboard.py

# Run tests
pytest tests/
```

### Adding New Features
1. **Backend**: Add endpoints in `enhanced_dashboard.py`
2. **Frontend**: Update `dashboard.html` and `dashboard.js`
3. **Styling**: Modify `dashboard.css`
4. **Configuration**: Update `enhanced_dashboard_config.py`

## 🐳 Docker Deployment

### Production Deployment
```bash
# Build and start services
docker-compose -f enhanced_dashboard_docker.yml up -d

# Scale dashboard instances
docker-compose -f enhanced_dashboard_docker.yml up -d --scale enhanced-dashboard=3

# Update services
docker-compose -f enhanced_dashboard_docker.yml pull
docker-compose -f enhanced_dashboard_docker.yml up -d
```

### Health Monitoring
```bash
# Check service health
docker-compose -f enhanced_dashboard_docker.yml ps

# View logs
docker-compose -f enhanced_dashboard_docker.yml logs -f enhanced-dashboard

# Monitor resources
docker stats claudebot-enhanced-dashboard
```

## 🔒 Security

### Security Features
- CORS protection
- Rate limiting
- Input validation
- SQL injection prevention
- XSS protection
- CSRF protection

### Production Security
- HTTPS support
- Security headers
- Access logging
- Error handling
- Input sanitization

## 📈 Performance

### Optimization Features
- Connection pooling
- Redis caching
- Gzip compression
- Static file caching
- Chart optimization
- Lazy loading

### Monitoring
- Health checks
- Performance metrics
- Error tracking
- Uptime monitoring
- Resource usage

## 🐛 Troubleshooting

### Common Issues

#### Database Connection
```bash
# Check database connectivity
python -c "from data.database import DatabaseManager; print('DB OK')"

# Test connection
psql -h localhost -U claudebot -d claudebot_db -c "SELECT 1;"
```

#### WebSocket Issues
```bash
# Check WebSocket connection
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Key: test" -H "Sec-WebSocket-Version: 13" http://localhost:8001/ws
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8001
lsof -i :8001

# Change port
export DASHBOARD_PORT=8002
```

### Logs
```bash
# Application logs
tail -f enhanced_dashboard.log

# Docker logs
docker logs claudebot-enhanced-dashboard -f

# Nginx logs
docker logs claudebot-nginx-enhanced -f
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/api/docs`

## 🔄 Updates

### Version 2.0.0
- Initial release of enhanced dashboard
- Real-time WebSocket support
- Advanced analytics and charts
- Professional UI design
- Docker containerization
- Comprehensive API

---

**Built with ❤️ for ClaudeBot Trading System**

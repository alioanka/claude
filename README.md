# 🚀 Advanced Crypto Trading Bot

An intelligent, fully automated cryptocurrency trading bot with machine learning capabilities, advanced risk management, and real-time monitoring.

## ✨ Key Features

- 🤖 **Fully Automated Trading** - Autonomous pair selection, position sizing, and execution
- 🧠 **Machine Learning Integration** - LSTM, XGBoost, and Transformer models
- 📊 **Multiple Trading Strategies** - Momentum, Mean Reversion, ML-based, and Arbitrage
- ⚡ **Real-time Data Processing** - WebSocket and REST API data collection
- 🛡️ **Advanced Risk Management** - Dynamic position sizing and portfolio optimization
- 📱 **Telegram Notifications** - Real-time alerts and performance updates
- 🔄 **Paper & Live Trading** - Seamless switching between modes
- 📈 **Performance Monitoring** - Comprehensive analytics and reporting
- 🐳 **Docker Deployment** - Production-ready containerized setup
- ☁️ **Cloud Ready** - VPS deployment with monitoring stack

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │ Strategy Layer  │    │ Execution Layer │
│                 │    │                 │    │                 │
│ • Market Data   │───▶│ • ML Models     │───▶│ • Order Manager │
│ • News Feed     │    │ • Risk Manager  │    │ • Portfolio Mgr │
│ • Sentiment     │    │ • Signal Gen    │    │ • Exchange API  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Git
- Binance account with API keys
- Python 3.9+ (for development)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

### 2. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (REQUIRED)
nano .env
```

### 3. Configure API Keys

Add your Binance API keys to `.env`:

```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET=your_secret_here
TRADING_MODE=paper  # Start with paper trading
```

### 4. Deploy with Docker

```bash
# Make script executable
chmod +x run_bot.sh

# Start the bot
./run_bot.sh start
```

### 5. Monitor Performance

```bash
# View real-time logs
./run_bot.sh logs

# Monitor performance
./run_bot.sh monitor

# Check health
./run_bot.sh health
```

## 📋 Configuration

### Trading Settings

```bash
# Trading Configuration
TRADING_MODE=paper          # paper or live
INITIAL_CAPITAL=10000       # Starting capital
MAX_POSITIONS=5             # Maximum concurrent positions
RISK_PER_TRADE=0.02        # 2% risk per trade
MAX_DRAWDOWN=0.15          # 15% maximum drawdown
```

### Strategy Allocation

Edit `config/strategies.json`:

```json
{
    "strategy_allocation": {
        "momentum_strategy": 0.3,
        "mean_reversion": 0.25,
        "ml_strategy": 0.35,
        "arbitrage_strategy": 0.1
    }
}
```

### Trading Pairs

Edit `config/trading_pairs.json`:

```json
{
    "active_pairs": [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "ADAUSDT",
        "XRPUSDT"
    ]
}
```

## 🤖 Trading Strategies

### 1. Momentum Strategy
- **Indicators**: EMA crossovers, RSI, MACD, Bollinger Bands
- **Logic**: Trend following with volume confirmation
- **Risk**: Trailing stops and volatility-based position sizing

### 2. Mean Reversion Strategy
- **Indicators**: Bollinger Bands, RSI, Stochastic, Z-Score
- **Logic**: Counter-trend trading on overextended moves
- **Risk**: Time-based exits and mean reversion signals

### 3. ML Strategy (Ensemble)
- **Models**: LSTM, XGBoost, Transformer
- **Features**: Technical indicators, market structure, sentiment
- **Logic**: Ensemble predictions with confidence thresholds

### 4. Statistical Arbitrage
- **Logic**: Pair correlation analysis and spread trading
- **Risk**: Correlation monitoring and position limits

## 🧠 Machine Learning Features

### Feature Engineering
- 50+ technical indicators
- Market microstructure features
- Sentiment analysis from news/social media
- Volume profile and order book imbalance

### Model Architecture
```python
# LSTM Model
- Sequence length: 60 candles
- Hidden layers: 128 -> 64 neurons
- Dropout: 20% regularization
- Output: Price direction probability

# XGBoost Model
- 100+ engineered features
- Tree depth: 6
- Learning rate: 0.1
- Regularization: L1/L2

# Transformer Model
- Attention heads: 8
- Multi-timeframe inputs
- Position encoding
```

### Model Retraining
- **Frequency**: Every 24 hours (configurable)
- **Data**: Rolling 30-day window
- **Validation**: Walk-forward analysis
- **Performance**: Automatic model selection

## 🛡️ Risk Management

### Position Sizing
- **Kelly Criterion**: Optimal position sizing based on win rate
- **Volatility Adjustment**: Dynamic sizing based on ATR
- **Correlation Limits**: Maximum 70% correlation between positions

### Risk Controls
- **Maximum Position Size**: 10% of portfolio
- **Stop Loss**: 3% (configurable)
- **Take Profit**: 6% (configurable)
- **Maximum Drawdown**: 15% portfolio stop
- **Daily Loss Limit**: 5% daily drawdown limit

### Portfolio Management
- **Rebalancing**: Hourly portfolio optimization
- **Diversification**: Sector and correlation limits
- **Leverage**: Dynamic leverage based on volatility

## 📊 Performance Monitoring

### Real-time Metrics
- Total P&L and daily performance
- Win rate and Sharpe ratio
- Active positions and exposure
- Strategy performance breakdown

### Notifications
- Trade entries and exits
- Risk alerts and portfolio updates
- System errors and maintenance
- Daily/weekly performance summaries

### Reporting
- Performance attribution by strategy
- Risk metrics and drawdown analysis
- Trading frequency and costs
- Model accuracy and predictions

## 🔧 Development Setup

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py
```

### Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Run specific test
pytest tests/test_strategies.py
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## 🚀 Deployment

### VPS Deployment

1. **Setup VPS** (Ubuntu 20.04+ recommended)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

2. **Clone and Configure**
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
cp .env.example .env
# Edit .env with your settings
```

3. **Deploy**
```bash
./run_bot.sh start
```

### Production Monitoring

```bash
# Install monitoring stack
./run_bot.sh monitoring

# Access Grafana dashboard
# http://your-vps-ip:3000 (admin/admin_password)
```

### SSL/Security Setup

```bash
# Install Nginx
sudo apt install nginx

# Configure SSL with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

## 📱 Management Commands

```bash
# Service Management
./run_bot.sh start          # Start all services
./run_bot.sh stop           # Stop all services
./run_bot.sh restart        # Restart services
./run_bot.sh status         # Show status

# Monitoring
./run_bot.sh logs           # Show logs
./run_bot.sh monitor        # Real-time monitoring
./run_bot.sh health         # Health check

# Maintenance
./run_bot.sh backup         # Create backup
./run_bot.sh restore <file> # Restore from backup
./run_bot.sh cleanup        # Clean old data
./run_bot.sh update         # Update bot code

# Emergency
./run_bot.sh emergency      # Emergency stop
```

## ⚠️ Important Warnings

### Live Trading Risks
- **Start with paper trading** to validate strategies
- **Never risk more than you can afford to lose**
- **Monitor positions regularly** during initial deployment
- **Test thoroughly** before switching to live mode
- **Keep API keys secure** and use IP restrictions

### Performance Expectations
- **Realistic targets**: 10-20% monthly returns (not 500% weekly)
- **Market conditions**: Performance varies with market volatility
- **Drawdown periods**: Expect temporary losses during optimization
- **Continuous monitoring**: Automated doesn't mean unsupervised

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **GitHub Issues**: Bug reports and feature requests
- **Telegram**: @YourTradingBotSupport
- **Email**: support@yourdomain.com

## 🙏 Acknowledgments

- Binance API for market data
- ccxt library for exchange integration
- Various ML libraries (TensorFlow, scikit-learn, XGBoost)
- Docker for containerization

---

## ⚡ Quick Commands Reference

```bash
# Setup
git clone <repo> && cd crypto-trading-bot
cp .env.example .env && nano .env
./run_bot.sh start

# Monitor
./run_bot.sh logs
./run_bot.sh status
./run_bot.sh health

# Maintenance
./run_bot.sh backup
./run_bot.sh cleanup
./run_bot.sh update
```

**Happy Trading! 🚀📈**
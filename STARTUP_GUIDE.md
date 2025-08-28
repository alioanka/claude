# 🚀 Trading Bot Startup Guide

## Quick Setup (5 minutes)

### 1. Prerequisites Check
```bash
# Check Docker
docker --version
docker-compose --version

# Check Git
git --version
```

### 2. Clone and Setup
```bash
# Clone repository
git clone https://github.com/your-username/crypto-trading-bot.git
cd crypto-trading-bot

# Copy environment file
cp .env.example .env

# Create required directories
mkdir -p logs storage/{historical,models,backups,exports}
```

### 3. Configure Environment
Edit `.env` file with your settings:

```bash
# REQUIRED: Exchange API credentials
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET=your_secret_key_here

# REQUIRED: Trading mode
TRADING_MODE=paper

# REQUIRED: Initial capital
INITIAL_CAPITAL=10000

# OPTIONAL: Telegram notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 4. Start the Bot
```bash
# Make script executable
chmod +x run_bot.sh

# Start everything
./run_bot.sh start
```

### 5. Verify Setup
```bash
# Check logs
./run_bot.sh logs

# Check health
./run_bot.sh health

# Check status
./run_bot.sh status
```

---

## 🔧 Troubleshooting Common Issues

### Issue: No Logs Appearing

**Problem**: The `logs/` directory is empty after starting.

**Solutions**:
1. Check directory permissions:
```bash
sudo chown -R $USER:$USER logs/
chmod 755 logs/
```

2. Check Docker volume mounting:
```bash
docker-compose down
docker-compose up trading-bot -d
docker-compose logs trading-bot
```

3. Force log creation:
```bash
# Create log files manually
touch logs/trading_bot.log logs/error.log logs/trades.log
chmod 666 logs/*.log
```

### Issue: Database Connection Errors

**Problem**: `postgresql connection failed` or similar.

**Solutions**:
1. Check PostgreSQL container:
```bash
docker-compose ps postgres
docker-compose logs postgres
```

2. Wait for PostgreSQL to be ready:
```bash
# PostgreSQL might take 30-60 seconds to initialize
./run_bot.sh health
```

3. Reset database:
```bash
docker-compose down
docker volume rm crypto-trading-bot_postgres_data
docker-compose up postgres -d
# Wait 60 seconds, then start bot
./run_bot.sh start
```

### Issue: Binance API Errors

**Problem**: `API key invalid` or `Permission denied`.

**Solutions**:
1. Verify API keys in `.env`:
```bash
grep -E "(BINANCE_API_KEY|BINANCE_SECRET)" .env
```

2. Check Binance API permissions:
   - Enable "Spot & Margin Trading" for live trading
   - Enable "Futures Trading" if using futures
   - Add your server's IP to whitelist

3. Test API connection:
```bash
# Enter the container and test
docker-compose exec trading-bot python -c "
import ccxt
import os
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET'),
    'sandbox': False
})
print(exchange.fetch_balance())
"
```

### Issue: High Memory/CPU Usage

**Problem**: System running slow or containers crashing.

**Solutions**:
1. Check resource usage:
```bash
docker stats
```

2. Reduce data collection frequency:
```bash
# In .env file
DATA_COLLECTION_INTERVAL=300  # 5 minutes instead of 1 minute
```

3. Limit ML model complexity:
```bash
# In .env file
ML_RETRAIN_INTERVAL=168  # Weekly instead of daily
FEATURE_LOOKBACK=50     # Reduce from 100
```

### Issue: Telegram Notifications Not Working

**Problem**: Bot starts but no Telegram messages.

**Solutions**:
1. Verify bot token and chat ID:
```bash
# Test with curl
curl -X GET "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getMe"
```

2. Get your chat ID:
```bash
# Send a message to your bot first, then:
curl -X GET "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates"
```

3. Check firewall:
```bash
# Ensure outbound HTTPS is allowed
curl -I https://api.telegram.org
```

### Issue: Paper Trading Not Working

**Problem**: Bot starts but no trades are executed in paper mode.

**Solutions**:
1. Check trading pair configuration:
```bash
# View config/trading_pairs.json
cat config/trading_pairs.json
```

2. Verify data collection:
```bash
docker-compose logs trading-bot | grep -i "data"
```

3. Check strategy configuration:
```bash
# View config/strategies.json
cat config/strategies.json
```

### Issue: Docker Build Failures

**Problem**: `docker-compose build` fails.

**Solutions**:
1. Update Docker:
```bash
sudo apt update
sudo apt install docker.io docker-compose
```

2. Clear Docker cache:
```bash
docker system prune -a
```

3. Build with no cache:
```bash
docker-compose build --no-cache
```

---

## 📊 Monitoring and Maintenance

### Daily Checks
```bash
# Check bot status
./run_bot.sh status

# View recent logs
./run_bot.sh logs | tail -50

# Check performance
./run_bot.sh monitor
```

### Weekly Maintenance
```bash
# Create backup
./run_bot.sh backup

# Clean old data
./run_bot.sh cleanup

# Update bot
./run_bot.sh update
```

### Performance Monitoring
```bash
# Enable monitoring stack
./run_bot.sh monitoring

# Access Grafana dashboard
# http://localhost:3000 (admin/admin_password)
```

---

## 🆘 Emergency Procedures

### Emergency Stop
```bash
# Immediate stop (closes positions if in live mode)
./run_bot.sh emergency
```

### Restore from Backup
```bash
# List available backups
ls storage/backups/

# Restore from backup
./run_bot.sh restore storage/backups/backup_YYYYMMDD_HHMMSS.tar.gz
```

### Reset Everything
```bash
# DANGER: This will delete all data
docker-compose down
docker volume prune -f
docker system prune -f
rm -rf logs/* storage/backups/*
./run_bot.sh start
```

---

## 📞 Getting Help

### Check Logs First
```bash
# Bot logs
./run_bot.sh logs

# All service logs
docker-compose logs

# Specific service logs
docker-compose logs postgres
docker-compose logs redis
```

### Collect Debug Information
```bash
# System info
./run_bot.sh health

# Docker info
docker system info
docker-compose ps

# Resource usage
docker stats --no-stream
```

### Log Locations
- **Bot logs**: `logs/trading_bot.log`
- **Error logs**: `logs/error.log`
- **Trade logs**: `logs/trades.log`
- **Docker logs**: `docker-compose logs [service_name]`

### Configuration Files
- **Environment**: `.env`
- **Trading pairs**: `config/trading_pairs.json`
- **Strategies**: `config/strategies.json`
- **Risk management**: `config/risk_management.json`

---

## 🔐 Security Best Practices

1. **API Keys**: 
   - Never commit `.env` to version control
   - Use IP restrictions on Binance
   - Rotate keys regularly

2. **Network Security**:
   - Use firewall rules
   - Limit exposed ports
   - Use VPN for remote access

3. **Data Protection**:
   - Encrypt backups
   - Secure log files
   - Regular security updates

---

## 📈 Performance Optimization

### For High-Frequency Trading
```bash
# In .env file
DATA_COLLECTION_INTERVAL=10
HEALTH_CHECK_INTERVAL=60
LOG_LEVEL=WARNING
```

### For Low-Resource Systems
```bash
# In .env file
ML_RETRAIN_INTERVAL=168
FEATURE_LOOKBACK=30
MAX_POSITIONS=3
```

### For Production Deployment
```bash
# Use production profile
docker-compose --profile production up -d
```

---

This guide covers the most common setup and troubleshooting scenarios. For specific issues not covered here, check the detailed logs and error messages for more context.
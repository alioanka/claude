#!/usr/bin/env python3
"""
Enhanced Dashboard Configuration
Configuration file for the ClaudeBot Enhanced Dashboard
"""

import os
from pathlib import Path

# Dashboard Configuration
DASHBOARD_CONFIG = {
    # Server Configuration
    'host': '0.0.0.0',
    'port': 8001,  # Different from default dashboard (8000)
    'debug': False,
    'reload': False,
    
    # Database Configuration
    'database': {
        'url': os.getenv('DATABASE_URL', 'postgresql://trader:secure_password@postgres:5432/trading_bot'),
        'pool_size': 10,
        'max_overflow': 20,
        'pool_timeout': 30,
        'pool_recycle': 3600
    },
    
    # Redis Configuration (for caching)
    'redis': {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', 6379)),
        'db': 1,  # Use different DB from main bot
        'password': os.getenv('REDIS_PASSWORD', None),
        'decode_responses': True
    },
    
    # WebSocket Configuration
    'websocket': {
        'ping_interval': 20,
        'ping_timeout': 10,
        'close_timeout': 10,
        'max_size': 2**20,  # 1MB
        'max_queue': 32
    },
    
    # Caching Configuration
    'cache': {
        'ttl': 300,  # 5 minutes
        'max_size': 1000,
        'cleanup_interval': 600  # 10 minutes
    },
    
    # Security Configuration
    'security': {
        'cors_origins': ['*'],
        'cors_methods': ['*'],
        'cors_headers': ['*'],
        'rate_limit': {
            'enabled': True,
            'requests_per_minute': 60,
            'burst_size': 10
        }
    },
    
    # Logging Configuration
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'enhanced_dashboard.log',
        'max_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    },
    
    # Chart Configuration
    'charts': {
        'update_interval': 30,  # seconds
        'max_data_points': 100,
        'default_timeframe': '7d'
    },
    
    # Performance Configuration
    'performance': {
        'max_concurrent_requests': 100,
        'request_timeout': 30,
        'background_task_interval': 60
    }
}

# API Endpoints Configuration
API_ENDPOINTS = {
    'account': '/api/account',
    'positions': '/api/positions',
    'trades': '/api/trades',
    'performance': '/api/performance',
    'analytics': {
        'strategy_performance': '/api/analytics/strategy-performance',
        'risk_metrics': '/api/analytics/risk-metrics',
        'portfolio_charts': '/api/analytics/portfolio-charts'
    },
    'market_data': '/api/market-data',
    'config': '/api/config',
    'websocket': '/ws'
}

# Chart Types Configuration
CHART_TYPES = {
    'portfolio_performance': {
        'type': 'line',
        'height': 300,
        'responsive': True,
        'interactive': True
    },
    'strategy_distribution': {
        'type': 'doughnut',
        'height': 250,
        'responsive': True,
        'interactive': True
    },
    'pnl_distribution': {
        'type': 'histogram',
        'height': 300,
        'responsive': True,
        'interactive': True
    },
    'drawdown': {
        'type': 'area',
        'height': 300,
        'responsive': True,
        'interactive': True
    }
}

# Strategy Configuration
STRATEGY_CONFIG = {
    'momentum_strategy': {
        'name': 'Momentum Strategy',
        'enabled': True,
        'allocation': 0.6,
        'color': '#4e73df',
        'description': 'Trend-following strategy based on price momentum'
    },
    'mean_reversion': {
        'name': 'Mean Reversion',
        'enabled': True,
        'allocation': 0.3,
        'color': '#1cc88a',
        'description': 'Mean reversion strategy based on statistical analysis'
    },
    'arbitrage_strategy': {
        'name': 'Arbitrage Strategy',
        'enabled': True,
        'allocation': 0.1,
        'color': '#f6c23e',
        'description': 'Cross-exchange and statistical arbitrage opportunities'
    }
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_positions': 50,
    'max_daily_loss': 0.03,  # 3%
    'max_position_size': 0.01,  # 1%
    'stop_loss_percent': 0.01,  # 1%
    'take_profit_percent': 0.025,  # 2.5%
    'max_correlation': 0.7,
    'volatility_filter': True,
    'min_volume_ratio': 1.5
}

# UI Configuration
UI_CONFIG = {
    'theme': 'light',
    'primary_color': '#4e73df',
    'success_color': '#1cc88a',
    'warning_color': '#f6c23e',
    'danger_color': '#e74a3b',
    'info_color': '#36b9cc',
    'refresh_interval': 30000,  # 30 seconds
    'animation_duration': 500,  # milliseconds
    'chart_animation': True,
    'responsive_breakpoints': {
        'xs': 576,
        'sm': 768,
        'md': 992,
        'lg': 1200,
        'xl': 1400
    }
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    'enabled': True,
    'types': {
        'position_opened': True,
        'position_closed': True,
        'trade_executed': True,
        'error_occurred': True,
        'risk_alert': True,
        'performance_milestone': True
    },
    'channels': {
        'toast': True,
        'console': True,
        'websocket': True
    }
}

# Export Configuration
EXPORT_CONFIG = {
    'formats': ['csv', 'json', 'xlsx'],
    'max_records': 10000,
    'include_metadata': True,
    'compression': True
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'health_check_interval': 60,  # seconds
    'metrics_collection': True,
    'performance_tracking': True,
    'error_tracking': True,
    'uptime_monitoring': True
}

def get_config():
    """Get the complete configuration"""
    return {
        'dashboard': DASHBOARD_CONFIG,
        'api_endpoints': API_ENDPOINTS,
        'chart_types': CHART_TYPES,
        'strategy_config': STRATEGY_CONFIG,
        'risk_config': RISK_CONFIG,
        'ui_config': UI_CONFIG,
        'notification_config': NOTIFICATION_CONFIG,
        'export_config': EXPORT_CONFIG,
        'monitoring_config': MONITORING_CONFIG
    }

def get_database_url():
    """Get database URL from environment or config"""
    return DASHBOARD_CONFIG['database']['url']

def get_redis_url():
    """Get Redis URL from environment or config"""
    redis_config = DASHBOARD_CONFIG['redis']
    if redis_config['password']:
        return f"redis://:{redis_config['password']}@{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
    else:
        return f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"

def is_development():
    """Check if running in development mode"""
    return os.getenv('ENVIRONMENT', 'production').lower() == 'development'

def is_production():
    """Check if running in production mode"""
    return os.getenv('ENVIRONMENT', 'production').lower() == 'production'

def get_log_level():
    """Get logging level"""
    if is_development():
        return 'DEBUG'
    return DASHBOARD_CONFIG['logging']['level']

def get_port():
    """Get dashboard port"""
    return int(os.getenv('DASHBOARD_PORT', DASHBOARD_CONFIG['port']))

def get_host():
    """Get dashboard host"""
    return os.getenv('DASHBOARD_HOST', DASHBOARD_CONFIG['host'])

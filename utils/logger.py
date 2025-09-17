"""
Logger Configuration - Advanced logging system
Provides structured logging with different levels and outputs.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional
import structlog

from config.config import config

# Add these module-level singletons near the top of the file (once):
_ROOT_LOGGER_NAME = "trading_bot"
_ROOT_INITIALIZED = False

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Set up a single rotating handler on the root, and let children propagate."""
    global _ROOT_INITIALIZED

    os.makedirs('logs', exist_ok=True)

    # structlog stays as-is
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 1) Initialize the ONE root only once
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if not _ROOT_INITIALIZED:
        log_level = (level or os.getenv('LOG_LEVEL', 'INFO')).upper()
        root.setLevel(getattr(logging, log_level))

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # IMPORTANT: delay=True and encoding to reduce contention and mojibake
        file_handler = logging.handlers.RotatingFileHandler(
            'logs/trading_bot.log', maxBytes=10*1024*1024, backupCount=5,
            delay=True, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)

        # Error and warning handler - captures both ERROR and WARNING levels
        error_handler = logging.handlers.RotatingFileHandler(
            'logs/error.log', maxBytes=10*1024*1024, backupCount=5,
            delay=True, encoding='utf-8'
        )
        error_handler.setLevel(logging.WARNING)  # Changed from ERROR to WARNING

        # Risk management handler - separate file for risk rejections
        risk_handler = logging.handlers.RotatingFileHandler(
            'logs/risk.log', maxBytes=5*1024*1024, backupCount=10,
            delay=True, encoding='utf-8'
        )
        risk_handler.setLevel(logging.WARNING)
        risk_handler.addFilter(lambda r: (
            'risk' in r.name.lower() or 
            'max position' in r.message.lower() or
            'risk management rejection' in r.message.lower() or
            'signal blocked by risk management' in r.message.lower() or
            'max positions' in r.message.lower() or
            'position limit' in r.message.lower() or
            'exposure limit' in r.message.lower() or
            'drawdown limit' in r.message.lower()
        ))

        # Trades go to a separate file; filter so only trade/portfolio logs hit it
        trade_handler = logging.handlers.RotatingFileHandler(
            'logs/trades.log', maxBytes=10*1024*1024, backupCount=10,
            delay=True, encoding='utf-8'
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.addFilter(lambda r: ('trade' in r.name.lower()) or ('portfolio' in r.name.lower()))

        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        )
        console_formatter = ColoredFormatter('%(asctime)s | %(levelname_colored)s | %(name_colored)s | %(message)s')
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )

        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(detailed_formatter)
        error_handler.setFormatter(detailed_formatter)
        risk_handler.setFormatter(detailed_formatter)
        trade_handler.setFormatter(json_formatter)

        root.addHandler(console_handler)
        root.addHandler(file_handler)
        root.addHandler(error_handler)
        root.addHandler(risk_handler)
        root.addHandler(trade_handler)

        _ROOT_INITIALIZED = True

    # 2) Return a child logger that just propagates to the root handlers
    child = logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
    child.propagate = True
    child.handlers = []  # ensure no duplicate handlers on children
    return child


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    
    NAME_COLORS = {
        'main': '\033[94m',           # Blue
        'core.bot': '\033[95m',       # Magenta
        'data.collector': '\033[96m', # Cyan
        'strategies': '\033[92m',     # Light Green
        'risk': '\033[93m',           # Light Yellow
        'ml': '\033[91m',             # Light Red
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # Add colored level name
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname_colored = f"{level_color}{self.BOLD}{record.levelname:8s}{self.RESET}"
        
        # Add colored logger name
        logger_name = record.name.split('.')[-1] if '.' in record.name else record.name
        name_color = self.NAME_COLORS.get(record.name, '\033[37m')  # Default white
        record.name_colored = f"{name_color}{logger_name:15s}{self.RESET}"
        
        return super().format(record)

class RiskLogger:
    """Specialized logger for risk management operations"""
    
    def __init__(self, name: str):
        self.logger = setup_logger(name)
        self.rejection_count = 0
        self.session_start = datetime.utcnow()
    
    def log_position_rejection(self, symbol: str, reason: str, current_positions: int, max_positions: int):
        """Log position limit rejection"""
        self.rejection_count += 1
        self.logger.warning(
            f"🚫 POSITION REJECTION #{self.rejection_count}: {symbol} - {reason} "
            f"(Current: {current_positions}/{max_positions})"
        )
    
    def log_exposure_rejection(self, symbol: str, current_exposure: float, max_exposure: float, reason: str):
        """Log exposure limit rejection"""
        self.rejection_count += 1
        exposure_pct = (current_exposure / max_exposure) * 100 if max_exposure > 0 else 0
        self.logger.warning(
            f"🚫 EXPOSURE REJECTION #{self.rejection_count}: {symbol} - {reason} "
            f"(Exposure: {exposure_pct:.1f}% of limit)"
        )
    
    def log_drawdown_rejection(self, symbol: str, current_drawdown: float, max_drawdown: float, reason: str = "Drawdown limit exceeded"):
        """Log drawdown limit rejection"""
        self.rejection_count += 1
        drawdown_pct = (current_drawdown / max_drawdown) * 100 if max_drawdown > 0 else 0
        self.logger.warning(
            f"🚫 DRAWDOWN REJECTION #{self.rejection_count}: {symbol} - {reason} "
            f"(Drawdown {drawdown_pct:.1f}% exceeds limit)"
        )
    
    def log_correlation_rejection(self, symbol: str, correlated_symbol: str, correlation: float, threshold: float, reason: str = "High correlation"):
        """Log correlation limit rejection"""
        self.rejection_count += 1
        self.logger.warning(
            f"🚫 CORRELATION REJECTION #{self.rejection_count}: {symbol} - {reason} "
            f"(Correlation with {correlated_symbol}: {correlation:.3f}, threshold: {threshold:.3f})"
        )
    
    def log_volatility_rejection(self, symbol: str, volatility: float, threshold: float, reason: str):
        """Log volatility-based rejection"""
        self.rejection_count += 1
        self.logger.warning(
            f"🚫 VOLATILITY REJECTION #{self.rejection_count}: {symbol} - {reason} "
            f"(Volatility: {volatility:.3f}, threshold: {threshold:.3f})"
        )
    
    def log_risk_override(self, symbol: str, original_decision: str, new_decision: str, reason: str):
        """Log risk management override"""
        self.logger.info(
            f"🔄 RISK OVERRIDE: {symbol} - {original_decision} → {new_decision} ({reason})"
        )
    
    def log_risk_metrics(self, total_exposure: float, max_exposure: float, 
                        current_drawdown: float, max_drawdown: float, 
                        positions_count: int, max_positions: int):
        """Log current risk metrics"""
        exposure_pct = (total_exposure / max_exposure) * 100 if max_exposure > 0 else 0
        drawdown_pct = (current_drawdown / max_drawdown) * 100 if max_drawdown > 0 else 0
        
        self.logger.info(
            f"📊 RISK METRICS: Exposure {exposure_pct:.1f}% | "
            f"Drawdown {drawdown_pct:.1f}% | Positions {positions_count}/{max_positions}"
        )
    
    def get_session_stats(self):
        """Get session statistics"""
        runtime = datetime.utcnow() - self.session_start
        return {
            'session_start': self.session_start.isoformat(),
            'runtime_hours': runtime.total_seconds() / 3600,
            'rejection_count': self.rejection_count
        }

class TradingLogger:
    """Specialized logger for trading operations"""
    
    def __init__(self, name: str):
        self.logger = setup_logger(name)
        self.trade_count = 0
        self.session_start = datetime.utcnow()
    
    def log_trade_entry(self, symbol: str, side: str, amount: float, 
                       price: float, strategy: str):
        """Log trade entry"""
        self.trade_count += 1
        self.logger.info(
            f"🔵 ENTRY #{self.trade_count}: {symbol} {side.upper()} "
            f"{amount:.6f} @ ${price:.4f} | Strategy: {strategy}"
        )
    
    def log_trade_exit(self, symbol: str, side: str, amount: float, 
                      price: float, pnl: float, pnl_pct: float):
        """Log trade exit"""
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        self.logger.info(
            f"{pnl_emoji} EXIT: {symbol} {side.upper()} "
            f"{amount:.6f} @ ${price:.4f} | "
            f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)"
        )
    
    def log_position_update(self, symbol: str, unrealized_pnl: float, 
                           unrealized_pnl_pct: float):
        """Log position update"""
        pnl_emoji = "📈" if unrealized_pnl >= 0 else "📉"
        self.logger.debug(
            f"{pnl_emoji} {symbol} Unrealized: "
            f"${unrealized_pnl:.2f} ({unrealized_pnl_pct:+.2f}%)"
        )
    
    def log_portfolio_update(self, total_balance: float, daily_pnl: float, 
                           daily_pnl_pct: float, positions_count: int):
        """Log portfolio update"""
        balance_emoji = "💰" if daily_pnl >= 0 else "💸"
        self.logger.info(
            f"{balance_emoji} Portfolio: ${total_balance:.2f} | "
            f"Daily: ${daily_pnl:.2f} ({daily_pnl_pct:+.2f}%) | "
            f"Positions: {positions_count}"
        )
    
    def log_strategy_performance(self, strategy: str, win_rate: float, 
                               total_pnl: float, trades_count: int):
        """Log strategy performance"""
        performance_emoji = "🎯" if win_rate >= 0.6 else "🎲"
        self.logger.info(
            f"{performance_emoji} {strategy}: Win Rate {win_rate:.1%} | "
            f"Total PnL: ${total_pnl:.2f} | Trades: {trades_count}"
        )
    
    def log_risk_alert(self, alert_type: str, symbol: str, details: str):
        """Log risk management alerts"""
        self.logger.warning(f"⚠️ RISK ALERT [{alert_type}] {symbol}: {details}")
    
    def log_system_status(self, status: str, details: str = ""):
        """Log system status"""
        status_emojis = {
            'starting': '🚀',
            'running': '✅',
            'stopping': '🛑',
            'error': '❌',
            'warning': '⚠️'
        }
        emoji = status_emojis.get(status.lower(), '📊')
        self.logger.info(f"{emoji} SYSTEM {status.upper()}: {details}")
    
    def log_ml_update(self, model_name: str, accuracy: float, 
                     predictions_count: int):
        """Log ML model updates"""
        self.logger.info(
            f"🧠 {model_name}: Accuracy {accuracy:.1%} | "
            f"Predictions: {predictions_count}"
        )
    
    def log_market_analysis(self, analysis_type: str, symbol: str, 
                          confidence: float, details: str):
        """Log market analysis"""
        confidence_emoji = "🔥" if confidence >= 0.8 else "📊" if confidence >= 0.6 else "🤔"
        self.logger.info(
            f"{confidence_emoji} {analysis_type.upper()} {symbol}: "
            f"Confidence {confidence:.1%} | {details}"
        )
    
    def get_session_stats(self):
        """Get session statistics"""
        runtime = datetime.utcnow() - self.session_start
        return {
            'session_start': self.session_start.isoformat(),
            'runtime_hours': runtime.total_seconds() / 3600,
            'trades_count': self.trade_count
        }

# Performance monitoring logger
class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self, name: str):
        self.logger = setup_logger(f"performance.{name}")
        self.metrics = {}
    
    def log_execution_time(self, operation: str, execution_time: float):
        """Log operation execution time"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(execution_time)
        
        if execution_time > 5.0:  # Log slow operations
            self.logger.warning(
                f"⏱️ SLOW OPERATION: {operation} took {execution_time:.2f}s"
            )
        else:
            self.logger.debug(f"⚡ {operation}: {execution_time:.3f}s")
    
    def log_api_call(self, endpoint: str, response_time: float, status_code: int):
        """Log API call metrics"""
        if status_code >= 400:
            self.logger.error(f"🔴 API ERROR: {endpoint} - {status_code} ({response_time:.2f}s)")
        elif response_time > 2.0:
            self.logger.warning(f"🟡 SLOW API: {endpoint} - {response_time:.2f}s")
        else:
            self.logger.debug(f"🟢 API: {endpoint} - {response_time:.3f}s")
    
    def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage"""
        if memory_mb > 500:  # Log high memory usage
            self.logger.warning(f"🧠 HIGH MEMORY: {operation} using {memory_mb:.1f}MB")
        else:
            self.logger.debug(f"💾 {operation}: {memory_mb:.1f}MB")
    
    def get_performance_summary(self):
        """Get performance summary"""
        summary = {}
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times)
                }
        return summary

# Audit logger for compliance
class AuditLogger:
    """Audit logger for compliance and trade tracking"""
    
    def __init__(self):
        self.logger = setup_logger("audit")
        
        # Create separate audit log file
        os.makedirs('logs/audit', exist_ok=True)
        
        audit_handler = logging.handlers.RotatingFileHandler(
            'logs/audit/audit.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=20
        )
        audit_handler.setLevel(logging.INFO)
        
        audit_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "event": "%(message)s"}'
        )
        audit_handler.setFormatter(audit_formatter)
        self.logger.addHandler(audit_handler)
    
    def log_trade_decision(self, symbol: str, action: str, reasoning: str, 
                          confidence: float, strategy: str):
        """Log trade decision for audit trail"""
        event = {
            'type': 'trade_decision',
            'symbol': symbol,
            'action': action,
            'reasoning': reasoning,
            'confidence': confidence,
            'strategy': strategy,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(event)
    
    def log_risk_override(self, symbol: str, risk_type: str, 
                         original_limit: float, new_limit: float, reason: str):
        """Log risk management overrides"""
        event = {
            'type': 'risk_override',
            'symbol': symbol,
            'risk_type': risk_type,
            'original_limit': original_limit,
            'new_limit': new_limit,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.warning(event)
    
    def log_system_change(self, change_type: str, component: str, 
                         old_value: str, new_value: str):
        """Log system configuration changes"""
        event = {
            'type': 'system_change',
            'change_type': change_type,
            'component': component,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(event)
    
    def log_balance_change(self, balance_before: float, balance_after: float, 
                          change_type: str, details: str):
        """Log balance changes"""
        event = {
            'type': 'balance_change',
            'balance_before': balance_before,
            'balance_after': balance_after,
            'change': balance_after - balance_before,
            'change_type': change_type,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        }
        self.logger.info(event)

# Error notification logger
class ErrorNotificationLogger:
    """Logger that sends critical errors to notification systems"""
    
    def __init__(self):
        self.logger = setup_logger("error_notifications")
        self.notification_manager = None
        self.critical_errors = []
        self.error_count = 0
    
    def set_notification_manager(self, notification_manager):
        """Set notification manager for sending alerts"""
        self.notification_manager = notification_manager
    
    async def log_critical_error(self, component: str, error: str, 
                                impact: str = "Unknown"):
        """Log critical error and send notification"""
        self.error_count += 1
        
        error_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'component': component,
            'error': error,
            'impact': impact,
            'error_number': self.error_count
        }
        
        self.critical_errors.append(error_info)
        
        # Keep only last 100 errors in memory
        if len(self.critical_errors) > 100:
            self.critical_errors.pop(0)
        
        self.logger.critical(f"🚨 CRITICAL ERROR #{self.error_count}: [{component}] {error}")
        
        # Send notification if manager is available
        if self.notification_manager:
            try:
                await self.notification_manager.send_error_notification(error_info)
            except Exception as e:
                self.logger.error(f"Failed to send error notification: {e}")
    
    async def log_system_failure(self, system: str, details: str):
        """Log system-wide failures"""
        await self.log_critical_error(
            component=system,
            error=f"System failure: {details}",
            impact="High - System may not function properly"
        )
    
    def get_recent_errors(self, count: int = 10):
        """Get recent critical errors"""
        return self.critical_errors[-count:]

# Global logger instances
main_logger = setup_logger("main")
trading_logger = TradingLogger("trading")
risk_logger = RiskLogger("risk")
performance_logger = PerformanceLogger("main")
audit_logger = AuditLogger()
error_notification_logger = ErrorNotificationLogger()

# Context manager for timing operations
import time
from contextlib import asynccontextmanager, contextmanager

@contextmanager
def log_execution_time(operation_name: str, logger_instance=None):
    """Context manager to log execution time of operations"""
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        logger_to_use = logger_instance or performance_logger
        logger_to_use.log_execution_time(operation_name, execution_time)

@asynccontextmanager
async def log_async_execution_time(operation_name: str, logger_instance=None):
    """Async context manager to log execution time of async operations"""
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        logger_to_use = logger_instance or performance_logger
        logger_to_use.log_execution_time(operation_name, execution_time)

# Decorator for automatic logging
def log_function_calls(logger_name: str = None):
    """Decorator to automatically log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = setup_logger(logger_name or func.__module__)
            func_logger.debug(f"📞 Calling {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"✅ {func.__name__} completed successfully")
                return result
            except Exception as e:
                func_logger.error(f"❌ {func.__name__} failed: {e}")
                raise
        
        # Handle async functions
        import asyncio
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                func_logger = setup_logger(logger_name or func.__module__)
                func_logger.debug(f"📞 Calling async {func.__name__}")
                
                try:
                    result = await func(*args, **kwargs)
                    func_logger.debug(f"✅ {func.__name__} completed successfully")
                    return result
                except Exception as e:
                    func_logger.error(f"❌ {func.__name__} failed: {e}")
                    raise
            
            return async_wrapper
        
        return wrapper
    return decorator

# Log rotation management
def setup_log_rotation():
    """Setup log rotation for all log files"""
    import glob
    
    log_files = glob.glob('logs/*.log')
    total_size = sum(os.path.getsize(f) for f in log_files if os.path.exists(f))
    
    main_logger.info(f"📊 Log files total size: {total_size / (1024*1024):.1f}MB")
    
    # Compress old logs if total size > 100MB
    if total_size > 100 * 1024 * 1024:
        compress_old_logs()

def compress_old_logs():
    """Compress old log files to save space"""
    import gzip
    import glob
    from pathlib import Path
    
    try:
        old_logs = glob.glob('logs/*.log.*')
        compressed_count = 0
        
        for log_file in old_logs:
            if not log_file.endswith('.gz'):
                with open(log_file, 'rb') as f_in:
                    with gzip.open(f"{log_file}.gz", 'wb') as f_out:
                        f_out.writelines(f_in)
                
                os.remove(log_file)
                compressed_count += 1
        
        if compressed_count > 0:
            main_logger.info(f"🗜️ Compressed {compressed_count} old log files")
    
    except Exception as e:
        main_logger.error(f"❌ Log compression failed: {e}")

# Initialize logging system
def initialize_logging_system():
    """Initialize the complete logging system"""
    try:
        # Create log directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('logs/audit', exist_ok=True)
        os.makedirs('logs/archive', exist_ok=True)
        
        # Setup log rotation
        setup_log_rotation()
        
        main_logger.info("📊 Logging system initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize logging system: {e}")
        return False

# Export main functions and classes
__all__ = [
    'setup_logger',
    'TradingLogger',
    'RiskLogger',
    'PerformanceLogger',
    'AuditLogger',
    'ErrorNotificationLogger',
    'log_execution_time',
    'log_async_execution_time',
    'log_function_calls',
    'initialize_logging_system',
    'main_logger',
    'trading_logger',
    'risk_logger',
    'performance_logger',
    'audit_logger',
    'error_notification_logger'
]
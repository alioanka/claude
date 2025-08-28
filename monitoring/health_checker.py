"""
Health Checker - Monitors system health and performance
Provides continuous monitoring of bot components and alerts.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system metrics will be limited")

from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self, bot=None, notification_manager=None):
        self.bot = bot
        self.notification_manager = notification_manager
        self.is_monitoring = False
        self.check_interval = 60  # seconds
        self.health_status = {
            'overall': 'healthy',
            'components': {},
            'last_check': None,
            'issues': []
        }
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage_percent': 80,
            'memory_usage_percent': 85,
            'disk_usage_percent': 90,
            'response_time_seconds': 5.0
        }
        
    async def monitor(self):
        """Main monitoring loop"""
        try:
            self.is_monitoring = True
            logger.info("🏥 Health monitoring started")
            
            while self.is_monitoring:
                try:
                    await self.perform_health_check()
                    await asyncio.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"❌ Health check error: {e}")
                    await asyncio.sleep(30)  # Shorter sleep on error
                    
        except Exception as e:
            logger.error(f"❌ Health monitoring failed: {e}")
        finally:
            self.is_monitoring = False
            logger.info("🏥 Health monitoring stopped")
    
    async def perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            issues = []
            components = {}
            
            # System resource checks (if psutil available)
            if PSUTIL_AVAILABLE:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                
                try:
                    disk_info = psutil.disk_usage('/')
                    disk_usage_pct = disk_info.percent
                except:
                    disk_usage_pct = 0
                
                components['system'] = {
                    'status': 'healthy',
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_info.percent,
                    'disk_usage': disk_usage_pct,
                    'uptime': self.get_uptime()
                }
                
                # Check resource thresholds
                if cpu_usage > self.thresholds['cpu_usage_percent']:
                    issues.append(f"High CPU usage: {cpu_usage:.1f}%")
                    components['system']['status'] = 'warning'
                
                if memory_info.percent > self.thresholds['memory_usage_percent']:
                    issues.append(f"High memory usage: {memory_info.percent:.1f}%")
                    components['system']['status'] = 'warning'
                
                if disk_usage_pct > self.thresholds['disk_usage_percent']:
                    issues.append(f"High disk usage: {disk_usage_pct:.1f}%")
                    components['system']['status'] = 'critical'
            else:
                components['system'] = {
                    'status': 'unknown',
                    'message': 'System metrics unavailable (psutil not installed)'
                }
            
            # Bot component checks
            if self.bot:
                bot_status = await self.check_bot_health()
                components['bot'] = bot_status
                
                if bot_status['status'] != 'healthy':
                    issues.extend(bot_status.get('issues', []))
            
            # Update health status
            self.health_status = {
                'overall': 'critical' if any('critical' in str(comp) for comp in components.values()) 
                          else 'warning' if issues else 'healthy',
                'components': components,
                'last_check': datetime.utcnow().isoformat(),
                'issues': issues
            }
            
            # Send alerts for critical issues
            if issues and self.notification_manager:
                await self.send_health_alert(issues)
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            self.health_status = {
                'overall': 'error',
                'error': str(e),
                'last_check': datetime.utcnow().isoformat()
            }
    
    async def check_bot_health(self) -> Dict[str, Any]:
        """Check trading bot specific health"""
        try:
            status = {
                'status': 'healthy',
                'is_running': getattr(self.bot, 'is_running', False),
                'issues': []
            }
            
            # Check if bot is running
            if hasattr(self.bot, 'is_running') and not self.bot.is_running:
                status['status'] = 'critical'
                status['issues'].append("Bot is not running")
            
            # Check exchange connectivity
            if hasattr(self.bot, 'exchange_manager'):
                try:
                    if hasattr(self.bot.exchange_manager, 'health_check'):
                        health_check = await self.bot.exchange_manager.health_check()
                        if not health_check:
                            status['status'] = 'warning'
                            status['issues'].append("Exchange connectivity issues")
                except Exception as e:
                    status['status'] = 'critical'
                    status['issues'].append(f"Exchange manager error: {e}")
            
            # Check data collection
            if hasattr(self.bot, 'data_collector'):
                if hasattr(self.bot.data_collector, 'is_collecting'):
                    if not self.bot.data_collector.is_collecting:
                        status['status'] = 'warning'
                        status['issues'].append("Data collection stopped")
            
            return status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'issues': [f"Health check error: {e}"]
            }
    
    async def send_health_alert(self, issues: List[str]):
        """Send health alert notification"""
        try:
            if not self.notification_manager:
                return
                
            severity = 'critical' if any('critical' in issue.lower() for issue in issues) else 'warning'
            
            issues_text = "\n".join([f"• {issue}" for issue in issues[:5]])  # Limit to 5 issues
            
            await self.notification_manager.send_notification(
                f"🏥 Health Alert\n\n{issues_text}",
                priority=severity,
                category='system',
                immediate=True
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to send health alert: {e}")
    
    def get_uptime(self) -> str:
        """Get system uptime"""
        try:
            if PSUTIL_AVAILABLE:
                boot_time = psutil.boot_time()
                uptime_seconds = datetime.utcnow().timestamp() - boot_time
                uptime_hours = uptime_seconds / 3600
                return f"{uptime_hours:.1f} hours"
            else:
                return "Unknown (psutil not available)"
        except Exception as e:
            logger.error(f"Uptime calculation error: {e}")
            return "Unknown"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health_status.copy()
    
    async def stop(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        logger.info("🏥 Health monitoring stop requested")
#!/usr/bin/env python3
"""
Crypto Trading Bot - Main Entry Point
Advanced automated trading system with ML integration and risk management.
"""

import asyncio
import signal
import sys
import os
import logging
from datetime import datetime
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import config
from core.bot import TradingBot
from utils.logger import setup_logger, initialize_logging_system
from utils.notifications import NotificationManager
from monitoring.health_checker import HealthChecker
from monitoring.dashboard import DashboardManager

# Setup logging first
initialize_logging_system()
logger = setup_logger(__name__)

class TradingBotApplication:
    """Main application class for the trading bot"""
    
    def __init__(self):
        self.bot: Optional[TradingBot] = None
        self.health_checker: Optional[HealthChecker] = None
        self.notification_manager: Optional[NotificationManager] = None
        self.shutdown_event = asyncio.Event()
        self.startup_time = datetime.utcnow()
        self.dashboard_manager: Optional[DashboardManager] = None
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🚀 Initializing Crypto Trading Bot...")
            logger.info(f"📊 Trading Mode: {config.trading.mode.upper()}")
            logger.info(f"💰 Initial Capital: ${config.trading.initial_capital:,.2f}")
            logger.info(f"🎯 Risk Per Trade: {config.trading.risk_per_trade:.1%}")
            
            # Validate configuration
            if not config.validate_config():
                logger.error("❌ Configuration validation failed!")
                return False
            
            # Initialize notification manager first
            try:
                self.notification_manager = NotificationManager()
                await self.notification_manager.initialize()
                logger.info("✅ Notification manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Notification manager failed to initialize: {e}")
                self.notification_manager = None
            
            # Send startup notification
            if self.notification_manager:
                try:
                    await self.notification_manager.send_startup_message()
                except Exception as e:
                    logger.warning(f"⚠️ Startup notification failed: {e}")
            
            # Initialize trading bot
            try:
                self.bot = TradingBot()
                await self.bot.initialize()
                logger.info("✅ Trading bot initialized")
            except Exception as e:
                logger.error(f"❌ Trading bot initialization failed: {e}")
                if self.notification_manager:
                    await self.notification_manager.send_error_message(f"Bot initialization failed: {e}")
                return False
            
            # Initialize dashboard
            try:
                self.dashboard_manager = DashboardManager(
                    self.bot.portfolio_manager.db_manager,
                    self.bot.portfolio_manager
                )
#                dashboard.trade_executor = bot.trade_executor
#                dashboard.bot = bot
                logger.info("✅ Dashboard manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Dashboard manager failed to initialize: {e}")
                self.dashboard_manager = None
            
            # Initialize health checker
            try:
                self.health_checker = HealthChecker(self.bot, self.notification_manager)
                logger.info("✅ Health checker initialized")
            except Exception as e:
                logger.warning(f"⚠️ Health checker failed to initialize: {e}")
                self.health_checker = None
            
            logger.info("✅ All components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            if self.notification_manager:
                try:
                    await self.notification_manager.send_error_message(f"Bot initialization failed: {e}")
                except:
                    pass
            return False
    
    async def run(self):
        """Main run loop"""
        try:
            logger.info("🤖 Starting Trading Bot...")
            
            # Start all components
            tasks = []
            
            # Start the main trading bot
            if self.bot:
                tasks.append(asyncio.create_task(self.bot.run()))
                logger.info("📈 Trading bot task started")
            
            # Start health monitoring
            if self.health_checker:
                tasks.append(asyncio.create_task(self.health_checker.monitor()))
                logger.info("🏥 Health monitoring task started")

            # Start dashboard server
            if self.dashboard_manager:
                tasks.append(asyncio.create_task(self.dashboard_manager.start_dashboard(
                    host="0.0.0.0", port=8000
                )))
                logger.info("📊 Dashboard task started on :8000")

            
            # Add shutdown handler task
            tasks.append(asyncio.create_task(self.wait_for_shutdown()))
            
            # Add periodic status update task
            tasks.append(asyncio.create_task(self.status_update_loop()))
            
            if not tasks:
                logger.error("❌ No tasks to run - initialization may have failed")
                return
            
            logger.info(f"🚀 Trading bot fully started with {len(tasks)} tasks")
            
            # Wait for any task to complete (usually shutdown)
            try:
                done, pending = await asyncio.wait(
                    tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                logger.info("📴 First task completed, initiating shutdown...")
                
                # Cancel remaining tasks
                for task in pending:
                    if not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=5.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                
            except Exception as e:
                logger.error(f"❌ Task execution error: {e}")
            
            logger.info("🛑 All tasks completed")
            
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
            if self.notification_manager:
                try:
                    await self.notification_manager.send_error_message(f"Runtime error: {e}")
                except:
                    pass
    
    async def status_update_loop(self):
        """Periodic status updates"""
        try:
            while not self.shutdown_event.is_set():
                try:
                    await asyncio.sleep(300)  # 5 minutes
                    
                    if self.bot and hasattr(self.bot, 'portfolio_manager') and self.bot.portfolio_manager:
                        # Get basic status
                        balance = self.bot.portfolio_manager.get_total_balance()
                        daily_pnl = self.bot.portfolio_manager.get_daily_pnl_pct()
                        positions = self.bot.portfolio_manager.get_position_count()
                        
                        logger.info(f"📊 Status Update - Balance: ${balance:.2f}, Daily PnL: {daily_pnl:+.2f}%, Positions: {positions}")
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug(f"Status update error: {e}")
                    
        except asyncio.CancelledError:
            logger.info("📊 Status update loop cancelled")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            await self.shutdown_event.wait()
            logger.info("🛑 Shutdown signal received")
        except asyncio.CancelledError:
            logger.info("🛑 Shutdown wait cancelled")
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            logger.info("🔄 Initiating graceful shutdown...")
            
            # Calculate uptime
            uptime = datetime.utcnow() - self.startup_time
            logger.info(f"⏱️ Bot uptime: {uptime}")
            
            # Close all positions if in live trading mode
            if self.bot and config.is_live_trading():
                try:
                    logger.info("📤 Closing all open positions (live trading mode)...")
                    await self.bot.close_all_positions()
                    logger.info("✅ All positions closed")
                except Exception as e:
                    logger.error(f"❌ Error closing positions: {e}")
            
            # Shutdown components in reverse order
            if self.health_checker:
                try:
                    await self.health_checker.stop()
                    logger.info("✅ Health checker stopped")
                except Exception as e:
                    logger.error(f"❌ Health checker shutdown error: {e}")
            
            if self.bot:
                try:
                    await self.bot.shutdown()
                    logger.info("✅ Trading bot stopped")
                except Exception as e:
                    logger.error(f"❌ Trading bot shutdown error: {e}")
            
            # Send shutdown notification
            if self.notification_manager:
                try:
                    await self.notification_manager.send_shutdown_message()
                    logger.info("✅ Shutdown notification sent")
                except Exception as e:
                    logger.warning(f"⚠️ Shutdown notification failed: {e}")
                
                try:
                    await self.notification_manager.cleanup()
                    logger.info("✅ Notification manager cleaned up")
                except Exception as e:
                    logger.error(f"❌ Notification cleanup error: {e}")
            
            # Set shutdown event
            self.shutdown_event.set()
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
        finally:
            # Ensure shutdown event is set
            self.shutdown_event.set()

# Global application instance
app = None

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"📡 Received signal {signum}, initiating shutdown...")
    
    if app and not app.shutdown_event.is_set():
        # Create a task to handle shutdown
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        
        if loop and loop.is_running():
            loop.create_task(app.shutdown())
        else:
            # If no loop is running, just set the shutdown event
            app.shutdown_event.set()

async def create_health_check_server():
    """Create a simple health check HTTP server"""
    try:
        from aiohttp import web
        
        async def health_check(request):
            """Health check endpoint"""
            try:
                status = {
                    'status': 'healthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'uptime_seconds': (datetime.utcnow() - app.startup_time).total_seconds() if app else 0
                }
                
                if app and app.bot:
                    if hasattr(app.bot, 'portfolio_manager') and app.bot.portfolio_manager:
                        status['balance'] = app.bot.portfolio_manager.get_total_balance()
                        status['positions'] = app.bot.portfolio_manager.get_position_count()
                
                return web.json_response(status)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return web.json_response(
                    {'status': 'error', 'error': str(e)}, 
                    status=500
                )
        
        # Create web application
        web_app = web.Application()
        web_app.router.add_get('/health', health_check)
        web_app.router.add_get('/', health_check)  # Root also serves health check
        
        # Start server
        runner = web.AppRunner(web_app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', 9090)
        await site.start()
        
        logger.info("🌐 Health check server started on http://0.0.0.0:9090/health")
        return runner
        
    except Exception as e:
        logger.warning(f"⚠️ Could not start health check server: {e}")
        return None

async def main():
    """Main async function"""
    global app
    
    try:
        # Create application instance
        app = TradingBotApplication()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start health check server
        health_server = await create_health_check_server()
        
        # Initialize and run the application
        if await app.initialize():
            logger.info("🎉 Bot initialization successful, starting main loop...")
            await app.run()
        else:
            logger.error("💥 Bot initialization failed")
            sys.exit(1)
        
        # Cleanup health server
        if health_server:
            try:
                await health_server.cleanup()
            except:
                pass
        
    except KeyboardInterrupt:
        logger.info("👋 Keyboard interrupt received")
        if app:
            await app.shutdown()
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        if app:
            try:
                await app.shutdown()
            except:
                pass
        sys.exit(1)
    finally:
        logger.info("👋 Trading bot exiting")

if __name__ == "__main__":
    # Print startup banner
    print("""
    ╔═══════════════════════════════════════╗
    ║        🚀 CRYPTO TRADING BOT 🚀       ║
    ║                                       ║
    ║  Advanced ML-Powered Trading System   ║
    ║        Built for Performance          ║
    ╚═══════════════════════════════════════╝
    """)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required!")
        sys.exit(1)
    
    # Create required directories
    required_dirs = [
        'logs',
        'storage',
        'storage/historical',
        'storage/models',
        'storage/backups',
        'storage/exports',
        'storage/performance'
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
#!/usr/bin/env python3
"""
Crypto Trading Bot - Main Entry Point
Advanced automated trading system with ML integration and risk management.
"""

import asyncio
import signal
import sys
import logging
from datetime import datetime
from typing import Optional

from config.config import config
from core.bot import TradingBot
from utils.logger import setup_logger
from utils.notifications import NotificationManager
from monitoring.health_checker import HealthChecker

# Setup logging
logger = setup_logger(__name__)

class TradingBotApplication:
    """Main application class for the trading bot"""
    
    def __init__(self):
        self.bot: Optional[TradingBot] = None
        self.health_checker: Optional[HealthChecker] = None
        self.notification_manager: Optional[NotificationManager] = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🚀 Initializing Crypto Trading Bot...")
            logger.info(f"📊 Trading Mode: {config.trading.mode.upper()}")
            logger.info(f"💰 Initial Capital: ${config.trading.initial_capital:,.2f}")
            
            # Validate configuration
            if not config.validate_config():
                logger.error("❌ Configuration validation failed!")
                sys.exit(1)
            
            # Initialize notification manager
            self.notification_manager = NotificationManager()
            await self.notification_manager.initialize()
            
            # Send startup notification
            await self.notification_manager.send_startup_message()
            
            # Initialize trading bot
            self.bot = TradingBot()
            await self.bot.initialize()
            
            # Initialize health checker
            self.health_checker = HealthChecker(self.bot, self.notification_manager)
            
            logger.info("✅ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            if self.notification_manager:
                await self.notification_manager.send_error_message(f"Bot initialization failed: {e}")
            raise
    
    async def run(self):
        """Main run loop"""
        try:
            logger.info("🤖 Starting Trading Bot...")
            
            # Start all components
            tasks = []
            
            # Start the main trading bot
            if self.bot:
                tasks.append(asyncio.create_task(self.bot.run()))
            
            # Start health monitoring
            if self.health_checker:
                tasks.append(asyncio.create_task(self.health_checker.monitor()))
            
            # Add shutdown handler task
            tasks.append(asyncio.create_task(self.wait_for_shutdown()))
            
            # Wait for any task to complete (usually shutdown)
            done, pending = await asyncio.wait(
                tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            logger.info("🛑 Trading Bot stopped")
            
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
            if self.notification_manager:
                await self.notification_manager.send_error_message(f"Runtime error: {e}")
            raise
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            logger.info("🔄 Initiating graceful shutdown...")
            
            # Close all positions if in live trading mode
            if self.bot and config.is_live_trading():
                logger.info("📤 Closing all open positions...")
                await self.bot.close_all_positions()
            
            # Shutdown components
            if self.bot:
                await self.bot.shutdown()
            
            if self.health_checker:
                await self.health_checker.stop()
            
            # Send shutdown notification
            if self.notification_manager:
                await self.notification_manager.send_shutdown_message()
                await self.notification_manager.cleanup()
            
            # Set shutdown event
            self.shutdown_event.set()
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# Global application instance
app = TradingBotApplication()

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"📡 Received signal {signum}, initiating shutdown...")
    asyncio.create_task(app.shutdown())

async def main():
    """Main async function"""
    try:
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize and run the application
        await app.initialize()
        await app.run()
        
    except KeyboardInterrupt:
        logger.info("👋 Keyboard interrupt received")
        await app.shutdown()
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        await app.shutdown()
        sys.exit(1)

if __name__ == "__main__":
    # Print startup banner
    print("""
    ╔═══════════════════════════════════════╗
    ║        🚀 CRYPTO TRADING BOT 🚀        ║
    ║                                       ║
    ║  Advanced ML-Powered Trading System   ║
    ║        Built for Performance          ║
    ╚═══════════════════════════════════════╝
    """)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required!")
        sys.exit(1)
    
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
"""
Core Trading Bot - Main Orchestrator
Manages all trading operations, strategies, and risk management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from config.config import config, TRADING_PAIRS
from core.exchange_manager import ExchangeManager
from core.portfolio_manager import PortfolioManager
from core.trade_executor import TradeExecutor
from data.data_collector import DataCollector
from strategies.strategy_manager import StrategyManager
from risk.risk_manager import RiskManager
from ml.predictor import MLPredictor
from monitoring.performance_tracker import PerformanceTracker
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    side: str  # 'long' or 'short'
    confidence: float  # 0.0 to 1.0
    strategy_name: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    leverage: float = 1.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        self.exchange_manager: Optional[ExchangeManager] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.trade_executor: Optional[TradeExecutor] = None
        self.data_collector: Optional[DataCollector] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.ml_predictor: Optional[MLPredictor] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        
        # Bot state
        self.is_running = False
        self.last_strategy_update = datetime.utcnow()
        self.active_signals: Dict[str, TradingSignal] = {}
        self.trading_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0
        }
        
    async def initialize(self):
        """Initialize all bot components"""
        try:
            logger.info("🔧 Initializing Trading Bot components...")
            
            # Initialize exchange manager
            self.exchange_manager = ExchangeManager()
            await self.exchange_manager.initialize()
            
            # Initialize data collector
            self.data_collector = DataCollector(self.exchange_manager)
            await self.data_collector.initialize()
            
            # Initialize portfolio manager
            self.portfolio_manager = PortfolioManager(
                self.exchange_manager, 
                config.trading.initial_capital
            )
            await self.portfolio_manager.initialize()
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.portfolio_manager)
            
            # Initialize trade executor
            self.trade_executor = TradeExecutor(
                self.exchange_manager,
                self.portfolio_manager,
                self.risk_manager
            )
            
            # Initialize ML predictor
            self.ml_predictor = MLPredictor()
            await self.ml_predictor.initialize()
            
            # Initialize strategy manager
            self.strategy_manager = StrategyManager(
                self.data_collector,
                self.ml_predictor,
                self.risk_manager
            )
            await self.strategy_manager.initialize()
            
            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker(self.portfolio_manager)
            
            # Start data collection
            await self.data_collector.start_collection()
            
            logger.info("✅ All bot components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            raise
    
    async def run(self):
        """Main trading loop"""
        try:
            self.is_running = True
            logger.info("🚀 Starting main trading loop...")
            
            # Main trading loop
            while self.is_running:
                try:
                    # Update market data
                    await self.update_market_data()
                    
                    # Generate trading signals
                    signals = await self.generate_signals()
                    
                    # Process signals and execute trades
                    await self.process_signals(signals)
                    
                    # Update portfolio and risk metrics
                    await self.update_portfolio_metrics()
                    
                    # Retrain ML models if needed
                    await self.update_ml_models()
                    
                    # Update performance tracking
                    await self.performance_tracker.update()
                    
                    # Log status
                    await self.log_status()
                    
                    # Sleep before next iteration
                    await asyncio.sleep(60)  # 1-minute intervals
                    
                except Exception as e:
                    logger.error(f"❌ Error in trading loop: {e}")
                    await asyncio.sleep(30)  # Shorter sleep on error
                    
        except Exception as e:
            logger.error(f"💥 Fatal error in trading loop: {e}")
            raise
    
    async def update_market_data(self):
        """Update market data for all trading pairs"""
        try:
            # Data is automatically updated by DataCollector
            # Just ensure it's running properly
            if not self.data_collector.is_collecting:
                logger.warning("⚠️ Data collector stopped, restarting...")
                await self.data_collector.start_collection()
                
        except Exception as e:
            logger.error(f"❌ Failed to update market data: {e}")
    
    async def generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals from all strategies"""
        try:
            all_signals = []
            
            for symbol in TRADING_PAIRS:
                # Get latest market data
                market_data = await self.data_collector.get_latest_data(symbol)
                if not market_data:
                    continue
                
                # Generate signals from strategy manager
                strategy_signals = await self.strategy_manager.generate_signals(
                    symbol, market_data
                )
                
                # Add ML predictions
                ml_signal = await self.ml_predictor.predict(symbol, market_data)
                if ml_signal:
                    strategy_signals.append(ml_signal)
                
                # Filter and validate signals
                valid_signals = await self.validate_signals(strategy_signals, symbol)
                all_signals.extend(valid_signals)
            
            logger.debug(f"📊 Generated {len(all_signals)} trading signals")
            return all_signals
            
        except Exception as e:
            logger.error(f"❌ Failed to generate signals: {e}")
            return []
    
    async def validate_signals(self, signals: List[Dict], symbol: str) -> List[TradingSignal]:
        """Validate and convert raw signals to TradingSignal objects"""
        validated_signals = []
        
        try:
            for signal_data in signals:
                # Skip weak signals
                if signal_data.get('confidence', 0) < config.ml.confidence_threshold:
                    continue
                
                # Create TradingSignal object
                signal = TradingSignal(
                    symbol=symbol,
                    side=signal_data['side'],
                    confidence=signal_data['confidence'],
                    strategy_name=signal_data['strategy'],
                    entry_price=signal_data['entry_price'],
                    stop_loss=signal_data.get('stop_loss', 0),
                    take_profit=signal_data.get('take_profit', 0),
                    position_size=signal_data.get('position_size', 0),
                    leverage=signal_data.get('leverage', 1.0)
                )
                
                # Risk validation
                if await self.risk_manager.validate_signal(signal):
                    validated_signals.append(signal)
                else:
                    logger.debug(f"🚫 Signal rejected by risk manager: {symbol} {signal.side}")
            
            return validated_signals
            
        except Exception as e:
            logger.error(f"❌ Signal validation failed: {e}")
            return []
    
    async def process_signals(self, signals: List[TradingSignal]):
        """Process validated signals and execute trades"""
        try:
            if not signals:
                return
            
            # Sort signals by confidence (highest first)
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            for signal in signals:
                try:
                    # Check if we already have a position in this symbol
                    if await self.portfolio_manager.has_position(signal.symbol):
                        continue
                    
                    # Calculate position size
                    position_size = await self.risk_manager.calculate_position_size(
                        signal, self.portfolio_manager.get_available_balance()
                    )
                    
                    if position_size <= 0:
                        continue
                    
                    # Update signal with calculated position size
                    signal.position_size = position_size
                    
                    # Execute the trade
                    trade_result = await self.trade_executor.execute_signal(signal)
                    
                    if trade_result['success']:
                        # Store active signal
                        self.active_signals[signal.symbol] = signal
                        
                        # Update trading stats
                        self.trading_stats['total_trades'] += 1
                        
                        logger.info(
                            f"✅ Trade executed: {signal.symbol} {signal.side} "
                            f"${position_size:.2f} @ {signal.entry_price:.4f}"
                        )
                    else:
                        logger.warning(
                            f"⚠️ Trade execution failed: {signal.symbol} - "
                            f"{trade_result.get('error', 'Unknown error')}"
                        )
                        
                except Exception as e:
                    logger.error(f"❌ Failed to process signal for {signal.symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to process signals: {e}")
    
    async def update_portfolio_metrics(self):
        """Update portfolio metrics and close positions if needed"""
        try:
            # Update portfolio manager
            await self.portfolio_manager.update()
            
            # Check active positions for exit conditions
            positions = await self.portfolio_manager.get_positions()
            
            for position in positions:
                symbol = position['symbol']
                
                # Check if we should close the position
                should_close = await self.should_close_position(symbol, position)
                
                if should_close:
                    await self.trade_executor.close_position(symbol)
                    
                    # Remove from active signals
                    if symbol in self.active_signals:
                        del self.active_signals[symbol]
                    
                    logger.info(f"🔄 Position closed: {symbol}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio metrics: {e}")
    
    async def should_close_position(self, symbol: str, position: Dict) -> bool:
        """Determine if a position should be closed"""
        try:
            # Get current market price
            current_price = await self.exchange_manager.get_current_price(symbol)
            if not current_price:
                return False
            
            entry_price = position['entry_price']
            side = position['side']
            
            # Calculate current PnL
            if side == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # short
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop loss
            if symbol in self.active_signals:
                signal = self.active_signals[symbol]
                
                if side == 'long' and current_price <= signal.stop_loss:
                    return True
                elif side == 'short' and current_price >= signal.stop_loss:
                    return True
                
                # Check take profit
                if side == 'long' and current_price >= signal.take_profit:
                    return True
                elif side == 'short' and current_price <= signal.take_profit:
                    return True
            
            # Check maximum drawdown
            if pnl_pct < -config.trading.max_drawdown:
                return True
            
            # Check time-based exit (24 hours max)
            position_age = datetime.utcnow() - position['timestamp']
            if position_age > timedelta(hours=24):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking position closure for {symbol}: {e}")
            return False
    
    async def update_ml_models(self):
        """Update ML models if retrain interval has passed"""
        try:
            current_time = datetime.utcnow()
            last_retrain = self.ml_predictor.last_retrain_time
            
            retrain_interval = timedelta(hours=config.ml.retrain_interval)
            
            if current_time - last_retrain >= retrain_interval:
                logger.info("🧠 Retraining ML models...")
                await self.ml_predictor.retrain_models()
                logger.info("✅ ML models retrained successfully!")
                
        except Exception as e:
            logger.error(f"❌ Failed to update ML models: {e}")
    
    async def log_status(self):
        """Log current bot status"""
        try:
            # get_total_balance is SYNC; get_positions is ASYNC
            balance = self.portfolio_manager.get_total_balance()
            positions_list = await self.portfolio_manager.get_positions()
            positions = len(positions_list)

            # performance_tracker daily PnL is sync (returns a float)
            daily_pnl = self.performance_tracker.get_daily_pnl()

            logger.info(
                f"📊 Status: Balance: ${balance:.2f} | "
                f"Positions: {positions} | "
                f"Daily PnL: {daily_pnl:.2f}% | "
                f"Total Trades: {self.trading_stats['total_trades']}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to log status: {e}")

    
    async def close_all_positions(self):
        """Close all open positions (used during shutdown)"""
        try:
            logger.info("🔄 Closing all positions...")
            
            positions = await self.portfolio_manager.get_positions()
            
            for position in positions:
                symbol = position['symbol']
                await self.trade_executor.close_position(symbol)
                logger.info(f"✅ Closed position: {symbol}")
            
            # Clear active signals
            self.active_signals.clear()
            
        except Exception as e:
            logger.error(f"❌ Failed to close all positions: {e}")
    
    async def shutdown(self):
        """Shutdown the bot gracefully"""
        try:
            logger.info("🛑 Shutting down Trading Bot...")
            
            self.is_running = False
            
            # Stop data collection
            if self.data_collector:
                await self.data_collector.stop_collection()
            
            # Close exchange connections
            if self.exchange_manager:
                await self.exchange_manager.close()
            
            # Save final performance metrics
            if self.performance_tracker:
                await self.performance_tracker.save_final_report()
            
            logger.info("✅ Trading Bot shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
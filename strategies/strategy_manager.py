"""
Strategy Manager - Manages all trading strategies
Coordinates strategy execution, performance tracking, and allocation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import importlib
import json

from config.config import config, TRADING_PAIRS
from strategies.base_strategy import BaseStrategy, StrategySignal
from strategies.momentum_strategy import MomentumStrategy
from utils.logger import setup_logger

logger = setup_logger(__name__)

class StrategyManager:
    """Main strategy management class"""
    
    def __init__(self, data_collector=None, ml_predictor=None, risk_manager=None):
        self.data_collector = data_collector
        self.ml_predictor = ml_predictor
        self.risk_manager = risk_manager
        
        # Strategy instances
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_allocation = {}
        
        # Performance tracking
        self.strategy_performance = {}
        self.total_signals_generated = 0
        self.last_rebalance = datetime.utcnow()
        
        # Load strategy configuration
        self.load_strategy_config()
    
    def load_strategy_config(self):
        """Load strategy configuration from config"""
        try:
            if hasattr(config, 'strategies') and config.strategies:
                self.strategy_allocation = config.strategies.get('strategy_allocation', {
                    'momentum_strategy': 0.5,
                    'mean_reversion': 0.3,
                    'ml_strategy': 0.2
                })
            else:
                # Default allocation
                self.strategy_allocation = {
                    'momentum_strategy': 0.6,
                    'simple_ma': 0.4
                }
            
            logger.info(f"📊 Strategy allocation loaded: {self.strategy_allocation}")
            
        except Exception as e:
            logger.error(f"❌ Strategy config loading failed: {e}")
            self.strategy_allocation = {'momentum_strategy': 1.0}
    
    async def initialize(self):
        """Initialize all strategies"""
        try:
            logger.info("🎯 Initializing Strategy Manager...")
            
            # Initialize available strategies
            await self.load_strategies()
            
            # Set initial performance metrics
            for strategy_name in self.strategies:
                self.strategy_performance[strategy_name] = {
                    'signals_generated': 0,
                    'successful_signals': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'last_signal_time': None,
                    'allocation': self.strategy_allocation.get(strategy_name, 0.0)
                }
            
            logger.info(f"✅ Strategy Manager initialized with {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"❌ Strategy Manager initialization failed: {e}")
            # Create at least one default strategy
            await self.create_default_strategy()
    
    async def load_strategies(self):
        """Load and initialize trading strategies"""
        try:
            # Initialize Momentum Strategy (already implemented)
            if 'momentum_strategy' in self.strategy_allocation:
                momentum_params = {}
                if hasattr(config, 'strategies') and 'momentum_strategy' in config.strategies:
                    momentum_params = config.strategies['momentum_strategy'].get('parameters', {})
                
                self.strategies['momentum_strategy'] = MomentumStrategy(momentum_params)
                logger.info("✅ Momentum Strategy loaded")
            
            # Initialize Simple Moving Average Strategy (basic implementation)
            if 'simple_ma' in self.strategy_allocation:
                self.strategies['simple_ma'] = SimpleMAStrategy()
                logger.info("✅ Simple MA Strategy loaded")
            
            # Initialize Mean Reversion Strategy (basic implementation)
            if 'mean_reversion' in self.strategy_allocation:
                self.strategies['mean_reversion'] = MeanReversionStrategy()
                logger.info("✅ Mean Reversion Strategy loaded")
            
            # Initialize ML Strategy if ML predictor is available
            if 'ml_strategy' in self.strategy_allocation and self.ml_predictor:
                self.strategies['ml_strategy'] = MLStrategy(self.ml_predictor)
                logger.info("✅ ML Strategy loaded")
            
        except Exception as e:
            logger.error(f"❌ Strategy loading failed: {e}")
    
    async def create_default_strategy(self):
        """Create default strategy as fallback"""
        try:
            self.strategies['default'] = SimpleMAStrategy()
            self.strategy_allocation = {'default': 1.0}
            logger.info("✅ Default strategy created")
        except Exception as e:
            logger.error(f"❌ Default strategy creation failed: {e}")
    
    async def generate_signals(self, symbol: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from all active strategies"""
        try:
            all_signals = []
            
            for strategy_name, strategy in self.strategies.items():
                if not strategy.enabled:
                    continue
                
                allocation = self.strategy_allocation.get(strategy_name, 0.0)
                if allocation <= 0:
                    continue
                
                try:
                    # Generate signal from strategy
                    signal = await strategy.analyze(symbol, market_data)
                    
                    if signal and signal.action != 'hold':
                        # Convert to dict format for compatibility
                        signal_dict = {
                            'symbol': signal.symbol,
                            'side': 'buy' if signal.action == 'buy' else 'sell',
                            'confidence': signal.confidence,
                            'strategy': signal.strategy_name,
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'position_size': signal.position_size,
                            'reasoning': signal.reasoning,
                            'allocation_weight': allocation
                        }
                        
                        all_signals.append(signal_dict)
                        
                        # Update performance tracking
                        self.strategy_performance[strategy_name]['signals_generated'] += 1
                        self.strategy_performance[strategy_name]['last_signal_time'] = datetime.utcnow().isoformat()
                        
                        logger.debug(f"📊 Signal from {strategy_name}: {signal.action} {symbol} (confidence: {signal.confidence:.2f})")
                
                except Exception as e:
                    logger.error(f"❌ Signal generation failed for {strategy_name}: {e}")
                    continue
            
            self.total_signals_generated += len(all_signals)
            
            return all_signals
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed for {symbol}: {e}")
            return []
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all strategies"""
        return {
            'total_signals_generated': self.total_signals_generated,
            'active_strategies': len([s for s in self.strategies.values() if s.enabled]),
            'strategy_allocation': self.strategy_allocation.copy(),
            'individual_performance': self.strategy_performance.copy(),
            'last_rebalance': self.last_rebalance.isoformat()
        }
    
    def update_strategy_performance(self, strategy_name: str, pnl: float, success: bool):
        """Update strategy performance metrics"""
        try:
            if strategy_name not in self.strategy_performance:
                return
            
            perf = self.strategy_performance[strategy_name]
            perf['total_pnl'] += pnl
            
            if success:
                perf['successful_signals'] += 1
            
            # Update win rate
            if perf['signals_generated'] > 0:
                perf['win_rate'] = perf['successful_signals'] / perf['signals_generated']
            
            logger.debug(f"📈 Updated {strategy_name} performance: PnL={pnl:.2f}, Success={success}")
            
        except Exception as e:
            logger.error(f"❌ Performance update failed for {strategy_name}: {e}")
    
    async def rebalance_strategies(self):
        """Rebalance strategy allocations based on performance"""
        try:
            # Simple rebalancing logic - in production you'd want more sophisticated methods
            current_time = datetime.utcnow()
            
            # Only rebalance once per day
            if (current_time - self.last_rebalance).total_seconds() < 86400:
                return
            
            logger.info("🔄 Rebalancing strategy allocations...")
            
            # Calculate new allocations based on performance
            total_performance_score = 0
            performance_scores = {}
            
            for strategy_name, perf in self.strategy_performance.items():
                # Simple scoring: win_rate * (1 + total_pnl_ratio)
                win_rate = perf.get('win_rate', 0.5)
                pnl_ratio = max(-0.5, min(0.5, perf.get('total_pnl', 0) / 1000))  # Normalize PnL
                
                score = win_rate * (1 + pnl_ratio)
                performance_scores[strategy_name] = max(0.1, score)  # Minimum 10%
                total_performance_score += performance_scores[strategy_name]
            
            # Normalize to sum to 1.0
            if total_performance_score > 0:
                new_allocation = {}
                for strategy_name, score in performance_scores.items():
                    new_allocation[strategy_name] = score / total_performance_score
                
                # Apply the new allocation
                self.strategy_allocation = new_allocation
                self.last_rebalance = current_time
                
                logger.info(f"✅ Strategy rebalancing completed: {self.strategy_allocation}")
            
        except Exception as e:
            logger.error(f"❌ Strategy rebalancing failed: {e}")
    
    def enable_strategy(self, strategy_name: str):
        """Enable a specific strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enable()
            logger.info(f"✅ Strategy enabled: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a specific strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].disable()
            logger.info(f"⏸️ Strategy disabled: {strategy_name}")


# Basic Strategy Implementations for missing strategies

class SimpleMAStrategy(BaseStrategy):
    """Simple Moving Average Strategy"""
    
    def __init__(self):
        super().__init__("Simple MA Strategy", timeframes=["1h"], parameters={'ma_fast': 10, 'ma_slow': 20})
    
    def get_required_data(self) -> Dict[str, Any]:
        return {
            'timeframes': self.timeframes,
            'lookback_period': 50,
            'required_indicators': ['sma_10', 'sma_20']
        }
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> StrategySignal:
        try:
            df = self.preprocess_data(market_data)
            if df.empty or len(df) < 20:
                return self._no_signal(symbol, "Insufficient data")
            
            # Calculate moving averages
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # Get current values
            current_price = df['close'].iloc[-1]
            sma_10 = df['sma_10'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            
            # Simple crossover logic
            if sma_10 > sma_20 and df['sma_10'].iloc[-2] <= df['sma_20'].iloc[-2]:
                # Bullish crossover
                return StrategySignal(
                    symbol=symbol,
                    action="buy",
                    confidence=0.7,
                    entry_price=current_price,
                    reasoning="MA bullish crossover",
                    strategy_name=self.name
                )
            elif sma_10 < sma_20 and df['sma_10'].iloc[-2] >= df['sma_20'].iloc[-2]:
                # Bearish crossover
                return StrategySignal(
                    symbol=symbol,
                    action="sell",
                    confidence=0.7,
                    entry_price=current_price,
                    reasoning="MA bearish crossover",
                    strategy_name=self.name
                )
            else:
                return self._no_signal(symbol, "No crossover detected")
                
        except Exception as e:
            return self._no_signal(symbol, f"Analysis error: {e}")
    
    def _no_signal(self, symbol: str, reason: str) -> StrategySignal:
        return StrategySignal(
            symbol=symbol,
            action="hold",
            confidence=0.0,
            reasoning=reason,
            strategy_name=self.name
        )


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy"""
    
    def __init__(self):
        super().__init__("Mean Reversion Strategy", timeframes=["15m"], 
                        parameters={'bb_period': 20, 'bb_std': 2, 'rsi_period': 14})
    
    def get_required_data(self) -> Dict[str, Any]:
        return {
            'timeframes': self.timeframes,
            'lookback_period': 50,
            'required_indicators': ['bb_upper', 'bb_lower', 'rsi']
        }
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> StrategySignal:
        try:
            df = self.preprocess_data(market_data)
            if df.empty or len(df) < 20:
                return self._no_signal(symbol, "Insufficient data")
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            if 'bb_upper' not in df.columns or 'rsi' not in df.columns:
                return self._no_signal(symbol, "Missing indicators")
            
            # Get current values
            current_price = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            # Mean reversion logic
            if current_price < bb_lower and rsi < 30:
                # Oversold condition
                return StrategySignal(
                    symbol=symbol,
                    action="buy",
                    confidence=0.75,
                    entry_price=current_price,
                    reasoning="Oversold - below BB lower, RSI < 30",
                    strategy_name=self.name
                )
            elif current_price > bb_upper and rsi > 70:
                # Overbought condition
                return StrategySignal(
                    symbol=symbol,
                    action="sell",
                    confidence=0.75,
                    entry_price=current_price,
                    reasoning="Overbought - above BB upper, RSI > 70",
                    strategy_name=self.name
                )
            else:
                return self._no_signal(symbol, "No extreme conditions")
                
        except Exception as e:
            return self._no_signal(symbol, f"Analysis error: {e}")
    
    def _no_signal(self, symbol: str, reason: str) -> StrategySignal:
        return StrategySignal(
            symbol=symbol,
            action="hold",
            confidence=0.0,
            reasoning=reason,
            strategy_name=self.name
        )


class MLStrategy(BaseStrategy):
    """ML-based Strategy"""
    
    def __init__(self, ml_predictor):
        super().__init__("ML Strategy", timeframes=["1h"], parameters={})
        self.ml_predictor = ml_predictor
    
    def get_required_data(self) -> Dict[str, Any]:
        return {
            'timeframes': self.timeframes,
            'lookback_period': 100,
            'required_indicators': []
        }
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> StrategySignal:
        try:
            if not self.ml_predictor:
                return self._no_signal(symbol, "No ML predictor available")
            
            # Get ML prediction
            ml_signal = await self.ml_predictor.predict(symbol, market_data)
            
            if not ml_signal or ml_signal['action'] == 'hold':
                return self._no_signal(symbol, "ML model suggests hold")
            
            # Convert ML signal to StrategySignal
            action = ml_signal['action']
            confidence = ml_signal['confidence']
            
            # Get current price for entry
            current_price = None
            if market_data and 'candles' in market_data and market_data['candles']:
                current_price = float(market_data['candles'][-1][4])  # Close price
            
            return StrategySignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=current_price,
                reasoning=ml_signal.get('reasoning', 'ML prediction'),
                strategy_name=self.name
            )
            
        except Exception as e:
            return self._no_signal(symbol, f"ML analysis error: {e}")
    
    def _no_signal(self, symbol: str, reason: str) -> StrategySignal:
        return StrategySignal(
            symbol=symbol,
            action="hold",
            confidence=0.0,
            reasoning=reason,
            strategy_name=self.name
        )
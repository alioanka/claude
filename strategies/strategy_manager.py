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
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from strategies.arbitrage_strategy import ArbitrageStrategy


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
                    'momentum_strategy': 0.3,
                    'mean_reversion': 0.25,
                    'ml_strategy': 0.35,
                    'arbitrage_strategy': 0.1
                })
            else:
                # Default allocation
                self.strategy_allocation = {
                    'momentum_strategy': 0.6,
                    'mean_reversion': 0.3,
                    'arbitrage_strategy': 0.1
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
            # Initialize available strategies
            await self.load_strategies()

            # Normalize allocation strictly over enabled & loaded strategies
            active_names = [name for name, strat in self.strategies.items() if getattr(strat, "enabled", True)]
            total_alloc = sum(max(0.0, float(self.strategy_allocation.get(n, 0.0))) for n in active_names)

            if total_alloc <= 0:
                # fallback: equal-weight active strategies
                equal = 1.0 / max(1, len(active_names))
                self.strategy_allocation = {n: (equal if n in active_names else 0.0) for n in self.strategies.keys()}
            else:
                # rescale only active strategies; keep disabled at 0.0
                self.strategy_allocation = {
                    n: (max(0.0, float(self.strategy_allocation.get(n, 0.0))) / total_alloc if n in active_names else 0.0)
                    for n in self.strategies.keys()
                }

            logger.info(f"📊 Strategy allocation normalized over active set: {self.strategy_allocation}")
                            
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
            

            
            # Initialize Mean Reversion Strategy (basic implementation)
            if 'mean_reversion' in self.strategy_allocation:
                self.strategies['mean_reversion'] = MeanReversionStrategy()
                logger.info("✅ Mean Reversion Strategy loaded")

            # Initialize Enhanced Arbitrage Strategy (if allocated)
            if 'arbitrage_strategy' in self.strategy_allocation and (
                hasattr(config, 'strategies') and config.strategies.get('arbitrage_strategy', {}).get('enabled', True)
            ):
                arb_params = {}
                if hasattr(config, 'strategies') and 'arbitrage_strategy' in getattr(config, 'strategies', {}):
                    arb_params = config.strategies['arbitrage_strategy'].get('parameters', {})
                
                # Add exchange configurations
                exchanges_config = {}
                for exchange_name in ['binance', 'kucoin', 'bybit']:
                    exchange_config = config.get_exchange_config(exchange_name)
                    if exchange_config and exchange_config.get('enabled', False):
                        exchanges_config[exchange_name] = {
                            'enabled': exchange_config.get('enabled', False),
                            'api_key': exchange_config.get('api_key', ''),
                            'secret': exchange_config.get('api_secret', ''),
                            'sandbox': exchange_config.get('sandbox', True)
                        }
                
                arb_params['exchanges'] = exchanges_config
                self.strategies['arbitrage_strategy'] = ArbitrageStrategy(arb_params)
                logger.info("✅ Enhanced Arbitrage Strategy loaded with exchange configs")


            # Initialize ML Strategy (config-driven)
            if 'ml_strategy' in self.strategy_allocation and (
                hasattr(config, 'strategies') and config.strategies.get('ml_strategy', {}).get('enabled', True)
            ):
                ml_params = {}
                if hasattr(config, 'strategies') and 'ml_strategy' in getattr(config, 'strategies', {}):
                    ml_params = config.strategies['ml_strategy']
                self.strategies['ml_strategy'] = MLStrategy(ml_params)
                logger.info("✅ ML Strategy loaded")


            
        except Exception as e:
            logger.error(f"❌ Strategy loading failed: {e}")
    
    async def create_default_strategy(self):
        """Create default strategy as fallback"""
        try:
            self.strategies['default'] = MomentumStrategy({})
            self.strategy_allocation = {'default': 1.0}
            logger.info("✅ Default strategy created (Momentum)")
        except Exception as e:
            logger.error(f"❌ Default strategy creation failed: {e}")

    
    # strategies/strategy_manager.py
    #from datetime import datetime

    async def generate_signals(self, symbol: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from all active strategies"""
        try:
            all_signals: List[Dict[str, Any]] = []

            for strategy_name, strategy in self.strategies.items():
                if not strategy.enabled:
                    self._log_rejection(strategy_name, symbol, "strategy disabled")
                    continue

                allocation = self.strategy_allocation.get(strategy_name, 0.0)
                if allocation <= 0:
                    self._log_rejection(strategy_name, symbol, f"allocation is {allocation}")
                    continue

                try:
                    # 1) Ask strategy for a signal
                    signal = await strategy.analyze(symbol, market_data)

                    # 2) Normalize outcomes and log WHY anything is skipped
                    if signal is None:
                        self._log_rejection(strategy_name, symbol, "strategy returned None")
                        continue

                    action = getattr(signal, "action", "hold")
                    if action is None:
                        action = "hold"
                    action = str(action).lower()

                    # Treat long/buy and short/sell equivalently here
                    if action in ("long", "buy"):
                        norm_side = "buy"
                    elif action in ("short", "sell"):
                        norm_side = "sell"
                    else:
                        # action == hold or anything unknown
                        # action == hold or anything unknown
                        # Keep the shape: reason=<string>; enrich with compact details if present
                        # Keep the shape: reason=<string>; enrich with compact details if present
                        try:
                            _detail = getattr(signal, "detail", None)
                            base_reason = str(getattr(signal, "reasoning", "") or "hold")
                            if isinstance(_detail, dict) and _detail:
                                keys = [k for k in ("timeframe","bars","vol","avg_vol","atr","spread","z","trend","rsi","bb_width")
                                        if k in _detail][:6]
                                extra = "; ".join(f"{k}={_detail[k]}" for k in keys)
                                reason_txt = f"action=hold; reason={base_reason}; {extra}"
                            else:
                                reason_txt = f"action=hold; reason={base_reason}"
                        except Exception:
                            reason_txt = f"action=hold; reason={getattr(signal, 'reasoning', '')}"

                        self._log_rejection(
                            strategy_name, symbol,
                            reason_txt,
                            confidence=getattr(signal, "confidence", 0.0)
                        )
                        continue


                    # 3) Strategy-level validation (logs details at its own level)
                    if not strategy.validate_signal(signal, market_data):
                        self._log_rejection(
                            strategy_name, symbol,
                            f"validate_signal=False (conf={getattr(signal, 'confidence', 0):.3f}, action={action})",
                            confidence=getattr(signal, "confidence", 0.0)
                        )
                        continue

                    # 4) Convert to dict format for the rest of the pipeline
                    conf_val = getattr(signal, "confidence", 0.0)
                    try:
                        conf_val = float(conf_val)
                    except Exception:
                        conf_val = 0.0

                    signal_dict = {
                        "symbol": signal.symbol,
                        "side": norm_side,  # 'buy' or 'sell'
                        "confidence": conf_val,
                        "strategy": getattr(signal, "strategy_name", strategy_name),
                        "entry_price": getattr(signal, "entry_price", None),
                        "stop_loss": getattr(signal, "stop_loss", None),
                        "take_profit": getattr(signal, "take_profit", None),
                        "position_size": getattr(signal, "position_size", None),
                        "reasoning": getattr(signal, "reasoning", ""),
                        "allocation_weight": allocation,
                        # helpful extras for the dashboard/analytics
                        "timeframe": getattr(signal, "timeframe", "1m"),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    all_signals.append(signal_dict)

                    # Update performance tracking
                    self.strategy_performance[strategy_name]["signals_generated"] += 1
                    self.strategy_performance[strategy_name]["last_signal_time"] = datetime.utcnow().isoformat()

                    logger.debug(
                        "📊 Signal from %s: %s %s (confidence: %.3f)",
                        strategy_name, norm_side, symbol, conf_val
                    )

                except Exception as e:
                    logger.error("❌ Signal generation failed for %s: %s", strategy_name, e)
                    continue

            self.total_signals_generated += len(all_signals)
            return all_signals

        except Exception as e:
            logger.error("❌ Signal generation failed for %s: %s", symbol, e)
            return []

    def _log_rejection(self, strategy_name: str, symbol: str, reason: str, confidence: float | int | str = 0.0) -> None:
        """Uniform logging for why a signal wasn't accepted/appended.

        IMPORTANT: Use INFO level and a strict, parseable format so the dashboard
        /api/rejections fallback (log parser) can display entries.
        """
        try:
            conf = float(confidence)
        except Exception:
            conf = 0.0

        # Do NOT change this prefix/shape unless you also update the dashboard parser.
        # Log as ERROR if it's an actual error, INFO if it's a normal rejection
        if "error:" in str(reason).lower():
            logger.error(
                "REJECTION | strategy=%s symbol=%s reason=%s conf=%.3f",
                strategy_name, symbol, str(reason).strip(), conf
            )
        else:
            logger.info(
                "REJECTION | strategy=%s symbol=%s reason=%s conf=%.3f",
                strategy_name, symbol, str(reason).strip(), conf
            )

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


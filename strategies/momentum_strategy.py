"""
Momentum Strategy - Trend-following strategy with multiple confirmations
Uses EMA crossovers, RSI, MACD, and volume confirmation for signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from strategies.base_strategy import BaseStrategy, StrategySignal
from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MomentumStrategy(BaseStrategy):
    """Enhanced momentum strategy with multiple confirmations"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'ema_fast': 12,
            'ema_slow': 26,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'volume_threshold': 1.5,  # Volume must be 1.5x average
            'atr_period': 14,
            'min_trend_strength': 0.02,  # 2% minimum trend strength
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            name="Enhanced Momentum Strategy",
            timeframes=["15m", "1h", "4h"],
            parameters=default_params
        )
        
        # Strategy-specific settings
        self.requires_trend = True
        self.min_confidence = 0.65
        self.stop_loss_pct = 0.025  # 2.5% stop loss
        self.take_profit_pct = 0.055  # 5.5% take profit
    
    def get_required_data(self) -> Dict[str, Any]:
        """Get data requirements"""
        return {
            'timeframes': self.timeframes,
            'lookback_period': max(self.parameters['ema_slow'], 50),
            'required_indicators': [
                'ema_fast', 'ema_slow', 'rsi', 'macd', 'macd_signal',
                'volume_sma', 'atr', 'bb_upper', 'bb_lower'
            ]
        }
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> StrategySignal:
        """Analyze market data and generate momentum signal"""
        try:
            # Preprocess data
            df = self.preprocess_data(market_data)
            if df.empty or len(df) < self.parameters['ema_slow']:
                return self._no_signal(symbol, "Insufficient data")
            
            # Calculate indicators
            df = self._calculate_momentum_indicators(df)
            
            # Check if market conditions are suitable
            if not self.is_market_condition_suitable(df):
                return self._no_signal(symbol, "Unsuitable market conditions")
            
            # Generate signal
            signal = self._generate_momentum_signal(symbol, df)
            
            # Validate signal
            if not self.validate_signal(signal, market_data):
                return self._no_signal(symbol, "Signal validation failed")
            
            # Set stop loss and take profit
            signal = self.set_stop_loss_take_profit(signal)
            
            # Update state
            self.state.last_signal = signal
            self.state.last_update = datetime.utcnow()
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Momentum analysis failed for {symbol}: {e}")
            return self._no_signal(symbol, f"Analysis error: {e}")
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-specific indicators"""
        try:
            # EMAs
            df[f'ema_{self.parameters["ema_fast"]}'] = df['close'].ewm(
                span=self.parameters['ema_fast']
            ).mean()
            df[f'ema_{self.parameters["ema_slow"]}'] = df['close'].ewm(
                span=self.parameters['ema_slow']
            ).mean()
            
            # RSI
            df = self.indicators.add_rsi(df, self.parameters['rsi_period'])
            
            # MACD
            df = self.indicators.add_macd(
                df, 
                self.parameters['macd_fast'],
                self.parameters['macd_slow'],
                self.parameters['macd_signal']
            )
            
            # Volume indicators
            df = self.indicators.add_volume_sma(df, 20)
            
            # Volatility
            df = self.indicators.add_atr(df, self.parameters['atr_period'])
            
            # Bollinger Bands for additional confirmation
            df = self.indicators.add_bollinger_bands(df)
            
            # Trend strength
            df['trend_strength'] = abs(
                df[f'ema_{self.parameters["ema_fast"]}'] - 
                df[f'ema_{self.parameters["ema_slow"]}']
            ) / df['close']
            
            # Price momentum
            df['price_momentum'] = df['close'].pct_change(periods=10)
            
            # Volume momentum
            df['volume_momentum'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Momentum indicators calculation failed: {e}")
            return df
    
    def _generate_momentum_signal(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        """Generate momentum trading signal"""
        try:
            current_idx = len(df) - 1
            prev_idx = current_idx - 1
            
            if prev_idx < 0:
                return self._no_signal(symbol, "Insufficient data for comparison")
            
            # Get current values
            current_price = df['close'].iloc[current_idx]
            ema_fast = df[f'ema_{self.parameters["ema_fast"]}'].iloc[current_idx]
            ema_slow = df[f'ema_{self.parameters["ema_slow"]}'].iloc[current_idx]
            rsi = df['rsi'].iloc[current_idx]
            macd = df['macd'].iloc[current_idx]
            macd_signal = df['macd_signal'].iloc[current_idx]
            macd_histogram = df['macd_histogram'].iloc[current_idx]
            volume_momentum = df['volume_momentum'].iloc[current_idx]
            trend_strength = df['trend_strength'].iloc[current_idx]
            price_momentum = df['price_momentum'].iloc[current_idx]
            
            # Get previous values for trend detection
            prev_ema_fast = df[f'ema_{self.parameters["ema_fast"]}'].iloc[prev_idx]
            prev_ema_slow = df[f'ema_{self.parameters["ema_slow"]}'].iloc[prev_idx]
            prev_macd = df['macd'].iloc[prev_idx]
            prev_macd_signal = df['macd_signal'].iloc[prev_idx]
            
            # Initialize signal components
            bullish_signals = 0
            bearish_signals = 0
            signal_strength = 0.0
            reasoning_parts = []
            
            # 1. EMA Crossover Analysis
            ema_bullish = ema_fast > ema_slow
            ema_bearish = ema_fast < ema_slow
            
            # Check for fresh crossover
            prev_ema_bullish = prev_ema_fast > prev_ema_slow
            fresh_bullish_cross = ema_bullish and not prev_ema_bullish
            fresh_bearish_cross = ema_bearish and not prev_ema_bullish
            
            if fresh_bullish_cross:
                bullish_signals += 2
                signal_strength += 0.3
                reasoning_parts.append("Fresh bullish EMA crossover")
            elif ema_bullish and trend_strength > self.parameters['min_trend_strength']:
                bullish_signals += 1
                signal_strength += 0.15
                reasoning_parts.append("EMA bullish trend confirmed")
            
            if fresh_bearish_cross:
                bearish_signals += 2
                signal_strength += 0.3
                reasoning_parts.append("Fresh bearish EMA crossover")
            elif ema_bearish and trend_strength > self.parameters['min_trend_strength']:
                bearish_signals += 1
                signal_strength += 0.15
                reasoning_parts.append("EMA bearish trend confirmed")
            
            # 2. RSI Analysis
            if self.parameters['rsi_oversold'] < rsi < self.parameters['rsi_overbought']:
                if rsi > 50 and ema_bullish:
                    bullish_signals += 1
                    signal_strength += 0.1
                    reasoning_parts.append(f"RSI bullish ({rsi:.1f})")
                elif rsi < 50 and ema_bearish:
                    bearish_signals += 1
                    signal_strength += 0.1
                    reasoning_parts.append(f"RSI bearish ({rsi:.1f})")
            
            # 3. MACD Analysis
            macd_bullish = macd > macd_signal and macd_histogram > 0
            macd_bearish = macd < macd_signal and macd_histogram < 0
            
            # Check for MACD crossover
            prev_macd_bullish = prev_macd > prev_macd_signal
            macd_cross_bullish = macd_bullish and not prev_macd_bullish
            macd_cross_bearish = macd_bearish and prev_macd_bullish
            
            if macd_cross_bullish:
                bullish_signals += 2
                signal_strength += 0.25
                reasoning_parts.append("MACD bullish crossover")
            elif macd_bullish:
                bullish_signals += 1
                signal_strength += 0.12
                reasoning_parts.append("MACD bullish trend")
            
            if macd_cross_bearish:
                bearish_signals += 2
                signal_strength += 0.25
                reasoning_parts.append("MACD bearish crossover")
            elif macd_bearish:
                bearish_signals += 1
                signal_strength += 0.12
                reasoning_parts.append("MACD bearish trend")
            
            # 4. Volume Confirmation
            volume_confirmed = volume_momentum > self.parameters['volume_threshold']
            if volume_confirmed:
                if bullish_signals > bearish_signals:
                    bullish_signals += 1
                    signal_strength += 0.15
                    reasoning_parts.append(f"Volume confirmed ({volume_momentum:.1f}x)")
                elif bearish_signals > bullish_signals:
                    bearish_signals += 1
                    signal_strength += 0.15
                    reasoning_parts.append(f"Volume confirmed ({volume_momentum:.1f}x)")
            
            # 5. Price Momentum
            if abs(price_momentum) > 0.01:  # Minimum 1% price movement
                if price_momentum > 0 and bullish_signals > bearish_signals:
                    bullish_signals += 1
                    signal_strength += 0.1
                    reasoning_parts.append(f"Strong price momentum (+{price_momentum*100:.1f}%)")
                elif price_momentum < 0 and bearish_signals > bullish_signals:
                    bearish_signals += 1
                    signal_strength += 0.1
                    reasoning_parts.append(f"Strong price momentum ({price_momentum*100:.1f}%)")
            
            # 6. Bollinger Bands confirmation
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_upper = df['bb_upper'].iloc[current_idx]
                bb_lower = df['bb_lower'].iloc[current_idx]
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                
                if bb_position > 0.8:  # Near upper band
                    if bearish_signals > bullish_signals:
                        bearish_signals += 1
                        signal_strength += 0.08
                        reasoning_parts.append("Near BB upper band")
                elif bb_position < 0.2:  # Near lower band
                    if bullish_signals > bearish_signals:
                        bullish_signals += 1
                        signal_strength += 0.08
                        reasoning_parts.append("Near BB lower band")
            
            # Determine final signal
            net_signals = bullish_signals - bearish_signals
            
            # Require minimum signal strength and clear direction
            min_net_signals = 2  # Require at least 2 net signals
            
            if net_signals >= min_net_signals and signal_strength >= 0.4:
                action = "buy"
                confidence = min(0.95, 0.5 + signal_strength)
                reasoning = f"Bullish momentum: {', '.join(reasoning_parts)}"
                
            elif net_signals <= -min_net_signals and signal_strength >= 0.4:
                action = "sell"
                confidence = min(0.95, 0.5 + signal_strength)
                reasoning = f"Bearish momentum: {', '.join(reasoning_parts)}"
                
            else:
                return self._no_signal(symbol, f"Weak signal: B{bullish_signals}/B{bearish_signals}, strength={signal_strength:.2f}")
            
            # Additional confidence adjustments
            if trend_strength > 0.05:  # Strong trend
                confidence += 0.05
            
            if volume_confirmed:
                confidence += 0.05
            
            # Fresh crossovers get bonus confidence
            if fresh_bullish_cross or fresh_bearish_cross or macd_cross_bullish or macd_cross_bearish:
                confidence += 0.1
            
            # Cap confidence
            confidence = min(0.98, confidence)
            
            return StrategySignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=current_price,
                reasoning=reasoning,
                timeframe=self.timeframes[0],
                strategy_name=self.name
            )
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return self._no_signal(symbol, f"Signal generation error: {e}")
    
    def _no_signal(self, symbol: str, reason: str) -> StrategySignal:
        """Generate no-action signal"""
        return StrategySignal(
            symbol=symbol,
            action="hold",
            confidence=0.0,
            reasoning=reason,
            timeframe=self.timeframes[0],
            strategy_name=self.name
        )
    
    def is_market_condition_suitable(self, df: pd.DataFrame) -> bool:
        """Check if market conditions are suitable for momentum strategy"""
        try:
            if df.empty or len(df) < 20:
                return False
            
            # Check for sufficient volatility
            if 'atr' in df.columns:
                recent_atr = df['atr'].iloc[-5:].mean()
                current_price = df['close'].iloc[-1]
                volatility_pct = (recent_atr / current_price) * 100
                
                # Momentum strategies need some volatility
                if volatility_pct < 0.8:  # Less than 0.8% volatility
                    logger.debug(f"Low volatility for momentum: {volatility_pct:.2f}%")
                    return False
            
            # Check for trending conditions
            if f'ema_{self.parameters["ema_fast"]}' in df.columns and f'ema_{self.parameters["ema_slow"]}' in df.columns:
                ema_fast = df[f'ema_{self.parameters["ema_fast"]}'].iloc[-10:]
                ema_slow = df[f'ema_{self.parameters["ema_slow"]}'].iloc[-10:]
                
                # Calculate trend consistency
                fast_trend = (ema_fast.iloc[-1] - ema_fast.iloc[0]) / ema_fast.iloc[0]
                slow_trend = (ema_slow.iloc[-1] - ema_slow.iloc[0]) / ema_slow.iloc[0]
                
                # Both EMAs should be trending in same direction for momentum
                if abs(fast_trend) < 0.01 and abs(slow_trend) < 0.01:
                    logger.debug("Market not trending enough for momentum strategy")
                    return False
            
            # Check for sufficient volume
            if 'volume_sma' in df.columns:
                recent_volume = df['volume'].iloc[-5:].mean()
                avg_volume = df['volume_sma'].iloc[-1]
                
                if recent_volume < avg_volume * 0.5:  # Less than 50% of average volume
                    logger.debug("Insufficient volume for momentum strategy")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Market condition check failed: {e}")
            return False
    
    def set_stop_loss_take_profit(self, signal: StrategySignal) -> StrategySignal:
        """Set dynamic stop loss and take profit based on volatility"""
        try:
            if not signal.entry_price or signal.action == "hold":
                return signal
            
            # Base stop loss and take profit
            base_stop_loss = self.stop_loss_pct
            base_take_profit = self.take_profit_pct
            
            # Adjust based on confidence (higher confidence = tighter stops)
            confidence_factor = signal.confidence
            
            # Adjust stop loss based on confidence
            if confidence_factor > 0.8:
                stop_loss_pct = base_stop_loss * 0.8  # Tighter stop for high confidence
                take_profit_pct = base_take_profit * 1.2  # Higher target
            elif confidence_factor < 0.6:
                stop_loss_pct = base_stop_loss * 1.2  # Wider stop for low confidence
                take_profit_pct = base_take_profit * 0.9  # Lower target
            else:
                stop_loss_pct = base_stop_loss
                take_profit_pct = base_take_profit
            
            # Set stop loss and take profit
            if signal.action == "buy":
                signal.stop_loss = signal.entry_price * (1 - stop_loss_pct)
                signal.take_profit = signal.entry_price * (1 + take_profit_pct)
            elif signal.action == "sell":
                signal.stop_loss = signal.entry_price * (1 + stop_loss_pct)
                signal.take_profit = signal.entry_price * (1 - take_profit_pct)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Stop loss/take profit setting failed: {e}")
            return signal
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get detailed strategy summary"""
        base_summary = self.get_performance_summary()
        
        momentum_specific = {
            'strategy_type': 'Momentum/Trend Following',
            'required_market_conditions': 'Trending markets with volume',
            'typical_holding_period': '2-8 hours',
            'risk_level': 'Medium-High',
            'best_markets': 'Volatile trending markets',
            'avoid_markets': 'Ranging/sideways markets',
            'parameters': {
                'ema_fast': self.parameters['ema_fast'],
                'ema_slow': self.parameters['ema_slow'],
                'rsi_period': self.parameters['rsi_period'],
                'volume_threshold': self.parameters['volume_threshold'],
                'min_trend_strength': self.parameters['min_trend_strength']
            },
            'signal_components': [
                'EMA crossovers',
                'RSI confirmation',
                'MACD signals',
                'Volume confirmation',
                'Price momentum',
                'Bollinger Bands position'
            ]
        }
        
        base_summary.update(momentum_specific)
        return base_summary
    
    def optimize_parameters(self, historical_data: pd.DataFrame, 
                          optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Optimize strategy parameters using historical data"""
        try:
            logger.info(f"🔧 Optimizing {self.name} parameters...")
            
            # Parameter ranges to test
            param_ranges = {
                'ema_fast': [8, 10, 12, 15],
                'ema_slow': [21, 26, 30, 35],
                'rsi_period': [10, 14, 18, 21],
                'volume_threshold': [1.2, 1.5, 1.8, 2.0],
                'min_trend_strength': [0.01, 0.02, 0.03, 0.04]
            }
            
            best_params = self.parameters.copy()
            best_score = -float('inf')
            optimization_results = []
            
            # Grid search (simplified for demonstration)
            import itertools
            
            # Generate parameter combinations (limit to avoid excessive computation)
            param_combinations = list(itertools.product(
                param_ranges['ema_fast'][:2],  # Limit combinations
                param_ranges['ema_slow'][:2],
                param_ranges['rsi_period'][:2],
                param_ranges['volume_threshold'][:2],
                param_ranges['min_trend_strength'][:2]
            ))
            
            for i, params in enumerate(param_combinations[:20]):  # Test first 20 combinations
                test_params = {
                    'ema_fast': params[0],
                    'ema_slow': params[1],
                    'rsi_period': params[2],
                    'volume_threshold': params[3],
                    'min_trend_strength': params[4]
                }
                
                # Temporarily update parameters
                original_params = self.parameters.copy()
                self.parameters.update(test_params)
                
                # Backtest with these parameters
                backtest_score = self._backtest_parameters(historical_data, optimization_metric)
                
                optimization_results.append({
                    'parameters': test_params.copy(),
                    'score': backtest_score
                })
                
                if backtest_score > best_score:
                    best_score = backtest_score
                    best_params = test_params.copy()
                
                # Restore original parameters
                self.parameters = original_params
                
                logger.debug(f"Tested params {i+1}/20: score = {backtest_score:.4f}")
            
            # Update with best parameters
            self.parameters.update(best_params)
            
            logger.info(f"✅ Optimization completed. Best score: {best_score:.4f}")
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'optimization_results': optimization_results,
                'improvement': best_score  # Would compare to baseline
            }
            
        except Exception as e:
            logger.error(f"❌ Parameter optimization failed: {e}")
            return {'error': str(e)}
    
    def _backtest_parameters(self, data: pd.DataFrame, metric: str) -> float:
        """Simplified backtesting for parameter optimization"""
        try:
            # This is a simplified version - in practice, you'd run full backtest
            df = self._calculate_momentum_indicators(data)
            
            if df.empty or len(df) < 100:
                return -1.0
            
            # Generate signals for the data
            signals = []
            for i in range(50, len(df)):  # Start after warm-up period
                test_data = df.iloc[:i+1]
                try:
                    # Simplified signal generation
                    if len(test_data) >= self.parameters['ema_slow']:
                        signal = self._generate_momentum_signal("TEST", test_data)
                        if signal.action != "hold" and signal.confidence > self.min_confidence:
                            signals.append({
                                'index': i,
                                'action': signal.action,
                                'confidence': signal.confidence,
                                'price': test_data['close'].iloc[-1]
                            })
                except:
                    continue
            
            if not signals:
                return -1.0
            
            # Calculate simple performance metric
            if metric == 'signal_count':
                return len(signals)
            elif metric == 'avg_confidence':
                return np.mean([s['confidence'] for s in signals])
            else:
                # Default to signal quality score
                return len(signals) * np.mean([s['confidence'] for s in signals])
                
        except Exception as e:
            logger.error(f"❌ Backtest error: {e}")
            return -1.0
    
    def get_market_regime_suitability(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how suitable current market regime is for this strategy"""
        try:
            df = self.preprocess_data(market_data)
            if df.empty:
                return {'suitability': 'unknown', 'score': 0.0}
            
            df = self._calculate_momentum_indicators(df)
            
            suitability_score = 0.0
            factors = []
            
            # Check trend strength
            if 'trend_strength' in df.columns:
                avg_trend_strength = df['trend_strength'].iloc[-10:].mean()
                if avg_trend_strength > 0.03:
                    suitability_score += 0.3
                    factors.append(f"Strong trend ({avg_trend_strength:.1%})")
                elif avg_trend_strength > 0.01:
                    suitability_score += 0.1
                    factors.append(f"Moderate trend ({avg_trend_strength:.1%})")
            
            # Check volatility
            if 'atr' in df.columns:
                current_atr = df['atr'].iloc[-1]
                current_price = df['close'].iloc[-1]
                volatility_pct = (current_atr / current_price) * 100
                
                if volatility_pct > 2.0:
                    suitability_score += 0.3
                    factors.append(f"High volatility ({volatility_pct:.1f}%)")
                elif volatility_pct > 1.0:
                    suitability_score += 0.2
                    factors.append(f"Moderate volatility ({volatility_pct:.1f}%)")
            
            # Check volume
            if 'volume_momentum' in df.columns:
                avg_volume_momentum = df['volume_momentum'].iloc[-5:].mean()
                if avg_volume_momentum > 1.5:
                    suitability_score += 0.2
                    factors.append(f"Strong volume ({avg_volume_momentum:.1f}x)")
                elif avg_volume_momentum > 1.0:
                    suitability_score += 0.1
                    factors.append(f"Good volume ({avg_volume_momentum:.1f}x)")
            
            # Check MACD momentum
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd_momentum = df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]
                if abs(macd_momentum) > df['close'].iloc[-1] * 0.001:
                    suitability_score += 0.2
                    factors.append("Strong MACD momentum")
            
            # Determine suitability level
            if suitability_score >= 0.7:
                suitability = 'excellent'
            elif suitability_score >= 0.5:
                suitability = 'good'
            elif suitability_score >= 0.3:
                suitability = 'moderate'
            elif suitability_score >= 0.1:
                suitability = 'poor'
            else:
                suitability = 'unsuitable'
            
            return {
                'suitability': suitability,
                'score': suitability_score,
                'factors': factors,
                'recommendation': self._get_regime_recommendation(suitability)
            }
            
        except Exception as e:
            logger.error(f"❌ Market regime analysis failed: {e}")
            return {'suitability': 'unknown', 'score': 0.0, 'error': str(e)}
    
    def _get_regime_recommendation(self, suitability: str) -> str:
        """Get recommendation based on market regime suitability"""
        recommendations = {
            'excellent': "Perfect conditions for momentum trading. Increase position size.",
            'good': "Favorable conditions. Normal position sizing recommended.",
            'moderate': "Mixed conditions. Consider reducing position size or waiting.",
            'poor': "Unfavorable conditions. Avoid new positions or use minimal size.",
            'unsuitable': "Very poor conditions. Disable strategy or wait for better setup."
        }
        
        return recommendations.get(suitability, "Unknown market conditions.")
    
    def __str__(self) -> str:
        return f"MomentumStrategy(EMA: {self.parameters['ema_fast']}/{self.parameters['ema_slow']}, signals: {self.state.signals_generated})"
"""
Mean Reversion Strategy - Counter-trend trading strategy
Uses statistical measures to identify overbought/oversold conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from strategies.base_strategy import BaseStrategy, StrategySignal
from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MeanReversionStrategy(BaseStrategy):
    """Statistical mean reversion strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'rsi_period': 14,
            'rsi_overbought': 75,
            'rsi_oversold': 25,
            'stoch_period': 14,
            'stoch_d_period': 3,
            'z_score_period': 50,
            'z_score_threshold': 2.0,
            'volume_confirmation': True,
            'volume_threshold': 1.2,
            'max_holding_time': 240,  # minutes
            'profit_target_pct': 0.03,  # 3%
            'stop_loss_pct': 0.025,    # 2.5%
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            name="Mean Reversion Strategy",
            timeframes=["5m", "15m", "1h"],
            parameters=default_params
        )
        
        # Strategy-specific settings
        self.requires_trend = False  # Works better in ranging markets
        self.min_confidence = 0.6
        self.stop_loss_pct = self.parameters['stop_loss_pct']
        self.take_profit_pct = self.parameters['profit_target_pct']
        
        # Track position entry times for time-based exits
        self.position_entry_times = {}
    
    def get_required_data(self) -> Dict[str, Any]:
        """Get data requirements"""
        return {
            'timeframes': self.timeframes,
            'lookback_period': max(self.parameters['bb_period'], self.parameters['z_score_period'], 100),
            'required_indicators': [
                'bb_upper', 'bb_lower', 'bb_middle',
                'rsi', 'stoch_k', 'stoch_d',
                'volume_sma', 'atr'
            ]
        }
    
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> StrategySignal:
        """Analyze market data and generate mean reversion signal"""
        try:
            # Preprocess data
            df = self.preprocess_data(market_data)
            if df.empty or len(df) < self.parameters['bb_period']:
                return self._no_signal(symbol, "Insufficient data")
            
            # Calculate indicators
            df = self._calculate_mean_reversion_indicators(df)
            
            # Check market suitability for mean reversion
            if not self._is_suitable_for_mean_reversion(df):
                return self._no_signal(symbol, "Market not suitable for mean reversion")
            
            # Generate signal
            signal = self._generate_mean_reversion_signal(symbol, df)
            
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
            logger.error(f"❌ Mean reversion analysis failed for {symbol}: {e}")
            return self._no_signal(symbol, f"Analysis error: {e}")
    
    def _calculate_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion specific indicators"""
        try:
            # Bollinger Bands
            df = self.indicators.add_bollinger_bands(df, self.parameters['bb_period'], self.parameters['bb_std_dev'])
            
            # RSI
            df = self.indicators.add_rsi(df, self.parameters['rsi_period'])
            
            # Stochastic
            df = self.indicators.add_stochastic(df, self.parameters['stoch_period'], self.parameters['stoch_d_period'])
            
            # Volume
            df = self.indicators.add_volume_sma(df, 20)
            
            # ATR for volatility
            df = self.indicators.add_atr(df, 14)
            
            # Z-Score of price relative to moving average
            sma = df['close'].rolling(self.parameters['z_score_period']).mean()
            std = df['close'].rolling(self.parameters['z_score_period']).std()
            df['z_score'] = (df['close'] - sma) / std
            
            # Distance from Bollinger Bands
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['distance_to_upper'] = (df['bb_upper'] - df['close']) / df['close']
                df['distance_to_lower'] = (df['close'] - df['bb_lower']) / df['close']
            
            # Mean reversion oscillator (custom)
            short_ma = df['close'].rolling(5).mean()
            long_ma = df['close'].rolling(20).mean()
            df['mr_oscillator'] = (short_ma - long_ma) / long_ma
            
            # Volatility normalization
            df['normalized_price'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            
            # Price extremes
            df['is_price_extreme_high'] = df['close'] >= df['close'].rolling(20).max()
            df['is_price_extreme_low'] = df['close'] <= df['close'].rolling(20).min()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Mean reversion indicators calculation failed: {e}")
            return df
    
    def _is_suitable_for_mean_reversion(self, df: pd.DataFrame) -> bool:
        """Check if market conditions are suitable for mean reversion"""
        try:
            if len(df) < 50:
                return False
            
            # Check for ranging market (not strongly trending)
            recent_data = df.tail(20)
            
            # Calculate trend strength
            if 'close' in recent_data.columns:
                price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
                
                # Avoid strong trends (threshold: 5% over 20 periods)
                if abs(price_change) > 0.05:
                    logger.debug("Market trending too strongly for mean reversion")
                    return False
            
            # Check volatility (need some volatility for mean reversion opportunities)
            if 'atr' in df.columns:
                recent_atr = df['atr'].tail(10).mean()
                current_price = df['close'].iloc[-1]
                volatility_pct = (recent_atr / current_price) * 100
                
                # Need minimum volatility (0.5%) but not too high (8%)
                if volatility_pct < 0.5 or volatility_pct > 8.0:
                    logger.debug(f"Volatility unsuitable for mean reversion: {volatility_pct:.2f}%")
                    return False
            
            # Check Bollinger Band width (need sufficient width)
            if 'bb_width' in df.columns:
                recent_bb_width = df['bb_width'].tail(5).mean()
                if recent_bb_width < 0.02:  # Less than 2%
                    logger.debug("Bollinger Bands too narrow for mean reversion")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Market suitability check failed: {e}")
            return False
    
    def _generate_mean_reversion_signal(self, symbol: str, df: pd.DataFrame) -> StrategySignal:
        """Generate mean reversion trading signal"""
        try:
            current_idx = len(df) - 1
            current_price = df['close'].iloc[current_idx]
            
            # Get current indicator values
            bb_upper = df['bb_upper'].iloc[current_idx] if 'bb_upper' in df.columns else None
            bb_lower = df['bb_lower'].iloc[current_idx] if 'bb_lower' in df.columns else None
            bb_middle = df['bb_middle'].iloc[current_idx] if 'bb_middle' in df.columns else None
            bb_position = df['bb_position'].iloc[current_idx] if 'bb_position' in df.columns else 0.5
            
            rsi = df['rsi'].iloc[current_idx] if 'rsi' in df.columns else 50
            stoch_k = df['stoch_k'].iloc[current_idx] if 'stoch_k' in df.columns else 50
            stoch_d = df['stoch_d'].iloc[current_idx] if 'stoch_d' in df.columns else 50
            z_score = df['z_score'].iloc[current_idx] if 'z_score' in df.columns else 0
            
            volume_ratio = df['volume_ratio'].iloc[current_idx] if 'volume_ratio' in df.columns else 1.0
            
            # Initialize signal scoring
            buy_score = 0
            sell_score = 0
            reasoning_parts = []
            
            # 1. Bollinger Bands Analysis
            if bb_upper and bb_lower and bb_middle:
                if current_price <= bb_lower:
                    buy_score += 3
                    reasoning_parts.append(f"Price at BB lower band (${current_price:.4f} <= ${bb_lower:.4f})")
                elif bb_position < 0.2:
                    buy_score += 2
                    reasoning_parts.append(f"Price near BB lower (position: {bb_position:.2f})")
                
                if current_price >= bb_upper:
                    sell_score += 3
                    reasoning_parts.append(f"Price at BB upper band (${current_price:.4f} >= ${bb_upper:.4f})")
                elif bb_position > 0.8:
                    sell_score += 2
                    reasoning_parts.append(f"Price near BB upper (position: {bb_position:.2f})")
            
            # 2. RSI Analysis
            if rsi <= self.parameters['rsi_oversold']:
                buy_score += 2
                reasoning_parts.append(f"RSI oversold ({rsi:.1f})")
            elif rsi <= 35:
                buy_score += 1
                reasoning_parts.append(f"RSI low ({rsi:.1f})")
            
            if rsi >= self.parameters['rsi_overbought']:
                sell_score += 2
                reasoning_parts.append(f"RSI overbought ({rsi:.1f})")
            elif rsi >= 65:
                sell_score += 1
                reasoning_parts.append(f"RSI high ({rsi:.1f})")
            
            # 3. Stochastic Analysis
            if stoch_k < 20 and stoch_d < 20:
                buy_score += 2
                reasoning_parts.append(f"Stochastic oversold (K:{stoch_k:.1f}, D:{stoch_d:.1f})")
            elif stoch_k < 30:
                buy_score += 1
                reasoning_parts.append(f"Stochastic low ({stoch_k:.1f})")
            
            if stoch_k > 80 and stoch_d > 80:
                sell_score += 2
                reasoning_parts.append(f"Stochastic overbought (K:{stoch_k:.1f}, D:{stoch_d:.1f})")
            elif stoch_k > 70:
                sell_score += 1
                reasoning_parts.append(f"Stochastic high ({stoch_k:.1f})")
            
            # 4. Z-Score Analysis
            z_threshold = self.parameters['z_score_threshold']
            if z_score <= -z_threshold:
                buy_score += 2
                reasoning_parts.append(f"Z-Score extreme low ({z_score:.2f})")
            elif z_score <= -1.5:
                buy_score += 1
                reasoning_parts.append(f"Z-Score low ({z_score:.2f})")
            
            if z_score >= z_threshold:
                sell_score += 2
                reasoning_parts.append(f"Z-Score extreme high ({z_score:.2f})")
            elif z_score >= 1.5:
                sell_score += 1
                reasoning_parts.append(f"Z-Score high ({z_score:.2f})")
            
            # 5. Volume Confirmation (if enabled)
            volume_confirmed = True
            if self.parameters['volume_confirmation']:
                if volume_ratio >= self.parameters['volume_threshold']:
                    if buy_score > sell_score:
                        buy_score += 1
                        reasoning_parts.append(f"Volume confirmed ({volume_ratio:.1f}x)")
                    elif sell_score > buy_score:
                        sell_score += 1
                        reasoning_parts.append(f"Volume confirmed ({volume_ratio:.1f}x)")
                else:
                    volume_confirmed = False
                    reasoning_parts.append(f"Low volume ({volume_ratio:.1f}x)")
            
            # 6. Multiple timeframe confirmation (simplified)
            # In a full implementation, you'd check higher timeframes
            if len(df) >= 20:
                sma_20 = df['close'].rolling(20).mean().iloc[current_idx]
                if buy_score > sell_score and current_price < sma_20 * 0.98:
                    buy_score += 1
                    reasoning_parts.append("Below 20-SMA support")
                elif sell_score > buy_score and current_price > sma_20 * 1.02:
                    sell_score += 1
                    reasoning_parts.append("Above 20-SMA resistance")
            
            # 7. Recent price action
            if len(df) >= 5:
                recent_low = df['low'].tail(5).min()
                recent_high = df['high'].tail(5).max()
                
                if current_price <= recent_low and buy_score > sell_score:
                    buy_score += 1
                    reasoning_parts.append("At recent support")
                elif current_price >= recent_high and sell_score > buy_score:
                    sell_score += 1
                    reasoning_parts.append("At recent resistance")
            
            # Determine final signal
            min_score = 4  # Minimum score required for signal
            net_score = buy_score - sell_score
            
            if buy_score >= min_score and net_score >= 2:
                action = "buy"
                base_confidence = min(0.95, 0.5 + (buy_score * 0.08))
                reasoning = f"Mean reversion BUY: {', '.join(reasoning_parts[:4])}"
                
            elif sell_score >= min_score and net_score <= -2:
                action = "sell"
                base_confidence = min(0.95, 0.5 + (sell_score * 0.08))
                reasoning = f"Mean reversion SELL: {', '.join(reasoning_parts[:4])}"
                
            else:
                return self._no_signal(symbol, f"Weak signals: Buy({buy_score}) vs Sell({sell_score})")
            
            # Adjust confidence based on various factors
            confidence = base_confidence
            
            # Volume confirmation adjustment
            if not volume_confirmed:
                confidence *= 0.9
            
            # Multiple indicator confirmation bonus
            confirming_indicators = 0
            if 'bb_position' in locals():
                if (action == "buy" and bb_position < 0.3) or (action == "sell" and bb_position > 0.7):
                    confirming_indicators += 1
            
            if (action == "buy" and rsi < 40) or (action == "sell" and rsi > 60):
                confirming_indicators += 1
            
            if (action == "buy" and z_score < -1) or (action == "sell" and z_score > 1):
                confirming_indicators += 1
            
            if confirming_indicators >= 2:
                confidence += 0.1
            
            # Market volatility adjustment
            if 'atr' in df.columns:
                atr = df['atr'].iloc[current_idx]
                vol_pct = (atr / current_price) * 100
                if vol_pct > 3:  # High volatility
                    confidence += 0.05
                elif vol_pct < 1:  # Low volatility
                    confidence *= 0.95
            
            # Cap confidence
            confidence = max(0.6, min(0.95, confidence))
            
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
        """Check if market conditions are suitable for mean reversion"""
        return self._is_suitable_for_mean_reversion(df)
    
    def set_stop_loss_take_profit(self, signal: StrategySignal) -> StrategySignal:
        """Set dynamic stop loss and take profit for mean reversion"""
        try:
            if not signal.entry_price or signal.action == "hold":
                return signal
            
            # Base levels
            base_stop_pct = self.parameters['stop_loss_pct']
            base_profit_pct = self.parameters['profit_target_pct']
            
            # Adjust based on confidence
            confidence_factor = signal.confidence
            
            # Higher confidence = tighter stops (expect quicker reversal)
            if confidence_factor > 0.8:
                stop_pct = base_stop_pct * 0.9
                profit_pct = base_profit_pct * 1.1
            elif confidence_factor < 0.65:
                stop_pct = base_stop_pct * 1.2
                profit_pct = base_profit_pct * 0.9
            else:
                stop_pct = base_stop_pct
                profit_pct = base_profit_pct
            
            # Set levels
            if signal.action == "buy":
                signal.stop_loss = signal.entry_price * (1 - stop_pct)
                signal.take_profit = signal.entry_price * (1 + profit_pct)
            elif signal.action == "sell":
                signal.stop_loss = signal.entry_price * (1 + stop_pct)
                signal.take_profit = signal.entry_price * (1 - profit_pct)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Stop/profit setting failed: {e}")
            return signal
    
    def should_exit_position(self, symbol: str, entry_time: datetime, 
                           current_price: float, entry_price: float, side: str) -> bool:
        """Check if position should be exited based on time or mean reversion"""
        try:
            # Time-based exit
            position_age = (datetime.utcnow() - entry_time).total_seconds() / 60  # minutes
            if position_age > self.parameters['max_holding_time']:
                logger.info(f"🕐 Time-based exit for {symbol}: {position_age:.1f} minutes")
                return True
            
            # Mean reversion completion check
            # If price has moved back towards the mean, consider exit
            price_move = (current_price - entry_price) / entry_price
            
            if side == "long" and price_move > 0.015:  # 1.5% profit on mean reversion
                logger.info(f"📈 Mean reversion profit target hit: {symbol}")
                return True
            elif side == "short" and price_move < -0.015:
                logger.info(f"📉 Mean reversion profit target hit: {symbol}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Exit check failed: {e}")
            return False
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get detailed strategy summary"""
        base_summary = self.get_performance_summary()
        
        mean_reversion_specific = {
            'strategy_type': 'Mean Reversion/Counter-Trend',
            'required_market_conditions': 'Ranging/sideways markets with volatility',
            'typical_holding_period': f'{self.parameters["max_holding_time"]} minutes max',
            'risk_level': 'Medium',
            'best_markets': 'Range-bound volatile markets',
            'avoid_markets': 'Strong trending markets',
            'parameters': {
                'bb_period': self.parameters['bb_period'],
                'bb_std_dev': self.parameters['bb_std_dev'],
                'rsi_overbought': self.parameters['rsi_overbought'],
                'rsi_oversold': self.parameters['rsi_oversold'],
                'z_score_threshold': self.parameters['z_score_threshold'],
                'volume_threshold': self.parameters['volume_threshold'],
                'max_holding_time': self.parameters['max_holding_time']
            },
            'signal_components': [
                'Bollinger Bands extremes',
                'RSI overbought/oversold',
                'Stochastic oscillator',
                'Z-Score deviation',
                'Volume confirmation',
                'Support/resistance levels'
            ]
        }
        
        base_summary.update(mean_reversion_specific)
        return base_summary
    
    def get_current_market_assessment(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current market for mean reversion opportunities"""
        try:
            df = self.preprocess_data(market_data)
            if df.empty:
                return {'assessment': 'insufficient_data'}
            
            df = self._calculate_mean_reversion_indicators(df)
            
            current_price = df['close'].iloc[-1]
            bb_position = df['bb_position'].iloc[-1] if 'bb_position' in df.columns else 0.5
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            z_score = df['z_score'].iloc[-1] if 'z_score' in df.columns else 0
            
            # Assess mean reversion opportunity
            opportunity_score = 0
            assessment_factors = []
            
            # Bollinger Band position
            if bb_position < 0.2:
                opportunity_score += 3
                assessment_factors.append(f"Near BB lower band (pos: {bb_position:.2f})")
            elif bb_position > 0.8:
                opportunity_score += 3
                assessment_factors.append(f"Near BB upper band (pos: {bb_position:.2f})")
            
            # RSI extremes
            if rsi < 30 or rsi > 70:
                opportunity_score += 2
                assessment_factors.append(f"RSI extreme ({rsi:.1f})")
            
            # Z-Score extremes
            if abs(z_score) > 2:
                opportunity_score += 2
                assessment_factors.append(f"Z-Score extreme ({z_score:.2f})")
            
            # Market suitability
            suitable = self._is_suitable_for_mean_reversion(df)
            
            if opportunity_score >= 5 and suitable:
                assessment = 'excellent'
            elif opportunity_score >= 3 and suitable:
                assessment = 'good'
            elif opportunity_score >= 1 and suitable:
                assessment = 'moderate'
            elif suitable:
                assessment = 'poor'
            else:
                assessment = 'unsuitable'
            
            return {
                'assessment': assessment,
                'opportunity_score': opportunity_score,
                'suitable_for_mean_reversion': suitable,
                'current_metrics': {
                    'bb_position': bb_position,
                    'rsi': rsi,
                    'z_score': z_score,
                    'price': current_price
                },
                'factors': assessment_factors
            }
            
        except Exception as e:
            logger.error(f"❌ Market assessment failed: {e}")
            return {'assessment': 'error', 'error': str(e)}
    
    def __str__(self) -> str:
        return f"MeanReversionStrategy(BB: {self.parameters['bb_period']}/{self.parameters['bb_std_dev']}, RSI: {self.parameters['rsi_overbought']}/{self.parameters['rsi_oversold']})"
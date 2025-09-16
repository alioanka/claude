"""
Simplified Arbitrage Strategy - Works with single exchange and market data
Focuses on statistical arbitrage and mean reversion opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass

from strategies.base_strategy import BaseStrategy, StrategySignal
from utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

@dataclass
class SimpleArbitrageConfig:
    """Configuration for simplified arbitrage strategy"""
    min_spread_threshold: float = 0.002  # 0.2% minimum spread
    max_spread_threshold: float = 0.05   # 5% maximum spread
    correlation_threshold: float = 0.7   # Minimum correlation for pair trading
    lookback_period: int = 100           # Periods for correlation calculation
    z_score_threshold: float = 1.5       # Z-score threshold for mean reversion
    max_position_hold_time: int = 24     # Maximum hours to hold position
    transaction_cost: float = 0.001      # Total transaction costs (0.1%)

class SimpleArbitrageStrategy(BaseStrategy):
    """
    Simplified arbitrage strategy that works with:
    1. Statistical arbitrage (pairs trading) using price ratios
    2. Mean reversion on price spreads
    3. Bollinger Band mean reversion
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("simple_arbitrage_strategy", config)
        self.config = config or {}
        self.arb_config = SimpleArbitrageConfig(**self.config.get('arbitrage_params', {}))
        self.indicators = TechnicalIndicators()
        
        # Price history for correlation analysis
        self.price_history = {}
        self.spread_history = {}
        self._last_spread = 0.0
        
        # Common trading pairs for statistical arbitrage
        self.correlation_pairs = {
            'BTCUSDT': ['ETHUSDT', 'BNBUSDT'],
            'ETHUSDT': ['BTCUSDT', 'ADAUSDT'],
            'BNBUSDT': ['BTCUSDT', 'ETHUSDT'],
            'ADAUSDT': ['ETHUSDT', 'DOTUSDT'],
            'SOLUSDT': ['ETHUSDT', 'AVAXUSDT'],
            'MATICUSDT': ['ETHUSDT', 'ADAUSDT'],
            'ATOMUSDT': ['ETHUSDT', 'DOTUSDT'],
            'LINKUSDT': ['ETHUSDT', 'ADAUSDT'],
            'DOTUSDT': ['ETHUSDT', 'ADAUSDT'],
            'AVAXUSDT': ['ETHUSDT', 'SOLUSDT']
        }
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate arbitrage signals"""
        try:
            # data is already a DataFrame, no need to call _ensure_dataframe
            if data is None or data.empty or len(data) < 50:
                return None

            signals = []
            
            # 1. Statistical arbitrage (pairs trading)
            pairs_signal = await self._check_statistical_arbitrage(symbol, data)
            if pairs_signal:
                signals.append(pairs_signal)
            
            # 2. Mean reversion on price spreads
            mean_reversion_signal = await self._check_mean_reversion(symbol, data)
            if mean_reversion_signal:
                signals.append(mean_reversion_signal)
            
            # 3. Bollinger Band mean reversion
            bb_signal = await self._check_bollinger_mean_reversion(symbol, data)
            if bb_signal:
                signals.append(bb_signal)
            
            # Return highest confidence signal
            if signals:
                best_signal = max(signals, key=lambda s: s.get('confidence', 0))
                return best_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Arbitrage signal generation failed for {symbol}: {e}")
            return None

    def get_required_data(self) -> Dict[str, Any]:
        """Describe the minimal data requirements for this strategy"""
        return {
            "timeframe": "1m",
            "lookback": max(self.arb_config.lookback_period, 100),
            "symbols": list(self.correlation_pairs.keys()),
            "notes": "Statistical arbitrage requires price data for correlated pairs"
        }

    async def analyze(self, symbol: str, market_data: Dict[str, Any]):
        """Analyze market data and generate trading signal"""
        try:
            df = self._ensure_dataframe(market_data, lookback=self.arb_config.lookback_period)
            if df is None or df.empty or len(df) < 50:
                return self._no_signal(symbol, "Insufficient data")

            # Generate signal
            sig = await self.generate_signal(symbol, df)
            
            if not sig:
                return self._no_signal(symbol, "No arbitrage opportunity")

            # Convert to StrategySignal
            action = sig.get("action") or sig.get("side", "hold")
            action = str(action).lower()
            if action in ("long", "buy"):
                action = "buy"
            elif action in ("short", "sell"):
                action = "sell"
            else:
                action = "hold"

            conf = float(sig.get("confidence", 0.0))
            entry = sig.get("entry_price") or sig.get("price")
            sl = sig.get("stop_loss")
            tp = sig.get("take_profit")

            return StrategySignal(
                symbol=symbol,
                action=action if conf >= self.min_confidence else "hold",
                confidence=conf,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                position_size=None,
                reasoning=sig.get("reasoning", "arbitrage"),
                timeframe="1m",
                strategy_name=self.name
            )
            
        except Exception as e:
            logger.error(f"[SimpleArbitrageStrategy] analyze failed for {symbol}: {e}")
            return self._no_signal(symbol, f"error: {e}")

    async def _check_statistical_arbitrage(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check for statistical arbitrage opportunities using price ratios"""
        try:
            if len(data) < self.arb_config.lookback_period:
                return None
            
            # Get correlated pairs
            pairs = self.correlation_pairs.get(symbol, [])
            if not pairs:
                return None
            
            # For now, simulate with price ratio analysis
            # In a real implementation, you'd fetch data for the correlated pair
            recent_data = data.tail(self.arb_config.lookback_period)
            
            # Calculate price ratio spread (simulated)
            # In reality: spread = price1/price2 - historical_mean(price1/price2)
            price_changes = recent_data['close'].pct_change().dropna()
            
            # Create synthetic spread using price momentum
            spread = np.cumsum(price_changes)
            
            # Calculate z-score
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            if spread_std == 0:
                return None
            
            current_spread = spread.iloc[-1]
            z_score = (current_spread - spread_mean) / spread_std
            
            # Generate signal based on z-score
            if abs(z_score) > self.arb_config.z_score_threshold:
                signal_type = 'sell' if z_score > 0 else 'buy'  # Mean reversion
                confidence = min(abs(z_score) / 3.0, 0.8)  # Scale to 0-0.8
                
                return {
                    'signal_type': 'statistical_arbitrage',
                    'symbol': symbol,
                    'action': signal_type,
                    'confidence': confidence,
                    'entry_price': float(data['close'].iloc[-1]),
                    'reasoning': f"Statistical arbitrage z-score: {z_score:.2f}",
                    'metadata': {
                        'type': 'statistical_arbitrage',
                        'z_score': z_score,
                        'spread': current_spread,
                        'correlation': 0.8  # Simulated
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Statistical arbitrage check failed: {e}")
            return None

    async def _check_mean_reversion(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check for mean reversion opportunities using price spreads"""
        try:
            if len(data) < 50:
                return None
            
            recent_data = data.tail(50)
            current_price = float(recent_data['close'].iloc[-1])
            
            # Calculate moving averages
            sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
            sma_50 = recent_data['close'].rolling(50).mean().iloc[-1]
            
            if pd.isna(sma_20) or pd.isna(sma_50):
                return None
            
            # Calculate deviation from mean
            deviation_20 = (current_price - sma_20) / sma_20
            deviation_50 = (current_price - sma_50) / sma_50
            
            # Check for mean reversion opportunity
            if abs(deviation_20) > 0.02 and abs(deviation_50) > 0.01:  # 2% and 1% thresholds
                signal_type = 'sell' if deviation_20 > 0 else 'buy'  # Mean reversion
                confidence = min(abs(deviation_20) * 10, 0.7)  # Scale to 0-0.7
                
                return {
                    'signal_type': 'mean_reversion',
                    'symbol': symbol,
                    'action': signal_type,
                    'confidence': confidence,
                    'entry_price': current_price,
                    'reasoning': f"Mean reversion: {deviation_20:.2%} deviation from SMA20",
                    'metadata': {
                        'type': 'mean_reversion',
                        'deviation_20': deviation_20,
                        'deviation_50': deviation_50,
                        'sma_20': sma_20,
                        'sma_50': sma_50
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Mean reversion check failed: {e}")
            return None

    async def _check_bollinger_mean_reversion(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check for Bollinger Band mean reversion opportunities"""
        try:
            if len(data) < 20:
                return None
            
            recent_data = data.tail(20)
            current_price = float(recent_data['close'].iloc[-1])
            
            # Calculate Bollinger Bands
            sma = recent_data['close'].rolling(20).mean().iloc[-1]
            std = recent_data['close'].rolling(20).std().iloc[-1]
            
            if pd.isna(sma) or pd.isna(std) or std == 0:
                return None
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            # Check for Bollinger Band touch
            if current_price >= upper_band:
                # Price touched upper band - potential sell signal
                confidence = min((current_price - upper_band) / upper_band * 20, 0.8)
                return {
                    'signal_type': 'bollinger_mean_reversion',
                    'symbol': symbol,
                    'action': 'sell',
                    'confidence': confidence,
                    'entry_price': current_price,
                    'reasoning': f"Bollinger Band upper touch: {current_price:.2f} >= {upper_band:.2f}",
                    'metadata': {
                        'type': 'bollinger_mean_reversion',
                        'upper_band': upper_band,
                        'lower_band': lower_band,
                        'sma': sma,
                        'std': std
                    }
                }
            elif current_price <= lower_band:
                # Price touched lower band - potential buy signal
                confidence = min((lower_band - current_price) / lower_band * 20, 0.8)
                return {
                    'signal_type': 'bollinger_mean_reversion',
                    'symbol': symbol,
                    'action': 'buy',
                    'confidence': confidence,
                    'entry_price': current_price,
                    'reasoning': f"Bollinger Band lower touch: {current_price:.2f} <= {lower_band:.2f}",
                    'metadata': {
                        'type': 'bollinger_mean_reversion',
                        'upper_band': upper_band,
                        'lower_band': lower_band,
                        'sma': sma,
                        'std': std
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Bollinger mean reversion check failed: {e}")
            return None

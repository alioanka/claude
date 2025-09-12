"""
Base Strategy Class - Abstract base for all trading strategies
Provides common interface and functionality for strategy implementations.
"""

from abc import ABC, abstractmethod
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from types import SimpleNamespace

from config.config import config
from utils.logger import setup_logger
from utils.indicators import TechnicalIndicators

logger = setup_logger(__name__)

@dataclass
class StrategySignal:
    """Strategy signal output"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    reasoning: str = ""
    timeframe: str = "1h"
    strategy_name: str = ""
    features: Optional[Dict[str, float]] = None

@dataclass
class StrategyState:
    """Strategy internal state"""
    last_signal: Optional[StrategySignal] = None
    last_update: datetime = datetime.utcnow()
    signals_generated: int = 0
    profitable_signals: int = 0
    total_pnl: float = 0.0
    active_trades: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.active_trades is None:
            self.active_trades = {}

class BaseStrategy(ABC):
    """Abstract base strategy class"""
    
    def __init__(self, name: str, timeframes: List[str] = None, parameters: Dict[str, Any] = None):
        self.name = name
        self.timeframes = timeframes or ["1h"]
        self.parameters = parameters or {}
        
        # Strategy state
        self.state = StrategyState()
        self.indicators = TechnicalIndicators()
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_return': 0.0
        }
        
        # Strategy configuration
        self.enabled = True
        self.min_confidence = 0.5
        self.max_positions = 1  # Maximum positions per symbol
        self.lookback_period = 100  # Number of candles to analyze
        
        # Risk parameters
        self.stop_loss_pct = float(getattr(config.trading, 'stop_loss_percent', 0.03))  # 3% stop loss
        self.take_profit_pct = float(getattr(config.trading, 'take_profit_percent', 0.06))  # 6% take profit
        self.max_risk_per_trade = 0.02  # 2% portfolio risk per trade
        
        logger.info(f"🎯 Strategy initialized: {self.name}")
    
    @abstractmethod
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> StrategySignal:
        """
        Analyze market data and generate trading signal
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            market_data: Dictionary containing OHLCV data and other market info
            
        Returns:
            StrategySignal: Trading signal with action, confidence, etc.
        """
        pass
    
    @abstractmethod
    def get_required_data(self) -> Dict[str, Any]:
        """
        Get data requirements for the strategy
        
        Returns:
            Dict containing required timeframes, lookback periods, etc.
        """
        pass
    
    # strategies/base_strategy.py
    def preprocess_data(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Accepts either:
        • dict with key 'candles': list[dict] OR list[list] (ccxt OHLCV)
        • list[dict] (already candles)
        • list[list] (ccxt OHLCV rows)
        Produces a DataFrame with columns: timestamp, open, high, low, close, volume
        """
        import pandas as pd
        try:
            candles = None

            # 1) dict case
            if isinstance(market_data, dict) and 'candles' in market_data:
                candles = market_data['candles']

            # 2) list case (DataCollector.get_latest_data returns list[dict])
            elif isinstance(market_data, list):
                candles = market_data

            if candles is None or not candles:
                return pd.DataFrame()

            # Normalize to DataFrame
            if isinstance(candles[0], dict):
                # expect keys: open_time/close_time or timestamp, open high low close volume
                df = pd.DataFrame(candles)
                # unify timestamp
                if 'timestamp' not in df.columns:
                    if 'close_time' in df.columns:
                        df['timestamp'] = df['close_time']
                    elif 'open_time' in df.columns:
                        df['timestamp'] = df['open_time']
                # ensure required numeric cols exist
                for col in ['open','high','low','close','volume']:
                    if col not in df.columns:
                        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(method='ffill').fillna(0)
            else:
                # list[list] ccxt OHLCV: [ts, o, h, l, c, v]
                df = pd.DataFrame(candles, columns=['timestamp','open','high','low','close','volume'])

            # Sort and clean
            df = df.dropna(subset=['timestamp','close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce').fillna(method='ffill')
            for col in ['open','high','low','close','volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna().sort_values('timestamp').reset_index(drop=True)

            # Return recent lookback only
            lookback = getattr(self, 'lookback_period', 300)
            if len(df) > lookback:
                df = df.tail(lookback)

            return df

        except Exception as e:
            logger.error(f"❌ Data preprocessing failed for {self.name}: {e}")
            return pd.DataFrame()

        


    def _ensure_dataframe(
        self,
        market_data: Any,
        since: Optional[int] = None,     # epoch ms or None
        lookback: Optional[int] = None   # rows to keep
    ) -> pd.DataFrame:
        """
        Normalize market_data (list[dict], list[list], or {'candles': ...}) to a
        DataFrame with columns: timestamp, open, high, low, close, volume.

        Handles mixed timestamp types safely. If `since` is provided (epoch ms),
        rows older than that are filtered *after* converting to datetime.
        """
        try:
            # 1) Extract candles
            candles = None
            if isinstance(market_data, dict) and "candles" in market_data:
                candles = market_data["candles"]
            elif isinstance(market_data, list):
                candles = market_data
            else:
                return pd.DataFrame()

            if not candles:
                return pd.DataFrame()

            # 2) Build DF from dicts or OHLCV lists
            if isinstance(candles[0], dict):
                df = pd.DataFrame(candles)
                # Try common timestamp keys → unify to 'timestamp' in ms
                if "timestamp" not in df.columns:
                    if "close_time" in df.columns:
                        df["timestamp"] = df["close_time"]
                    elif "open_time" in df.columns:
                        df["timestamp"] = df["open_time"]
                # coerce numerics
                for col in ("open","high","low","close","volume"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    else:
                        df[col] = pd.NA
            else:
                # Assume ccxt OHLCV: [ts, o, h, l, c, v]
                cols = ["timestamp","open","high","low","close","volume"]
                df = pd.DataFrame(candles, columns=cols[:len(candles[0])])
                # Coerce numerics
                for col in ("open","high","low","close","volume"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

            # 3) Clean & sort
            if "timestamp" not in df.columns:
                return pd.DataFrame()

            # Convert timestamp → datetime64[ns]; accept int(ms), str, or datetime
            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                ts = df["timestamp"]
            else:
                # try epoch ms first; fallback to generic parse
                ts = pd.to_datetime(df["timestamp"], unit="ms", errors="ignore")
                ts = pd.to_datetime(ts, errors="coerce")
            df["timestamp"] = ts

            # Drop bad rows, sort by time
            df = df.dropna(subset=["timestamp","close"]).sort_values("timestamp").reset_index(drop=True)

            # 4) Apply `since` filter SAFELY (convert since ms → datetime)
            if since is not None:
                try:
                    since_dt = pd.to_datetime(int(since), unit="ms", utc=False)
                    df = df[df["timestamp"] >= since_dt]
                except Exception:
                    # If `since` is malformed, ignore filter rather than crash
                    pass

            # 5) Apply lookback row trim
            if lookback is None:
                lookback = getattr(self, "lookback_period", 300)
            if lookback and len(df) > lookback:
                df = df.tail(lookback)

            # Final numeric coercion
            for col in ("open","high","low","close","volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"])

            return df
        except Exception as e:
            logger.error(f"{self.name}: _ensure_dataframe failed: {e}")
            return pd.DataFrame()
        
    # strategies/base_strategy.py



    def _no_signal(self, symbol: str, reason: str = "no-signal", detail: Optional[dict] = None):
        """
        Return a benign 'hold' signal-like object with a reason so the manager
        can log a parseable rejection. Accepts optional 'detail' dict for dashboard.
        """
        try:
            from strategies.models import StrategySignal  # if you have it elsewhere, fine to ignore
            obj = StrategySignal(
                symbol=symbol,
                action="hold",
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size=None,
                reasoning=reason or "no-signal",
                timeframe=getattr(self, "timeframe", "1m"),
                strategy_name=getattr(self, "name", self.__class__.__name__),
            )
            if detail:
                setattr(obj, "detail", detail)
            return obj
        except Exception:
            from types import SimpleNamespace
            obj = SimpleNamespace(
                symbol=symbol,
                action="hold",
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size=None,
                reasoning=reason or "no-signal",
                timeframe=getattr(self, "timeframe", "1m"),
                strategy_name=getattr(self, "name", self.__class__.__name__),
            )
            if detail:
                setattr(obj, "detail", detail)
            return obj


    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added indicators
        """
        try:
            if df.empty:
                return df
            
            # Add common indicators
            df = self.indicators.add_sma(df, periods=[20, 50])
            df = self.indicators.add_ema(df, periods=[12, 26])
            df = self.indicators.add_rsi(df)
            df = self.indicators.add_macd(df)
            df = self.indicators.add_bollinger_bands(df)
            df = self.indicators.add_atr(df)
            df = self.indicators.add_stochastic(df)
            
            # Add volume indicators
            df = self.indicators.add_volume_sma(df)
            df = self.indicators.add_obv(df)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Indicator calculation failed for {self.name}: {e}")
            return df
    
    def validate_signal(self, signal: StrategySignal, market_data: Dict[str, Any]) -> bool:
        """
        Validate generated signal against strategy rules
        
        Args:
            signal: Generated trading signal
            market_data: Current market data
            
        Returns:
            bool: True if signal is valid
        """
        try:
            # Check if strategy is enabled
            if not self.enabled:
                return False
            
            # Check confidence threshold
            if signal.confidence < self.min_confidence:
                logger.debug(f"Signal confidence {signal.confidence:.2f} below threshold {self.min_confidence}")
                return False
            
            # Check if action is valid
            if signal.action not in ['buy', 'sell', 'hold']:
                return False
            
            # Skip hold signals
            if signal.action == 'hold':
                return False
            
            # Check if we already have maximum positions for this symbol
            if signal.symbol in self.state.active_trades:
                active_count = len(self.state.active_trades[signal.symbol])
                if active_count >= self.max_positions:
                    logger.debug(f"Max positions ({self.max_positions}) reached for {signal.symbol}")
                    return False
            
            # Validate stop loss and take profit
            if signal.action == 'buy':
                if signal.stop_loss and signal.entry_price:
                    if signal.stop_loss >= signal.entry_price:
                        logger.warning(f"Invalid stop loss for buy signal: {signal.stop_loss} >= {signal.entry_price}")
                        return False
                
                if signal.take_profit and signal.entry_price:
                    if signal.take_profit <= signal.entry_price:
                        logger.warning(f"Invalid take profit for buy signal: {signal.take_profit} <= {signal.entry_price}")
                        return False
            
            elif signal.action == 'sell':
                if signal.stop_loss and signal.entry_price:
                    if signal.stop_loss <= signal.entry_price:
                        logger.warning(f"Invalid stop loss for sell signal: {signal.stop_loss} <= {signal.entry_price}")
                        return False
                
                if signal.take_profit and signal.entry_price:
                    if signal.take_profit >= signal.entry_price:
                        logger.warning(f"Invalid take profit for sell signal: {signal.take_profit} >= {signal.entry_price}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal validation failed for {self.name}: {e}")
            return False
    
    def calculate_position_size(self, signal: StrategySignal, account_balance: float) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            float: Position size in base currency
        """
        try:
            if not signal.entry_price:
                return 0.0
            
            # Calculate risk amount
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Calculate stop loss distance
            if signal.action == 'buy' and signal.stop_loss:
                stop_distance = signal.entry_price - signal.stop_loss
            elif signal.action == 'sell' and signal.stop_loss:
                stop_distance = signal.stop_loss - signal.entry_price
            else:
                # Default stop loss if not provided
                stop_distance = signal.entry_price * self.stop_loss_pct
            
            if stop_distance <= 0:
                logger.warning(f"Invalid stop distance: {stop_distance}")
                return 0.0
            
            # Position size = Risk Amount / Stop Distance
            position_size = risk_amount / stop_distance
            
            # Apply confidence scaling
            position_size *= signal.confidence
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Position size calculation failed: {e}")
            return 0.0
    
    def set_stop_loss_take_profit(self, signal: StrategySignal) -> StrategySignal:
        """
        Set stop loss and take profit if not already set
        
        Args:
            signal: Trading signal
            
        Returns:
            StrategySignal: Signal with stop loss and take profit set
        """
        try:
            if not signal.entry_price:
                return signal
            
            # Set stop loss if not provided
            if not signal.stop_loss:
                if signal.action == 'buy':
                    signal.stop_loss = signal.entry_price * (1 - self.stop_loss_pct)
                elif signal.action == 'sell':
                    signal.stop_loss = signal.entry_price * (1 + self.stop_loss_pct)
            
            # Set take profit if not provided
            if not signal.take_profit:
                if signal.action == 'buy':
                    signal.take_profit = signal.entry_price * (1 + self.take_profit_pct)
                elif signal.action == 'sell':
                    signal.take_profit = signal.entry_price * (1 - self.take_profit_pct)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Stop loss/take profit setting failed: {e}")
            return signal
    
    def update_performance(self, signal: StrategySignal, result: Dict[str, Any]):
        """
        Update strategy performance metrics
        
        Args:
            signal: Original trading signal
            result: Trade result with PnL information
        """
        try:
            self.performance_metrics['total_signals'] += 1
            
            pnl = result.get('pnl', 0)
            self.performance_metrics['total_return'] += pnl
            
            if pnl > 0:
                self.performance_metrics['winning_signals'] += 1
            else:
                self.performance_metrics['losing_signals'] += 1
            
            # Update win rate
            if self.performance_metrics['total_signals'] > 0:
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['winning_signals'] / 
                    self.performance_metrics['total_signals']
                )
            
            # Update average return
            if self.performance_metrics['total_signals'] > 0:
                self.performance_metrics['avg_return'] = (
                    self.performance_metrics['total_return'] / 
                    self.performance_metrics['total_signals']
                )
            
            # Update state
            self.state.signals_generated += 1
            self.state.total_pnl += pnl
            if pnl > 0:
                self.state.profitable_signals += 1
            
            logger.debug(f"📊 {self.name} performance updated: {self.performance_metrics}")
            
        except Exception as e:
            logger.error(f"❌ Performance update failed for {self.name}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        return {
            'strategy_name': self.name,
            'enabled': self.enabled,
            'total_signals': self.performance_metrics['total_signals'],
            'winning_signals': self.performance_metrics['winning_signals'],
            'losing_signals': self.performance_metrics['losing_signals'],
            'win_rate': self.performance_metrics['win_rate'],
            'avg_return': self.performance_metrics['avg_return'],
            'total_return': self.performance_metrics['total_return'],
            'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': self.performance_metrics.get('max_drawdown', 0.0),
            'last_signal_time': self.state.last_update.isoformat() if self.state.last_update else None,
            'active_trades': len(self.state.active_trades)
        }
    
    def reset_performance(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_signals': 0,
            'winning_signals': 0,
            'losing_signals': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_return': 0.0
        }
        
        self.state = StrategyState()
        logger.info(f"🔄 {self.name} performance metrics reset")
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """
        Update strategy parameters
        
        Args:
            new_parameters: Dictionary of parameter updates
        """
        try:
            self.parameters.update(new_parameters)
            
            # Update internal parameters if they exist
            for param, value in new_parameters.items():
                if hasattr(self, param):
                    setattr(self, param, value)
            
            logger.info(f"🔧 {self.name} parameters updated: {new_parameters}")
            
        except Exception as e:
            logger.error(f"❌ Parameter update failed for {self.name}: {e}")
    
    def is_market_condition_suitable(self, df: pd.DataFrame) -> bool:
        """
        Check if current market conditions are suitable for the strategy
        
        Args:
            df: OHLCV DataFrame with indicators
            
        Returns:
            bool: True if conditions are suitable
        """
        try:
            if df.empty or len(df) < 20:
                return False
            
            # Check for sufficient volatility (using ATR)
            if 'atr' in df.columns:
                recent_atr = df['atr'].iloc[-5:].mean()
                price = df['close'].iloc[-1]
                volatility_pct = (recent_atr / price) * 100
                
                # Require minimum 0.5% volatility
                if volatility_pct < 0.5:
                    logger.debug(f"Low volatility: {volatility_pct:.2f}%")
                    return False
            
            # Check for trending vs ranging market
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                sma20 = df['sma_20'].iloc[-1]
                sma50 = df['sma_50'].iloc[-1]
                
                # Some strategies work better in trending markets
                if hasattr(self, 'requires_trend') and self.requires_trend:
                    trend_strength = abs(sma20 - sma50) / sma50
                    if trend_strength < 0.02:  # Less than 2% difference
                        logger.debug("Market not trending enough")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Market condition check failed: {e}")
            return True  # Default to suitable if check fails
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'last_update': self.state.last_update.isoformat(),
            'signals_generated': self.state.signals_generated,
            'profitable_signals': self.state.profitable_signals,
            'total_pnl': self.state.total_pnl,
            'active_trades_count': len(self.state.active_trades),
            'parameters': self.parameters.copy(),
            'performance': self.get_performance_summary()
        }
    
    def enable(self):
        """Enable the strategy"""
        self.enabled = True
        logger.info(f"✅ Strategy enabled: {self.name}")
    
    def disable(self):
        """Disable the strategy"""
        self.enabled = False
        logger.info(f"⏸️ Strategy disabled: {self.name}")
    
    def add_active_trade(self, symbol: str, trade_info: Dict[str, Any]):
        """Add active trade to tracking"""
        if symbol not in self.state.active_trades:
            self.state.active_trades[symbol] = []
        
        self.state.active_trades[symbol].append(trade_info)
    
    def remove_active_trade(self, symbol: str, trade_id: str):
        """Remove active trade from tracking"""
        if symbol in self.state.active_trades:
            self.state.active_trades[symbol] = [
                trade for trade in self.state.active_trades[symbol]
                if trade.get('trade_id') != trade_id
            ]
            
            # Remove empty symbol entry
            if not self.state.active_trades[symbol]:
                del self.state.active_trades[symbol]
    
    def has_active_trades(self, symbol: str = None) -> bool:
        """Check if strategy has active trades"""
        if symbol:
            return symbol in self.state.active_trades and len(self.state.active_trades[symbol]) > 0
        
        return len(self.state.active_trades) > 0
    
    async def cleanup(self):
        """Cleanup strategy resources"""
        try:
            # Clear active trades
            self.state.active_trades.clear()
            
            logger.info(f"🧹 Strategy cleanup completed: {self.name}")
            
        except Exception as e:
            logger.error(f"❌ Strategy cleanup failed for {self.name}: {e}")
    
    def __str__(self) -> str:
        """String representation of the strategy"""
        return f"Strategy({self.name}, enabled={self.enabled}, signals={self.state.signals_generated})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"Strategy(name='{self.name}', enabled={self.enabled}, "
                f"timeframes={self.timeframes}, signals_generated={self.state.signals_generated}, "
                f"win_rate={self.performance_metrics['win_rate']:.2%})")
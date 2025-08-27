"""
Technical Indicators - Comprehensive collection of trading indicators
Provides optimized calculations for various technical analysis indicators.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple
import talib
from scipy.signal import argrelextrema

class TechnicalIndicators:
    """Collection of technical indicators for trading strategies"""
    
    def __init__(self):
        self.indicators_cache = {}
    
    # Moving Averages
    def add_sma(self, df: pd.DataFrame, periods: Union[int, List[int]] = 20) -> pd.DataFrame:
        """Simple Moving Average"""
        try:
            periods_list = periods if isinstance(periods, list) else [periods]
            
            for period in periods_list:
                if period <= len(df):
                    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            return df
        except Exception as e:
            print(f"SMA calculation error: {e}")
            return df
    
    def add_ema(self, df: pd.DataFrame, periods: Union[int, List[int]] = 20) -> pd.DataFrame:
        """Exponential Moving Average"""
        try:
            periods_list = periods if isinstance(periods, list) else [periods]
            
            for period in periods_list:
                if period <= len(df):
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            return df
        except Exception as e:
            print(f"EMA calculation error: {e}")
            return df
    
    def add_wma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Weighted Moving Average"""
        try:
            if period <= len(df):
                weights = np.arange(1, period + 1)
                df[f'wma_{period}'] = df['close'].rolling(period).apply(
                    lambda prices: np.dot(prices, weights) / weights.sum(), raw=True
                )
            return df
        except Exception as e:
            print(f"WMA calculation error: {e}")
            return df
    
    def add_hull_ma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Hull Moving Average"""
        try:
            if period <= len(df):
                half_period = period // 2
                sqrt_period = int(np.sqrt(period))
                
                wma_half = df['close'].rolling(half_period).apply(
                    lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
                )
                wma_full = df['close'].rolling(period).apply(
                    lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
                )
                
                hull_values = 2 * wma_half - wma_full
                df[f'hull_{period}'] = hull_values.rolling(sqrt_period).apply(
                    lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
                )
            
            return df
        except Exception as e:
            print(f"Hull MA calculation error: {e}")
            return df
    
    # Oscillators
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Relative Strength Index"""
        try:
            if period <= len(df):
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
        except Exception as e:
            print(f"RSI calculation error: {e}")
            return df
    
    def add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator"""
        try:
            if k_period <= len(df):
                lowest_low = df['low'].rolling(window=k_period).min()
                highest_high = df['high'].rolling(window=k_period).max()
                
                df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
                df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
            
            return df
        except Exception as e:
            print(f"Stochastic calculation error: {e}")
            return df
    
    def add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Williams %R"""
        try:
            if period <= len(df):
                highest_high = df['high'].rolling(window=period).max()
                lowest_low = df['low'].rolling(window=period).min()
                
                df['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            return df
        except Exception as e:
            print(f"Williams %R calculation error: {e}")
            return df
    
    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Commodity Channel Index"""
        try:
            if period <= len(df):
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                sma_tp = typical_price.rolling(window=period).mean()
                mad = typical_price.rolling(window=period).apply(
                    lambda x: np.mean(np.abs(x - np.mean(x)))
                )
                
                df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
            
            return df
        except Exception as e:
            print(f"CCI calculation error: {e}")
            return df
    
    # MACD
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)"""
        try:
            if slow <= len(df):
                ema_fast = df['close'].ewm(span=fast).mean()
                ema_slow = df['close'].ewm(span=slow).mean()
                
                df['macd'] = ema_fast - ema_slow
                df['macd_signal'] = df['macd'].ewm(span=signal).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            return df
        except Exception as e:
            print(f"MACD calculation error: {e}")
            return df
    
    # Bollinger Bands
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Bollinger Bands"""
        try:
            if period <= len(df):
                sma = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                
                df['bb_upper'] = sma + (std * std_dev)
                df['bb_middle'] = sma
                df['bb_lower'] = sma - (std * std_dev)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
        except Exception as e:
            print(f"Bollinger Bands calculation error: {e}")
            return df
    
    # Keltner Channels
    def add_keltner_channels(self, df: pd.DataFrame, period: int = 20, atr_mult: float = 2) -> pd.DataFrame:
        """Keltner Channels"""
        try:
            if period <= len(df):
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                kc_middle = typical_price.rolling(window=period).mean()
                
                # Add ATR if not present
                if 'atr' not in df.columns:
                    df = self.add_atr(df, period)
                
                df['kc_upper'] = kc_middle + (df['atr'] * atr_mult)
                df['kc_middle'] = kc_middle
                df['kc_lower'] = kc_middle - (df['atr'] * atr_mult)
            
            return df
        except Exception as e:
            print(f"Keltner Channels calculation error: {e}")
            return df
    
    # Volatility Indicators
    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range"""
        try:
            if period <= len(df) and len(df) > 1:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                df['atr'] = pd.Series(true_range).rolling(window=period).mean()
            
            return df
        except Exception as e:
            print(f"ATR calculation error: {e}")
            return df
    
    def add_historical_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Historical Volatility"""
        try:
            if period <= len(df):
                returns = np.log(df['close'] / df['close'].shift())
                df['hist_vol'] = returns.rolling(window=period).std() * np.sqrt(252) * 100
            
            return df
        except Exception as e:
            print(f"Historical Volatility calculation error: {e}")
            return df
    
    # Volume Indicators
    def add_volume_sma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Volume Simple Moving Average"""
        try:
            if period <= len(df):
                df['volume_sma'] = df['volume'].rolling(window=period).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
        except Exception as e:
            print(f"Volume SMA calculation error: {e}")
            return df
    
    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """On Balance Volume"""
        try:
            if len(df) > 1:
                price_change = df['close'].diff()
                volume_direction = np.where(price_change > 0, df['volume'], 
                                          np.where(price_change < 0, -df['volume'], 0))
                df['obv'] = volume_direction.cumsum()
            
            return df
        except Exception as e:
            print(f"OBV calculation error: {e}")
            return df
    
    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume Weighted Average Price"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap_numerator = (typical_price * df['volume']).cumsum()
            vwap_denominator = df['volume'].cumsum()
            
            df['vwap'] = vwap_numerator / vwap_denominator
            
            return df
        except Exception as e:
            print(f"VWAP calculation error: {e}")
            return df
    
    def add_money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Money Flow Index"""
        try:
            if period <= len(df) and len(df) > 1:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                money_flow = typical_price * df['volume']
                
                positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
                negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
                
                positive_mf = positive_flow.rolling(window=period).sum()
                negative_mf = negative_flow.rolling(window=period).sum()
                
                money_ratio = positive_mf / negative_mf
                df['mfi'] = 100 - (100 / (1 + money_ratio))
            
            return df
        except Exception as e:
            print(f"MFI calculation error: {e}")
            return df
    
    # Trend Indicators
    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index"""
        try:
            if period <= len(df) and len(df) > 1:
                # Calculate True Range
                if 'atr' not in df.columns:
                    df = self.add_atr(df, period)
                
                # Calculate Directional Movement
                high_diff = df['high'].diff()
                low_diff = df['low'].diff()
                
                plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
                minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
                
                # Smooth the values
                plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
                minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
                
                # Calculate DI+ and DI-
                df['di_plus'] = 100 * (plus_dm_smooth / df['atr'])
                df['di_minus'] = 100 * (minus_dm_smooth / df['atr'])
                
                # Calculate DX and ADX
                dx = 100 * np.abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
                df['adx'] = dx.rolling(window=period).mean()
            
            return df
        except Exception as e:
            print(f"ADX calculation error: {e}")
            return df
    
    def add_parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """Parabolic SAR"""
        try:
            if len(df) > 1:
                sar = []
                af = af_start
                ep = df['high'].iloc[0]  # Extreme Point
                trend = 1  # 1 for uptrend, -1 for downtrend
                
                sar.append(df['low'].iloc[0])
                
                for i in range(1, len(df)):
                    if trend == 1:  # Uptrend
                        sar_value = sar[i-1] + af * (ep - sar[i-1])
                        
                        if df['low'].iloc[i] <= sar_value:
                            trend = -1
                            sar_value = ep
                            ep = df['low'].iloc[i]
                            af = af_start
                        else:
                            if df['high'].iloc[i] > ep:
                                ep = df['high'].iloc[i]
                                af = min(af + af_start, af_max)
                    
                    else:  # Downtrend
                        sar_value = sar[i-1] - af * (sar[i-1] - ep)
                        
                        if df['high'].iloc[i] >= sar_value:
                            trend = 1
                            sar_value = ep
                            ep = df['high'].iloc[i]
                            af = af_start
                        else:
                            if df['low'].iloc[i] < ep:
                                ep = df['low'].iloc[i]
                                af = min(af + af_start, af_max)
                    
                    sar.append(sar_value)
                
                df['sar'] = sar
            
            return df
        except Exception as e:
            print(f"Parabolic SAR calculation error: {e}")
            return df
    
    # Support and Resistance
    def add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot Points (Traditional)"""
        try:
            if len(df) > 1:
                # Use previous day's data for pivot calculation
                prev_high = df['high'].shift()
                prev_low = df['low'].shift()
                prev_close = df['close'].shift()
                
                pivot = (prev_high + prev_low + prev_close) / 3
                
                df['pivot'] = pivot
                df['r1'] = 2 * pivot - prev_low
                df['s1'] = 2 * pivot - prev_high
                df['r2'] = pivot + (prev_high - prev_low)
                df['s2'] = pivot - (prev_high - prev_low)
                df['r3'] = prev_high + 2 * (pivot - prev_low)
                df['s3'] = prev_low - 2 * (prev_high - pivot)
            
            return df
        except Exception as e:
            print(f"Pivot Points calculation error: {e}")
            return df
    
    def add_support_resistance(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Dynamic Support and Resistance Levels"""
        try:
            if len(df) >= window * 2:
                # Find local maxima and minima
                highs = df['high'].values
                lows = df['low'].values
                
                # Find resistance levels (local maxima)
                resistance_indices = argrelextrema(highs, np.greater, order=window)[0]
                support_indices = argrelextrema(lows, np.less, order=window)[0]
                
                # Create arrays for support and resistance
                resistance = np.full(len(df), np.nan)
                support = np.full(len(df), np.nan)
                
                # Fill resistance levels
                for idx in resistance_indices:
                    resistance[idx] = highs[idx]
                
                # Fill support levels
                for idx in support_indices:
                    support[idx] = lows[idx]
                
                # Forward fill the levels
                df['resistance'] = pd.Series(resistance).fillna(method='ffill')
                df['support'] = pd.Series(support).fillna(method='ffill')
            
            return df
        except Exception as e:
            print(f"Support/Resistance calculation error: {e}")
            return df
    
    # Custom Indicators
    def add_squeeze_momentum(self, df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20) -> pd.DataFrame:
        """Squeeze Momentum Indicator"""
        try:
            if bb_period <= len(df) and kc_period <= len(df):
                # Add Bollinger Bands and Keltner Channels if not present
                if 'bb_upper' not in df.columns:
                    df = self.add_bollinger_bands(df, bb_period)
                if 'kc_upper' not in df.columns:
                    df = self.add_keltner_channels(df, kc_period)
                
                # Squeeze occurs when BB are inside KC
                df['squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
                
                # Momentum calculation
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                sma_tp = typical_price.rolling(window=bb_period).mean()
                df['squeeze_momentum'] = typical_price - sma_tp
            
            return df
        except Exception as e:
            print(f"Squeeze Momentum calculation error: {e}")
            return df
    
    def add_awesome_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Awesome Oscillator"""
        try:
            if len(df) >= 34:
                median_price = (df['high'] + df['low']) / 2
                ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
                df['ao'] = ao
            
            return df
        except Exception as e:
            print(f"Awesome Oscillator calculation error: {e}")
            return df
    
    def add_elder_ray(self, df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """Elder Ray Index"""
        try:
            if period <= len(df):
                ema = df['close'].ewm(span=period).mean()
                df['bull_power'] = df['high'] - ema
                df['bear_power'] = df['low'] - ema
            
            return df
        except Exception as e:
            print(f"Elder Ray calculation error: {e}")
            return df
    
    # Price Patterns
    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic Candlestick Patterns"""
        try:
            if len(df) > 1:
                # Doji
                body_size = abs(df['close'] - df['open'])
                range_size = df['high'] - df['low']
                df['doji'] = (body_size / range_size) < 0.1
                
                # Hammer
                lower_shadow = df['open'].combine(df['close'], min) - df['low']
                upper_shadow = df['high'] - df['open'].combine(df['close'], max)
                df['hammer'] = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
                
                # Shooting Star
                df['shooting_star'] = (upper_shadow > 2 * body_size) & (lower_shadow < body_size)
                
                # Engulfing patterns
                prev_body = abs(df['close'].shift() - df['open'].shift())
                current_body = abs(df['close'] - df['open'])
                
                bullish_engulfing = (df['close'] > df['open']) & \
                                  (df['close'].shift() < df['open'].shift()) & \
                                  (df['close'] > df['open'].shift()) & \
                                  (df['open'] < df['close'].shift()) & \
                                  (current_body > prev_body)
                
                bearish_engulfing = (df['close'] < df['open']) & \
                                  (df['close'].shift() > df['open'].shift()) & \
                                  (df['close'] < df['open'].shift()) & \
                                  (df['open'] > df['close'].shift()) & \
                                  (current_body > prev_body)
                
                df['bullish_engulfing'] = bullish_engulfing
                df['bearish_engulfing'] = bearish_engulfing
            
            return df
        except Exception as e:
            print(f"Candlestick patterns calculation error: {e}")
            return df
    
    # Market Structure
    def add_market_structure(self, df: pd.DataFrame, swing_length: int = 5) -> pd.DataFrame:
        """Market Structure Analysis"""
        try:
            if len(df) >= swing_length * 2:
                # Higher Highs and Lower Lows
                df['hh'] = df['high'] > df['high'].rolling(swing_length*2+1, center=True).max().shift()
                df['ll'] = df['low'] < df['low'].rolling(swing_length*2+1, center=True).min().shift()
                
                # Higher Lows and Lower Highs
                df['hl'] = df['low'] > df['low'].rolling(swing_length*2+1, center=True).min().shift()
                df['lh'] = df['high'] < df['high'].rolling(swing_length*2+1, center=True).max().shift()
                
                # Trend direction
                uptrend = df['hh'].rolling(swing_length).sum() > 0
                downtrend = df['ll'].rolling(swing_length).sum() > 0
                
                df['trend'] = np.where(uptrend, 1, np.where(downtrend, -1, 0))
            
            return df
        except Exception as e:
            print(f"Market structure calculation error: {e}")
            return df
    
    # Composite Indicators
    def add_composite_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite Momentum Score"""
        try:
            # Add required indicators
            df = self.add_rsi(df)
            df = self.add_macd(df)
            df = self.add_stochastic(df)
            
            if all(col in df.columns for col in ['rsi', 'macd', 'stoch_k']):
                # Normalize indicators to -1 to 1 scale
                rsi_norm = (df['rsi'] - 50) / 50
                macd_norm = np.tanh(df['macd'] / df['close'] * 1000)  # Scale MACD
                stoch_norm = (df['stoch_k'] - 50) / 50
                
                # Composite momentum (equally weighted)
                df['momentum_composite'] = (rsi_norm + macd_norm + stoch_norm) / 3
            
            return df
        except Exception as e:
            print(f"Composite momentum calculation error: {e}")
            return df
    
    def add_volatility_bands(self, df: pd.DataFrame, period: int = 20, multiplier: float = 1.5) -> pd.DataFrame:
        """Custom Volatility Bands"""
        try:
            if period <= len(df):
                # Calculate price volatility
                returns = df['close'].pct_change()
                volatility = returns.rolling(period).std()
                
                # Calculate bands
                sma = df['close'].rolling(period).mean()
                vol_adjustment = volatility * multiplier * df['close']
                
                df['vol_upper'] = sma + vol_adjustment
                df['vol_lower'] = sma - vol_adjustment
                df['vol_middle'] = sma
                df['vol_position'] = (df['close'] - df['vol_lower']) / (df['vol_upper'] - df['vol_lower'])
            
            return df
        except Exception as e:
            print(f"Volatility bands calculation error: {e}")
            return df
    
    # Utility Methods
    def normalize_indicator(self, series: pd.Series, method: str = 'minmax') -> pd.Series:
        """Normalize indicator values"""
        try:
            if method == 'minmax':
                return (series - series.min()) / (series.max() - series.min())
            elif method == 'zscore':
                return (series - series.mean()) / series.std()
            elif method == 'robust':
                median = series.median()
                mad = (series - median).abs().median()
                return (series - median) / mad
            else:
                return series
        except Exception as e:
            print(f"Normalization error: {e}")
            return series
    
    def get_signal_strength(self, df: pd.DataFrame, indicators: List[str]) -> pd.Series:
        """Calculate signal strength from multiple indicators"""
        try:
            signals = []
            
            for indicator in indicators:
                if indicator in df.columns:
                    # Convert indicator to signal (-1 to 1)
                    if indicator == 'rsi':
                        signal = np.where(df[indicator] > 70, -1, np.where(df[indicator] < 30, 1, 0))
                    elif indicator.startswith('macd'):
                        signal = np.where(df[indicator] > 0, 1, -1)
                    elif indicator == 'bb_percent':
                        signal = np.where(df[indicator] > 0.8, -1, np.where(df[indicator] < 0.2, 1, 0))
                    else:
                        # Generic normalization
                        normalized = self.normalize_indicator(df[indicator], 'zscore')
                        signal = np.tanh(normalized)  # Bound between -1 and 1
                    
                    signals.append(signal)
            
            if signals:
                # Average all signals
                composite_signal = np.mean(signals, axis=0)
                return pd.Series(composite_signal, index=df.index)
            
            return pd.Series(0, index=df.index)
            
        except Exception as e:
            print(f"Signal strength calculation error: {e}")
            return pd.Series(0, index=df.index)
    
    def detect_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 5) -> pd.Series:
        """Detect bullish/bearish divergence"""
        try:
            # Find peaks and troughs
            price_peaks = argrelextrema(price.values, np.greater, order=window)[0]
            price_troughs = argrelextrema(price.values, np.less, order=window)[0]
            
            indicator_peaks = argrelextrema(indicator.values, np.greater, order=window)[0]
            indicator_troughs = argrelextrema(indicator.values, np.less, order=window)[0]
            
            divergence = np.zeros(len(price))
            
            # Check for bullish divergence (price lower low, indicator higher low)
            for i in range(1, len(price_troughs)):
                if i < len(indicator_troughs):
                    price_idx1, price_idx2 = price_troughs[i-1], price_troughs[i]
                    ind_idx1, ind_idx2 = indicator_troughs[i-1], indicator_troughs[i]
                    
                    if (price.iloc[price_idx2] < price.iloc[price_idx1] and 
                        indicator.iloc[ind_idx2] > indicator.iloc[ind_idx1]):
                        divergence[price_troughs[i]] = 1  # Bullish divergence
            
            # Check for bearish divergence (price higher high, indicator lower high)
            for i in range(1, len(price_peaks)):
                if i < len(indicator_peaks):
                    price_idx1, price_idx2 = price_peaks[i-1], price_peaks[i]
                    ind_idx1, ind_idx2 = indicator_peaks[i-1], indicator_peaks[i]
                    
                    if (price.iloc[price_idx2] > price.iloc[price_idx1] and 
                        indicator.iloc[ind_idx2] < indicator.iloc[ind_idx1]):
                        divergence[price_peaks[i]] = -1  # Bearish divergence
            
            return pd.Series(divergence, index=price.index)
            
        except Exception as e:
            print(f"Divergence detection error: {e}")
            return pd.Series(0, index=price.index)
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available indicators"""
        try:
            # Moving Averages
            df = self.add_sma(df, [10, 20, 50, 200])
            df = self.add_ema(df, [12, 26, 50])
            
            # Oscillators
            df = self.add_rsi(df)
            df = self.add_stochastic(df)
            df = self.add_williams_r(df)
            df = self.add_cci(df)
            
            # MACD
            df = self.add_macd(df)
            
            # Bands
            df = self.add_bollinger_bands(df)
            df = self.add_keltner_channels(df)
            
            # Volatility
            df = self.add_atr(df)
            df = self.add_historical_volatility(df)
            
            # Volume
            df = self.add_volume_sma(df)
            df = self.add_obv(df)
            df = self.add_vwap(df)
            df = self.add_money_flow_index(df)
            
            # Trend
            df = self.add_adx(df)
            df = self.add_parabolic_sar(df)
            
            # Support/Resistance
            df = self.add_pivot_points(df)
            
            # Custom
            df = self.add_squeeze_momentum(df)
            df = self.add_composite_momentum(df)
            df = self.add_candlestick_patterns(df)
            df = self.add_market_structure(df)
            
            return df
            
        except Exception as e:
            print(f"Calculate all indicators error: {e}")
            return df
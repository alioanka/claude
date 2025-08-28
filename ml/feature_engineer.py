"""
Feature Engineer - Creates ML features from market data
Handles feature extraction, transformation, and selection for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from utils.indicators import TechnicalIndicators
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """Main feature engineering class"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.feature_names = []
        self.scaler_params = {}
        
    def engineer_features(self, df: pd.DataFrame, symbol: str = '') -> pd.DataFrame:
        """Main feature engineering pipeline"""
        try:
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient data for feature engineering: {len(df)} rows")
                return pd.DataFrame()
            
            # Make a copy to avoid modifying original data
            features_df = df.copy()
            
            # Add basic price features
            features_df = self._add_price_features(features_df)
            
            # Add technical indicators
            features_df = self._add_technical_indicators(features_df)
            
            # Add volume features
            features_df = self._add_volume_features(features_df)
            
            # Add time-based features
            features_df = self._add_time_features(features_df)
            
            # Add market structure features
            features_df = self._add_market_structure_features(features_df)
            
            # Add volatility features
            features_df = self._add_volatility_features(features_df)
            
            # Add momentum features
            features_df = self._add_momentum_features(features_df)
            
            # Clean and finalize
            features_df = self._clean_features(features_df)
            
            # Store feature names for later use
            self.feature_names = [col for col in features_df.columns 
                                if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            
            logger.info(f"Generated {len(self.feature_names)} features for {symbol}")
            return features_df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return pd.DataFrame()
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            # Basic price changes
            df['price_change_1'] = df['close'].pct_change(1)
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            df['price_change_20'] = df['close'].pct_change(20)
            
            # Log returns
            df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
            df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
            
            # Price position relative to high/low
            df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['close_to_open'] = (df['close'] - df['open']) / df['open']
            
            # Gap features
            df['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).clip(lower=0)
            df['gap_down'] = ((df['close'].shift(1) - df['open']) / df['close'].shift(1)).clip(lower=0)
            
            # Price ratios
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_high_ratio'] = df['close'] / df['high']
            df['close_low_ratio'] = df['close'] / df['low']
            
            return df
            
        except Exception as e:
            logger.error(f"Price features error: {e}")
            return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            # Moving averages
            df = self.indicators.add_sma(df, [5, 10, 20, 50])
            df = self.indicators.add_ema(df, [5, 12, 26, 50])
            
            # Oscillators
            df = self.indicators.add_rsi(df, 14)
            df = self.indicators.add_stochastic(df)
            df = self.indicators.add_williams_r(df)
            
            # MACD
            df = self.indicators.add_macd(df)
            
            # Bollinger Bands
            df = self.indicators.add_bollinger_bands(df)
            
            # ATR for volatility
            df = self.indicators.add_atr(df)
            
            # Add indicator relationships
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
            
            if 'ema_12' in df.columns and 'ema_26' in df.columns:
                df['ema_12_26_ratio'] = df['ema_12'] / df['ema_26']
            
            # Price vs moving averages
            if 'sma_20' in df.columns:
                df['close_sma20_ratio'] = df['close'] / df['sma_20']
                df['close_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
            
            # Bollinger Band position
            if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            return df
            
        except Exception as e:
            logger.error(f"Technical indicators error: {e}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            # Volume moving averages
            df['volume_sma_5'] = df['volume'].rolling(5).mean()
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            
            # Volume ratios
            df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
            df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
            
            # Volume change
            df['volume_change'] = df['volume'].pct_change()
            
            # Price-volume features
            df['price_volume'] = df['close'] * df['volume']
            df['pv_trend'] = df['price_volume'].rolling(5).mean()
            
            # On Balance Volume
            df = self.indicators.add_obv(df)
            if 'obv' in df.columns:
                df['obv_sma'] = df['obv'].rolling(20).mean()
                df['obv_ratio'] = df['obv'] / df['obv_sma']
            
            # Volume-weighted price
            df['vwap_5'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
            df['close_vwap_ratio'] = df['close'] / df['vwap_5']
            
            # Volume spikes
            volume_mean = df['volume'].rolling(20).mean()
            volume_std = df['volume'].rolling(20).std()
            df['volume_spike'] = ((df['volume'] - volume_mean) / volume_std).clip(upper=3)
            
            return df
            
        except Exception as e:
            logger.error(f"Volume features error: {e}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            if 'timestamp' in df.index.names or isinstance(df.index, pd.DatetimeIndex):
                dt_index = df.index
            else:
                # Try to convert timestamp column
                if 'timestamp' in df.columns:
                    dt_index = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    return df
            
            # Hour of day (0-23)
            df['hour'] = dt_index.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Day of week (0-6)
            df['dayofweek'] = dt_index.dayofweek
            df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            
            # Market session indicators
            df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Time features error: {e}")
            return df
    
    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features"""
        try:
            # Support and resistance levels
            window = 10
            df['local_max'] = df['high'].rolling(window*2+1, center=True).max() == df['high']
            df['local_min'] = df['low'].rolling(window*2+1, center=True).min() == df['low']
            
            # Trend strength
            df['trend_strength_5'] = abs(df['close'].rolling(5).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
            ))
            
            df['trend_strength_20'] = abs(df['close'].rolling(20).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
            ))
            
            # Higher highs, lower lows
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            
            # Consecutive moves
            df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int)
            df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int)
            
            # Count consecutive moves
            df['up_streak'] = df['consecutive_up'].groupby(
                (df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()
            ).cumsum()
            
            df['down_streak'] = df['consecutive_down'].groupby(
                (df['consecutive_down'] != df['consecutive_down'].shift()).cumsum()
            ).cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Market structure features error: {e}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
            # True Range and ATR
            if 'atr' not in df.columns:
                df = self.indicators.add_atr(df)
            
            # Volatility measures
            df['volatility_5'] = df['close'].rolling(5).std()
            df['volatility_20'] = df['close'].rolling(20).std()
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
            
            # Intraday volatility
            df['intraday_vol'] = (df['high'] - df['low']) / df['open']
            df['intraday_vol_sma'] = df['intraday_vol'].rolling(20).mean()
            df['intraday_vol_ratio'] = df['intraday_vol'] / df['intraday_vol_sma']
            
            # Garman-Klass volatility
            df['gk_volatility'] = np.log(df['high']/df['low'])**2 - (2*np.log(2)-1) * np.log(df['close']/df['open'])**2
            df['gk_vol_sma'] = df['gk_volatility'].rolling(20).mean()
            
            # ATR-based features
            if 'atr' in df.columns:
                df['atr_ratio'] = df['atr'] / df['close']
                df['atr_percentile'] = df['atr'].rolling(50).rank() / 50
            
            # Volatility regimes
            vol_20 = df['close'].rolling(20).std()
            vol_threshold_high = vol_20.rolling(100).quantile(0.8)
            vol_threshold_low = vol_20.rolling(100).quantile(0.2)
            
            df['vol_regime_high'] = (vol_20 > vol_threshold_high).astype(int)
            df['vol_regime_low'] = (vol_20 < vol_threshold_low).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Volatility features error: {e}")
            return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        try:
            # Rate of Change (ROC)
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
            
            # Momentum oscillator
            for period in [10, 20]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period)
            
            # Price acceleration (second derivative)
            df['price_accel'] = df['close'].diff().diff()
            
            # Relative performance vs moving averages
            if 'sma_20' in df.columns:
                df['rel_perf_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
            
            # Volume momentum
            if 'volume_sma_20' in df.columns:
                df['volume_momentum'] = (df['volume'] - df['volume_sma_20']) / df['volume_sma_20']
            
            # MACD momentum
            if 'macd_histogram' in df.columns:
                df['macd_momentum'] = df['macd_histogram'].diff()
            
            # RSI momentum
            if 'rsi' in df.columns:
                df['rsi_momentum'] = df['rsi'].diff()
                df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
                df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Momentum features error: {e}")
            return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize features"""
        try:
            # Handle infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill then backward fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # For any remaining NaN values, fill with 0
            df = df.fillna(0)
            
            # Remove features with too many zeros or constant values
            features_to_remove = []
            for col in df.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
                    # Check if feature is mostly zeros or constant
                    if (df[col] == 0).sum() / len(df) > 0.95:
                        features_to_remove.append(col)
                    elif df[col].nunique() == 1:
                        features_to_remove.append(col)
            
            if features_to_remove:
                df = df.drop(columns=features_to_remove)
                logger.info(f"Removed {len(features_to_remove)} low-quality features")
            
            return df
            
        except Exception as e:
            logger.error(f"Feature cleaning error: {e}")
            return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names"""
        return self.feature_names.copy()
    
    def select_features(self, df: pd.DataFrame, target_col: str, method: str = 'correlation', 
                       n_features: int = 50) -> List[str]:
        """Select best features using various methods"""
        try:
            if target_col not in df.columns:
                logger.error(f"Target column {target_col} not found")
                return self.feature_names
            
            feature_cols = [col for col in df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', target_col]]
            
            if method == 'correlation':
                # Select features with highest correlation to target
                correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
                selected_features = correlations.head(n_features).index.tolist()
                
            elif method == 'variance':
                # Select features with highest variance
                variances = df[feature_cols].var().sort_values(ascending=False)
                selected_features = variances.head(n_features).index.tolist()
                
            else:
                # Default: return all features
                selected_features = feature_cols[:n_features]
            
            logger.info(f"Selected {len(selected_features)} features using {method} method")
            return selected_features
            
        except Exception as e:
            logger.error(f"Feature selection error: {e}")
            return self.feature_names[:n_features]
    
    def create_target_labels(self, df: pd.DataFrame, method: str = 'price_change', 
                           periods: int = 5, threshold: float = 0.02) -> pd.Series:
        """Create target labels for supervised learning"""
        try:
            if method == 'price_change':
                # Future price change
                future_returns = df['close'].shift(-periods) / df['close'] - 1
                
                # Create labels: 0=down, 1=hold, 2=up
                labels = pd.Series(1, index=df.index)  # Default hold
                labels[future_returns < -threshold] = 0  # Down
                labels[future_returns > threshold] = 2   # Up
                
            elif method == 'trend':
                # Trend direction
                future_price = df['close'].shift(-periods)
                labels = (future_price > df['close']).astype(int)
                
            elif method == 'volatility':
                # Future volatility regime
                future_vol = df['close'].rolling(periods).std().shift(-periods)
                vol_median = future_vol.median()
                labels = (future_vol > vol_median).astype(int)
                
            else:
                # Binary up/down
                future_returns = df['close'].shift(-periods) / df['close'] - 1
                labels = (future_returns > 0).astype(int)
            
            # Remove last 'periods' rows that don't have future data
            labels = labels[:-periods]
            
            logger.info(f"Created {method} labels with {len(labels)} samples")
            return labels
            
        except Exception as e:
            logger.error(f"Target label creation error: {e}")
            return pd.Series(dtype=int)
    
    def get_feature_importance_info(self) -> Dict[str, str]:
        """Get information about feature types for interpretation"""
        feature_info = {}
        
        for feature in self.feature_names:
            if 'price_change' in feature or 'return' in feature:
                feature_info[feature] = 'price_movement'
            elif 'volume' in feature:
                feature_info[feature] = 'volume_analysis'
            elif any(indicator in feature for indicator in ['rsi', 'macd', 'sma', 'ema', 'bb']):
                feature_info[feature] = 'technical_indicator'
            elif 'volatility' in feature or 'atr' in feature:
                feature_info[feature] = 'volatility_measure'
            elif 'momentum' in feature or 'roc' in feature:
                feature_info[feature] = 'momentum_indicator'
            elif any(time_feat in feature for time_feat in ['hour', 'day', 'session']):
                feature_info[feature] = 'time_based'
            else:
                feature_info[feature] = 'other'
        
        return feature_info
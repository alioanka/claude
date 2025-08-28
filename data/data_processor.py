"""
Data processing module for trading bot.
Handles data cleaning, transformation, and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from utils.helper import DataValidator, TechnicalIndicators, FileManager

logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing class for cleaning and transforming market data."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.validator = DataValidator()
        self.indicators = TechnicalIndicators()
        
    def clean_ohlcv_data(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Clean OHLCV data by removing anomalies and filling gaps.
        
        Args:
            df: Raw OHLCV DataFrame
            symbol: Symbol name for logging
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data for {symbol or 'unknown symbol'}")
        original_len = len(df)
        
        # Validate data format
        is_valid, issues = self.validator.validate_ohlcv_data(df)
        if not is_valid:
            logger.warning(f"Data validation issues: {issues}")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Remove duplicates
        cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
        
        # Sort by index (timestamp)
        cleaned_df = cleaned_df.sort_index()
        
        # Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        # Remove outliers
        cleaned_df = self._remove_outliers(cleaned_df)
        
        # Fill gaps if needed
        cleaned_df = self._fill_gaps(cleaned_df)
        
        # Final validation
        cleaned_df = self._final_validation(cleaned_df)
        
        logger.info(f"Data cleaning completed. Rows: {original_len} -> {len(cleaned_df)}")
        return cleaned_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in OHLCV data."""
        # Forward fill missing values (use previous valid observation)
        df = df.fillna(method='ffill')
        
        # Backward fill any remaining NaN values at the beginning
        df = df.fillna(method='bfill')
        
        # If still NaN values exist, drop those rows
        before_drop = len(df)
        df = df.dropna()
        
        if len(df) < before_drop:
            logger.warning(f"Dropped {before_drop - len(df)} rows with missing values")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
        """Remove extreme outliers using z-score method."""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        before_removal = len(df)
        
        for col in numeric_columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col]))
                outlier_mask = z_scores > z_threshold
                
                if outlier_mask.sum() > 0:
                    logger.info(f"Removing {outlier_mask.sum()} outliers from {col}")
                    df = df[~outlier_mask]
        
        if len(df) < before_removal:
            logger.info(f"Removed {before_removal - len(df)} outlier rows")
        
        return df
    
    def _fill_gaps(self, df: pd.DataFrame, max_gap_hours: int = 24) -> pd.DataFrame:
        """Fill small gaps in time series data."""
        if df.empty:
            return df
        
        # Detect the frequency of the data
        time_diffs = df.index.to_series().diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0] if not time_diffs.empty else pd.Timedelta(hours=1)
        
        # Create expected index
        expected_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=most_common_diff
        )
        
        # Reindex and fill small gaps
        df_reindexed = df.reindex(expected_index)
        
        # Only fill gaps smaller than max_gap_hours
        max_gap = pd.Timedelta(hours=max_gap_hours)
        gap_sizes = df_reindexed.index.to_series().diff()
        
        # Forward fill only small gaps
        fill_mask = gap_sizes <= max_gap
        for col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].fillna(method='ffill')
            # Remove fills that were too large gaps
            large_gap_mask = (~fill_mask) & df_reindexed[col].isna().shift(-1, fill_value=False)
            df_reindexed.loc[large_gap_mask, col] = np.nan
        
        # Remove rows that are still NaN (large gaps)
        df_filled = df_reindexed.dropna()
        
        gap_count = len(df_filled) - len(df)
        if gap_count > 0:
            logger.info(f"Filled {gap_count} data gaps")
        
        return df_filled
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform final validation and corrections."""
        # Ensure OHLC relationships are correct
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Fix high/low if needed
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Ensure positive values for prices and volume
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].abs()
        
        if 'volume' in df.columns:
            df['volume'] = df['volume'].abs()
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data.
        
        Args:
            df: OHLCV DataFrame
            indicators: List of indicators to add
            
        Returns:
            DataFrame with added indicators
        """
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'bollinger', 'macd']
        
        df_with_indicators = df.copy()
        
        if 'close' not in df.columns:
            logger.error("Close price column required for technical indicators")
            return df_with_indicators
        
        close_prices = df['close']
        
        # Simple Moving Averages
        if 'sma_20' in indicators:
            df_with_indicators['sma_20'] = self.indicators.sma(close_prices, 20)
        if 'sma_50' in indicators:
            df_with_indicators['sma_50'] = self.indicators.sma(close_prices, 50)
        
        # Exponential Moving Averages
        if 'ema_12' in indicators:
            df_with_indicators['ema_12'] = self.indicators.ema(close_prices, 12)
        if 'ema_26' in indicators:
            df_with_indicators['ema_26'] = self.indicators.ema(close_prices, 26)
        
        # RSI
        if 'rsi' in indicators:
            df_with_indicators['rsi'] = self.indicators.rsi(close_prices, 14)
        
        # Bollinger Bands
        if 'bollinger' in indicators:
            bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(close_prices)
            df_with_indicators['bb_upper'] = bb_upper
            df_with_indicators['bb_middle'] = bb_middle
            df_with_indicators['bb_lower'] = bb_lower
        
        # MACD
        if 'macd' in indicators:
            macd_line, signal_line, histogram = self.indicators.macd(close_prices)
            df_with_indicators['macd'] = macd_line
            df_with_indicators['macd_signal'] = signal_line
            df_with_indicators['macd_histogram'] = histogram
        
        # Price-based indicators
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # Average True Range
            if 'atr' in indicators:
                df_with_indicators['atr'] = self._calculate_atr(df)
            
            # Stochastic Oscillator
            if 'stochastic' in indicators:
                stoch_k, stoch_d = self._calculate_stochastic(df)
                df_with_indicators['stoch_k'] = stoch_k
                df_with_indicators['stoch_d'] = stoch_d
        
        # Volume-based indicators
        if 'volume' in df.columns and 'volume_sma' in indicators:
            df_with_indicators['volume_sma'] = self.indicators.sma(df['volume'], 20)
        
        logger.info(f"Added {len([i for i in indicators if i in df_with_indicators.columns])} technical indicators")
        return df_with_indicators
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.DataFrame({
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        }).max(axis=1)
        
        return true_range.rolling(window=period).mean()
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def calculate_returns(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate various types of returns."""
        df_returns = df.copy()
        
        if price_col not in df.columns:
            logger.error(f"Price column '{price_col}' not found")
            return df_returns
        
        prices = df[price_col]
        
        # Simple returns
        df_returns['returns'] = prices.pct_change()
        
        # Log returns
        df_returns['log_returns'] = np.log(prices / prices.shift(1))
        
        # Rolling returns
        for period in [5, 10, 20]:
            df_returns[f'returns_{period}d'] = prices.pct_change(periods=period)
        
        # Rolling volatility
        df_returns['volatility_20d'] = df_returns['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df_returns
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df_time = df.copy()
        
        # Extract time components
        df_time['hour'] = df_time.index.hour
        df_time['day_of_week'] = df_time.index.dayofweek
        df_time['day_of_month'] = df_time.index.day
        df_time['month'] = df_time.index.month
        df_time['quarter'] = df_time.index.quarter
        
        # Market session indicators
        df_time['is_market_open'] = (df_time['hour'] >= 9) & (df_time['hour'] < 16) & (df_time['day_of_week'] < 5)
        df_time['is_weekend'] = df_time['day_of_week'] >= 5
        
        # Time-based cyclical features
        df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
        df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
        df_time['day_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
        df_time['day_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
        
        return df_time
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax', columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize numerical data.
        
        Args:
            df: Input DataFrame
            method: Normalization method ('minmax', 'zscore', 'robust')
            columns: Columns to normalize (None for all numerical)
            
        Returns:
            Normalized DataFrame
        """
        df_norm = df.copy()
        
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df_norm.columns:
                continue
                
            if method == 'minmax':
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val > 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
            
            elif method == 'robust':
                median_val = df_norm[col].median()
                mad = np.median(np.abs(df_norm[col] - median_val))
                if mad > 0:
                    df_norm[col] = (df_norm[col] - median_val) / (1.4826 * mad)
        
        logger.info(f"Normalized {len(columns)} columns using {method} method")
        return df_norm
    
    def resample_data(self, df: pd.DataFrame, freq: str = '1H', agg_method: str = 'ohlc') -> pd.DataFrame:
        """
        Resample OHLCV data to different frequency.
        
        Args:
            df: Input OHLCV DataFrame
            freq: Target frequency (e.g., '1H', '4H', '1D')
            agg_method: Aggregation method
            
        Returns:
            Resampled DataFrame
        """
        if agg_method == 'ohlc' and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # OHLC aggregation
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            
            if 'volume' in df.columns:
                agg_dict['volume'] = 'sum'
            
            # Add other columns with mean aggregation
            for col in df.columns:
                if col not in agg_dict:
                    agg_dict[col] = 'mean'
            
            resampled = df.resample(freq).agg(agg_dict)
        
        else:
            # Simple mean aggregation
            resampled = df.resample(freq).mean()
        
        # Remove rows with NaN values
        resampled = resampled.dropna()
        
        logger.info(f"Resampled data to {freq} frequency. Rows: {len(df)} -> {len(resampled)}")
        return resampled
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15, 
                   test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets chronologically.
        
        Args:
            df: Input DataFrame
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        n_samples = len(df)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def process_multiple_symbols(self, data_dict: Dict[str, pd.DataFrame], 
                                operations: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Process multiple symbols with specified operations.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            operations: List of operations to perform
            
        Returns:
            Dictionary of processed DataFrames
        """
        if operations is None:
            operations = ['clean', 'indicators', 'returns', 'time_features']
        
        processed_data = {}
        
        for symbol, df in data_dict.items():
            logger.info(f"Processing {symbol}")
            processed_df = df.copy()
            
            if 'clean' in operations:
                processed_df = self.clean_ohlcv_data(processed_df, symbol)
            
            if 'indicators' in operations:
                processed_df = self.add_technical_indicators(processed_df)
            
            if 'returns' in operations:
                processed_df = self.calculate_returns(processed_df)
            
            if 'time_features' in operations:
                processed_df = self.add_time_features(processed_df)
            
            if 'normalize' in operations:
                # Normalize only price and volume columns
                price_vol_cols = ['open', 'high', 'low', 'close', 'volume']
                cols_to_norm = [col for col in price_vol_cols if col in processed_df.columns]
                processed_df = self.normalize_data(processed_df, columns=cols_to_norm)
            
            processed_data[symbol] = processed_df
        
        logger.info(f"Processed {len(processed_data)} symbols")
        return processed_data

class DataPipeline:
    """Data processing pipeline for streamlined data preparation."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.processor = DataProcessor(config)
        self.file_manager = FileManager()
    
    def run_pipeline(self, 
                     raw_data: Dict[str, pd.DataFrame],
                     operations: List[str] = None,
                     save_path: str = None) -> Dict[str, pd.DataFrame]:
        """
        Run complete data processing pipeline.
        
        Args:
            raw_data: Raw market data
            operations: Processing operations to perform
            save_path: Path to save processed data
            
        Returns:
            Processed data dictionary
        """
        logger.info("Starting data processing pipeline")
        
        # Process all symbols
        processed_data = self.processor.process_multiple_symbols(raw_data, operations)
        
        # Save processed data if path provided
        if save_path:
            self._save_processed_data(processed_data, save_path)
        
        logger.info("Data processing pipeline completed")
        return processed_data
    
    def _save_processed_data(self, data: Dict[str, pd.DataFrame], save_path: str):
        """Save processed data to files."""
        for symbol, df in data.items():
            file_path = f"{save_path}/{symbol.replace('/', '_')}_processed.parquet"
            self.file_manager.save_dataframe(df, file_path)
            logger.info(f"Saved processed data for {symbol} to {file_path}")
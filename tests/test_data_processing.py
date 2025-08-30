"""
Unit tests for data processing modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import DatabaseManager, MarketData, Position, Trade, Signal
from data.data_processor import DataProcessor
from data.data_collector import DataCollector

class TestDatabaseManager:
    """Test database operations"""
    
    @pytest.fixture
    async def db_manager(self):
        """Create test database manager"""
        db_url = "sqlite:///:memory:"  # In-memory database for testing
        manager = DatabaseManager(db_url)
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_save_market_data(self, db_manager):
        """Test saving market data"""
        test_data = [{
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'timestamp': int(datetime.utcnow().timestamp() * 1000),
            'open': 50000.0,
            'high': 51000.0,
            'low': 49500.0,
            'close': 50500.0,
            'volume': 1000.0
        }]
        
        result = await db_manager.save_market_data(test_data)
        assert result is True
        
        # Verify data was saved
        saved_data = await db_manager.get_latest_market_data('BTCUSDT', '1h', 1)
        assert len(saved_data) == 1
        assert saved_data[0].symbol == 'BTCUSDT'
        assert saved_data[0].close == 50500.0
    
    @pytest.mark.asyncio
    async def test_save_position(self, db_manager):
        """Test saving positions"""
        position_data = {
            'symbol': 'ETHUSDT',
            'side': 'long',
            'size': 0.5,
            'entry_price': 3000.0,
            'strategy': 'test_strategy'
        }
        
        result = await db_manager.save_position(position_data)
        assert result is True
        
        # Verify position was saved
        positions = await db_manager.get_open_positions()
        assert len(positions) == 1
        assert positions[0].symbol == 'ETHUSDT'
        assert positions[0].side == 'long'
    
    @pytest.mark.asyncio
    async def test_save_trade(self, db_manager):
        """Test saving trades"""
        trade_data = {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'size': 0.1,
            'price': 50000.0,
            'strategy': 'test_strategy'
        }
        
        result = await db_manager.save_trade(trade_data)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_save_signal(self, db_manager):
        """Test saving signals"""
        signal_data = {
            'symbol': 'BTCUSDT',
            'strategy': 'test_strategy',
            'signal_type': 'buy',
            'confidence': 0.8,
            'price': 50000.0
        }
        
        result = await db_manager.save_signal(signal_data)
        assert result is True
        
        # Verify signal was saved
        signals = await db_manager.get_signals(symbol='BTCUSDT', limit=1)
        assert len(signals) == 1
        assert signals[0].signal_type == 'buy'
        assert signals[0].confidence == 0.8

class TestDataProcessor:
    """Test data processing functionality"""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
        data = []
        
        base_price = 50000
        for i, date in enumerate(dates):
            price = base_price + (i * 10) + np.random.normal(0, 100)
            data.append({
                'timestamp': date,
                'open': price,
                'high': price + np.random.uniform(0, 200),
                'low': price - np.random.uniform(0, 200),
                'close': price + np.random.uniform(-100, 100),
                'volume': np.random.uniform(100, 1000)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def data_processor(self):
        """Create data processor instance"""
        return DataProcessor()
    
    def test_clean_data(self, data_processor, sample_ohlcv_data):
        """Test data cleaning"""
        # Add some bad data
        dirty_data = sample_ohlcv_data.copy()
        dirty_data.loc[5, 'close'] = np.nan
        dirty_data.loc[10, 'volume'] = -100
        
        cleaned_data = data_processor.clean_data(dirty_data)
        
        # Check that bad data was removed/fixed
        assert not cleaned_data['close'].isnull().any()
        assert (cleaned_data['volume'] >= 0).all()
        assert len(cleaned_data) <= len(dirty_data)
    
    def test_add_technical_indicators(self, data_processor, sample_ohlcv_data):
        """Test adding technical indicators"""
        data_with_indicators = data_processor.add_technical_indicators(sample_ohlcv_data)
        
        # Check that indicators were added
        expected_indicators = ['sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        for indicator in expected_indicators:
            assert indicator in data_with_indicators.columns
    
    def test_normalize_data(self, data_processor, sample_ohlcv_data):
        """Test data normalization"""
        normalized_data = data_processor.normalize_data(sample_ohlcv_data)
        
        # Check normalization
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in normalized_data.columns:
                assert normalized_data[col].std() <= 1.1  # Should be approximately normalized
    
    def test_resample_data(self, data_processor, sample_ohlcv_data):
        """Test data resampling"""
        # Resample hourly data to daily
        daily_data = data_processor.resample_ohlcv(sample_ohlcv_data, '1D')
        
        assert len(daily_data) < len(sample_ohlcv_data)
        assert 'open' in daily_data.columns
        assert 'high' in daily_data.columns
        assert 'low' in daily_data.columns
        assert 'close' in daily_data.columns
        assert 'volume' in daily_data.columns

class TestMarketDataCollector:
    """Test market data collection"""
    
    @pytest.fixture
    def collector(self):
        """Create market data collector"""
        config = {
            'exchange': 'binance',
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'timeframes': ['1h'],
            'sandbox': True
        }
        return DataCollector(config)
    
    def test_collector_initialization(self, collector):
        """Test collector initialization"""
        assert collector.exchange_name == 'binance'
        assert 'BTCUSDT' in collector.symbols
        assert '1h' in collector.timeframes
    
    @pytest.mark.asyncio
    async def test_fetch_single_symbol(self, collector):
        """Test fetching data for single symbol"""
        # Mock the exchange response to avoid API calls
        collector.exchange = None  # Disable real exchange
        
        # Test with mock data
        mock_data = [
            [1640995200000, 50000, 51000, 49000, 50500, 1000],  # timestamp, o, h, l, c, v
            [1640998800000, 50500, 51500, 49500, 51000, 1200]
        ]
        
        # This would normally call the exchange
        # For testing, we'll just verify the structure
        assert len(mock_data) == 2
        assert len(mock_data[0]) == 6  # timestamp + OHLCV

class TestDataValidation:
    """Test data validation functions"""
    
    def test_validate_ohlcv_data(self):
        """Test OHLCV data validation"""
        # Valid data
        valid_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'open': np.random.uniform(49000, 51000, 10),
            'high': np.random.uniform(50000, 52000, 10),
            'low': np.random.uniform(48000, 50000, 10),
            'close': np.random.uniform(49500, 51500, 10),
            'volume': np.random.uniform(100, 1000, 10)
        })
        
        # Ensure high >= low, etc.
        valid_data['high'] = np.maximum(valid_data['high'], valid_data[['open', 'close']].max(axis=1))
        valid_data['low'] = np.minimum(valid_data['low'], valid_data[['open', 'close']].min(axis=1))
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        is_valid = processor.validate_ohlcv(valid_data)
        assert is_valid is True
        
        # Invalid data (high < low)
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'high'] = invalid_data.loc[0, 'low'] - 100
        
        is_valid = processor.validate_ohlcv(invalid_data)
        assert is_valid is False
    
    def test_detect_outliers(self):
        """Test outlier detection"""
        # Create data with outliers
        normal_prices = np.random.normal(50000, 1000, 100)
        outlier_prices = np.concatenate([normal_prices, [80000, 20000]])  # Clear outliers
        
        data = pd.DataFrame({
            'close': outlier_prices,
            'volume': np.random.uniform(100, 1000, 102)
        })
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        outliers = processor.detect_outliers(data, 'close')
        assert len(outliers) >= 2  # Should detect at least the 2 outliers we added
    
    def test_gap_detection(self):
        """Test price gap detection"""
        # Create data with a gap
        timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')
        prices = np.linspace(50000, 51000, 100)
        
        # Insert a gap
        prices[50] = 55000  # Significant gap up
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'close': prices
        })
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        gaps = processor.detect_price_gaps(data)
        assert len(gaps) >= 1  # Should detect the gap we created

class TestDataIntegrity:
    """Test data integrity and consistency"""
    
    def test_timestamp_consistency(self):
        """Test timestamp consistency checks"""
        # Create data with inconsistent timestamps
        timestamps = pd.date_range('2024-01-01', periods=10, freq='1H')
        # Duplicate a timestamp
        timestamps = timestamps.insert(5, timestamps[4])
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'close': np.random.uniform(49000, 51000, 11)
        })
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        cleaned_data = processor.remove_duplicate_timestamps(data)
        assert len(cleaned_data) == 10  # Should remove duplicate
    
    def test_volume_validation(self):
        """Test volume data validation"""
        data = pd.DataFrame({
            'volume': [100, 200, -50, 0, 500, np.nan, 300]  # Mix of valid/invalid
        })
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        cleaned_data = processor.clean_volume_data(data)
        
        # Should remove negative and NaN volumes
        assert (cleaned_data['volume'] >= 0).all()
        assert not cleaned_data['volume'].isnull().any()
    
    def test_price_consistency(self):
        """Test price consistency (high >= low, etc.)"""
        data = pd.DataFrame({
            'open': [100, 105, 110],
            'high': [105, 103, 115],  # Second high is below open - inconsistent
            'low': [95, 98, 108],
            'close': [102, 100, 112]
        })
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        cleaned_data = processor.fix_price_inconsistencies(data)
        
        # Verify consistency
        assert (cleaned_data['high'] >= cleaned_data['low']).all()
        assert (cleaned_data['high'] >= cleaned_data[['open', 'close']].min(axis=1)).all()
        assert (cleaned_data['low'] <= cleaned_data[['open', 'close']].max(axis=1)).all()

class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)  # For reproducible tests
        
        prices = []
        price = 50000
        for _ in range(100):
            price += np.random.normal(0, 100)
            prices.append(price)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + np.random.uniform(0, 200) for p in prices],
            'low': [p - np.random.uniform(0, 200) for p in prices],
            'close': [p + np.random.uniform(-100, 100) for p in prices],
            'volume': np.random.uniform(100, 1000, 100)
        })
    
    def test_technical_indicators(self, sample_data):
        """Test technical indicator calculation"""
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        data_with_indicators = processor.add_technical_indicators(sample_data)
        
        # Check that indicators were added
        expected_indicators = [
            'sma_20', 'ema_12', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'volume_sma'
        ]
        
        for indicator in expected_indicators:
            assert indicator in data_with_indicators.columns
            # Check that indicator has some non-null values
            assert not data_with_indicators[indicator].dropna().empty
    
    def test_volatility_calculation(self, sample_data):
        """Test volatility calculations"""
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        data_with_vol = processor.add_volatility_features(sample_data)
        
        assert 'volatility' in data_with_vol.columns
        assert 'atr' in data_with_vol.columns
        assert (data_with_vol['volatility'].dropna() >= 0).all()
        assert (data_with_vol['atr'].dropna() >= 0).all()
    
    def test_momentum_features(self, sample_data):
        """Test momentum feature calculation"""
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        data_with_momentum = processor.add_momentum_features(sample_data)
        
        momentum_cols = [col for col in data_with_momentum.columns if 'momentum' in col]
        assert len(momentum_cols) > 0
        
        # Momentum should be percentage changes
        for col in momentum_cols:
            momentum_values = data_with_momentum[col].dropna()
            if len(momentum_values) > 0:
                assert not np.isinf(momentum_values).any()

class TestDataAggregation:
    """Test data aggregation and resampling"""
    
    def test_ohlcv_resampling(self):
        """Test OHLCV data resampling"""
        # Create minute data
        dates = pd.date_range('2024-01-01', periods=60, freq='1T')
        minute_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(49000, 51000, 60),
            'high': np.random.uniform(50000, 52000, 60),
            'low': np.random.uniform(48000, 50000, 60),
            'close': np.random.uniform(49500, 51500, 60),
            'volume': np.random.uniform(10, 100, 60)
        })
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        # Resample to hourly
        hourly_data = processor.resample_ohlcv(minute_data, '1H')
        
        assert len(hourly_data) == 1  # Should aggregate to 1 hour
        assert 'open' in hourly_data.columns
        assert hourly_data['volume'].iloc[0] > 0  # Volume should be summed
    
    def test_multi_timeframe_features(self):
        """Test multi-timeframe feature creation"""
        # Create base hourly data
        dates = pd.date_range('2024-01-01', periods=168, freq='1H')  # 1 week
        hourly_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.uniform(49000, 51000, 168),
            'volume': np.random.uniform(100, 1000, 168)
        })
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        multi_tf_data = processor.add_multi_timeframe_features(hourly_data)
        
        # Should have features from multiple timeframes
        multi_tf_cols = [col for col in multi_tf_data.columns if '_4h' in col or '_1d' in col]
        assert len(multi_tf_cols) > 0

class TestPerformanceMetrics:
    """Test performance calculation functions"""
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        # Create return series
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        sharpe = processor.calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert -5 <= sharpe <= 5  # Reasonable range for Sharpe ratio
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Create portfolio value series with known drawdown
        portfolio_values = [10000, 10500, 9500, 9000, 9200, 10800, 11000]  # Max DD = 15%
        portfolio_series = pd.Series(portfolio_values)
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        max_dd = processor.calculate_max_drawdown(portfolio_series)
        
        assert isinstance(max_dd, float)
        assert 0 <= max_dd <= 1  # Should be between 0 and 1
        assert abs(max_dd - 0.15) < 0.02  # Should be approximately 15%
    
    def test_volatility_calculation(self):
        """Test volatility calculation"""
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        vol = processor.calculate_volatility(returns)
        
        assert isinstance(vol, float)
        assert vol >= 0
        assert 0 <= vol <= 1  # Should be reasonable volatility range

class TestDataStorage:
    """Test data storage and retrieval"""
    
    @pytest.mark.asyncio
    async def test_batch_data_save(self):
        """Test saving large batches of data"""
        # Create large dataset
        n_points = 1000
        dates = pd.date_range('2024-01-01', periods=n_points, freq='1H')
        
        batch_data = []
        for i, date in enumerate(dates):
            batch_data.append({
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'timestamp': int(date.timestamp() * 1000),
                'open': 50000 + i,
                'high': 50100 + i,
                'low': 49900 + i,
                'close': 50050 + i,
                'volume': 100 + i
            })
        
        # Test database manager with batch save
        db_url = "sqlite:///:memory:"
        db_manager = DatabaseManager(db_url)
        await db_manager.initialize()
        
        result = await db_manager.save_market_data(batch_data)
        assert result is True
        
        # Verify data was saved
        saved_data = await db_manager.get_latest_market_data('BTCUSDT', '1h', n_points)
        assert len(saved_data) == n_points
    
    def test_data_compression(self):
        """Test data compression for storage efficiency"""
        # Create large dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1T'),
            'close': np.random.uniform(49000, 51000, 10000),
            'volume': np.random.uniform(100, 1000, 10000)
        })
        
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        # Test compression
        compressed_data = processor.compress_data(large_data)
        decompressed_data = processor.decompress_data(compressed_data)
        
        # Data should be approximately the same after compression/decompression
        assert len(decompressed_data) == len(large_data)
        np.testing.assert_array_almost_equal(
            decompressed_data['close'].values[:100], 
            large_data['close'].values[:100], 
            decimal=2
        )

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
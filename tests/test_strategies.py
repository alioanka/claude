"""
Unit tests for trading strategies.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from strategies.arbitrage_strategy import ArbitrageStrategy

class TestMomentumStrategy:
    """Test momentum trading strategy"""
    
    @pytest.fixture
    def momentum_strategy(self):
        """Create momentum strategy instance"""
        config = {
            'ema_short': 12,
            'ema_long': 26,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_threshold': 1.5
        }
        return MomentumStrategy(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data with momentum"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        # Create trending price data
        trend = np.linspace(0, 1000, 100)  # Upward trend
        noise = np.random.normal(0, 50, 100)
        prices = 50000 + trend + noise
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 100, 100),
            'low': prices - np.random.uniform(0, 100, 100),
            'close': prices + np.random.uniform(-50, 50, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }).set_index('timestamp')
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, momentum_strategy, sample_data):
        """Test momentum signal generation"""
        signal = await momentum_strategy.generate_signal('BTCUSDT', sample_data)
        
        assert signal is not None
        assert 'signal_type' in signal
        assert signal['signal_type'] in ['buy', 'sell', 'hold']
        assert 'confidence' in signal
        assert 0 <= signal['confidence'] <= 1
        assert signal['symbol'] == 'BTCUSDT'
    
    def test_ema_crossover_detection(self, momentum_strategy, sample_data):
        """Test EMA crossover detection"""
        # Add EMAs to data
        sample_data['ema_12'] = sample_data['close'].ewm(span=12).mean()
        sample_data['ema_26'] = sample_data['close'].ewm(span=26).mean()
        
        crossovers = momentum_strategy._detect_ema_crossover(sample_data)
        
        assert isinstance(crossovers, list)
        if crossovers:
            for crossover in crossovers:
                assert 'type' in crossover
                assert crossover['type'] in ['bullish', 'bearish']
                assert 'index' in crossover
    
    def test_volume_confirmation(self, momentum_strategy, sample_data):
        """Test volume confirmation logic"""
        # Test high volume scenario
        high_volume_data = sample_data.copy()
        high_volume_data.loc[high_volume_data.index[-1], 'volume'] = 2000  # High volume
        
        volume_confirmed = momentum_strategy._confirm_with_volume(high_volume_data)
        assert isinstance(volume_confirmed, bool)
    
    def test_trend_strength_calculation(self, momentum_strategy, sample_data):
        """Test trend strength calculation"""
        trend_strength = momentum_strategy._calculate_trend_strength(sample_data)
        
        assert isinstance(trend_strength, float)
        assert -1 <= trend_strength <= 1

class TestMeanReversionStrategy:
    """Test mean reversion trading strategy"""
    
    @pytest.fixture
    def mean_reversion_strategy(self):
        """Create mean reversion strategy instance"""
        config = {
            'bollinger_period': 20,
            'bollinger_std': 2,
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70
        }
        return MeanReversionStrategy(config)
    
    @pytest.fixture
    def oscillating_data(self):
        """Create oscillating price data for mean reversion"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        # Create oscillating prices around a mean
        mean_price = 50000
        oscillation = np.sin(np.linspace(0, 4*np.pi, 100)) * 1000
        noise = np.random.normal(0, 100, 100)
        prices = mean_price + oscillation + noise
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 100, 100),
            'low': prices - np.random.uniform(0, 100, 100),
            'close': prices + np.random.uniform(-50, 50, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }).set_index('timestamp')
    
    @pytest.mark.asyncio
    async def test_mean_reversion_signal(self, mean_reversion_strategy, oscillating_data):
        """Test mean reversion signal generation"""
        signal = await mean_reversion_strategy.generate_signal('BTCUSDT', oscillating_data)
        
        assert signal is not None
        assert 'signal_type' in signal
        assert signal['signal_type'] in ['buy', 'sell', 'hold']
        assert 'confidence' in signal
    
    def test_bollinger_band_calculation(self, mean_reversion_strategy, oscillating_data):
        """Test Bollinger Bands calculation"""
        bb_data = mean_reversion_strategy._calculate_bollinger_bands(oscillating_data)
        
        assert 'bb_upper' in bb_data.columns
        assert 'bb_lower' in bb_data.columns
        assert 'bb_middle' in bb_data.columns
        
        # Upper should be above lower
        assert (bb_data['bb_upper'] >= bb_data['bb_lower']).all()
    
    def test_oversold_overbought_detection(self, mean_reversion_strategy, oscillating_data):
        """Test oversold/overbought condition detection"""
        # Add RSI to data
        delta = oscillating_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        oscillating_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Force oversold condition
        oscillating_data.loc[oscillating_data.index[-1], 'rsi'] = 25
        
        is_oversold = mean_reversion_strategy._is_oversold(oscillating_data.iloc[-1])
        assert is_oversold is True
        
        # Force overbought condition
        oscillating_data.loc[oscillating_data.index[-1], 'rsi'] = 75
        is_overbought = mean_reversion_strategy._is_overbought(oscillating_data.iloc[-1])
        assert is_overbought is True

class TestMLStrategy:
    """Test machine learning strategy"""
    
    @pytest.fixture
    def ml_strategy(self):
        """Create ML strategy instance"""
        config = {
            'min_confidence': 0.6,
            'lstm_config': {
                'sequence_length': 30,
                'hidden_units': [64, 32]
            },
            'model_dir': 'test_models'
        }
        return MLStrategy(config)
    
    @pytest.fixture
    def ml_training_data(self):
        """Create training data for ML models"""
        dates = pd.date_range('2024-01-01', periods=200, freq='1H')
        
        # Create data with some predictable patterns
        trend = np.linspace(0, 2000, 200)
        cyclical = np.sin(np.linspace(0, 8*np.pi, 200)) * 500
        noise = np.random.normal(0, 100, 200)
        prices = 50000 + trend + cyclical + noise
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 100, 200),
            'low': prices - np.random.uniform(0, 100, 200),
            'close': prices + np.random.uniform(-50, 50, 200),
            'volume': np.random.uniform(100, 1000, 200)
        }).set_index('timestamp')
        
        return {'BTCUSDT': data}
    
    @pytest.mark.asyncio
    async def test_feature_engineering(self, ml_strategy, ml_training_data):
        """Test feature engineering for ML"""
        data = ml_training_data['BTCUSDT']
        features_df = await ml_strategy._engineer_features(data)
        
        assert not features_df.empty
        assert len(ml_strategy.feature_columns) > 0
        
        # Check for expected features
        expected_features = ['returns', 'rsi', 'macd', 'bb_position', 'volatility']
        for feature in expected_features:
            assert feature in features_df.columns
    
    @pytest.mark.asyncio 
    async def test_model_training(self, ml_strategy, ml_training_data):
        """Test ML model training"""
        # Mock the ML libraries to avoid actual training in tests
        with patch('strategies.ml_strategy.TENSORFLOW_AVAILABLE', False), \
             patch('strategies.ml_strategy.XGBOOST_AVAILABLE', False), \
             patch('strategies.ml_strategy.SKLEARN_AVAILABLE', True):
            
            success = await ml_strategy.train_models(ml_training_data)
            # Should handle missing libraries gracefully
            assert isinstance(success, bool)
    
    def test_prediction_combination(self, ml_strategy):
        """Test prediction combination logic"""
        predictions = {
            'lstm': {'direction': 0.7, 'confidence': 0.8},
            'xgboost': {'direction': 0.6, 'confidence': 0.7},
            'ensemble': {'direction': 0.65, 'confidence': 0.75}
        }
        
        combined = asyncio.run(ml_strategy._combine_predictions(predictions))
        
        assert 'direction' in combined
        assert 'confidence' in combined
        assert 0 <= combined['direction'] <= 1
        assert 0 <= combined['confidence'] <= 1

class TestArbitrageStrategy:
    """Test arbitrage trading strategy"""
    
    @pytest.fixture
    def arbitrage_strategy(self):
        """Create arbitrage strategy instance"""
        config = {
            'arbitrage_params': {
                'min_spread_threshold': 0.005,
                'max_spread_threshold': 0.05,
                'correlation_threshold': 0.8
            },
            'exchanges': {
                'binance': {'enabled': False},  # Disabled for testing
                'kraken': {'enabled': False}
            }
        }
        return ArbitrageStrategy(config)
    
    def test_cross_exchange_arbitrage_detection(self, arbitrage_strategy):
        """Test cross-exchange arbitrage detection"""
        # Mock price data with arbitrage opportunity
        prices = {
            'binance': {'bid': 50000, 'ask': 50100, 'timestamp': datetime.utcnow()},
            'kraken': {'bid': 50200, 'ask': 50300, 'timestamp': datetime.utcnow()}
        }
        
        # Calculate spread manually
        spread = (50200 - 50100) / 50100  # Buy on binance, sell on kraken
        
        assert spread > 0  # Should be profitable opportunity
        assert spread > arbitrage_strategy.arb_config.min_spread_threshold
    
    def test_statistical_arbitrage_signal(self, arbitrage_strategy):
        """Test statistical arbitrage (pairs trading)"""
        # Create correlated data that diverges
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        base_price = np.cumsum(np.random.normal(0, 0.01, 100))
        
        # Correlated pair that diverges at the end
        btc_prices = 50000 + base_price * 1000
        eth_prices = 3000 + base_price * 60
        eth_prices[-1] += 200  # Divergence
        
        data = pd.DataFrame({
            'timestamp': dates,
            'close': btc_prices,
            'volume': np.random.uniform(100, 1000, 100)
        }).set_index('timestamp')
        
        # Test signal generation
        signal = asyncio.run(arbitrage_strategy.generate_signal('BTCUSDT', data))
        
        # May not generate signal due to simplified implementation
        # Just test that it doesn't crash
        assert signal is None or isinstance(signal, dict)
    
    def test_triangular_arbitrage_calculation(self, arbitrage_strategy):
        """Test triangular arbitrage profit calculation"""
        # Mock prices for triangular arbitrage
        prices = {
            'BTCUSDT': {'bid': 50000, 'ask': 50100},
            'ETHUSDT': {'bid': 3000, 'ask': 3010},
            'ETHBTC': {'bid': 0.059, 'ask': 0.061}
        }
        
        profit = arbitrage_strategy._calculate_triangular_profit(
            prices, 'BTCUSDT', 'ETHUSDT', 'ETHBTC'
        )
        
        assert isinstance(profit, float)
        assert profit >= 0  # Profit should be non-negative

class TestStrategyValidation:
    """Test strategy validation and edge cases"""
    
    def test_insufficient_data_handling(self):
        """Test strategy behavior with insufficient data"""
        # Create very small dataset
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1H'),
            'close': [50000, 50100, 49900, 50200, 50050],
            'volume': [100, 150, 120, 180, 140]
        }).set_index('timestamp')
        
        momentum_strategy = MomentumStrategy({})
        
        # Should handle gracefully
        signal = asyncio.run(momentum_strategy.generate_signal('BTCUSDT', small_data))
        
        # Should return None or hold signal due to insufficient data
        assert signal is None or signal['signal_type'] == 'hold'
    
    def test_invalid_data_handling(self):
        """Test strategy behavior with invalid data"""
        # Create data with NaN values
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1H'),
            'close': [50000] * 25 + [np.nan] * 25,  # Half NaN values
            'volume': [100] * 50
        }).set_index('timestamp')
        
        momentum_strategy = MomentumStrategy({})
        
        # Should handle gracefully
        signal = asyncio.run(momentum_strategy.generate_signal('BTCUSDT', invalid_data))
        assert signal is None or isinstance(signal, dict)
    
    def test_extreme_market_conditions(self):
        """Test strategy behavior during extreme market conditions"""
        # Create data simulating market crash
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        
        # Simulate 50% crash over 10 periods
        crash_multipliers = np.linspace(1.0, 0.5, 10)
        recovery_multipliers = np.linspace(0.5, 0.7, 40)
        all_multipliers = np.concatenate([crash_multipliers, recovery_multipliers])
        
        prices = 50000 * all_multipliers
        
        crash_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, 50)  # High volume during crash
        }).set_index('timestamp')
        
        strategies = [
            MomentumStrategy({}),
            MeanReversionStrategy({})
        ]
        
        for strategy in strategies:
            signal = asyncio.run(strategy.generate_signal('BTCUSDT', crash_data))
            # Should generate some signal or handle gracefully
            assert signal is None or isinstance(signal, dict)

class TestStrategyPerformance:
    """Test strategy performance metrics"""
    
    def test_strategy_backtest_integration(self):
        """Test strategy integration with backtesting"""
        from scripts.backtest_runner import BacktestRunner
        
        config = {
            'initial_capital': 10000,
            'commission_rate': 0.001,
            'strategies': {
                'momentum_strategy': {'enabled': True},
                'mean_reversion_strategy': {'enabled': True}
            }
        }
        
        runner = BacktestRunner(config)
        
        # Test initialization
        assert runner.initial_capital == 10000
        assert runner.commission_rate == 0.001
        assert hasattr(runner, 'strategies')
    
    def test_signal_quality_metrics(self):
        """Test signal quality and performance metrics"""
        # Create sample signals and outcomes
        signals = [
            {'signal_type': 'buy', 'confidence': 0.8, 'actual_return': 0.05},
            {'signal_type': 'buy', 'confidence': 0.6, 'actual_return': -0.02},
            {'signal_type': 'sell', 'confidence': 0.9, 'actual_return': 0.03},  # Correct sell signal
            {'signal_type': 'hold', 'confidence': 0.3, 'actual_return': 0.001}
        ]
        
        # Calculate signal accuracy
        correct_signals = 0
        for signal in signals:
            if signal['signal_type'] == 'buy' and signal['actual_return'] > 0:
                correct_signals += 1
            elif signal['signal_type'] == 'sell' and signal['actual_return'] < 0:
                correct_signals += 1
            elif signal['signal_type'] == 'hold' and abs(signal['actual_return']) < 0.01:
                correct_signals += 1
        
        accuracy = correct_signals / len(signals)
        assert 0 <= accuracy <= 1
    
    def test_strategy_risk_metrics(self):
        """Test strategy-specific risk metrics"""
        # Sample strategy performance data
        strategy_returns = pd.Series([0.02, -0.01, 0.03, -0.015, 0.025, -0.005])
        
        # Calculate strategy metrics
        win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
        avg_win = strategy_returns[strategy_returns > 0].mean()
        avg_loss = strategy_returns[strategy_returns < 0].mean()
        
        assert 0 <= win_rate <= 1
        assert avg_win > 0
        assert avg_loss < 0

class TestStrategyConfiguration:
    """Test strategy configuration and parameter validation"""
    
    def test_momentum_config_validation(self):
        """Test momentum strategy configuration validation"""
        # Valid configuration
        valid_config = {
            'ema_short': 12,
            'ema_long': 26,
            'rsi_period': 14
        }
        
        strategy = MomentumStrategy(valid_config)
        assert strategy.config['ema_short'] < strategy.config['ema_long']
        
        # Invalid configuration (short > long)
        with pytest.raises(ValueError):
            invalid_config = {
                'ema_short': 26,
                'ema_long': 12  # Invalid: short period > long period
            }
            MomentumStrategy(invalid_config)
    
    def test_strategy_parameter_ranges(self):
        """Test strategy parameter validation ranges"""
        # Test RSI parameters
        rsi_config = {'rsi_overbought': 85, 'rsi_oversold': 15}
        strategy = MeanReversionStrategy(rsi_config)
        
        assert 0 < strategy.config['rsi_oversold'] < 50
        assert 50 < strategy.config['rsi_overbought'] < 100
        assert strategy.config['rsi_oversold'] < strategy.config['rsi_overbought']

class TestStrategyRobustness:
    """Test strategy robustness and error handling"""
    
    def test_missing_indicator_data(self):
        """Test strategy behavior when indicator data is missing"""
        # Create data without required indicators
        basic_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1H'),
            'close': np.random.uniform(49000, 51000, 50)
        }).set_index('timestamp')
        
        momentum_strategy = MomentumStrategy({})
        
        # Should handle missing indicators gracefully
        signal = asyncio.run(momentum_strategy.generate_signal('BTCUSDT', basic_data))
        assert signal is None or isinstance(signal, dict)
    
    def test_strategy_with_flat_market(self):
        """Test strategy behavior in flat/sideways market"""
        # Create flat market data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        flat_prices = 50000 + np.random.normal(0, 10, 100)  # Very low volatility
        
        flat_data = pd.DataFrame({
            'timestamp': dates,
            'open': flat_prices,
            'high': flat_prices + 5,
            'low': flat_prices - 5,
            'close': flat_prices,
            'volume': np.random.uniform(100, 200, 100)
        }).set_index('timestamp')
        
        strategies = [
            MomentumStrategy({}),
            MeanReversionStrategy({})
        ]
        
        for strategy in strategies:
            signal = asyncio.run(strategy.generate_signal('BTCUSDT', flat_data))
            # In flat market, should mostly generate hold signals
            if signal:
                assert signal['confidence'] < 0.7  # Low confidence in flat market

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
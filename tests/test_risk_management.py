"""
Unit tests for risk management modules.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.risk_manager import RiskManager
from risk.portfolio_optimizer import PortfolioOptimizer, OptimizationConfig

class TestRiskManager:
    """Test risk management functionality"""
    
    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance"""
        config = {
            'max_position_size': 0.1,
            'max_daily_loss': 0.05,
            'max_drawdown': 0.15,
            'risk_per_trade': 0.02,
            'correlation_limit': 0.7
        }
        return RiskManager(config)
    
    def test_position_size_calculation(self, risk_manager):
        """Test position size calculation"""
        portfolio_value = 10000
        confidence = 0.8
        volatility = 0.02
        
        position_size = risk_manager.calculate_position_size(
            portfolio_value, confidence, volatility
        )
        
        assert position_size > 0
        assert position_size <= portfolio_value * 0.1  # Max 10% position
        assert isinstance(position_size, float)
    
    def test_risk_limits_check(self, risk_manager):
        """Test risk limits validation"""
        # Test within limits
        positions = [
            {'symbol': 'BTCUSDT', 'size': 500, 'pnl': 50},
            {'symbol': 'ETHUSDT', 'size': 300, 'pnl': -20}
        ]
        portfolio_value = 10000
        
        within_limits = risk_manager.check_risk_limits(positions, portfolio_value)
        assert within_limits is True
        
        # Test exceeding limits
        risky_positions = [
            {'symbol': 'BTCUSDT', 'size': 2000, 'pnl': -800},  # Large loss
            {'symbol': 'ETHUSDT', 'size': 1500, 'pnl': -600}
        ]
        
        exceeds_limits = risk_manager.check_risk_limits(risky_positions, portfolio_value)
        assert exceeds_limits is False
    
    def test_stop_loss_calculation(self, risk_manager):
        """Test stop loss calculation"""
        entry_price = 50000
        position_size = 0.1
        portfolio_value = 10000
        
        stop_loss = risk_manager.calculate_stop_loss(
            entry_price, position_size, portfolio_value, side='long'
        )
        
        assert stop_loss < entry_price  # Stop loss should be below entry for long
        assert stop_loss > 0
        
        # Test short position
        stop_loss_short = risk_manager.calculate_stop_loss(
            entry_price, position_size, portfolio_value, side='short'
        )
        
        assert stop_loss_short > entry_price  # Stop loss should be above entry for short
    
    def test_correlation_check(self, risk_manager):
        """Test position correlation checking"""
        # Mock price data for correlation calculation
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        # Highly correlated data
        base_prices = np.random.normal(0, 0.02, 100)
        correlated_data = {
            'BTCUSDT': pd.Series(base_prices, index=dates),
            'ETHUSDT': pd.Series(base_prices * 0.8 + np.random.normal(0, 0.005, 100), index=dates)
        }
        
        correlation = risk_manager.calculate_correlation(
            correlated_data['BTCUSDT'], 
            correlated_data['ETHUSDT']
        )
        
        assert 0 <= abs(correlation) <= 1
        assert correlation > 0.5  # Should be positively correlated
    
    def test_volatility_adjustment(self, risk_manager):
        """Test volatility-based position adjustment"""
        # High volatility scenario
        high_vol_returns = np.random.normal(0, 0.05, 20)  # 5% daily volatility
        high_vol_adjustment = risk_manager.adjust_for_volatility(high_vol_returns)
        
        # Low volatility scenario
        low_vol_returns = np.random.normal(0, 0.01, 20)  # 1% daily volatility
        low_vol_adjustment = risk_manager.adjust_for_volatility(low_vol_returns)
        
        # High volatility should result in smaller position multiplier
        assert high_vol_adjustment < low_vol_adjustment
        assert 0 < high_vol_adjustment <= 1
        assert 0 < low_vol_adjustment <= 1

class TestPortfolioOptimizer:
    """Test portfolio optimization functionality"""
    
    @pytest.fixture
    def optimizer(self):
        """Create portfolio optimizer"""
        config = OptimizationConfig(
            max_position_size=0.15,
            max_portfolio_risk=0.02,
            correlation_threshold=0.7
        )
        return PortfolioOptimizer(config)
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample positions"""
        return [
            {
                'symbol': 'BTCUSDT',
                'side': 'long',
                'size': 0.1,
                'entry_price': 50000,
                'current_price': 51000,
                'pnl': 100,
                'is_open': True
            },
            {
                'symbol': 'ETHUSDT',
                'side': 'long',
                'size': 0.5,
                'entry_price': 3000,
                'current_price': 2950,
                'pnl': -25,
                'is_open': True
            }
        ]
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals"""
        return [
            {
                'symbol': 'BTCUSDT',
                'signal_type': 'buy',
                'confidence': 0.8,
                'strategy': 'momentum'
            },
            {
                'symbol': 'ETHUSDT',
                'signal_type': 'sell',
                'confidence': 0.6,
                'strategy': 'mean_reversion'
            },
            {
                'symbol': 'ADAUSDT',
                'signal_type': 'buy',
                'confidence': 0.7,
                'strategy': 'ml_strategy'
            }
        ]
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        market_data = {}
        for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
            base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1
            prices = base_price + np.cumsum(np.random.normal(0, base_price * 0.001, 100))
            
            market_data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'close': prices,
                'volume': np.random.uniform(100, 1000, 100)
            })
        
        return market_data
    
    @pytest.mark.asyncio
    async def test_position_optimization(self, optimizer, sample_positions, sample_signals, sample_market_data):
        """Test position optimization"""
        portfolio_value = 10000
        
        optimal_weights = await optimizer.optimize_positions(
            sample_positions,
            sample_signals,
            sample_market_data,
            portfolio_value
        )
        
        assert isinstance(optimal_weights, dict)
        
        # Check weight constraints
        for symbol, weight in optimal_weights.items():
            assert abs(weight) <= optimizer.config.max_position_size
    
    def test_diversification_calculation(self, optimizer):
        """Test diversification score calculation"""
        # Well diversified portfolio
        diversified_weights = {
            'BTCUSDT': 0.25,
            'ETHUSDT': 0.25,
            'ADAUSDT': 0.25,
            'BNBUSDT': 0.25
        }
        
        diversification_score = optimizer.calculate_diversification_score(diversified_weights)
        
        assert 0 <= diversification_score <= 1
        assert diversification_score > 0.8  # Should be highly diversified
        
        # Concentrated portfolio
        concentrated_weights = {
            'BTCUSDT': 0.9,
            'ETHUSDT': 0.1
        }
        
        concentrated_score = optimizer.calculate_diversification_score(concentrated_weights)
        assert concentrated_score < diversification_score  # Should be less diversified
    
    def test_correlation_matrix_calculation(self, optimizer, sample_market_data):
        """Test correlation matrix calculation"""
        returns_data = {}
        
        for symbol, df in sample_market_data.items():
            returns_data[symbol] = df['close'].pct_change().dropna()
        
        returns_df = pd.DataFrame(returns_data)
        
        correlation_matrix = asyncio.run(optimizer._calculate_correlations(returns_df))
        
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        assert correlation_matrix.shape[0] == len(sample_market_data)
        
        # Diagonal should be 1 (perfect self-correlation)
        np.testing.assert_array_almost_equal(np.diag(correlation_matrix), 1.0, decimal=2)
    
    def test_risk_budget_allocation(self, optimizer):
        """Test risk budget allocation"""
        signals = {
            'BTCUSDT': 0.8,
            'ETHUSDT': 0.6,
            'ADAUSDT': 0.4
        }
        
        volatilities = {
            'BTCUSDT': 0.03,
            'ETHUSDT': 0.04,
            'ADAUSDT': 0.05
        }
        
        risk_allocations = asyncio.run(optimizer.get_risk_budget_allocation(signals, volatilities))
        
        assert isinstance(risk_allocations, dict)
        assert all(allocation >= 0 for allocation in risk_allocations.values())
        
        # Total risk should not exceed budget
        total_risk = sum(risk_allocations.values())
        assert total_risk <= optimizer.config.max_portfolio_risk * 1.1  # Small tolerance

class TestPositionSizing:
    """Test position sizing algorithms"""
    
    def test_kelly_criterion(self):
        """Test Kelly criterion position sizing"""
        from risk.risk_manager import RiskManager
        
        config = {'risk_per_trade': 0.02}
        rm = RiskManager(config)
        
        # Test parameters
        win_rate = 0.6
        avg_win = 0.03
        avg_loss = 0.02
        
        kelly_fraction = rm.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        assert 0 <= kelly_fraction <= 1
        assert isinstance(kelly_fraction, float)
    
    def test_volatility_adjustment(self):
        """Test volatility-based position adjustment"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        # High volatility should reduce position size
        high_vol = 0.05
        low_vol = 0.01
        
        high_vol_multiplier = rm.volatility_adjustment(high_vol)
        low_vol_multiplier = rm.volatility_adjustment(low_vol)
        
        assert high_vol_multiplier < low_vol_multiplier
        assert 0 < high_vol_multiplier <= 1
        assert 0 < low_vol_multiplier <= 1
    
    def test_position_sizing_edge_cases(self):
        """Test position sizing edge cases"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({'max_position_size': 0.1, 'min_position_size': 0.001})
        
        # Zero confidence should result in zero position
        zero_conf_size = rm.calculate_position_size(10000, 0.0, 0.02)
        assert zero_conf_size == 0
        
        # Very high confidence should be capped
        high_conf_size = rm.calculate_position_size(10000, 2.0, 0.02)
        max_allowed = 10000 * 0.1
        assert high_conf_size <= max_allowed

class TestRiskMetrics:
    """Test risk metric calculations"""
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns
        portfolio_value = 10000
        
        var_95 = rm.calculate_var(returns, portfolio_value, confidence=0.95)
        var_99 = rm.calculate_var(returns, portfolio_value, confidence=0.99)
        
        assert var_95 > 0
        assert var_99 > var_95  # 99% VaR should be higher than 95% VaR
        assert var_95 < portfolio_value  # VaR shouldn't exceed portfolio value
    
    def test_expected_shortfall(self):
        """Test Expected Shortfall (Conditional VaR) calculation"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        portfolio_value = 10000
        
        es = rm.calculate_expected_shortfall(returns, portfolio_value)
        var = rm.calculate_var(returns, portfolio_value)
        
        assert es >= var  # ES should be at least as large as VaR
        assert es > 0
    
    def test_beta_calculation(self):
        """Test beta calculation against market"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        # Create correlated returns
        market_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        asset_returns = 1.5 * market_returns + pd.Series(np.random.normal(0, 0.01, 100))
        
        beta = rm.calculate_beta(asset_returns, market_returns)
        
        assert isinstance(beta, float)
        assert 1.0 <= beta <= 2.0  # Should be around 1.5 given our setup

class TestDrawdownAnalysis:
    """Test drawdown analysis functions"""
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        # Create portfolio values with known drawdown
        portfolio_values = pd.Series([10000, 10500, 9000, 8500, 9500, 11000])
        
        max_drawdown = rm.calculate_max_drawdown(portfolio_values)
        
        assert 0 <= max_drawdown <= 1
        # Max drawdown should be around 19% (from 10500 to 8500)
        assert 0.15 <= max_drawdown <= 0.25
    
    def test_drawdown_duration(self):
        """Test drawdown duration calculation"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        # Create portfolio with extended drawdown
        values = [10000] + [9000] * 10 + [11000]  # 10-period drawdown
        portfolio_values = pd.Series(values)
        
        drawdown_duration = rm.calculate_drawdown_duration(portfolio_values)
        
        assert drawdown_duration >= 10
        assert isinstance(drawdown_duration, int)
    
    def test_underwater_curve(self):
        """Test underwater curve calculation"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        portfolio_values = pd.Series([10000, 10500, 9500, 9000, 9800, 11000])
        underwater_curve = rm.calculate_underwater_curve(portfolio_values)
        
        assert len(underwater_curve) == len(portfolio_values)
        assert (underwater_curve <= 0).all()  # Underwater should be negative or zero
        assert underwater_curve.iloc[0] == 0  # First value should be zero

class TestRiskAlerts:
    """Test risk alerting system"""
    
    def test_risk_threshold_breach(self):
        """Test risk threshold breach detection"""
        from risk.risk_manager import RiskManager
        
        config = {
            'max_daily_loss': 0.05,
            'max_drawdown': 0.15,
            'correlation_limit': 0.7
        }
        rm = RiskManager(config)
        
        # Test daily loss breach
        positions_with_loss = [
            {'symbol': 'BTCUSDT', 'pnl': -600},
            {'symbol': 'ETHUSDT', 'pnl': -400}
        ]
        
        alerts = rm.check_risk_alerts(positions_with_loss, portfolio_value=10000)
        
        assert 'daily_loss_exceeded' in alerts
        assert alerts['daily_loss_exceeded'] is True
    
    def test_position_concentration_alert(self):
        """Test position concentration alerts"""
        from risk.risk_manager import RiskManager
        
        config = {'max_position_size': 0.1}
        rm = RiskManager(config)
        
        # Large position that exceeds limit
        large_position = [
            {'symbol': 'BTCUSDT', 'size': 0.15, 'current_price': 50000}  # 15% of portfolio
        ]
        
        alerts = rm.check_position_concentration(large_position, portfolio_value=10000)
        
        assert 'concentration_risk' in alerts
        assert len(alerts['oversized_positions']) > 0

class TestStopLossManagement:
    """Test stop loss and take profit management"""
    
    def test_trailing_stop_loss(self):
        """Test trailing stop loss calculation"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({'trailing_stop_percent': 0.05})
        
        # Long position with price moving up
        entry_price = 50000
        current_price = 52000
        trail_percent = 0.05
        
        trailing_stop = rm.calculate_trailing_stop(
            entry_price, current_price, trail_percent, side='long'
        )
        
        expected_stop = current_price * (1 - trail_percent)
        assert abs(trailing_stop - expected_stop) < 1
        
        # Short position
        trailing_stop_short = rm.calculate_trailing_stop(
            entry_price, 48000, trail_percent, side='short'
        )
        
        expected_stop_short = 48000 * (1 + trail_percent)
        assert abs(trailing_stop_short - expected_stop_short) < 1
    
    def test_dynamic_stop_loss(self):
        """Test dynamic stop loss based on volatility"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        # High volatility should result in wider stop loss
        high_vol_stop = rm.calculate_dynamic_stop_loss(50000, 0.05, side='long')
        low_vol_stop = rm.calculate_dynamic_stop_loss(50000, 0.01, side='long')
        
        # High volatility stop should be further from entry price
        assert (50000 - high_vol_stop) > (50000 - low_vol_stop)
        assert high_vol_stop < 50000  # Stop loss below entry for long
        assert low_vol_stop < 50000

class TestRiskModelValidation:
    """Test risk model validation and backtesting"""
    
    def test_var_model_backtesting(self):
        """Test VaR model backtesting"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        # Create returns with known characteristics
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        
        # Calculate VaR violations
        var_95 = rm.calculate_var(returns, 10000, confidence=0.95)
        violations = rm.backtest_var_model(returns, var_95, confidence=0.95)
        
        # Should have approximately 5% violations for 95% VaR
        violation_rate = len(violations) / len(returns)
        assert 0.02 <= violation_rate <= 0.08  # Reasonable range around 5%
    
    def test_stress_testing(self):
        """Test stress testing scenarios"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        positions = [
            {'symbol': 'BTCUSDT', 'size': 0.1, 'current_price': 50000},
            {'symbol': 'ETHUSDT', 'size': 0.3, 'current_price': 3000}
        ]
        
        # Test market crash scenario (-20% across all positions)
        stress_scenarios = {
            'market_crash': -0.20,
            'crypto_winter': -0.50,
            'flash_crash': -0.10
        }
        
        stress_results = rm.run_stress_tests(positions, stress_scenarios, portfolio_value=10000)
        
        assert isinstance(stress_results, dict)
        for scenario, result in stress_results.items():
            assert 'portfolio_loss' in result
            assert 'loss_percentage' in result
            assert result['portfolio_loss'] <= 0  # Should be negative (loss)

class TestRiskReporting:
    """Test risk reporting and metrics"""
    
    def test_risk_dashboard_data(self):
        """Test risk dashboard data generation"""
        from risk.risk_manager import RiskManager
        
        rm = RiskManager({})
        
        positions = [
            {'symbol': 'BTCUSDT', 'size': 0.1, 'pnl': 100, 'current_price': 51000},
            {'symbol': 'ETHUSDT', 'size': 0.2, 'pnl': -50, 'current_price': 2950}
        ]
        
        dashboard_data = rm.generate_risk_dashboard_data(positions, portfolio_value=10000)
        
        required_fields = [
            'total_exposure', 'var_95', 'max_drawdown', 'concentration_risk',
            'correlation_warnings', 'daily_pnl', 'position_count'
        ]
        
        for field in required_fields:
            assert field in dashboard_data
    
    def test_risk_alert_generation(self):
        """Test risk alert generation"""
        from risk.risk_manager import RiskManager
        
        config = {
            'max_daily_loss': 0.03,
            'max_position_size': 0.1,
            'correlation_limit': 0.8
        }
        rm = RiskManager(config)
        
        # Positions that breach multiple limits
        risky_positions = [
            {'symbol': 'BTCUSDT', 'size': 0.15, 'pnl': -400},  # Too large, losing
            {'symbol': 'ETHUSDT', 'size': 0.12, 'pnl': -200}   # Too large, losing
        ]
        
        alerts = rm.generate_risk_alerts(risky_positions, portfolio_value=10000)
        
        assert len(alerts) > 0
        assert any('position_size' in alert for alert in alerts)
        assert any('daily_loss' in alert for alert in alerts)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
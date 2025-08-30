"""
Portfolio optimization module for dynamic position sizing and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization"""
    max_position_size: float = 0.1  # 10% max per position
    max_portfolio_risk: float = 0.02  # 2% daily VaR
    rebalance_frequency: int = 24  # hours
    correlation_threshold: float = 0.7
    min_position_size: float = 0.01  # 1% minimum
    lookback_period: int = 30  # days for correlation calculation

class PortfolioOptimizer:
    """Advanced portfolio optimization using Modern Portfolio Theory"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.position_history = {}
        self.correlation_matrix = None
        self.last_optimization = None
        
    async def optimize_positions(self, 
                               current_positions: List[Dict],
                               signals: List[Dict],
                               market_data: Dict[str, pd.DataFrame],
                               portfolio_value: float) -> Dict[str, float]:
        """
        Optimize position sizes based on signals and risk constraints
        
        Returns:
            Dict mapping symbol to optimal position size (as fraction of portfolio)
        """
        try:
            # Calculate returns and correlations
            returns_data = await self._calculate_returns(market_data)
            correlation_matrix = await self._calculate_correlations(returns_data)
            
            # Get signal strengths
            signal_strengths = self._process_signals(signals)
            
            # Current exposures
            current_exposures = self._get_current_exposures(current_positions, portfolio_value)
            
            # Optimize allocation
            optimal_weights = await self._optimize_weights(
                signal_strengths,
                correlation_matrix,
                returns_data,
                current_exposures
            )
            
            # Apply risk constraints
            constrained_weights = self._apply_risk_constraints(optimal_weights, correlation_matrix)
            
            logger.info(f"Portfolio optimization completed. Symbols: {len(constrained_weights)}")
            return constrained_weights
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {}
    
    async def _calculate_returns(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate returns for each symbol"""
        returns_data = {}
        
        for symbol, df in market_data.items():
            if len(df) < 2:
                continue
                
            # Calculate log returns
            df = df.sort_values('timestamp')
            returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            returns_data[symbol] = returns
        
        return pd.DataFrame(returns_data).fillna(0)
    
    async def _calculate_correlations(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate correlation matrix using Ledoit-Wolf shrinkage"""
        if returns_data.empty or len(returns_data.columns) < 2:
            return np.eye(1)
            
        try:
            # Use Ledoit-Wolf shrinkage for better correlation estimation
            lw = LedoitWolf()
            correlation_matrix = lw.fit(returns_data).covariance_
            
            # Normalize to correlation matrix
            D = np.diag(1 / np.sqrt(np.diag(correlation_matrix)))
            correlation_matrix = D @ correlation_matrix @ D
            
            self.correlation_matrix = correlation_matrix
            return correlation_matrix
            
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return np.eye(len(returns_data.columns))
    
    def _process_signals(self, signals: List[Dict]) -> Dict[str, float]:
        """Process trading signals to get signal strengths"""
        signal_strengths = {}
        
        for signal in signals:
            symbol = signal.get('symbol')
            signal_type = signal.get('signal_type', 'hold')
            confidence = signal.get('confidence', 0.0)
            
            if signal_type == 'buy':
                strength = confidence
            elif signal_type == 'sell':
                strength = -confidence
            else:  # hold
                strength = 0.0
            
            # Aggregate multiple signals for same symbol
            if symbol in signal_strengths:
                signal_strengths[symbol] = (signal_strengths[symbol] + strength) / 2
            else:
                signal_strengths[symbol] = strength
        
        return signal_strengths
    
    def _get_current_exposures(self, positions: List[Dict], portfolio_value: float) -> Dict[str, float]:
        """Get current position exposures as portfolio fractions"""
        exposures = {}
        
        for position in positions:
            if not position.get('is_open', True):
                continue
                
            symbol = position['symbol']
            position_value = position['size'] * position.get('current_price', position['entry_price'])
            exposure = position_value / portfolio_value if portfolio_value > 0 else 0
            
            exposures[symbol] = exposure
        
        return exposures
    
    async def _optimize_weights(self, 
                               signal_strengths: Dict[str, float],
                               correlation_matrix: np.ndarray,
                               returns_data: pd.DataFrame,
                               current_exposures: Dict[str, float]) -> Dict[str, float]:
        """Optimize portfolio weights using mean-variance optimization"""
        
        symbols = list(signal_strengths.keys())
        if not symbols:
            return {}
        
        # Expected returns based on signals
        expected_returns = np.array([signal_strengths.get(symbol, 0) for symbol in symbols])
        
        # Risk model
        if len(symbols) == 1:
            risk_matrix = np.array([[0.01]])  # Single asset
        else:
            # Use correlation matrix subset
            symbol_indices = [i for i, col in enumerate(returns_data.columns) if col in symbols]
            if len(symbol_indices) == len(symbols):
                risk_matrix = correlation_matrix[np.ix_(symbol_indices, symbol_indices)]
            else:
                risk_matrix = np.eye(len(symbols)) * 0.01
        
        # Objective function: maximize return - risk penalty
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(risk_matrix, weights)))
            return -(portfolio_return - 0.5 * portfolio_risk)  # Risk-adjusted return
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1.0},  # Sum of absolute weights = 1
        ]
        
        # Bounds for each weight
        bounds = [(-self.config.max_position_size, self.config.max_position_size) for _ in symbols]
        
        # Initial guess - equal weight
        x0 = np.array([1.0/len(symbols)] * len(symbols))
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = dict(zip(symbols, result.x))
            return optimal_weights
        else:
            logger.warning("Optimization failed, using signal-based weights")
            return self._fallback_weights(signal_strengths)
    
    def _fallback_weights(self, signal_strengths: Dict[str, float]) -> Dict[str, float]:
        """Fallback weight calculation based on signal strengths"""
        total_strength = sum(abs(s) for s in signal_strengths.values())
        if total_strength == 0:
            return {}
        
        weights = {}
        for symbol, strength in signal_strengths.items():
            weight = (strength / total_strength) * 0.8  # Conservative scaling
            weights[symbol] = max(min(weight, self.config.max_position_size), -self.config.max_position_size)
        
        return weights
    
    def _apply_risk_constraints(self, weights: Dict[str, float], correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Apply risk constraints to portfolio weights"""
        constrained_weights = weights.copy()
        
        # Check correlation constraints
        symbols = list(weights.keys())
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                if i < len(correlation_matrix) and j < len(correlation_matrix):
                    correlation = correlation_matrix[i][j]
                    
                    if abs(correlation) > self.config.correlation_threshold:
                        # Reduce weights for highly correlated positions
                        reduction_factor = 1 - (abs(correlation) - self.config.correlation_threshold)
                        constrained_weights[symbol1] *= reduction_factor
                        constrained_weights[symbol2] *= reduction_factor
        
        # Apply individual position limits
        for symbol in constrained_weights:
            weight = constrained_weights[symbol]
            constrained_weights[symbol] = max(
                min(weight, self.config.max_position_size), 
                -self.config.max_position_size
            )
            
            # Remove tiny positions
            if abs(constrained_weights[symbol]) < self.config.min_position_size:
                constrained_weights[symbol] = 0.0
        
        return constrained_weights
    
    def calculate_position_size(self, 
                               symbol: str, 
                               signal_confidence: float,
                               portfolio_value: float,
                               volatility: float,
                               optimal_weight: float = None) -> float:
        """
        Calculate optimal position size using Kelly Criterion and volatility adjustment
        """
        try:
            if optimal_weight is not None:
                # Use portfolio optimization result
                base_size = abs(optimal_weight) * portfolio_value
            else:
                # Fallback to Kelly Criterion
                # Simplified Kelly: f = (bp - q) / b
                # Where b = odds, p = win probability, q = loss probability
                win_rate = 0.55  # Conservative assumption
                avg_win = 0.03   # 3% average win
                avg_loss = 0.02  # 2% average loss
                
                kelly_fraction = ((win_rate * avg_win) - ((1 - win_rate) * avg_loss)) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                
                base_size = kelly_fraction * signal_confidence * portfolio_value
            
            # Volatility adjustment
            target_vol = 0.02  # 2% daily target volatility
            vol_adjustment = target_vol / max(volatility, 0.01)
            adjusted_size = base_size * vol_adjustment
            
            # Apply position limits
            max_size = self.config.max_position_size * portfolio_value
            min_size = self.config.min_position_size * portfolio_value
            
            final_size = max(min(adjusted_size, max_size), min_size)
            
            logger.debug(f"Position size for {symbol}: ${final_size:.2f} (confidence: {signal_confidence:.2f})")
            return final_size
            
        except Exception as e:
            logger.error(f"Position size calculation failed for {symbol}: {e}")
            return self.config.min_position_size * portfolio_value
    
    def calculate_portfolio_risk(self, positions: List[Dict], market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics"""
        try:
            if not positions:
                return {'var_95': 0.0, 'expected_shortfall': 0.0, 'max_drawdown': 0.0}
            
            # Calculate individual position risks
            position_risks = []
            for position in positions:
                symbol = position['symbol']
                if symbol in market_data and len(market_data[symbol]) > 1:
                    returns = market_data[symbol]['close'].pct_change().dropna()
                    position_var = np.percentile(returns, 5) * position['size'] * position['current_price']
                    position_risks.append(position_var)
            
            if not position_risks:
                return {'var_95': 0.0, 'expected_shortfall': 0.0, 'max_drawdown': 0.0}
            
            # Portfolio VaR (simplified - assumes correlation)
            portfolio_var = np.sqrt(np.sum(np.array(position_risks) ** 2)) * 0.8  # Correlation adjustment
            
            # Expected shortfall (conditional VaR)
            expected_shortfall = portfolio_var * 1.2
            
            return {
                'var_95': abs(portfolio_var),
                'expected_shortfall': abs(expected_shortfall),
                'max_drawdown': self._calculate_max_drawdown(positions)
            }
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return {'var_95': 0.0, 'expected_shortfall': 0.0, 'max_drawdown': 0.0}
    
    def _calculate_max_drawdown(self, positions: List[Dict]) -> float:
        """Calculate maximum drawdown from position history"""
        try:
            # Simplified calculation - would need historical portfolio values
            total_pnl = sum(pos.get('pnl', 0) for pos in positions)
            return abs(min(0, total_pnl)) / max(sum(pos.get('size', 0) * pos.get('entry_price', 0) for pos in positions), 1)
        except:
            return 0.0
    
    def should_rebalance(self, 
                        current_weights: Dict[str, float], 
                        optimal_weights: Dict[str, float]) -> bool:
        """Determine if portfolio should be rebalanced"""
        if not self.last_optimization:
            return True
        
        # Check time since last optimization
        time_diff = datetime.utcnow() - self.last_optimization
        if time_diff.total_seconds() / 3600 >= self.config.rebalance_frequency:
            return True
        
        # Check weight drift
        max_drift = 0.05  # 5% maximum drift before rebalancing
        for symbol in optimal_weights:
            current_weight = current_weights.get(symbol, 0)
            optimal_weight = optimal_weights[symbol]
            
            if abs(current_weight - optimal_weight) > max_drift:
                return True
        
        return False
    
    def get_rebalancing_trades(self, 
                              current_positions: Dict[str, float],
                              target_weights: Dict[str, float],
                              portfolio_value: float) -> List[Dict]:
        """Generate trades needed for rebalancing"""
        trades = []
        
        # Get all symbols (current + target)
        all_symbols = set(current_positions.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_positions.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            
            weight_diff = target_weight - current_weight
            
            # Skip tiny changes
            if abs(weight_diff) < self.config.min_position_size:
                continue
            
            # Calculate trade size in dollars
            trade_value = abs(weight_diff) * portfolio_value
            
            # Determine trade side
            if weight_diff > 0:
                side = 'buy'
            else:
                side = 'sell'
            
            trades.append({
                'symbol': symbol,
                'side': side,
                'value': trade_value,
                'weight_change': weight_diff,
                'priority': abs(weight_diff)  # Higher priority for larger changes
            })
        
        # Sort by priority
        trades.sort(key=lambda x: x['priority'], reverse=True)
        
        return trades
    
    def calculate_diversification_score(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio diversification score (0-1, higher is better)"""
        if not weights or len(weights) <= 1:
            return 0.0
        
        # Herfindahl-Hirschman Index for concentration
        weight_values = list(weights.values())
        hhi = sum(w**2 for w in weight_values)
        
        # Normalize to 0-1 scale (1 = perfectly diversified)
        max_hhi = 1.0  # All weight in one asset
        min_hhi = 1.0 / len(weights)  # Equal weight
        
        diversification_score = (max_hhi - hhi) / (max_hhi - min_hhi)
        return max(0, min(1, diversification_score))
    
    def check_concentration_risk(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Check for concentration risk in portfolio"""
        warnings = []
        
        # Check individual position sizes
        for symbol, weight in weights.items():
            if abs(weight) > self.config.max_position_size:
                warnings.append(f"{symbol}: Position size {weight:.1%} exceeds limit {self.config.max_position_size:.1%}")
        
        # Check correlation clustering
        if self.correlation_matrix is not None:
            high_corr_pairs = []
            symbols = list(weights.keys())
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if (i < len(self.correlation_matrix) and 
                        j < len(self.correlation_matrix) and
                        abs(self.correlation_matrix[i][j]) > self.config.correlation_threshold):
                        high_corr_pairs.append((symbol1, symbol2, self.correlation_matrix[i][j]))
            
            if high_corr_pairs:
                warnings.append(f"High correlation detected: {len(high_corr_pairs)} pairs above {self.config.correlation_threshold}")
        
        # Calculate diversification
        diversification = self.calculate_diversification_score(weights)
        
        return {
            'warnings': warnings,
            'diversification_score': diversification,
            'concentration_risk': diversification < 0.5,
            'total_exposure': sum(abs(w) for w in weights.values())
        }
    
    async def get_risk_budget_allocation(self, 
                                        signals: Dict[str, float],
                                        volatilities: Dict[str, float]) -> Dict[str, float]:
        """Allocate risk budget based on signal strength and volatility"""
        risk_allocations = {}
        total_risk_budget = self.config.max_portfolio_risk
        
        # Calculate risk contribution per symbol
        for symbol, signal_strength in signals.items():
            if symbol not in volatilities:
                continue
            
            volatility = volatilities[symbol]
            
            # Risk contribution = signal_strength / volatility
            risk_contribution = abs(signal_strength) / max(volatility, 0.01)
            risk_allocations[symbol] = risk_contribution
        
        # Normalize to risk budget
        total_risk = sum(risk_allocations.values())
        if total_risk > 0:
            risk_scalar = total_risk_budget / total_risk
            for symbol in risk_allocations:
                risk_allocations[symbol] *= risk_scalar
        
        return risk_allocations
    
    def update_optimization_history(self, weights: Dict[str, float]):
        """Update optimization history"""
        self.last_optimization = datetime.utcnow()
        
        # Store position history for analysis
        for symbol, weight in weights.items():
            if symbol not in self.position_history:
                self.position_history[symbol] = []
            
            self.position_history[symbol].append({
                'timestamp': self.last_optimization,
                'weight': weight
            })
            
            # Keep only recent history
            cutoff = self.last_optimization - timedelta(days=self.config.lookback_period)
            self.position_history[symbol] = [
                h for h in self.position_history[symbol] 
                if h['timestamp'] > cutoff
            ]
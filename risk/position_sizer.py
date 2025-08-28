"""
Position Sizer - Advanced position sizing algorithms
Implements various position sizing methods for optimal risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PositionSizer:
    """Advanced position sizing calculator"""
    
    def __init__(self, portfolio_manager=None):
        self.portfolio_manager = portfolio_manager
        
        # Default risk parameters
        self.default_risk_pct = 0.02  # 2% risk per trade
        self.max_position_pct = 0.10  # 10% max position size
        self.min_position_usd = 10    # $10 minimum position
        self.max_leverage = 3.0       # 3x max leverage
        
        # Position sizing methods
        self.methods = {
            'fixed_percent': self.fixed_percent_sizing,
            'risk_parity': self.risk_parity_sizing,
            'kelly_criterion': self.kelly_criterion_sizing,
            'volatility_adjusted': self.volatility_adjusted_sizing,
            'confidence_weighted': self.confidence_weighted_sizing,
            'martingale': self.martingale_sizing,
            'anti_martingale': self.anti_martingale_sizing,
            'optimal_f': self.optimal_f_sizing
        }
        
        # Performance tracking for Kelly Criterion
        self.trade_history = []
        self.max_history_length = 100
    
    def calculate_position_size(self, 
                              signal_data: Dict[str, Any], 
                              method: str = 'volatility_adjusted',
                              custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate position size using specified method"""
        try:
            # Extract signal information
            symbol = signal_data.get('symbol', '')
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit = signal_data.get('take_profit', 0)
            confidence = signal_data.get('confidence', 0.5)
            side = signal_data.get('side', 'long')
            strategy = signal_data.get('strategy', '')
            
            # Get portfolio information
            portfolio_value = self._get_portfolio_value()
            available_balance = self._get_available_balance()
            
            if portfolio_value <= 0 or entry_price <= 0:
                return self._create_result(0, "Invalid portfolio value or entry price")
            
            # Calculate using specified method
            if method in self.methods:
                result = self.methods[method](
                    signal_data, 
                    portfolio_value, 
                    available_balance, 
                    custom_params or {}
                )
            else:
                logger.warning(f"Unknown position sizing method: {method}, using default")
                result = self.volatility_adjusted_sizing(signal_data, portfolio_value, available_balance, {})
            
            # Apply final validations and adjustments
            final_result = self._apply_final_validations(result, signal_data, available_balance)
            
            logger.info(f"Position size calculated for {symbol}: {final_result['position_size']:.6f} (method: {method})")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return self._create_result(0, f"Calculation error: {e}")
    
    def fixed_percent_sizing(self, signal_data: Dict[str, Any], 
                           portfolio_value: float, 
                           available_balance: float, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        """Fixed percentage of portfolio"""
        try:
            risk_percent = params.get('risk_percent', self.default_risk_pct)
            entry_price = signal_data['entry_price']
            
            # Calculate position size as fixed percentage of portfolio
            risk_amount = portfolio_value * risk_percent
            position_size = risk_amount / entry_price
            
            return self._create_result(
                position_size, 
                f"Fixed {risk_percent:.1%} of portfolio",
                {'method': 'fixed_percent', 'risk_amount': risk_amount}
            )
            
        except Exception as e:
            logger.error(f"Fixed percent sizing error: {e}")
            return self._create_result(0, f"Fixed percent error: {e}")
    
    def risk_parity_sizing(self, signal_data: Dict[str, Any], 
                          portfolio_value: float, 
                          available_balance: float, 
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Risk parity based on stop loss distance"""
        try:
            entry_price = signal_data['entry_price']
            stop_loss = signal_data.get('stop_loss', 0)
            side = signal_data.get('side', 'long')
            
            # Calculate stop distance
            if stop_loss > 0:
                if side == 'long':
                    stop_distance = entry_price - stop_loss
                else:  # short
                    stop_distance = stop_loss - entry_price
            else:
                # Default 3% stop loss
                stop_distance = entry_price * 0.03
            
            if stop_distance <= 0:
                return self._create_result(0, "Invalid stop loss distance")
            
            # Risk amount (default 2% of portfolio)
            risk_percent = params.get('risk_percent', self.default_risk_pct)
            risk_amount = portfolio_value * risk_percent
            
            # Position size = Risk Amount / Stop Distance
            position_size = risk_amount / stop_distance
            
            return self._create_result(
                position_size,
                f"Risk parity: ${risk_amount:.2f} risk / ${stop_distance:.4f} stop distance",
                {
                    'method': 'risk_parity',
                    'risk_amount': risk_amount,
                    'stop_distance': stop_distance,
                    'risk_per_unit': stop_distance
                }
            )
            
        except Exception as e:
            logger.error(f"Risk parity sizing error: {e}")
            return self._create_result(0, f"Risk parity error: {e}")
    
    def kelly_criterion_sizing(self, signal_data: Dict[str, Any], 
                             portfolio_value: float, 
                             available_balance: float, 
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """Kelly Criterion based on historical performance"""
        try:
            if len(self.trade_history) < 10:
                # Fallback to risk parity if insufficient history
                return self.risk_parity_sizing(signal_data, portfolio_value, available_balance, params)
            
            # Calculate win rate and average win/loss
            wins = [trade for trade in self.trade_history if trade['pnl'] > 0]
            losses = [trade for trade in self.trade_history if trade['pnl'] < 0]
            
            if len(losses) == 0:
                # No losses yet, use conservative sizing
                return self.fixed_percent_sizing(signal_data, portfolio_value, available_balance, {'risk_percent': 0.01})
            
            win_rate = len(wins) / len(self.trade_history)
            avg_win = np.mean([trade['pnl_pct'] for trade in wins]) if wins else 0
            avg_loss = abs(np.mean([trade['pnl_pct'] for trade in losses]))
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            if avg_loss > 0:
                b = avg_win / avg_loss
                p = win_rate
                q = 1 - win_rate
                kelly_fraction = (b * p - q) / b
            else:
                kelly_fraction = 0.02
            
            # Apply conservative cap (max 25% Kelly)
            kelly_fraction = max(0.005, min(0.05, kelly_fraction * 0.25))
            
            entry_price = signal_data['entry_price']
            position_value = portfolio_value * kelly_fraction
            position_size = position_value / entry_price
            
            return self._create_result(
                position_size,
                f"Kelly Criterion: {kelly_fraction:.1%} of portfolio (WR:{win_rate:.1%}, Ratio:{avg_win/avg_loss:.2f})",
                {
                    'method': 'kelly_criterion',
                    'kelly_fraction': kelly_fraction,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'trade_count': len(self.trade_history)
                }
            )
            
        except Exception as e:
            logger.error(f"Kelly criterion sizing error: {e}")
            return self._create_result(0, f"Kelly criterion error: {e}")
    
    def volatility_adjusted_sizing(self, signal_data: Dict[str, Any], 
                                 portfolio_value: float, 
                                 available_balance: float, 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Volatility-adjusted position sizing"""
        try:
            # Get volatility estimate (simplified - would use ATR or realized vol in practice)
            symbol = signal_data.get('symbol', '')
            entry_price = signal_data['entry_price']
            
            # Default volatility estimate (2% daily)
            volatility_estimate = params.get('volatility_estimate', 0.02)
            
            # Get ATR-based volatility if available
            if 'atr' in signal_data and signal_data['atr'] > 0:
                volatility_estimate = signal_data['atr'] / entry_price
            
            # Base risk amount
            base_risk_pct = params.get('base_risk_pct', self.default_risk_pct)
            
            # Adjust for volatility
            # Higher volatility = smaller position size
            vol_adjustment = min(2.0, max(0.5, 0.02 / volatility_estimate))
            adjusted_risk_pct = base_risk_pct * vol_adjustment
            
            # Calculate position size
            risk_amount = portfolio_value * adjusted_risk_pct
            position_size = risk_amount / entry_price
            
            return self._create_result(
                position_size,
                f"Volatility adjusted: {adjusted_risk_pct:.2%} (vol: {volatility_estimate:.2%}, adj: {vol_adjustment:.2f}x)",
                {
                    'method': 'volatility_adjusted',
                    'volatility_estimate': volatility_estimate,
                    'vol_adjustment': vol_adjustment,
                    'base_risk_pct': base_risk_pct,
                    'adjusted_risk_pct': adjusted_risk_pct
                }
            )
            
        except Exception as e:
            logger.error(f"Volatility adjusted sizing error: {e}")
            return self._create_result(0, f"Volatility adjusted error: {e}")
    
    def confidence_weighted_sizing(self, signal_data: Dict[str, Any], 
                                 portfolio_value: float, 
                                 available_balance: float, 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Size positions based on signal confidence"""
        try:
            confidence = signal_data.get('confidence', 0.5)
            entry_price = signal_data['entry_price']
            
            # Base risk
            base_risk_pct = params.get('base_risk_pct', self.default_risk_pct)
            
            # Confidence scaling (0.5 confidence = 50% of base size)
            confidence_multiplier = max(0.25, min(2.0, confidence * 2))
            adjusted_risk_pct = base_risk_pct * confidence_multiplier
            
            # Additional boost for very high confidence
            if confidence > 0.9:
                adjusted_risk_pct *= 1.2
            
            risk_amount = portfolio_value * adjusted_risk_pct
            position_size = risk_amount / entry_price
            
            return self._create_result(
                position_size,
                f"Confidence weighted: {adjusted_risk_pct:.2%} (confidence: {confidence:.1%}, mult: {confidence_multiplier:.2f}x)",
                {
                    'method': 'confidence_weighted',
                    'confidence': confidence,
                    'confidence_multiplier': confidence_multiplier,
                    'base_risk_pct': base_risk_pct,
                    'adjusted_risk_pct': adjusted_risk_pct
                }
            )
            
        except Exception as e:
            logger.error(f"Confidence weighted sizing error: {e}")
    def martingale_sizing(self, signal_data: Dict[str, Any], 
                        portfolio_value: float, 
                        available_balance: float, 
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Martingale sizing - increase size after losses"""
        try:
            entry_price = signal_data['entry_price']
            base_risk_pct = params.get('base_risk_pct', self.default_risk_pct)
            
            # Count recent consecutive losses
            consecutive_losses = 0
            for trade in reversed(self.trade_history[-10:]):  # Check last 10 trades
                if trade['pnl'] < 0:
                    consecutive_losses += 1
                else:
                    break
            
            # Increase size after losses (dangerous - use with caution!)
            martingale_multiplier = min(3.0, 1.0 + (consecutive_losses * 0.5))
            adjusted_risk_pct = base_risk_pct * martingale_multiplier
            
            # Cap at maximum risk
            max_risk_pct = params.get('max_risk_pct', 0.05)  # 5% max
            adjusted_risk_pct = min(adjusted_risk_pct, max_risk_pct)
            
            risk_amount = portfolio_value * adjusted_risk_pct
            position_size = risk_amount / entry_price
            
            return self._create_result(
                position_size,
                f"Martingale: {adjusted_risk_pct:.2%} (losses: {consecutive_losses}, mult: {martingale_multiplier:.2f}x)",
                {
                    'method': 'martingale',
                    'consecutive_losses': consecutive_losses,
                    'martingale_multiplier': martingale_multiplier,
                    'adjusted_risk_pct': adjusted_risk_pct,
                    'warning': 'High risk method - use with extreme caution'
                }
            )
            
        except Exception as e:
            logger.error(f"Martingale sizing error: {e}")
            return self._create_result(0, f"Martingale error: {e}")
    
    def anti_martingale_sizing(self, signal_data: Dict[str, Any], 
                             portfolio_value: float, 
                             available_balance: float, 
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """Anti-martingale sizing - increase size after wins"""
        try:
            entry_price = signal_data['entry_price']
            base_risk_pct = params.get('base_risk_pct', self.default_risk_pct)
            
            # Count recent consecutive wins
            consecutive_wins = 0
            for trade in reversed(self.trade_history[-10:]):
                if trade['pnl'] > 0:
                    consecutive_wins += 1
                else:
                    break
            
            # Increase size after wins
            anti_martingale_multiplier = min(2.0, 1.0 + (consecutive_wins * 0.25))
            adjusted_risk_pct = base_risk_pct * anti_martingale_multiplier
            
            risk_amount = portfolio_value * adjusted_risk_pct
            position_size = risk_amount / entry_price
            
            return self._create_result(
                position_size,
                f"Anti-martingale: {adjusted_risk_pct:.2%} (wins: {consecutive_wins}, mult: {anti_martingale_multiplier:.2f}x)",
                {
                    'method': 'anti_martingale',
                    'consecutive_wins': consecutive_wins,
                    'anti_martingale_multiplier': anti_martingale_multiplier,
                    'adjusted_risk_pct': adjusted_risk_pct
                }
            )
            
        except Exception as e:
            logger.error(f"Anti-martingale sizing error: {e}")
            return self._create_result(0, f"Anti-martingale error: {e}")
    
    def optimal_f_sizing(self, signal_data: Dict[str, Any], 
                        portfolio_value: float, 
                        available_balance: float, 
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimal F sizing (similar to Kelly but for fixed fractional)"""
        try:
            if len(self.trade_history) < 20:
                return self.volatility_adjusted_sizing(signal_data, portfolio_value, available_balance, params)
            
            entry_price = signal_data['entry_price']
            
            # Calculate returns distribution
            returns = [trade['pnl_pct'] for trade in self.trade_history[-50:]]  # Last 50 trades
            
            # Find optimal f that maximizes geometric mean
            best_f = 0.02
            best_geom_mean = -float('inf')
            
            # Test different f values
            for f in np.arange(0.005, 0.1, 0.005):
                geom_returns = []
                for ret in returns:
                    # Geometric return with fraction f
                    geom_ret = 1 + (f * ret)
                    if geom_ret > 0:
                        geom_returns.append(geom_ret)
                
                if geom_returns:
                    geom_mean = np.exp(np.mean(np.log(geom_returns))) - 1
                    if geom_mean > best_geom_mean:
                        best_geom_mean = geom_mean
                        best_f = f
            
            # Conservative adjustment
            optimal_f = best_f * 0.5  # Use half of optimal for safety
            
            risk_amount = portfolio_value * optimal_f
            position_size = risk_amount / entry_price
            
            return self._create_result(
                position_size,
                f"Optimal F: {optimal_f:.2%} (calculated: {best_f:.2%})",
                {
                    'method': 'optimal_f',
                    'optimal_f': optimal_f,
                    'calculated_f': best_f,
                    'geometric_mean': best_geom_mean,
                    'sample_size': len(returns)
                }
            )
            
        except Exception as e:
            logger.error(f"Optimal F sizing error: {e}")
            return self._create_result(0, f"Optimal F error: {e}")
    
    def _apply_final_validations(self, result: Dict[str, Any], 
                               signal_data: Dict[str, Any], 
                               available_balance: float) -> Dict[str, Any]:
        """Apply final validations and constraints"""
        try:
            position_size = result['position_size']
            entry_price = signal_data['entry_price']
            
            # Calculate position value
            position_value = position_size * entry_price
            
            # Apply minimum position size
            if position_value < self.min_position_usd:
                min_size = self.min_position_usd / entry_price
                if min_size <= available_balance / entry_price:
                    result['position_size'] = min_size
                    result['adjustments'].append(f"Increased to minimum ${self.min_position_usd}")
                else:
                    result['position_size'] = 0
                    result['reason'] = "Below minimum size and insufficient balance"
                    return result
            
            # Apply maximum position size
            portfolio_value = self._get_portfolio_value()
            max_position_value = portfolio_value * self.max_position_pct
            
            if position_value > max_position_value:
                result['position_size'] = max_position_value / entry_price
                result['adjustments'].append(f"Reduced to max {self.max_position_pct:.1%} of portfolio")
                position_value = max_position_value
            
            # Check available balance
            if position_value > available_balance * 0.95:  # Leave 5% buffer
                result['position_size'] = (available_balance * 0.95) / entry_price
                result['adjustments'].append("Reduced to available balance")
                position_value = available_balance * 0.95
            
            # Apply leverage limits
            leverage = position_value / (available_balance if available_balance > 0 else portfolio_value)
            if leverage > self.max_leverage:
                result['position_size'] = result['position_size'] / (leverage / self.max_leverage)
                result['adjustments'].append(f"Reduced to max {self.max_leverage:.1f}x leverage")
            
            # Final position value
            result['position_value'] = result['position_size'] * entry_price
            result['portfolio_percentage'] = (result['position_value'] / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            return result
            
        except Exception as e:
            logger.error(f"Final validation error: {e}")
            result['position_size'] = 0
            result['reason'] = f"Validation error: {e}"
            return result
    
    def _create_result(self, position_size: float, reason: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create standardized result dictionary"""
        return {
            'position_size': max(0, position_size),
            'reason': reason,
            'metadata': metadata or {},
            'adjustments': [],
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            if self.portfolio_manager:
                return self.portfolio_manager.get_total_balance()
            return float(config.trading.initial_capital)
        except:
            return 10000.0  # Default fallback
    
    def _get_available_balance(self) -> float:
        """Get available balance for new positions"""
        try:
            if self.portfolio_manager:
                return self.portfolio_manager.get_available_balance()
            return self._get_portfolio_value() * 0.8  # Assume 80% available
        except:
            return 8000.0  # Default fallback
    
    def record_trade_result(self, trade_data: Dict[str, Any]):
        """Record trade result for Kelly Criterion and other adaptive methods"""
        try:
            trade_record = {
                'symbol': trade_data.get('symbol', ''),
                'pnl': trade_data.get('pnl', 0),
                'pnl_pct': trade_data.get('pnl_pct', 0),
                'duration': trade_data.get('duration_minutes', 0),
                'strategy': trade_data.get('strategy', ''),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.trade_history.append(trade_record)
            
            # Keep only recent history
            if len(self.trade_history) > self.max_history_length:
                self.trade_history.pop(0)
            
            logger.debug(f"Recorded trade result: {trade_record['symbol']} PnL: {trade_record['pnl']:.2f}")
            
        except Exception as e:
            logger.error(f"Trade result recording failed: {e}")
    
    def get_sizing_recommendation(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get position sizing recommendation with multiple methods"""
        try:
            symbol = signal_data.get('symbol', '')
            confidence = signal_data.get('confidence', 0.5)
            
            # Calculate using multiple methods
            methods_to_try = ['volatility_adjusted', 'confidence_weighted', 'risk_parity']
            
            if len(self.trade_history) >= 10:
                methods_to_try.append('kelly_criterion')
            
            results = {}
            for method in methods_to_try:
                try:
                    result = self.calculate_position_size(signal_data, method)
                    results[method] = result
                except Exception as e:
                    logger.warning(f"Method {method} failed: {e}")
                    continue
            
            if not results:
                return {'error': 'All sizing methods failed'}
            
            # Select best method based on context
            if confidence > 0.8 and 'confidence_weighted' in results:
                recommended_method = 'confidence_weighted'
            elif len(self.trade_history) >= 20 and 'kelly_criterion' in results:
                recommended_method = 'kelly_criterion'
            else:
                recommended_method = 'volatility_adjusted'
            
            recommendation = results.get(recommended_method, list(results.values())[0])
            
            return {
                'recommended_size': recommendation['position_size'],
                'recommended_method': recommended_method,
                'recommendation_reason': recommendation['reason'],
                'all_methods': results,
                'risk_assessment': self._assess_position_risk(recommendation, signal_data)
            }
            
        except Exception as e:
            logger.error(f"Sizing recommendation failed: {e}")
            return {'error': str(e)}
    
    def _assess_position_risk(self, sizing_result: Dict[str, Any], signal_data: Dict[str, Any]) -> str:
        """Assess the risk level of the position size"""
        try:
            portfolio_pct = sizing_result.get('portfolio_percentage', 0)
            position_size = sizing_result.get('position_size', 0)
            
            if portfolio_pct == 0 or position_size == 0:
                return "NO_POSITION"
            elif portfolio_pct < 2:
                return "LOW_RISK"
            elif portfolio_pct < 5:
                return "MEDIUM_RISK"
            elif portfolio_pct < 8:
                return "HIGH_RISK"
            else:
                return "VERY_HIGH_RISK"
                
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return "UNKNOWN_RISK"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for position sizing"""
        try:
            if not self.trade_history:
                return {'message': 'No trade history available'}
            
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
            losing_trades = total_trades - winning_trades
            
            total_pnl = sum(t['pnl'] for t in self.trade_history)
            avg_win = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] < 0]) if losing_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'avg_trade': total_pnl / total_trades if total_trades > 0 else 0,
                'max_history_length': self.max_history_length,
                'recommended_methods': self._get_recommended_methods()
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _get_recommended_methods(self) -> List[str]:
        """Get recommended sizing methods based on current performance"""
        recommendations = ['volatility_adjusted']  # Always available
        
        if len(self.trade_history) >= 10:
            recommendations.append('kelly_criterion')
            recommendations.append('confidence_weighted')
        
        if len(self.trade_history) >= 20:
            recommendations.append('optimal_f')
        
        return recommendations
    
    def update_risk_parameters(self, new_params: Dict[str, Any]):
        """Update risk parameters"""
        try:
            if 'default_risk_pct' in new_params:
                self.default_risk_pct = max(0.005, min(0.1, new_params['default_risk_pct']))
            
            if 'max_position_pct' in new_params:
                self.max_position_pct = max(0.01, min(0.5, new_params['max_position_pct']))
            
            if 'min_position_usd' in new_params:
                self.min_position_usd = max(1, new_params['min_position_usd'])
            
            if 'max_leverage' in new_params:
                self.max_leverage = max(1.0, min(10.0, new_params['max_leverage']))
            
            logger.info(f"Risk parameters updated: {new_params}")
            
        except Exception as e:
            logger.error(f"Risk parameter update failed: {e}")
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current position sizing settings"""
        return {
            'default_risk_pct': self.default_risk_pct,
            'max_position_pct': self.max_position_pct,
            'min_position_usd': self.min_position_usd,
            'max_leverage': self.max_leverage,
            'available_methods': list(self.methods.keys()),
            'trade_history_count': len(self.trade_history),
            'portfolio_value': self._get_portfolio_value(),
            'available_balance': self._get_available_balance()
        }
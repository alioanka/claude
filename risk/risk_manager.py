"""
Risk Manager - Comprehensive risk management system
Handles position sizing, portfolio risk, and trade validation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from config.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RiskManager:
    """Main risk management class"""
    
    def __init__(self, portfolio_manager=None):
        self.portfolio_manager = portfolio_manager
        
        # Risk limits from config
        self.risk_limits = {
            'max_position_size_percent': 10,  # 10% max per position
            'max_total_exposure_percent': 80,  # 80% max total exposure
            'max_daily_loss_percent': 5,      # 5% max daily loss
            'max_correlation': 0.7,           # 70% max correlation
            'min_position_size_usd': 10       # $10 minimum position
        }
        
        # Load risk rules from config
        if hasattr(config, 'risk_rules') and config.risk_rules:
            position_sizing = config.risk_rules.get('position_sizing', {})
            self.risk_limits.update({
                'max_position_size_percent': position_sizing.get('max_position_size_percent', 10),
                'max_total_exposure_percent': position_sizing.get('max_total_exposure_percent', 50),
                'min_position_size_usd': position_sizing.get('min_position_size_usd', 10)
            })
        
        # Performance tracking
        self.risk_events = []
        self.blocked_signals = 0
        self.total_signals_checked = 0
    
    async def validate_signal(self, signal) -> bool:
        """Validate trading signal against risk rules"""
        try:
            self.total_signals_checked += 1
            
            # Check if portfolio manager is available
            if not self.portfolio_manager:
                logger.warning("⚠️ No portfolio manager available for risk check")
                return True  # Allow trade if no risk manager
            
            # Get current portfolio state
            total_balance = self.portfolio_manager.get_total_balance()
            available_balance = self.portfolio_manager.get_available_balance()
            
            # Skip validation if no balance info
            if total_balance <= 0:
                return True
            
            # Calculate position size if not provided
            if not hasattr(signal, 'position_size') or not signal.position_size:
                position_size = await self.calculate_position_size(signal, available_balance)
                signal.position_size = position_size
            else:
                position_size = signal.position_size
            
            # Validation checks
            validation_checks = [
                self._check_minimum_position_size(position_size),
                self._check_maximum_position_size(signal, total_balance),
                self._check_available_balance(signal, available_balance),
                await self._check_total_exposure(signal, total_balance),
                await self._check_daily_loss_limit(),
                await self._check_correlation_risk(signal)
            ]
            
            # All checks must pass
            all_passed = all(validation_checks)
            
            if not all_passed:
                self.blocked_signals += 1
                logger.warning(f"🚫 Signal blocked by risk management: {signal.symbol}")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Risk validation error: {e}")
            # In case of error, be conservative and block the trade
            return False
    
    def _check_minimum_position_size(self, position_size: float) -> bool:
        """Check minimum position size"""
        min_size = self.risk_limits['min_position_size_usd']
        if position_size < min_size:
            logger.debug(f"Position size ${position_size:.2f} below minimum ${min_size}")
            return False
        return True
    
    def _check_maximum_position_size(self, signal, total_balance: float) -> bool:
        """Check maximum position size"""
        if not hasattr(signal, 'entry_price') or not signal.entry_price:
            return True
        
        position_value = signal.position_size * signal.entry_price
        max_position_value = total_balance * (self.risk_limits['max_position_size_percent'] / 100)
        
        if position_value > max_position_value:
            logger.debug(f"Position value ${position_value:.2f} exceeds maximum ${max_position_value:.2f}")
            return False
        return True
    
    def _check_available_balance(self, signal, available_balance: float) -> bool:
        """Check if sufficient balance available"""
        if not hasattr(signal, 'entry_price') or not signal.entry_price:
            return True
        
        required_balance = signal.position_size * signal.entry_price
        
        if required_balance > available_balance:
            logger.debug(f"Required balance ${required_balance:.2f} exceeds available ${available_balance:.2f}")
            return False
        return True
    
    async def _check_total_exposure(self, signal, total_balance: float) -> bool:
        """Check total portfolio exposure"""
        try:
            if not self.portfolio_manager:
                return True
            
            current_exposure = self.portfolio_manager.used_balance
            new_position_value = 0
            
            if hasattr(signal, 'entry_price') and signal.entry_price:
                new_position_value = signal.position_size * signal.entry_price
            
            total_exposure = current_exposure + new_position_value
            max_exposure = total_balance * (self.risk_limits['max_total_exposure_percent'] / 100)
            
            if total_exposure > max_exposure:
                logger.debug(f"Total exposure ${total_exposure:.2f} would exceed maximum ${max_exposure:.2f}")
                return False
            return True
            
        except Exception as e:
            logger.error(f"❌ Exposure check error: {e}")
            return False
    
    async def _check_daily_loss_limit(self) -> bool:
        """Check daily loss limit"""
        try:
            if not self.portfolio_manager:
                return True
            
            daily_pnl_pct = self.portfolio_manager.get_daily_pnl_pct()
            max_daily_loss = -self.risk_limits['max_daily_loss_percent']
            
            if daily_pnl_pct < max_daily_loss:
                logger.warning(f"Daily loss limit reached: {daily_pnl_pct:.2f}% < {max_daily_loss:.2f}%")
                return False
            return True
            
        except Exception as e:
            logger.error(f"❌ Daily loss check error: {e}")
            return True  # Allow trade if check fails
    
    async def _check_correlation_risk(self, signal) -> bool:
        """Check correlation risk (simplified version)"""
        try:
            if not self.portfolio_manager:
                return True
            
            # Get current positions
            positions = await self.portfolio_manager.get_positions()
            
            # If no positions, no correlation risk
            if len(positions) == 0:
                return True
            
            # Simplified correlation check - limit same-sector exposure
            # In a full implementation, you'd calculate actual price correlations
            current_symbols = [pos['symbol'] for pos in positions]
            
            # Check if adding another similar symbol (very basic check)
            symbol_base = signal.symbol.replace('USDT', '').replace('BUSD', '')
            similar_count = sum(1 for sym in current_symbols 
                              if sym.replace('USDT', '').replace('BUSD', '') == symbol_base)
            
            if similar_count > 0:
                logger.debug(f"Already have position in similar asset: {symbol_base}")
                # Allow but log the correlation risk
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Correlation check error: {e}")
            return True
    
    async def calculate_position_size(self, signal, available_balance: float) -> float:
        """Calculate optimal position size"""
        try:
            if not hasattr(signal, 'entry_price') or not signal.entry_price:
                return 0.0
            
            # Get portfolio balance
            if self.portfolio_manager:
                portfolio_value = self.portfolio_manager.get_total_balance()
            else:
                portfolio_value = available_balance
            
            # Base position size (2% of portfolio risk)
            base_risk = portfolio_value * config.trading.risk_per_trade
            
            # Calculate stop distance
            stop_distance = 0
            if hasattr(signal, 'stop_loss') and signal.stop_loss:
                if signal.action == 'buy':
                    stop_distance = signal.entry_price - signal.stop_loss
                elif signal.action == 'sell':
                    stop_distance = signal.stop_loss - signal.entry_price
            else:
                # Default 3% stop loss
                stop_distance = signal.entry_price * 0.03
            
            if stop_distance <= 0:
                stop_distance = signal.entry_price * 0.03
            
            # Position size = Risk Amount / Stop Distance
            position_size = base_risk / stop_distance
            
            # Apply confidence scaling if available
            if hasattr(signal, 'confidence'):
                confidence_multiplier = max(0.5, signal.confidence)  # Minimum 50%
                position_size *= confidence_multiplier
            
            # Ensure position size limits
            max_position_value = portfolio_value * (self.risk_limits['max_position_size_percent'] / 100)
            max_position_size = max_position_value / signal.entry_price
            
            position_size = min(position_size, max_position_size)
            
            # Ensure minimum position size
            min_size_by_value = self.risk_limits['min_position_size_usd'] / signal.entry_price
            position_size = max(position_size, min_size_by_value)
            
            # Ensure we don't exceed available balance
            max_affordable = (available_balance * 0.95) / signal.entry_price  # Leave 5% buffer
            position_size = min(position_size, max_affordable)
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            return 0.0
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        try:
            metrics = {
                'total_signals_checked': self.total_signals_checked,
                'blocked_signals': self.blocked_signals,
                'block_rate': (self.blocked_signals / max(1, self.total_signals_checked)) * 100,
                'risk_limits': self.risk_limits.copy(),
                'recent_events': len(self.risk_events)
            }
            
            if self.portfolio_manager:
                metrics.update({
                    'current_exposure_pct': (self.portfolio_manager.used_balance / 
                                           max(1, self.portfolio_manager.get_total_balance())) * 100,
                    'daily_pnl_pct': self.portfolio_manager.get_daily_pnl_pct(),
                    'active_positions': self.portfolio_manager.get_position_count(),
                    'available_balance': self.portfolio_manager.get_available_balance()
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Risk metrics calculation error: {e}")
            return {'error': str(e)}
    
    def log_risk_event(self, event_type: str, details: str):
        """Log risk management event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'details': details
        }
        
        self.risk_events.append(event)
        
        # Keep only last 100 events
        if len(self.risk_events) > 100:
            self.risk_events.pop(0)
        
        logger.info(f"🛡️ Risk Event: {event_type} - {details}")
    
    def update_risk_limits(self, new_limits: Dict[str, Any]):
        """Update risk limits"""
        self.risk_limits.update(new_limits)
        self.log_risk_event("LIMITS_UPDATED", f"New limits: {new_limits}")
    
    async def emergency_stop_check(self) -> bool:
        """Check if emergency stop should be triggered"""
        try:
            if not self.portfolio_manager:
                return False
            
            # Check for severe daily loss
            daily_pnl_pct = self.portfolio_manager.get_daily_pnl_pct()
            if daily_pnl_pct < -self.risk_limits['max_daily_loss_percent']:
                self.log_risk_event("EMERGENCY_STOP", f"Daily loss: {daily_pnl_pct:.2f}%")
                return True
            
            # Check for extreme drawdown
            max_drawdown_pct = self.portfolio_manager.get_max_drawdown_pct()
            if max_drawdown_pct > 25:  # 25% maximum drawdown
                self.log_risk_event("EMERGENCY_STOP", f"High drawdown: {max_drawdown_pct:.2f}%")
                return True
            
            # Check for system errors or anomalies
            current_balance = self.portfolio_manager.get_total_balance()
            if current_balance <= 0:
                self.log_risk_event("EMERGENCY_STOP", "Zero or negative balance detected")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Emergency stop check error: {e}")
            # In case of error, trigger emergency stop for safety
            return True
    
    async def get_position_size_recommendation(self, signal, available_balance: float) -> Dict[str, Any]:
        """Get detailed position size recommendation"""
        try:
            base_size = await self.calculate_position_size(signal, available_balance)
            
            # Risk assessment
            risk_factors = []
            size_multiplier = 1.0
            
            # Confidence-based adjustment
            if hasattr(signal, 'confidence'):
                if signal.confidence < 0.7:
                    size_multiplier *= 0.8
                    risk_factors.append("Low confidence")
                elif signal.confidence > 0.9:
                    size_multiplier *= 1.1
                    risk_factors.append("High confidence")
            
            # Portfolio exposure adjustment
            if self.portfolio_manager:
                exposure_pct = (self.portfolio_manager.used_balance / 
                              max(1, self.portfolio_manager.get_total_balance())) * 100
                
                if exposure_pct > 60:
                    size_multiplier *= 0.7
                    risk_factors.append("High portfolio exposure")
                elif exposure_pct < 20:
                    size_multiplier *= 1.1
                    risk_factors.append("Low portfolio exposure")
            
            recommended_size = base_size * size_multiplier
            
            return {
                'recommended_size': recommended_size,
                'base_size': base_size,
                'size_multiplier': size_multiplier,
                'risk_factors': risk_factors,
                'risk_assessment': self.assess_trade_risk(signal, recommended_size)
            }
            
        except Exception as e:
            logger.error(f"❌ Position size recommendation failed: {e}")
            return {
                'recommended_size': 0.0,
                'error': str(e)
            }
    
    def assess_trade_risk(self, signal, position_size: float) -> str:
        """Assess overall trade risk level"""
        try:
            risk_score = 0
            
            # Position size risk
            if hasattr(signal, 'entry_price') and signal.entry_price:
                position_value = position_size * signal.entry_price
                if self.portfolio_manager:
                    total_balance = self.portfolio_manager.get_total_balance()
                    position_pct = (position_value / total_balance) * 100
                    
                    if position_pct > 15:
                        risk_score += 3
                    elif position_pct > 10:
                        risk_score += 2
                    elif position_pct > 5:
                        risk_score += 1
            
            # Confidence risk
            if hasattr(signal, 'confidence'):
                if signal.confidence < 0.6:
                    risk_score += 2
                elif signal.confidence < 0.7:
                    risk_score += 1
            
            # Market conditions risk (simplified)
            if hasattr(signal, 'symbol'):
                # Add volatility-based risk assessment
                risk_score += 1  # Default market risk
            
            # Risk level classification
            if risk_score <= 2:
                return "LOW"
            elif risk_score <= 4:
                return "MEDIUM"
            elif risk_score <= 6:
                return "HIGH"
            else:
                return "VERY_HIGH"
                
        except Exception as e:
            logger.error(f"❌ Trade risk assessment failed: {e}")
            return "UNKNOWN"
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get risk management dashboard data"""
        try:
            dashboard = {
                'risk_limits': self.risk_limits.copy(),
                'current_metrics': self.get_risk_metrics(),
                'risk_events_count': len(self.risk_events),
                'recent_events': self.risk_events[-5:] if self.risk_events else [],
                'emergency_stop_status': False  # Would be updated by emergency check
            }
            
            if self.portfolio_manager:
                dashboard.update({
                    'portfolio_health': {
                        'total_balance': self.portfolio_manager.get_total_balance(),
                        'daily_pnl_pct': self.portfolio_manager.get_daily_pnl_pct(),
                        'max_drawdown_pct': self.portfolio_manager.get_max_drawdown_pct(),
                        'position_count': self.portfolio_manager.get_position_count(),
                        'exposure_pct': (self.portfolio_manager.used_balance / 
                                       max(1, self.portfolio_manager.get_total_balance())) * 100
                    }
                })
            
            return dashboard
            
        except Exception as e:
            logger.error(f"❌ Risk dashboard generation failed: {e}")
            return {'error': str(e)}
    
    async def validate_portfolio_health(self) -> Dict[str, Any]:
        """Comprehensive portfolio health validation"""
        try:
            if not self.portfolio_manager:
                return {'status': 'unknown', 'message': 'No portfolio manager available'}
            
            warnings = []
            critical_issues = []
            
            # Check daily loss
            daily_pnl_pct = self.portfolio_manager.get_daily_pnl_pct()
            if daily_pnl_pct < -3:
                if daily_pnl_pct < -self.risk_limits['max_daily_loss_percent']:
                    critical_issues.append(f"Daily loss exceeds limit: {daily_pnl_pct:.2f}%")
                else:
                    warnings.append(f"High daily loss: {daily_pnl_pct:.2f}%")
            
            # Check drawdown
            max_drawdown_pct = self.portfolio_manager.get_max_drawdown_pct()
            if max_drawdown_pct > 10:
                if max_drawdown_pct > 20:
                    critical_issues.append(f"Extreme drawdown: {max_drawdown_pct:.2f}%")
                else:
                    warnings.append(f"High drawdown: {max_drawdown_pct:.2f}%")
            
            # Check exposure
            total_balance = self.portfolio_manager.get_total_balance()
            exposure_pct = (self.portfolio_manager.used_balance / max(1, total_balance)) * 100
            
            if exposure_pct > self.risk_limits['max_total_exposure_percent']:
                critical_issues.append(f"Portfolio overexposed: {exposure_pct:.1f}%")
            elif exposure_pct > 70:
                warnings.append(f"High portfolio exposure: {exposure_pct:.1f}%")
            
            # Check position count
            position_count = self.portfolio_manager.get_position_count()
            if position_count > self.risk_limits.get('max_positions', 10):
                warnings.append(f"Too many positions: {position_count}")
            
            # Determine overall health status
            if critical_issues:
                status = 'critical'
                message = f"Critical issues detected: {'; '.join(critical_issues)}"
            elif warnings:
                status = 'warning'
                message = f"Warnings: {'; '.join(warnings)}"
            else:
                status = 'healthy'
                message = "Portfolio health is good"
            
            return {
                'status': status,
                'message': message,
                'warnings': warnings,
                'critical_issues': critical_issues,
                'metrics': {
                    'daily_pnl_pct': daily_pnl_pct,
                    'max_drawdown_pct': max_drawdown_pct,
                    'exposure_pct': exposure_pct,
                    'position_count': position_count,
                    'total_balance': total_balance
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio health validation failed: {e}")
            return {
                'status': 'error',
                'message': f"Health check failed: {e}"
            }
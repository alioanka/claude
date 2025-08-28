"""
Performance Tracker - Tracks and analyzes bot performance
Monitors trading metrics, portfolio performance, and generates reports.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import json
import os

from utils.logger import setup_logger

logger = setup_logger(__name__)

class PerformanceTracker:
    """Main performance tracking class"""
    
    def __init__(self, portfolio_manager=None):
        self.portfolio_manager = portfolio_manager
        
        # Performance data storage
        self.daily_snapshots = []
        self.trade_records = []
        self.strategy_performance = {}
        
        # Current session metrics
        self.session_start_time = datetime.utcnow()
        self.session_metrics = {
            'trades_executed': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_portfolio_value': 0.0
        }
        
        # Performance calculations
        self.returns_series = []
        self.portfolio_values = []
        self.timestamps = []
        
        # Benchmarking
        self.benchmark_returns = []  # For comparison with market
        
        # Reporting
        self.last_daily_report = datetime.utcnow().date() - timedelta(days=1)
        self.last_weekly_report = datetime.utcnow().date() - timedelta(days=7)
    
    async def initialize(self):
        """Initialize performance tracker"""
        try:
            logger.info("📊 Initializing Performance Tracker...")
            
            # Create performance data directory
            os.makedirs('storage/performance', exist_ok=True)
            
            # Load historical performance data if exists
            await self.load_historical_data()
            
            # Initialize with current portfolio state
            if self.portfolio_manager:
                initial_balance = self.portfolio_manager.get_total_balance()
                self.session_metrics['peak_portfolio_value'] = initial_balance
                
                # Add initial snapshot
                await self.record_snapshot()
            
            logger.info("✅ Performance Tracker initialized")
            
        except Exception as e:
            logger.error(f"❌ Performance Tracker initialization failed: {e}")
    
    async def load_historical_data(self):
        """Load historical performance data"""
        try:
            # Load daily snapshots
            snapshots_file = 'storage/performance/daily_snapshots.json'
            if os.path.exists(snapshots_file):
                with open(snapshots_file, 'r') as f:
                    self.daily_snapshots = json.load(f)
                logger.info(f"📈 Loaded {len(self.daily_snapshots)} daily snapshots")
            
            # Load trade records
            trades_file = 'storage/performance/trade_records.json'
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    self.trade_records = json.load(f)
                logger.info(f"💰 Loaded {len(self.trade_records)} trade records")
            
            # Load strategy performance
            strategy_file = 'storage/performance/strategy_performance.json'
            if os.path.exists(strategy_file):
                with open(strategy_file, 'r') as f:
                    self.strategy_performance = json.load(f)
                logger.info(f"🎯 Loaded performance for {len(self.strategy_performance)} strategies")
            
        except Exception as e:
            logger.error(f"❌ Historical data loading failed: {e}")
    
    async def update(self):
        """Main update method - called regularly by bot"""
        try:
            # Record current portfolio snapshot
            await self.record_snapshot()
            
            # Update session metrics
            await self.update_session_metrics()
            
            # Generate reports if needed
            await self.check_report_generation()
            
            # Save performance data periodically
            await self.save_performance_data()
            
        except Exception as e:
            logger.error(f"❌ Performance update failed: {e}")
    
    async def record_snapshot(self):
        """Record current portfolio snapshot"""
        try:
            if not self.portfolio_manager:
                return
            
            current_time = datetime.utcnow()
            
            snapshot = {
                'timestamp': current_time.isoformat(),
                'total_balance': self.portfolio_manager.get_total_balance(),
                'available_balance': self.portfolio_manager.get_available_balance(),
                'unrealized_pnl': self.portfolio_manager.unrealized_pnl,
                'daily_pnl': self.portfolio_manager.get_daily_pnl(),
                'daily_pnl_pct': self.portfolio_manager.get_daily_pnl_pct(),
                'position_count': self.portfolio_manager.get_position_count(),
                'portfolio_value': self.portfolio_manager.get_portfolio_value()
            }
            
            # Add to tracking arrays
            self.portfolio_values.append(snapshot['portfolio_value'])
            self.timestamps.append(current_time)
            
            # Calculate returns if we have previous data
            if len(self.portfolio_values) > 1:
                prev_value = self.portfolio_values[-2]
                if prev_value > 0:
                    return_pct = (snapshot['portfolio_value'] - prev_value) / prev_value
                    self.returns_series.append(return_pct)
            
            # Keep last 1440 snapshots (24 hours at 1-minute intervals)
            if len(self.portfolio_values) > 1440:
                self.portfolio_values.pop(0)
                self.timestamps.pop(0)
                if self.returns_series:
                    self.returns_series.pop(0)
            
        except Exception as e:
            logger.error(f"❌ Snapshot recording failed: {e}")
    
    async def update_session_metrics(self):
        """Update session performance metrics"""
        try:
            if not self.portfolio_manager:
                return
            
            current_value = self.portfolio_manager.get_portfolio_value()
            
            # Update peak portfolio value
            if current_value > self.session_metrics['peak_portfolio_value']:
                self.session_metrics['peak_portfolio_value'] = current_value
            
            # Calculate maximum drawdown
            peak_value = self.session_metrics['peak_portfolio_value']
            if peak_value > 0:
                drawdown = (peak_value - current_value) / peak_value
                if drawdown > self.session_metrics['max_drawdown']:
                    self.session_metrics['max_drawdown'] = drawdown
            
            # Update total PnL
            initial_capital = getattr(self.portfolio_manager, 'initial_capital', current_value)
            self.session_metrics['total_pnl'] = current_value - initial_capital
            
        except Exception as e:
            logger.error(f"❌ Session metrics update failed: {e}")
    
    def record_trade(self, trade_info: Dict[str, Any]):
        """Record completed trade for analysis"""
        try:
            trade_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': trade_info.get('symbol', ''),
                'side': trade_info.get('side', ''),
                'amount': trade_info.get('amount', 0),
                'entry_price': trade_info.get('entry_price', 0),
                'exit_price': trade_info.get('exit_price', 0),
                'pnl': trade_info.get('pnl', 0),
                'pnl_pct': trade_info.get('pnl_pct', 0),
                'strategy': trade_info.get('strategy', ''),
                'duration_minutes': trade_info.get('duration_minutes', 0),
                'commission': trade_info.get('commission', 0)
            }
            
            self.trade_records.append(trade_record)
            
            # Update session metrics
            self.session_metrics['trades_executed'] += 1
            
            if trade_record['pnl'] > 0:
                self.session_metrics['winning_trades'] += 1
            else:
                self.session_metrics['losing_trades'] += 1
            
            # Update strategy performance
            strategy_name = trade_record['strategy']
            if strategy_name:
                if strategy_name not in self.strategy_performance:
                    self.strategy_performance[strategy_name] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_pnl': 0.0,
                        'win_rate': 0.0,
                        'avg_return': 0.0
                    }
                
                strat_perf = self.strategy_performance[strategy_name]
                strat_perf['total_trades'] += 1
                strat_perf['total_pnl'] += trade_record['pnl']
                
                if trade_record['pnl'] > 0:
                    strat_perf['winning_trades'] += 1
                
                strat_perf['win_rate'] = strat_perf['winning_trades'] / strat_perf['total_trades']
                strat_perf['avg_return'] = strat_perf['total_pnl'] / strat_perf['total_trades']
            
            # Keep last 10000 trade records
            if len(self.trade_records) > 10000:
                self.trade_records.pop(0)
            
            logger.info(f"📝 Trade recorded: {trade_record['symbol']} PnL: ${trade_record['pnl']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Trade recording failed: {e}")
    
    def get_daily_pnl(self) -> float:
        """Get daily PnL percentage"""
        try:
            if not self.portfolio_manager:
                return 0.0
            return self.portfolio_manager.get_daily_pnl_pct()
        except:
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            current_time = datetime.utcnow()
            session_duration = (current_time - self.session_start_time).total_seconds() / 3600  # hours
            
            # Basic metrics
            metrics = {
                'session_duration_hours': session_duration,
                'trades_executed': self.session_metrics['trades_executed'],
                'winning_trades': self.session_metrics['winning_trades'],
                'losing_trades': self.session_metrics['losing_trades'],
                'win_rate': self.get_win_rate(),
                'total_pnl': self.session_metrics['total_pnl'],
                'max_drawdown_pct': self.session_metrics['max_drawdown'] * 100,
                'daily_pnl_pct': self.get_daily_pnl(),
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'sortino_ratio': self.calculate_sortino_ratio(),
                'profit_factor': self.calculate_profit_factor(),
                'avg_trade_duration': self.get_average_trade_duration(),
                'best_trade': self.get_best_trade(),
                'worst_trade': self.get_worst_trade(),
                'strategy_performance': self.strategy_performance.copy()
            }
            
            # Portfolio-specific metrics
            if self.portfolio_manager:
                metrics.update({
                    'current_balance': self.portfolio_manager.get_total_balance(),
                    'available_balance': self.portfolio_manager.get_available_balance(),
                    'position_count': self.portfolio_manager.get_position_count(),
                    'portfolio_value': self.portfolio_manager.get_portfolio_value()
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def get_win_rate(self) -> float:
        """Calculate win rate percentage"""
        total_trades = self.session_metrics['trades_executed']
        if total_trades == 0:
            return 0.0
        return (self.session_metrics['winning_trades'] / total_trades) * 100
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.returns_series) < 2:
                return 0.0
            
            returns_array = np.array(self.returns_series)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming daily returns)
            excess_return = mean_return - (risk_free_rate / 365)
            sharpe = (excess_return * np.sqrt(365)) / (std_return * np.sqrt(365))
            
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"❌ Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if len(self.returns_series) < 2:
                return 0.0
            
            returns_array = np.array(self.returns_series)
            mean_return = np.mean(returns_array)
            
            # Calculate downside deviation
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) == 0:
                return float('inf')  # No downside risk
            
            downside_deviation = np.std(negative_returns)
            if downside_deviation == 0:
                return float('inf')
            
            # Annualized Sortino ratio
            excess_return = mean_return - (risk_free_rate / 365)
            sortino = (excess_return * np.sqrt(365)) / (downside_deviation * np.sqrt(365))
            
            return float(sortino)
            
        except Exception as e:
            logger.error(f"❌ Sortino ratio calculation failed: {e}")
            return 0.0
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            if not self.trade_records:
                return 0.0
            
            gross_profit = sum(trade['pnl'] for trade in self.trade_records if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in self.trade_records if trade['pnl'] < 0))
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            return gross_profit / gross_loss
            
        except Exception as e:
            logger.error(f"❌ Profit factor calculation failed: {e}")
            return 0.0
    
    def get_average_trade_duration(self) -> float:
        """Get average trade duration in minutes"""
        try:
            if not self.trade_records:
                return 0.0
            
            durations = [trade.get('duration_minutes', 0) for trade in self.trade_records]
            valid_durations = [d for d in durations if d > 0]
            
            if not valid_durations:
                return 0.0
            
            return sum(valid_durations) / len(valid_durations)
            
        except Exception as e:
            logger.error(f"❌ Average duration calculation failed: {e}")
            return 0.0
    
    def get_best_trade(self) -> Dict[str, Any]:
        """Get best performing trade"""
        try:
            if not self.trade_records:
                return {}
            
            best_trade = max(self.trade_records, key=lambda x: x.get('pnl', 0))
            return {
                'symbol': best_trade.get('symbol', ''),
                'pnl': best_trade.get('pnl', 0),
                'pnl_pct': best_trade.get('pnl_pct', 0),
                'strategy': best_trade.get('strategy', ''),
                'timestamp': best_trade.get('timestamp', '')
            }
            
        except Exception as e:
            logger.error(f"❌ Best trade calculation failed: {e}")
            return {}
    
    def get_worst_trade(self) -> Dict[str, Any]:
        """Get worst performing trade"""
        try:
            if not self.trade_records:
                return {}
            
            worst_trade = min(self.trade_records, key=lambda x: x.get('pnl', 0))
            return {
                'symbol': worst_trade.get('symbol', ''),
                'pnl': worst_trade.get('pnl', 0),
                'pnl_pct': worst_trade.get('pnl_pct', 0),
                'strategy': worst_trade.get('strategy', ''),
                'timestamp': worst_trade.get('timestamp', '')
            }
            
        except Exception in e:
            logger.error(f"❌ Worst trade calculation failed: {e}")
            return {}
    
    async def check_report_generation(self):
        """Check if reports need to be generated"""
        try:
            current_date = datetime.utcnow().date()
            
            # Daily report
            if current_date > self.last_daily_report:
                await self.generate_daily_report()
                self.last_daily_report = current_date
            
            # Weekly report (Sundays)
            if current_date.weekday() == 6 and current_date > self.last_weekly_report:  # Sunday
                await self.generate_weekly_report()
                self.last_weekly_report = current_date
            
        except Exception as e:
            logger.error(f"❌ Report generation check failed: {e}")
    
    async def generate_daily_report(self):
        """Generate daily performance report"""
        try:
            logger.info("📊 Generating daily performance report...")
            
            metrics = self.get_performance_metrics()
            
            report = {
                'date': datetime.utcnow().date().isoformat(),
                'type': 'daily_report',
                'session_metrics': {
                    'trades_executed': metrics['trades_executed'],
                    'win_rate': metrics['win_rate'],
                    'total_pnl': metrics['total_pnl'],
                    'daily_pnl_pct': metrics['daily_pnl_pct'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown_pct': metrics['max_drawdown_pct']
                },
                'strategy_performance': metrics['strategy_performance'],
                'best_trade': metrics.get('best_trade', {}),
                'worst_trade': metrics.get('worst_trade', {})
            }
            
            # Save daily report
            report_file = f"storage/performance/daily_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("✅ Daily report generated")
            
        except Exception as e:
            logger.error(f"❌ Daily report generation failed: {e}")
    
    async def generate_weekly_report(self):
        """Generate weekly performance report"""
        try:
            logger.info("📈 Generating weekly performance report...")
            
            # Calculate weekly metrics
            week_start = datetime.utcnow().date() - timedelta(days=7)
            weekly_trades = [
                trade for trade in self.trade_records 
                if datetime.fromisoformat(trade['timestamp']).date() >= week_start
            ]
            
            weekly_pnl = sum(trade['pnl'] for trade in weekly_trades)
            weekly_win_rate = (
                sum(1 for trade in weekly_trades if trade['pnl'] > 0) / len(weekly_trades) * 100
                if weekly_trades else 0
            )
            
            report = {
                'week_ending': datetime.utcnow().date().isoformat(),
                'type': 'weekly_report',
                'weekly_metrics': {
                    'trades_executed': len(weekly_trades),
                    'weekly_pnl': weekly_pnl,
                    'win_rate': weekly_win_rate
                },
                'top_strategies': self.get_top_strategies(),
                'performance_summary': self.get_performance_metrics()
            }
            
            # Save weekly report
            report_file = f"storage/performance/weekly_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("✅ Weekly report generated")
            
        except Exception as e:
            logger.error(f"❌ Weekly report generation failed: {e}")
    
    def get_top_strategies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing strategies"""
        try:
            if not self.strategy_performance:
                return []
            
            # Sort strategies by total PnL
            sorted_strategies = sorted(
                self.strategy_performance.items(),
                key=lambda x: x[1].get('total_pnl', 0),
                reverse=True
            )
            
            return [
                {
                    'name': name,
                    'total_pnl': perf['total_pnl'],
                    'win_rate': perf['win_rate'],
                    'total_trades': perf['total_trades']
                }
                for name, perf in sorted_strategies[:limit]
            ]
            
        except Exception as e:
            logger.error(f"❌ Top strategies calculation failed: {e}")
            return []
    
    async def save_performance_data(self):
        """Save performance data to disk"""
        try:
            # Save every 100 updates to avoid excessive I/O
            if len(self.portfolio_values) % 100 != 0:
                return
            
            # Save daily snapshots
            snapshots_file = 'storage/performance/daily_snapshots.json'
            with open(snapshots_file, 'w') as f:
                json.dump(self.daily_snapshots[-1000:], f)  # Keep last 1000 snapshots
            
            # Save trade records
            trades_file = 'storage/performance/trade_records.json'
            with open(trades_file, 'w') as f:
                json.dump(self.trade_records[-5000:], f)  # Keep last 5000 trades
            
            # Save strategy performance
            strategy_file = 'storage/performance/strategy_performance.json'
            with open(strategy_file, 'w') as f:
                json.dump(self.strategy_performance, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Performance data saving failed: {e}")
    
    async def save_final_report(self):
        """Save final performance report on shutdown"""
        try:
            logger.info("💾 Saving final performance report...")
            
            final_metrics = self.get_performance_metrics()
            
            final_report = {
                'session_start': self.session_start_time.isoformat(),
                'session_end': datetime.utcnow().isoformat(),
                'final_metrics': final_metrics,
                'total_trades': len(self.trade_records),
                'strategy_breakdown': self.strategy_performance
            }
            
            report_file = f"storage/performance/final_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            # Also save all performance data
            await self.save_performance_data()
            
            logger.info("✅ Final performance report saved")
            
        except Exception as e:
            logger.error(f"❌ Final report saving failed: {e}")
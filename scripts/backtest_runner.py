"""
Backtesting runner for testing trading strategies on historical data.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path
import ccxt
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import DatabaseManager
from strategies.momentum_strategy import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy
from strategies.arbitrage_strategy import ArbitrageStrategy
from risk.risk_manager import RiskManager
from risk.portfolio_optimizer import PortfolioOptimizer
from utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class BacktestRunner:
    """Comprehensive backtesting engine for trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies = {}
        self.results = {}
        self.portfolio_history = []
        self.trade_history = []
        self.position_history = []
        
        # Initialize components
        self.risk_manager = RiskManager(config.get('risk_config', {}))
        self.portfolio_optimizer = PortfolioOptimizer()
        self.indicators = TechnicalIndicators()
        
        # Backtesting parameters
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission_rate = config.get('commission_rate', 0.001)
        self.slippage = config.get('slippage', 0.0005)
        
    async def run_backtest(self, 
                          symbols: List[str],
                          start_date: str,
                          end_date: str,
                          timeframe: str = '1h') -> Dict[str, Any]:
        """Run comprehensive backtest"""
        
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Initial capital: ${self.initial_capital}")
        
        # Load historical data
        market_data = await self._load_historical_data(symbols, start_date, end_date, timeframe)
        if not market_data:
            raise ValueError("No market data loaded")
        
        # Initialize strategies
        await self._initialize_strategies()
        
        # Run simulation
        results = await self._run_simulation(market_data, start_date, end_date)
        
        # Generate report
        report = await self._generate_report(results)
        
        logger.info("Backtest completed successfully")
        return report
    
    async def _load_historical_data(self, 
                                   symbols: List[str], 
                                   start_date: str, 
                                   end_date: str, 
                                   timeframe: str) -> Dict[str, pd.DataFrame]:
        """Load historical market data"""
        market_data = {}
        
        try:
            # Use ccxt to fetch historical data
            exchange = ccxt.binance()
            
            for symbol in symbols:
                logger.info(f"Loading data for {symbol}")
                
                # Convert date strings to timestamps
                start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
                end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
                
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, start_ts, limit=1000)
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    
                    # Add technical indicators
                    df = await self._add_technical_indicators(df)
                    
                    market_data[symbol] = df
                    logger.info(f"Loaded {len(df)} candles for {symbol}")
                
                await asyncio.sleep(0.1)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            # Fallback to dummy data for testing
            return self._generate_dummy_data(symbols, start_date, end_date, timeframe)
        
        return market_data
    
    async def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            # Moving averages
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to add technical indicators: {e}")
            return df
    
    async def _initialize_strategies(self):
        """Initialize trading strategies"""
        strategy_configs = self.config.get('strategies', {})
        
        if strategy_configs.get('momentum_strategy', {}).get('enabled', False):
            self.strategies['momentum'] = MomentumStrategy(strategy_configs['momentum_strategy'])
        
        if strategy_configs.get('mean_reversion_strategy', {}).get('enabled', False):
            self.strategies['mean_reversion'] = MeanReversionStrategy(strategy_configs['mean_reversion_strategy'])
        
        if strategy_configs.get('ml_strategy', {}).get('enabled', False):
            self.strategies['ml'] = MLStrategy(strategy_configs['ml_strategy'])
        
        if strategy_configs.get('arbitrage_strategy', {}).get('enabled', False):
            self.strategies['arbitrage'] = ArbitrageStrategy(strategy_configs['arbitrage_strategy'])
        
        logger.info(f"Initialized {len(self.strategies)} strategies: {list(self.strategies.keys())}")
    
    async def _run_simulation(self, 
                             market_data: Dict[str, pd.DataFrame], 
                             start_date: str, 
                             end_date: str) -> Dict[str, Any]:
        """Run the backtesting simulation"""
        
        # Portfolio state
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital,
            'trades': [],
            'signals': []
        }
        
        # Get all timestamps across all symbols
        all_timestamps = set()
        for df in market_data.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        
        # Run simulation step by step
        for i, timestamp in enumerate(timestamps):
            current_data = {}
            
            # Get current market data for all symbols
            for symbol, df in market_data.items():
                if timestamp in df.index:
                    current_data[symbol] = df.loc[timestamp].to_dict()
            
            if not current_data:
                continue
            
            # Update portfolio value
            await self._update_portfolio_value(portfolio, current_data)
            
            # Generate signals from all strategies
            signals = await self._generate_signals(current_data, market_data, timestamp)
            portfolio['signals'].extend(signals)
            
            # Execute trades based on signals
            trades = await self._execute_backtest_trades(portfolio, signals, current_data, timestamp)
            portfolio['trades'].extend(trades)
            
            # Record portfolio snapshot
            self.portfolio_history.append({
                'timestamp': timestamp,
                'total_value': portfolio['total_value'],
                'cash': portfolio['cash'],
                'positions_value': sum(pos['value'] for pos in portfolio['positions'].values()),
                'num_positions': len(portfolio['positions'])
            })
            
            # Progress logging
            if i % 100 == 0:
                progress = (i / len(timestamps)) * 100
                logger.info(f"Backtest progress: {progress:.1f}% - Portfolio: ${portfolio['total_value']:.2f}")
        
        return {
            'portfolio': portfolio,
            'portfolio_history': self.portfolio_history,
            'trade_history': portfolio['trades'],
            'signals_history': portfolio['signals']
        }
    
    async def _update_portfolio_value(self, portfolio: Dict, current_data: Dict[str, Dict]):
        """Update portfolio value based on current prices"""
        total_positions_value = 0
        
        for symbol, position in portfolio['positions'].items():
            if symbol in current_data:
                current_price = current_data[symbol]['close']
                position['current_price'] = current_price
                position['value'] = position['size'] * current_price
                position['pnl'] = position['value'] - position['cost_basis']
                position['pnl_percent'] = (position['pnl'] / position['cost_basis']) * 100
                
                total_positions_value += position['value']
        
        portfolio['total_value'] = portfolio['cash'] + total_positions_value
    
    async def _generate_signals(self, 
                               current_data: Dict[str, Dict], 
                               market_data: Dict[str, pd.DataFrame],
                               timestamp: datetime) -> List[Dict]:
        """Generate signals from all strategies"""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            for symbol in current_data.keys():
                if symbol in market_data:
                    # Get historical data up to current timestamp
                    historical_data = market_data[symbol].loc[:timestamp].copy()
                    
                    if len(historical_data) < 50:  # Need minimum data
                        continue
                    
                    try:
                        signal = await strategy.generate_signal(symbol, historical_data)
                        if signal and signal.get('signal_type') != 'hold':
                            signal.update({
                                'timestamp': timestamp,
                                'strategy': strategy_name,
                                'symbol': symbol
                            })
                            signals.append(signal)
                    except Exception as e:
                        logger.warning(f"Strategy {strategy_name} failed for {symbol}: {e}")
        
        return signals
    
    async def _execute_backtest_trades(self, 
                                      portfolio: Dict, 
                                      signals: List[Dict],
                                      current_data: Dict[str, Dict],
                                      timestamp: datetime) -> List[Dict]:
        """Execute trades based on signals in backtest"""
        trades = []
        
        for signal in signals:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            confidence = signal.get('confidence', 0.5)
            
            if symbol not in current_data:
                continue
            
            current_price = current_data[symbol]['close']
            
            # Apply slippage
            if signal_type == 'buy':
                execution_price = current_price * (1 + self.slippage)
            else:
                execution_price = current_price * (1 - self.slippage)
            
            # Calculate position size
            position_size = await self._calculate_backtest_position_size(
                portfolio, signal, execution_price
            )
            
            if position_size == 0:
                continue
            
            # Execute trade
            trade = await self._execute_trade(
                portfolio, symbol, signal_type, position_size, execution_price, timestamp
            )
            
            if trade:
                trades.append(trade)
        
        return trades
    
    async def _calculate_backtest_position_size(self, 
                                              portfolio: Dict, 
                                              signal: Dict, 
                                              price: float) -> float:
        """Calculate position size for backtest"""
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            confidence = signal.get('confidence', 0.5)
            
            # Risk-based position sizing
            risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 2% risk per trade
            portfolio_value = portfolio['total_value']
            
            # Base position size
            max_risk_amount = portfolio_value * risk_per_trade * confidence
            
            # Account for existing position
            existing_position = portfolio['positions'].get(symbol, {})
            
            if signal_type == 'buy':
                if existing_position:
                    # Don't add to existing long position
                    return 0
                position_size = max_risk_amount / price
                
            elif signal_type == 'sell':
                if not existing_position:
                    # Can't sell if no position
                    return 0
                position_size = existing_position['size']
            
            # Check if we have enough cash for buy orders
            if signal_type == 'buy':
                trade_value = position_size * price
                commission = trade_value * self.commission_rate
                total_cost = trade_value + commission
                
                if total_cost > portfolio['cash']:
                    position_size = (portfolio['cash'] * 0.95) / price  # Use 95% of available cash
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0
    
    async def _execute_trade(self, 
                            portfolio: Dict, 
                            symbol: str, 
                            side: str, 
                            size: float, 
                            price: float, 
                            timestamp: datetime) -> Optional[Dict]:
        """Execute a trade in the backtest"""
        try:
            trade_value = size * price
            commission = trade_value * self.commission_rate
            
            trade = {
                'id': f"bt_{timestamp.strftime('%Y%m%d_%H%M%S')}_{symbol}",
                'timestamp': timestamp,
                'symbol': symbol,
                'side': side,
                'size': size,
                'price': price,
                'value': trade_value,
                'commission': commission,
                'total_cost': trade_value + commission
            }
            
            if side == 'buy':
                # Open/add to position
                if trade['total_cost'] > portfolio['cash']:
                    logger.warning(f"Insufficient cash for trade: ${trade['total_cost']:.2f} > ${portfolio['cash']:.2f}")
                    return None
                
                portfolio['cash'] -= trade['total_cost']
                
                if symbol in portfolio['positions']:
                    # Add to existing position
                    existing = portfolio['positions'][symbol]
                    total_size = existing['size'] + size
                    total_cost = existing['cost_basis'] + trade_value
                    avg_price = total_cost / total_size
                    
                    portfolio['positions'][symbol].update({
                        'size': total_size,
                        'avg_price': avg_price,
                        'cost_basis': total_cost,
                        'last_update': timestamp
                    })
                else:
                    # New position
                    portfolio['positions'][symbol] = {
                        'size': size,
                        'avg_price': price,
                        'cost_basis': trade_value,
                        'current_price': price,
                        'value': trade_value,
                        'pnl': 0,
                        'pnl_percent': 0,
                        'opened_at': timestamp,
                        'last_update': timestamp
                    }
                
            elif side == 'sell':
                # Close/reduce position
                if symbol not in portfolio['positions']:
                    logger.warning(f"Cannot sell {symbol}: no position")
                    return None
                
                position = portfolio['positions'][symbol]
                if size > position['size']:
                    size = position['size']  # Can't sell more than we have
                    trade['size'] = size
                    trade['value'] = size * price
                    trade['total_cost'] = trade['value'] - commission
                
                # Calculate PnL
                cost_basis_sold = (size / position['size']) * position['cost_basis']
                pnl = (size * price) - cost_basis_sold - commission
                
                portfolio['cash'] += (size * price) - commission
                
                # Update position
                if size >= position['size']:
                    # Close entire position
                    del portfolio['positions'][symbol]
                else:
                    # Partial close
                    remaining_size = position['size'] - size
                    remaining_cost = position['cost_basis'] - cost_basis_sold
                    
                    portfolio['positions'][symbol].update({
                        'size': remaining_size,
                        'cost_basis': remaining_cost,
                        'avg_price': remaining_cost / remaining_size,
                        'last_update': timestamp
                    })
                
                trade['pnl'] = pnl
            
            return trade
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return None
    
    async def _generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive backtest report"""
        try:
            portfolio_history = pd.DataFrame(self.portfolio_history)
            trades = results['trade_history']
            
            if portfolio_history.empty:
                return {'error': 'No portfolio history available'}
            
            # Performance metrics
            initial_value = portfolio_history.iloc[0]['total_value']
            final_value = portfolio_history.iloc[-1]['total_value']
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate daily returns
            portfolio_history['daily_return'] = portfolio_history['total_value'].pct_change()
            daily_returns = portfolio_history['daily_return'].dropna()
            
            # Risk metrics
            volatility = daily_returns.std() * np.sqrt(365)  # Annualized
            sharpe_ratio = (daily_returns.mean() * 365) / (daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0
            
            # Drawdown analysis
            cumulative = (1 + daily_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Trade analysis
            profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            win_rate = len(profitable_trades) / len(trades) if trades else 0
            avg_win = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Strategy performance breakdown
            strategy_performance = {}
            for trade in trades:
                strategy = trade.get('strategy', 'unknown')
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'trades': 0, 'pnl': 0}
                strategy_performance[strategy]['trades'] += 1
                strategy_performance[strategy]['pnl'] += trade.get('pnl', 0)
            
            report = {
                'summary': {
                    'initial_capital': initial_value,
                    'final_value': final_value,
                    'total_return': total_return,
                    'total_return_percent': total_return * 100,
                    'annualized_return': ((final_value / initial_value) ** (365 / len(portfolio_history)) - 1) * 100,
                    'volatility': volatility * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': abs(max_drawdown) * 100,
                    'calmar_ratio': (total_return * 100) / abs(max_drawdown * 100) if max_drawdown != 0 else 0
                },
                'trades': {
                    'total_trades': len(trades),
                    'profitable_trades': len(profitable_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': win_rate * 100,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'total_commission': sum(t.get('commission', 0) for t in trades)
                },
                'strategy_performance': strategy_performance,
                'portfolio_history': portfolio_history.to_dict('records'),
                'trade_details': trades,
                'config': self.config
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_dummy_data(self, symbols: List[str], start_date: str, end_date: str, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Generate dummy data for testing when real data unavailable"""
        logger.warning("Generating dummy data for backtesting")
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Create date range
        if timeframe == '1h':
            freq = 'H'
        elif timeframe == '4h':
            freq = '4H'
        elif timeframe == '1d':
            freq = 'D'
        else:
            freq = 'H'
        
        dates = pd.date_range(start=start, end=end, freq=freq)
        
        market_data = {}
        
        for symbol in symbols:
            # Generate realistic price movements
            np.random.seed(42)  # For reproducible results
            
            base_price = 50000 if 'BTC' in symbol else 3000
            n_periods = len(dates)
            
            # Generate returns with some trend and volatility
            returns = np.random.normal(0.0001, 0.02, n_periods)  # Small positive drift, 2% volatility
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            # Create OHLCV data
            df_data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Simulate intraday moves
                high = price * (1 + np.random.uniform(0, 0.01))
                low = price * (1 - np.random.uniform(0, 0.01))
                open_price = prices[i-1] if i > 0 else price
                volume = np.random.uniform(1000, 10000)
                
                df_data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            df = df.set_index('timestamp')
            df = await self._add_technical_indicators(df)
            
            market_data[symbol] = df
        
        return market_data

def main():
    """Main function for running backtests from command line"""
    parser = argparse.ArgumentParser(description='Run crypto trading bot backtest')
    parser.add_argument('--config', required=True, help='Path to backtest config file')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], help='Trading symbols')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (1m, 5m, 15m, 1h, 4h, 1d)')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def run_backtest():
        runner = BacktestRunner(config)
        results = await runner.run_backtest(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe
        )
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        
        # Print summary
        summary = results.get('summary', {})
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Capital: ${summary.get('initial_capital', 0):,.2f}")
        print(f"Final Value: ${summary.get('final_value', 0):,.2f}")
        print(f"Total Return: {summary.get('total_return_percent', 0):.2f}%")
        print(f"Annualized Return: {summary.get('annualized_return', 0):.2f}%")
        print(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {summary.get('max_drawdown', 0):.2f}%")
        print(f"Win Rate: {results.get('trades', {}).get('win_rate', 0):.1f}%")
        print(f"Total Trades: {results.get('trades', {}).get('total_trades', 0)}")
        print("="*50)
    
    # Run the backtest
    asyncio.run(run_backtest())

if __name__ == "__main__":
    main()
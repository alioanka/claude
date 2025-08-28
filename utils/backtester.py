"""
Backtesting engine for trading strategies.
Provides comprehensive backtesting functionality with performance analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

from utils.helper import PerformanceCalculator, format_currency, format_percentage

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    order_type: OrderType = OrderType.MARKET
    timestamp: datetime = None
    order_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.order_id is None:
            self.order_id = f"{self.side.value}_{self.symbol}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"

@dataclass
class Trade:
    """Represents an executed trade."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def value(self) -> float:
        """Trade value excluding commission."""
        return self.quantity * self.price
    
    @property
    def net_value(self) -> float:
        """Trade value including commission."""
        return self.value - self.commission

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = None
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        price = self.current_price or self.entry_price
        return abs(self.quantity) * price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.current_price is None:
            return 0.0
        return self.quantity * (self.current_price - self.entry_price)
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

class Portfolio:
    """Manages portfolio state during backtesting."""
    
    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
    def add_trade(self, trade: Trade):
        """Add executed trade to portfolio."""
        self.trades.append(trade)
        
        # Update cash
        if trade.side == OrderSide.BUY:
            self.cash -= trade.net_value
        else:
            self.cash += trade.net_value
        
        # Update position
        symbol = trade.symbol
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if (position.is_long and trade.side == OrderSide.SELL) or \
               (position.is_short and trade.side == OrderSide.BUY):
                # Closing or reducing position
                if abs(trade.quantity) >= abs(position.quantity):
                    # Closing position completely
                    del self.positions[symbol]
                else:
                    # Reducing position
                    position.quantity += trade.quantity if trade.side == OrderSide.BUY else -trade.quantity
            else:
                # Adding to position
                total_quantity = position.quantity + (trade.quantity if trade.side == OrderSide.BUY else -trade.quantity)
                if total_quantity != 0:
                    # Calculate new average price
                    total_value = position.quantity * position.entry_price + trade.value
                    position.entry_price = total_value / total_quantity
                    position.quantity = total_quantity
        else:
            # New position
            quantity = trade.quantity if trade.side == OrderSide.BUY else -trade.quantity
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=trade.price,
                entry_time=trade.timestamp
            )
    
    def update_positions(self, prices: Dict[str, float], timestamp: datetime):
        """Update position current prices and record equity."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
        
        # Record equity curve
        total_equity = self.get_total_equity(prices)
        self.equity_curve.append((timestamp, total_equity))
    
    def get_total_equity(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate total portfolio equity."""
        equity = self.cash
        
        for position in self.positions.values():
            if current_prices and position.symbol in current_prices:
                position.current_price = current_prices[position.symbol]
            equity += position.market_value
        
        return equity
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)

class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.portfolio = Portfolio(initial_capital, commission_rate)
        
    def run_backtest(self, 
                     strategy,
                     data: Dict[str, pd.DataFrame],
                     start_date: str = None,
                     end_date: str = None) -> Dict:
        """
        Run backtest for given strategy and data.
        
        Args:
            strategy: Trading strategy instance
            data: Dictionary of symbol -> OHLCV DataFrame
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting backtest with initial capital: {format_currency(self.initial_capital)}")
        
        # Filter data by date range
        if start_date or end_date:
            data = self._filter_data_by_date(data, start_date, end_date)
        
        # Get all timestamps across all symbols
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        
        # Initialize strategy
        strategy.initialize(data)
        
        # Run backtest
        for i, timestamp in enumerate(timestamps):
            # Get current market data
            current_data = {}
            current_prices = {}
            
            for symbol, df in data.items():
                if timestamp in df.index:
                    current_data[symbol] = df.loc[df.index <= timestamp]
                    current_prices[symbol] = df.loc[timestamp, 'close']
            
            # Update portfolio positions with current prices
            self.portfolio.update_positions(current_prices, timestamp)
            
            # Get strategy signals
            signals = strategy.generate_signals(current_data, timestamp)
            
            # Execute orders
            if signals:
                for signal in signals:
                    self._execute_order(signal, current_prices, timestamp)
            
            # Log progress periodically
            if i % 1000 == 0:
                current_equity = self.portfolio.get_total_equity(current_prices)
                logger.info(f"Progress: {i}/{len(timestamps)}, Equity: {format_currency(current_equity)}")
        
        # Generate results
        results = self._generate_results()
        logger.info("Backtest completed")
        
        return results
    
    def _filter_data_by_date(self, data: Dict[str, pd.DataFrame], 
                           start_date: str = None, 
                           end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Filter data by date range."""
        filtered_data = {}
        
        for symbol, df in data.items():
            filtered_df = df.copy()
            
            if start_date:
                filtered_df = filtered_df[filtered_df.index >= start_date]
            if end_date:
                filtered_df = filtered_df[filtered_df.index <= end_date]
            
            filtered_data[symbol] = filtered_df
        
        return filtered_data
    
    def _execute_order(self, order: Order, current_prices: Dict[str, float], timestamp: datetime):
        """Execute trading order."""
        if order.symbol not in current_prices:
            logger.warning(f"No price data for {order.symbol} at {timestamp}")
            return
        
        # Calculate execution price (including slippage)
        market_price = current_prices[order.symbol]
        if order.order_type == OrderType.MARKET:
            execution_price = market_price
            if order.side == OrderSide.BUY:
                execution_price *= (1 + self.slippage_rate)
            else:
                execution_price *= (1 - self.slippage_rate)
        else:
            execution_price = order.price
        
        # Calculate commission
        trade_value = order.quantity * execution_price
        commission = trade_value * self.commission_rate
        
        # Check if we have enough cash/position for the trade
        if not self._can_execute_trade(order, execution_price, commission):
            logger.warning(f"Cannot execute trade: insufficient funds/position")
            return
        
        # Create and add trade
        trade = Trade(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=timestamp,
            commission=commission,
            slippage=abs(execution_price - market_price) * order.quantity
        )
        
        self.portfolio.add_trade(trade)
        logger.debug(f"Executed: {trade.side.value} {trade.quantity} {trade.symbol} @ {execution_price}")
    
    def _can_execute_trade(self, order: Order, price: float, commission: float) -> bool:
        """Check if trade can be executed."""
        if order.side == OrderSide.BUY:
            required_cash = order.quantity * price + commission
            return self.portfolio.cash >= required_cash
        else:
            # For sell orders, check if we have enough position
            position = self.portfolio.get_position(order.symbol)
            if position is None:
                return False  # Cannot sell without position
            return position.quantity >= order.quantity
    
    def _generate_results(self) -> Dict:
        """Generate comprehensive backtest results."""
        if not self.portfolio.equity_curve:
            return {"error": "No trades executed during backtest"}
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.portfolio.equity_curve, 
                               columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Calculate performance metrics
        calc = PerformanceCalculator()
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        results = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_trades': len(self.portfolio.trades),
                'winning_trades': len([t for t in self.portfolio.trades if self._trade_pnl(t) > 0]),
                'losing_trades': len([t for t in self.portfolio.trades if self._trade_pnl(t) < 0]),
            },
            'performance_metrics': {
                'sharpe_ratio': calc.calculate_sharpe_ratio(returns),
                'max_drawdown': calc.calculate_max_drawdown(returns),
                'volatility': calc.calculate_volatility(returns),
                'var_5': calc.calculate_var(returns, 0.05),
                'win_rate': calc.calculate_win_rate(returns)
            },
            'equity_curve': equity_df,
            'trades': self.portfolio.trades,
            'positions': self.portfolio.positions,
            'returns': returns
        }
        
        # Add formatted summary for display
        results['formatted_summary'] = self._format_summary(results)
        
        return results
    
    def _trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a single trade (simplified)."""
        # This is a simplified calculation
        # In practice, you'd need to match buy/sell pairs
        return 0.0  # Placeholder
    
    def _format_summary(self, results: Dict) -> str:
        """Format results summary for display."""
        summary = results['summary']
        metrics = results['performance_metrics']
        
        formatted = f"""
BACKTEST RESULTS SUMMARY
========================

Capital:
  Initial Capital: {format_currency(summary['initial_capital'])}
  Final Equity: {format_currency(summary['final_equity'])}
  Total Return: {format_percentage(summary['total_return'])}

Trading Activity:
  Total Trades: {summary['total_trades']}
  Winning Trades: {summary['winning_trades']}
  Losing Trades: {summary['losing_trades']}
  Win Rate: {format_percentage(metrics['win_rate'])}

Performance Metrics:
  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
  Max Drawdown: {format_percentage(metrics['max_drawdown'])}
  Volatility: {format_percentage(metrics['volatility'])}
  VaR (5%): {format_percentage(metrics['var_5'])}
        """
        
        return formatted.strip()

def run_simple_backtest(strategy, data: Dict[str, pd.DataFrame], **kwargs) -> Dict:
    """
    Convenience function to run a simple backtest.
    
    Args:
        strategy: Trading strategy instance
        data: Dictionary of symbol -> OHLCV DataFrame
        **kwargs: Additional arguments for BacktestEngine
        
    Returns:
        Backtest results dictionary
    """
    engine = BacktestEngine(**kwargs)
    return engine.run_backtest(strategy, data)
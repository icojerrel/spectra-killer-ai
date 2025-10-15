"""
Portfolio management with comprehensive tracking and analytics
Handles multiple positions with proper risk management and performance metrics
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Union
import logging
from collections import defaultdict

from .position import Position, PositionStatus
from .events import EventBus, Event, EventType, event_bus
from ..utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown
from ..utils.helpers import format_currency, format_percentage

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    balance: Decimal
    equity: Decimal
    open_positions_count: int
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_exposure: Decimal


class Portfolio:
    """Advanced portfolio management with comprehensive tracking"""
    
    def __init__(self, initial_balance: Union[float, Decimal], currency: str = "USD"):
        """
        Initialize portfolio
        
        Args:
            initial_balance: Starting balance
            currency: Portfolio currency
        """
        self.initial_balance = Decimal(str(initial_balance))
        self.balance = self.initial_balance
        self.currency = currency
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Performance tracking
        self.total_pnl = Decimal('0')
        self.daily_pnl = Decimal('0')
        self.max_balance = self.initial_balance
        self.drawdown = Decimal('0')
        self.max_drawdown = Decimal('0')
        
        # Trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_volume = Decimal('0')
        
        # History tracking
        self.equity_curve: List[PortfolioSnapshot] = []
        self.daily_stats: Dict[str, dict] = {}
        
        # Risk metrics
        self.risk_metrics = {
            'var_95': Decimal('0'),  # Value at Risk 95%
            'beta': Decimal('0'),
            'alpha': Decimal('0'),
            'correlation_to_market': Decimal('0'),
        }
        
        # Subscribe to events
        event_bus.subscribe(EventType.POSITION_CLOSED, self._on_position_closed)
        event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
    
    async def _on_position_closed(self, event: Event):
        """Handle position closed event"""
        position_data = event.data
        position_id = position_data.get('position_id')
        
        if position_id and position_id in self.positions:
            position = self.positions[position_id]
            self._update_performance_metrics(position)
    
    async def _on_order_filled(self, event: Event):
        """Handle order filled event"""
        # Update total volume
        volume = Decimal(str(event.data.get('volume', 0)))
        self.total_volume += volume
    
    def open_position(self, position: Position) -> bool:
        """
        Open a new position
        
        Args:
            position: Position to open
            
        Returns:
            True if position opened successfully
        """
        try:
            # Validate position
            if position.position_id in self.positions:
                logger.error(f"Position {position.position_id} already exists")
                return False
            
            # Check margin requirements
            required_margin = position.entry_price * position.quantity * Decimal('0.1')  # 10% margin
            if required_margin > self.balance * Decimal('0.5'):  # Max 50% margin usage
                logger.warning(f"Insufficient margin for position {position.position_id}")
                return False
            
            # Add position
            self.positions[position.position_id] = position
            
            # Update statistics
            self.total_trades += 1
            
            logger.info(f"Opened position: {position}")
            
            # Publish event
            event_bus.publish(Event(
                type=EventType.POSITION_OPENED,
                data={'position_id': position.position_id, 'symbol': position.symbol},
                source="Portfolio"
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def close_position(self, position_id: str, exit_price: Union[float, Decimal]) -> Optional[Decimal]:
        """
        Close a position
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price
            
        Returns:
            Realized P&L if position closed successfully
        """
        if position_id not in self.positions:
            logger.error(f"Position {position_id} not found")
            return None
        
        try:
            position = self.positions[position_id]
            exit_price = Decimal(str(exit_price))
            
            # Close position
            realized_pnl = position.close(exit_price)
            
            # Update portfolio balance
            self.balance += realized_pnl
            self.total_pnl += realized_pnl
            self.daily_pnl += realized_pnl
            
            # Update statistics
            if realized_pnl > 0:
                self.winning_trades += 1
            elif realized_pnl < 0:
                self.losing_trades += 1
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position_id]
            
            # Update peak balance and drawdown
            self._update_drawdown()
            
            logger.info(f"Closed position {position_id} with P&L: {realized_pnl}")
            
            # Publish event
            event_bus.publish(Event(
                type=EventType.POSITION_CLOSED,
                data={
                    'position_id': position_id,
                    'realized_pnl': float(realized_pnl),
                    'exit_price': float(exit_price)
                },
                source="Portfolio"
            ))
            
            return realized_pnl
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return None
    
    def update_positions(self, market_data: Dict[str, Decimal]) -> None:
        """
        Update all open positions with current market data
        
        Args:
            market_data: Dictionary of symbol -> current price
        """
        for position in self.positions.values():
            if position.symbol in market_data:
                position.update_price(market_data[position.symbol])
                
                # Check for risk triggers
                if position.should_stop_loss():
                    self._trigger_stop_loss(position)
                elif position.should_take_profit():
                    self._trigger_take_profit(position)
    
    def _trigger_stop_loss(self, position: Position) -> None:
        """Trigger stop loss for position"""
        logger.warning(f"Stop loss triggered for position {position.position_id}")
        
        # Publish event
        event_bus.publish(Event(
            type=EventType.STOP_LOSS_TRIGGERED,
            data={
                'position_id': position.position_id,
                'stop_loss': float(position.stop_loss)
            },
            source="Portfolio"
        ))
    
    def _trigger_take_profit(self, position: Position) -> None:
        """Trigger take profit for position"""
        logger.info(f"Take profit triggered for position {position.position_id}")
        
        # Publish event
        event_bus.publish(Event(
            type=EventType.TAKE_PROFIT_TRIGGERED,
            data={
                'position_id': position.position_id,
                'take_profit': float(position.take_profit)
            },
            source="Portfolio"
        ))
    
    def get_equity(self, market_data: Optional[Dict[str, Decimal]] = None) -> Decimal:
        """
        Calculate total portfolio equity
        
        Args:
            market_data: Current market data for position updates
            
        Returns:
            Total equity (balance + unrealized P&L)
        """
        equity = self.balance
        
        if market_data:
            for position in self.positions.values():
                if position.symbol in market_data:
                    position.update_price(market_data[position.symbol])
                    equity += position.unrealized_pnl
        else:
            # Use current unrealized P&L
            equity += sum(p.unrealized_pnl for p in self.positions.values())
        
        return equity
    
    def get_exposure(self) -> Dict[str, Decimal]:
        """
        Get current exposure by symbol
        
        Returns:
            Dictionary of symbol -> exposure amount
        """
        exposure = defaultdict(Decimal)
        
        for position in self.positions.values():
            exposure[position.symbol] += position.quantity * position.current_price if position.current_price else Decimal('0')
        
        return dict(exposure)
    
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        return self.positions.get(position_id)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a symbol"""
        return [p for p in self.positions.values() if p.symbol == symbol]
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if p.is_open]
    
    def get_performance_metrics(self) -> Dict[str, Union[float, str]]:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        total_return = ((self.balance - self.initial_balance) / self.initial_balance * 100)
        
        # Win rate
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Average win/loss
        avg_win = self._calculate_average_win()
        avg_loss = self._calculate_average_loss()
        
        # Profit factor
        profit_factor = self._calculate_profit_factor()
        
        # Sharpe ratio (simplified, needs historical data)
        sharpe_ratio = 0.0  # Would need equity curve history
        
        return {
            'total_return_pct': float(total_return),
            'total_pnl': float(self.total_pnl),
            'daily_pnl': float(self.daily_pnl),
            'balance': float(self.balance),
            'initial_balance': float(self.initial_balance),
            'win_rate_pct': float(win_rate),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'max_drawdown_pct': float(self.max_drawdown * 100),
            'sharpe_ratio': sharpe_ratio,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'total_volume': float(self.total_volume),
        }
    
    def _calculate_average_win(self) -> Decimal:
        """Calculate average winning trade"""
        winning_pnls = [p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0]
        return sum(winning_pnls) / len(winning_pnls) if winning_pnls else Decimal('0')
    
    def _calculate_average_loss(self) -> Decimal:
        """Calculate average losing trade"""
        losing_pnls = [p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0]
        return sum(losing_pnls) / len(losing_pnls) if losing_pnls else Decimal('0')
    
    def _calculate_profit_factor(self) -> Decimal:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0)
        gross_loss = abs(sum(p.realized_pnl for p in self.closed_positions if p.realized_pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else Decimal('0')
    
    def _update_performance_metrics(self, position: Position) -> None:
        """Update performance metrics after position close"""
        # This could update various metrics like rolling returns, etc.
        pass
    
    def _update_drawdown(self) -> None:
        """Update maximum drawdown tracking"""
        if self.balance > self.max_balance:
            self.max_balance = self.balance
        else:
            self.drawdown = (self.max_balance - self.balance) / self.max_balance
            self.max_drawdown = max(self.max_drawdown, self.drawdown)
    
    def take_snapshot(self, market_data: Optional[Dict[str, Decimal]] = None) -> PortfolioSnapshot:
        """
        Take a portfolio snapshot
        
        Args:
            market_data: Current market data
            
        Returns:
            PortfolioSnapshot
        """
        equity = self.get_equity(market_data)
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        total_exposure = sum(self.get_exposure().values())
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            balance=self.balance,
            equity=equity,
            open_positions_count=len(self.positions),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.total_pnl,
            total_exposure=total_exposure
        )
        
        self.equity_curve.append(snapshot)
        
        # Keep only last 1000 snapshots
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]
        
        return snapshot
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics"""
        self.daily_pnl = Decimal('0')
        
        # Reset daily stats for positions
        for position in self.positions.values():
            # Could track daily unrealized P&L here
            pass
    
    def export_data(self) -> Dict:
        """Export portfolio data for analysis"""
        return {
            'portfolio_info': {
                'initial_balance': float(self.initial_balance),
                'current_balance': float(self.balance),
                'total_pnl': float(self.total_pnl),
                'currency': self.currency,
            },
            'performance_metrics': self.get_performance_metrics(),
            'positions': [p.to_dict() for p in self.positions.values()],
            'closed_positions': [p.to_dict() for p in self.closed_positions],
            'equity_curve': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'balance': float(s.balance),
                    'equity': float(s.equity),
                    'open_positions_count': s.open_positions_count,
                    'unrealized_pnl': float(s.unrealized_pnl),
                } for s in self.equity_curve
            ],
        }
    
    def __str__(self) -> str:
        """String representation"""
        return (
            f"Portfolio({self.currency} {self.balance:.2f}, "
            f"Return: {((self.balance - self.initial_balance) / self.initial_balance * 100):.1f}%, "
            f"Positions: {len(self.positions)})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"Portfolio(balance={self.balance}, initial_balance={self.initial_balance}, "
                f"total_pnl={self.total_pnl}, positions={len(self.positions)}, "
                f"closed_positions={len(self.closed_positions)})")

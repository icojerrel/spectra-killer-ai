"""
Position management with comprehensive tracking
Handles individual trading positions with proper risk management
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Position types"""
    LONG = "BUY"
    SHORT = "SELL"


class PositionStatus(Enum):
    """Position status states"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class TradeEntry:
    """Individual trade entry/exit record"""
    price: Decimal
    quantity: Decimal
    timestamp: datetime
    commission: Decimal = Decimal('0')
    slippage: Decimal = Decimal('0')


@dataclass
class Position:
    """Trading position with comprehensive tracking"""
    
    # Basic position info
    position_id: str
    symbol: str
    position_type: PositionType
    entry_price: Decimal
    quantity: Decimal
    entry_time: datetime = field(default_factory=datetime.now)
    status: PositionStatus = PositionStatus.OPEN
    
    # Entry details
    
    # Risk management
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop: Optional[Decimal] = None
    
    # Current tracking
    current_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    
    # Trade history
    entries: list[TradeEntry] = field(default_factory=list)
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    confidence: Optional[Decimal] = None
    tags: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize position with validation"""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.position_type not in PositionType:
            raise ValueError(f"Invalid position type: {self.position_type}")
        
        # Add entry to trade history
        self.entries.append(TradeEntry(
            price=self.entry_price,
            quantity=self.quantity,
            timestamp=self.entry_time
        ))
        
        # Setup risk levels based on position type
        self._setup_risk_levels()
    
    def _setup_risk_levels(self):
        """Setup initial stop loss and take profit if not provided"""
        if self.stop_loss is None or self.take_profit is None:
            # Default risk/reward ratio of 1:2
            risk_distance = Decimal('20')  # Default 20 pips
            reward_distance = Decimal('40')  # Default 40 pips
            
            if self.position_type == PositionType.LONG:
                if self.stop_loss is None:
                    self.stop_loss = self.entry_price - risk_distance
                if self.take_profit is None:
                    self.take_profit = self.entry_price + reward_distance
            else:  # SHORT
                if self.stop_loss is None:
                    self.stop_loss = self.entry_price + risk_distance
                if self.take_profit is None:
                    self.take_profit = self.entry_price - reward_distance
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.position_type == PositionType.LONG
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.position_type == PositionType.SHORT
    
    @property
    def is_open(self) -> bool:
        """Check if position is open"""
        return self.status == PositionStatus.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if position is closed"""
        return self.status == PositionStatus.CLOSED
    
    @property
    def duration(self) -> Optional[float]:
        """Get position duration in seconds"""
        if not self.exit_time:
            return None
        return (self.exit_time - self.entry_time).total_seconds()
    
    def update_price(self, current_price: Decimal) -> Decimal:
        """
        Update current price and calculate unrealized P&L
        
        Args:
            current_price: Current market price
            
        Returns:
            Updated unrealized P&L
        """
        if not self.is_open:
            return self.unrealized_pnl
        
        self.current_price = current_price
        
        if self.is_long:
            price_diff = current_price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - current_price
        
        self.unrealized_pnl = price_diff * self.quantity
        return self.unrealized_pnl
    
    def should_stop_loss(self, current_price: Optional[Decimal] = None) -> bool:
        """Check if stop loss should be triggered"""
        if self.stop_loss is None or not self.is_open:
            return False
        
        price = current_price or self.current_price
        if price is None:
            return False
        
        if self.is_long:
            return price <= self.stop_loss
        else:  # SHORT
            return price >= self.stop_loss
    
    def should_take_profit(self, current_price: Optional[Decimal] = None) -> bool:
        """Check if take profit should be triggered"""
        if self.take_profit is None or not self.is_open:
            return False
        
        price = current_price or self.current_price
        if price is None:
            return False
        
        if self.is_long:
            return price >= self.take_profit
        else:  # SHORT
            return price <= self.take_profit
    
    def close(self, exit_price: Decimal, exit_time: Optional[datetime] = None) -> Decimal:
        """
        Close position and calculate realized P&L
        
        Args:
            exit_price: Exit price
            exit_time: Exit time (defaults to now)
            
        Returns:
            Realized P&L
        """
        if not self.is_open:
            raise ValueError("Position is already closed")
        
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.status = PositionStatus.CLOSED
        
        # Calculate realized P&L
        if self.is_long:
            price_diff = self.exit_price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - self.exit_price
        
        self.realized_pnl = price_diff * self.quantity
        
        # Add exit to trade history
        self.entries.append(TradeEntry(
            price=self.exit_price,
            quantity=-self.quantity,  # Negative for exit
            timestamp=self.exit_time
        ))
        
        logger.info(f"Position {self.position_id} closed with P&L: {self.realized_pnl}")
        return self.realized_pnl
    
    def cancel(self, reason: str = "") -> None:
        """Cancel position"""
        if self.status != PositionStatus.OPEN:
            raise ValueError("Only open positions can be cancelled")
        
        self.status = PositionStatus.CANCELLED
        logger.info(f"Position {self.position_id} cancelled: {reason}")
    
    def modify_stop_loss(self, new_stop_loss: Decimal) -> None:
        """Modify stop loss level"""
        if not self.is_open:
            raise ValueError("Cannot modify closed position")
        
        self.stop_loss = new_stop_loss
        logger.debug(f"Modified stop loss for {self.position_id} to {new_stop_loss}")
    
    def modify_take_profit(self, new_take_profit: Decimal) -> None:
        """Modify take profit level"""
        if not self.is_open:
            raise ValueError("Cannot modify closed position")
        
        self.take_profit = new_take_profit
        logger.debug(f"Modified take profit for {self.position_id} to {new_take_profit}")
    
    def get_risk_metrics(self, current_price: Optional[Decimal] = None) -> dict:
        """Get risk metrics for the position"""
        if not self.is_open:
            return {}
        
        price = current_price or self.current_price
        if price is None:
            return {}
        
        # Current P&L
        current_pnl = self.update_price(price)
        
        # Risk metrics
        entry_value = self.entry_price * self.quantity
        current_value = price * self.quantity
        
        # Percentage change
        pct_change = ((price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0
        
        # Distance to stop/take profit
        stop_distance = None
        take_distance = None
        
        if self.stop_loss:
            if self.is_long:
                stop_distance = ((price - self.stop_loss) / price * 100)
            else:
                stop_distance = ((self.stop_loss - price) / price * 100)
        
        if self.take_profit:
            if self.is_long:
                take_distance = ((self.take_profit - price) / price * 100)
            else:
                take_distance = ((price - self.take_profit) / price * 100)
        
        return {
            'current_pnl': float(current_pnl),
            'current_value': float(current_value),
            'pct_change': float(pct_change),
            'stop_distance_pct': float(stop_distance) if stop_distance else None,
            'take_distance_pct': float(take_distance) if take_distance else None,
            'risk_reward_ratio': float(abs(self.take_profit - self.entry_price) / abs(self.stop_loss - self.entry_price)) if self.stop_loss else None,
        }
    
    def to_dict(self) -> dict:
        """Convert position to dictionary"""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'position_type': self.position_type.value,
            'status': self.status.value,
            'entry_price': float(self.entry_price),
            'quantity': float(self.quantity),
            'entry_time': self.entry_time.isoformat(),
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'current_price': float(self.current_price) if self.current_price else None,
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl': float(self.realized_pnl),
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'strategy_id': self.strategy_id,
            'confidence': float(self.confidence) if self.confidence else None,
            'tags': self.tags,
            'duration_seconds': self.duration,
        }
    
    def __str__(self) -> str:
        """String representation"""
        status_emoji = "🟢" if self.is_open else "🔴"
        direction_emoji = "📈" if self.is_long else "📉"
        
        return (
            f"{status_emoji} {direction_emoji} {self.position_type.value} "
            f"{self.quantity} {self.symbol} @ ${self.entry_price} "
            f"P&L: ${self.unrealized_pnl:.2f}"
        )
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"Position(id={self.position_id}, symbol={self.symbol}, "
                f"type={self.position_type.value}, status={self.status.value}, "
                f"quantity={self.quantity}, entry_price={self.entry_price}, "
                f"unrealized_pnl={self.unrealized_pnl})")

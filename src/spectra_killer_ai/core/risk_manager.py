"""
Advanced Risk Management System
Comprehensive risk monitoring, position sizing, and limit enforcement
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

from ..utils.helpers import calculate_position_risk, format_currency
from .events import EventBus, Event, EventType, event_bus

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimit:
    """Individual risk limit definition"""
    name: str
    limit_type: str  # 'position_size', 'daily_loss', 'portfolio_risk', etc.
    max_value: Union[float, Decimal]
    current_value: Union[float, Decimal] = 0
    enabled: bool = True
    
    @property
    def utilization(self) -> float:
        """Get utilization percentage"""
        if not self.enabled or self.max_value == 0:
            return 0.0
        return float(self.current_value) / float(self.max_value) * 100
    
    @property
    def is_exceeded(self) -> bool:
        """Check if limit is exceeded"""
        return self.enabled and self.current_value > self.max_value


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    timestamp: datetime = field(default_factory=datetime.now)
    portfolio_value: Decimal = Decimal('0')
    total_exposure: Decimal = Decimal('0')
    daily_pnl: Decimal = Decimal('0')
    daily_trades: int = 0
    open_positions_count: int = 0
    var_95: Decimal = Decimal('0')
    cvar_95: Decimal = Decimal('0')
    beta: Decimal = Decimal('0')
    correlation_to_market: Decimal = Decimal('0')
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': float(self.portfolio_value),
            'total_exposure': float(self.total_exposure),
            'daily_pnl': float(self.daily_pnl),
            'daily_trades': self.daily_trades,
            'open_positions_count': self.open_positions_count,
            'var_95': float(self.var_95),
            'cvar_95': float(self.cvar_95),
            'beta': float(self.beta),
            'correlation_to_market': float(self.correlation_to_market),
        }


class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager with configuration
        
        Args:
            config: Risk management configuration
        """
        self.config = config
        
        # Core risk parameters
        self.max_position_size = Decimal(str(config.get('max_position_size', 10000)))
        self.max_risk_per_trade = Decimal(str(config.get('max_risk_per_trade', 0.02)))  # 2%
        self.max_daily_loss = Decimal(str(config.get('max_daily_loss', 0.05)))  # 5%
        self.max_positions = config.get('max_positions', 3)
        self.max_portfolio_risk = Decimal(str(config.get('max_portfolio_risk', 0.10)))  # 10%
        
        # Risk limits tracking
        self.risk_limits: Dict[str, RiskLimit] = self._initialize_risk_limits()
        
        # Daily tracking
        self.daily_stats = {
            'trades': 0,
            'pnl': Decimal('0'),
            'volume': Decimal('0'),
            'losses': 0,
            'wins': 0,
            'last_reset': datetime.now().date()
        }
        
        # Risk metrics
        self.current_metrics = RiskMetrics()
        self.metrics_history: List[RiskMetrics] = []
        
        # Risk scoring
        self.risk_score = 0.0  # 0-100, higher = more risky
        self.risk_level = RiskLevel.LOW
        
        # Subscribe to events
        event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
        event_bus.subscribe(EventType.POSITION_OPENED, self._on_position_opened)
        event_bus.subscribe(EventType.POSITION_CLOSED, self._on_position_closed)
        event_bus.subscribe(EventType.RISK_LIMIT_EXCEEDED, self._on_risk_limit_exceeded)
    
    def _initialize_risk_limits(self) -> Dict[str, RiskLimit]:
        """Initialize risk limits from configuration"""
        limits = {}
        
        # Position size limits
        limits['max_position_size'] = RiskLimit(
            name="Maximum Position Size",
            limit_type="position_size",
            max_value=self.max_position_size
        )
        
        # Per-trade risk limits
        limits['max_risk_per_trade'] = RiskLimit(
            name="Maximum Risk Per Trade",
            limit_type="risk_per_trade",
            max_value=self.max_risk_per_trade
        )
        
        # Daily loss limit
        limits['max_daily_loss'] = RiskLimit(
            name="Maximum Daily Loss",
            limit_type="daily_loss",
            max_value=self.max_daily_loss
        )
        
        # Position count limits
        limits['max_positions'] = RiskLimit(
            name="Maximum Positions",
            limit_type="position_count",
            max_value=self.max_positions
        )
        
        # Portfolio risk limits
        limits['max_portfolio_risk'] = RiskLimit(
            name="Maximum Portfolio Risk",
            limit_type="portfolio_risk",
            max_value=self.max_portfolio_risk
        )
        
        return limits
    
    def calculate_position_size(self, account_balance: Decimal, signal_confidence: float,
                             current_price: Decimal, stop_loss: Decimal) -> Decimal:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            account_balance: Current account balance
            signal_confidence: Signal confidence (0-1)
            current_price: Current price
            stop_loss: Stop loss price
            
        Returns:
            Recommended position size
        """
        # Base risk amount (2% of account)
        base_risk = account_balance * self.max_risk_per_trade
        
        # Adjust for confidence (0.5x to 1.0x based on confidence)
        confidence_multiplier = 0.5 + 0.5 * signal_confidence
        adjusted_risk = base_risk * confidence_multiplier
        
        # Calculate position size based on stop loss distance
        stop_distance = abs(current_price - stop_loss)
        if stop_distance == 0:
            logger.warning("Stop loss distance is zero, using minimum position size")
            return Decimal('1')
        
        position_size = adjusted_risk / stop_distance
        
        # Apply maximum limits
        max_by_balance = account_balance * Decimal('0.2')  # Max 20% of account
        max_by_config = self.max_position_size
        
        # Use the most restrictive limit
        position_size = min(position_size, max_by_balance, max_by_config)
        
        # Minimum position size
        min_size = Decimal('1')
        position_size = max(position_size, min_size)
        
        return position_size.quantize(Decimal('0.01'))
    
    def calculate_stop_loss(self, entry_price: Decimal, position_type: str,
                          atr: Optional[Decimal] = None, volatility_factor: float = 2.0) -> Decimal:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            position_type: 'BUY' or 'SELL'
            atr: Average True Range for dynamic stops
            volatility_factor: ATR multiplier
            
        Returns:
            Stop loss price
        """
        if atr is not None:
            stop_distance = atr * Decimal(str(volatility_factor))
        else:
            # Default stop loss based on position type
            if position_type.upper() == 'BUY':
                stop_distance = Decimal('20')  # Default $20 for gold
            else:
                stop_distance = Decimal('20')
        
        if position_type.upper() == 'BUY':
            return entry_price - stop_distance
        else:  # SELL
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: Decimal, position_type: str,
                            stop_loss: Decimal, risk_reward_ratio: float = 2.0) -> Decimal:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            position_type: 'BUY' or 'SELL'
            stop_loss: Stop loss price
            risk_reward_ratio: Risk/reward ratio
            
        Returns:
            Take profit price
        """
        risk_distance = abs(entry_price - stop_loss)
        profit_distance = risk_distance * Decimal(str(risk_reward_ratio))
        
        if position_type.upper() == 'BUY':
            return entry_price + profit_distance
        else:  # SELL
            return entry_price - profit_distance
    
    def can_open_position(self, portfolio_value: Decimal, current_positions: int,
                         daily_pnl: Decimal, proposed_size: Decimal,
                         proposed_risk: Decimal) -> Tuple[bool, List[str]]:
        """
        Check if a new position can be opened
        
        Args:
            portfolio_value: Current portfolio value
            current_positions: Number of current positions
            daily_pnl: Current daily P&L
            proposed_size: Proposed position size
            proposed_risk: Proposed risk amount
            
        Returns:
            Tuple of (can_open, list_of_reasons)
        """
        reasons = []
        
        # Check position count limit
        if current_positions >= self.max_positions:
            reasons.append(f"Maximum positions reached ({self.max_positions})")
        
        # Check daily loss limit
        daily_loss_percentage = -min(daily_pnl, 0) / portfolio_value
        if daily_loss_percentage >= self.max_daily_loss:
            reasons.append(f"Daily loss limit reached ({self.max_daily_loss * 100}%)")
        
        # Check position size limit
        if proposed_size > self.max_position_size:
            reasons.append(f"Position size exceeds maximum ({format_currency(self.max_position_size)})")
        
        # Check per-trade risk limit
        risk_percentage = proposed_risk / portfolio_value
        if risk_percentage > self.max_risk_per_trade:
            reasons.append(f"Per-trade risk exceeds maximum ({self.max_risk_per_trade * 100}%)")
        
        # Check portfolio risk limit
        portfolio_risk_usage = (current_positions * self.max_risk_per_trade) + risk_percentage
        if portfolio_risk_usage > self.max_portfolio_risk:
            reasons.append(f"Portfolio risk would exceed maximum ({self.max_portfolio_risk * 100}%)")
        
        return len(reasons) == 0, reasons
    
    def update_daily_stats(self, trade_pnl: Decimal, trade_volume: Decimal) -> None:
        """
        Update daily trading statistics
        
        Args:
            trade_pnl: P&L from completed trade
            trade_volume: Volume of trade
        """
        self.daily_stats['trades'] += 1
        self.daily_stats['pnl'] += trade_pnl
        self.daily_stats['volume'] += trade_volume
        
        if trade_pnl > 0:
            self.daily_stats['wins'] += 1
        elif trade_pnl < 0:
            self.daily_stats['losses'] += 1
        
        # Update risk limits
        self.risk_limits['max_daily_loss'].current_value = -min(self.daily_stats['pnl'], 0) / Decimal('10000')  # Assuming 10k base
        
        # Check for daily loss limit breach
        if self.risk_limits['max_daily_loss'].is_exceeded:
            event_bus.publish(Event(
                type=EventType.RISK_LIMIT_EXCEEDED,
                data={
                    'limit_name': 'max_daily_loss',
                    'current_value': float(self.risk_limits['max_daily_loss'].current_value),
                    'max_value': float(self.risk_limits['max_daily_loss'].max_value)
                },
                source="RiskManager"
            ))
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (typically called at start of new day)"""
        self.daily_stats = {
            'trades': 0,
            'pnl': Decimal('0'),
            'volume': Decimal('0'),
            'losses': 0,
            'wins': 0,
            'last_reset': datetime.now().date()
        }
        
        # Reset daily limits
        for limit in self.risk_limits.values():
            if 'daily' in limit.limit_type:
                limit.current_value = 0
        
        logger.info("Daily risk statistics reset")
    
    def calculate_risk_score(self, portfolio_value: Decimal, positions_data: List[Dict]) -> float:
        """
        Calculate overall risk score (0-100)
        
        Args:
            portfolio_value: Current portfolio value
            positions_data: List of position data
            
        Returns:
            Risk score
        """
        score = 0.0
        
        # Position count contribution (0-15 points)
        position_count = len(positions_data)
        score += min(15, position_count * 3)
        
        # Daily loss contribution (0-25 points)
        daily_loss_pct = -min(self.daily_stats['pnl'], 0) / portfolio_value if portfolio_value > 0 else 0
        loss_score = min(25, daily_loss_pct * 500)  # 5% loss = 25 points
        score += loss_score
        
        # Portfolio risk usage contribution (0-20 points)
        portfolio_risk_usage = position_count * float(self.max_risk_per_trade)
        risk_score = min(20, portfolio_risk_usage * 200)
        score += risk_score
        
        # Volatility contribution (0-20 points)
        # This would be calculated from actual position volatility
        score += 10  # Placeholder
        
        # Concentration risk (0-20 points)
        # Check if positions are too concentrated in one symbol
        symbols = [p.get('symbol') for p in positions_data]
        if symbols:
            max_concentration = max([symbols.count(s) for s in set(symbols)]) / len(symbols)
            score += min(20, max_concentration * 40)
        
        return min(100, score)
    
    def update_risk_level(self, score: float) -> None:
        """Update risk level based on score"""
        if score >= 75:
            self.risk_level = RiskLevel.CRITICAL
        elif score >= 50:
            self.risk_level = RiskLevel.HIGH
        elif score >= 25:
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW
    
    def get_risk_report(self) -> Dict:
        """Get comprehensive risk report"""
        return {
            'risk_score': self.risk_score,
            'risk_level': self.risk_level.value,
            'limits': {
                name: {
                    'name': limit.name,
                    'current': float(limit.current_value),
                    'maximum': float(limit.max_value),
                    'utilization': limit.utilization,
                    'exceeded': limit.is_exceeded,
                    'enabled': limit.enabled
                } for name, limit in self.risk_limits.items()
            },
            'daily_stats': {
                'trades': self.daily_stats['trades'],
                'pnl': float(self.daily_stats['pnl']),
                'volume': float(self.daily_stats['volume']),
                'wins': self.daily_stats['wins'],
                'losses': self.daily_stats['losses'],
                'win_rate': (self.daily_stats['wins'] / self.daily_stats['trades'] * 100) if self.daily_stats['trades'] > 0 else 0,
                'last_reset': self.daily_stats['last_reset'].isoformat()
            },
            'current_metrics': self.current_metrics.to_dict()
        }
    
    async def _on_order_filled(self, event: Event) -> None:
        """Handle order filled event"""
        # Update daily statistics if this is a closing order
        order_data = event.data
        if order_data.get('pnl'):
            self.update_daily_stats(
                Decimal(str(order_data.get('pnl', 0))),
                Decimal(str(order_data.get('volume', 0)))
            )
    
    async def _on_position_opened(self, event: Event) -> None:
        """Handle position opened event"""
        # Update position count
        self.risk_limits['max_positions'].current_value += 1
    
    async def _on_position_closed(self, event: Event) -> None:
        """Handle position closed event"""
        # Update position count
        self.risk_limits['max_positions'].current_value -= 1
    
    async def _on_risk_limit_exceeded(self, event: Event) -> None:
        """Handle risk limit exceeded event"""
        limit_name = event.data.get('limit_name')
        logger.warning(f"Risk limit exceeded: {limit_name}")
        
        # Could implement automatic actions here, like reducing position size or stopping trading
        if limit_name == 'max_daily_loss':
            logger.critical("Daily loss limit exceeded - consider stopping trading")
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate risk management configuration
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Validate risk percentages
        if self.max_risk_per_trade <= 0 or self.max_risk_per_trade > 0.10:
            issues.append("max_risk_per_trade should be between 0 and 10%")
        
        if self.max_daily_loss <= 0 or self.max_daily_loss > 0.20:
            issues.append("max_daily_loss should be between 0 and 20%")
        
        # Validate position limits
        if self.max_positions <= 0 or self.max_positions > 20:
            issues.append("max_positions should be between 1 and 20")
        
        if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 0.50:
            issues.append("max_portfolio_risk should be between 0 and 50%")
        
        return len(issues) == 0, issues

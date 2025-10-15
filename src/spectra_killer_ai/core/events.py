"""
Event system for trading engine
Provides pub/sub architecture for decoupled communication
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types in the system"""
    # Market events
    MARKET_DATA = "market_data"
    PRICE_UPDATE = "price_update"
    
    # Trading events
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    
    # Risk events
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"
    
    # System events
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Event:
    """Base event class"""
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    
    def __post_init__(self):
        """Validate event data"""
        if not isinstance(self.type, EventType):
            raise ValueError(f"Event type must be EventType, got {type(self.type)}")


class EventBus:
    """Async event bus for trading system communication"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed {handler.__name__} to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """Unsubscribe from an event type"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed {handler.__name__} from {event_type.value}")
            except ValueError:
                pass  # Handler not found
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers"""
        logger.debug(f"Publishing event: {event.type.value}")
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify subscribers
        if event.type in self._subscribers:
            tasks = []
            for handler in self._subscribers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        tasks.append(asyncio.create_task(asyncio.to_thread(handler, event)))
                except Exception as e:
                    logger.error(f"Error creating task for handler {handler}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         limit: Optional[int] = None) -> List[Event]:
        """Get event history, optionally filtered by type"""
        history = self._event_history
        
        if event_type:
            history = [e for e in history if e.type == event_type]
        
        if limit:
            history = history[-limit:]
        
        return history.copy()
    
    def clear_history(self):
        """Clear event history"""
        self._event_history.clear()
        logger.info("Event history cleared")
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type"""
        return len(self._subscribers.get(event_type, []))


# Global event bus instance
event_bus = EventBus()

"""
Test position management functionality
"""

import pytest
from decimal import Decimal
from datetime import datetime

from spectra_killer_ai.core.position import Position, PositionType, PositionStatus


class TestPosition:
    """Test Position class functionality"""
    
    def test_position_creation(self):
        """Test basic position creation"""
        position = Position(
            position_id="test_pos_001",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.50'),
            quantity=Decimal('1.0')
        )
        
        assert position.position_id == "test_pos_001"
        assert position.symbol == "XAUUSD"
        assert position.position_type == PositionType.LONG
        assert position.entry_price == Decimal('2050.50')
        assert position.quantity == Decimal('1.0')
        assert position.is_open
        assert position.is_long
        assert not position.is_short
        assert position.status == PositionStatus.OPEN
    
    def test_position_short(self):
        """Test short position creation"""
        position = Position(
            position_id="test_short_001",
            symbol="XAUUSD",
            position_type=PositionType.SHORT,
            entry_price=Decimal('2050.50'),
            quantity=Decimal('2.0')
        )
        
        assert position.is_short
        assert not position.is_long
        assert position.position_type == PositionType.SHORT
    
    def test_position_update_price(self):
        """Test updating position price and P&L calculation"""
        position = Position(
            position_id="test_update_001",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0')
        )
        
        # Update price for long position
        new_price = Decimal('2060.00')
        pnl = position.update_price(new_price)
        
        expected_pnl = (new_price - position.entry_price) * position.quantity
        assert pnl == expected_pnl
        assert position.unrealized_pnl == expected_pnl
        assert position.current_price == new_price
    
    def test_position_update_price_short(self):
        """Test updating price for short position"""
        position = Position(
            position_id="test_short_update_001",
            symbol="XAUUSD",
            position_type=PositionType.SHORT,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0')
        )
        
        # Update price for short position (price down = profit)
        new_price = Decimal('2040.00')
        pnl = position.update_price(new_price)
        
        expected_pnl = (position.entry_price - new_price) * position.quantity
        assert pnl == expected_pnl
        assert position.unrealized_pnl == expected_pnl
    
    def test_stop_loss_trigger_long(self):
        """Test stop loss trigger for long position"""
        position = Position(
            position_id="test_sl_long",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0'),
            stop_loss=Decimal('2040.00')
        )
        
        # Price hits stop loss
        assert position.should_stop_loss(Decimal('2039.50'))
        
        # Price above stop loss
        assert not position.should_stop_loss(Decimal('2040.50'))
    
    def test_stop_loss_trigger_short(self):
        """Test stop loss trigger for short position"""
        position = Position(
            position_id="test_sl_short",
            symbol="XAUUSD",
            position_type=PositionType.SHORT,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0'),
            stop_loss=Decimal('2060.00')
        )
        
        # Price hits stop loss
        assert position.should_stop_loss(Decimal('2060.50'))
        
        # Price below stop loss
        assert not position.should_stop_loss(Decimal('2059.50'))
    
    def test_take_profit_trigger_long(self):
        """Test take profit trigger for long position"""
        position = Position(
            position_id="test_tp_long",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0'),
            take_profit=Decimal('2070.00')
        )
        
        # Price hits take profit
        assert position.should_take_profit(Decimal('2070.50'))
        
        # Price below take profit
        assert not position.should_take_profit(Decimal('2069.50'))
    
    def test_position_close(self):
        """Test position closing"""
        position = Position(
            position_id="test_close_001",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0')
        )
        
        exit_price = Decimal('2060.00')
        realized_pnl = position.close(exit_price)
        
        expected_pnl = (exit_price - position.entry_price) * position.quantity
        assert realized_pnl == expected_pnl
        assert position.realized_pnl == expected_pnl
        assert position.exit_price == exit_price
        assert position.is_closed
        assert not position.is_open
        assert position.status == PositionStatus.CLOSED
        assert position.duration is not None
    
    def test_position_close_short(self):
        """Test closing short position"""
        position = Position(
            position_id="test_close_short_001",
            symbol="XAUUSD",
            position_type=PositionType.SHORT,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0')
        )
        
        exit_price = Decimal('2040.00')  # Price down = profit for short
        realized_pnl = position.close(exit_price)
        
        expected_pnl = (position.entry_price - exit_price) * position.quantity
        assert realized_pnl == expected_pnl
        assert position.realized_pnl == expected_pnl
    
    def test_position_invalid_creation(self):
        """Test invalid position creation"""
        # Invalid quantity
        with pytest.raises(ValueError):
            Position(
                position_id="invalid_001",
                symbol="XAUUSD",
                position_type=PositionType.LONG,
                entry_price=Decimal('2050.00'),
                quantity=Decimal('0')  # Invalid
            )
        
        # Invalid position type
        with pytest.raises(ValueError):
            Position(
                position_id="invalid_002",
                symbol="XAUUSD",
                position_type="INVALID",  # Invalid
                entry_price=Decimal('2050.00'),
                quantity=Decimal('1.0')
            )
    
    def test_position_modify_stop_loss(self):
        """Test modifying stop loss"""
        position = Position(
            position_id="test_modify_sl",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0')
        )
        
        new_stop_loss = Decimal('2045.00')
        position.modify_stop_loss(new_stop_loss)
        
        assert position.stop_loss == new_stop_loss
    
    def test_position_modify_take_profit(self):
        """Test modifying take profit"""
        position = Position(
            position_id="test_modify_tp",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0')
        )
        
        new_take_profit = Decimal('2065.00')
        position.modify_take_profit(new_take_profit)
        
        assert position.take_profit == new_take_profit
    
    def test_position_risk_metrics(self):
        """Test risk metrics calculation"""
        position = Position(
            position_id="test_risk_metrics",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0'),
            stop_loss=Decimal('2040.00'),
            take_profit=Decimal('2070.00')
        )
        
        current_price = Decimal('2055.00')
        metrics = position.get_risk_metrics(current_price)
        
        assert 'current_pnl' in metrics
        assert 'pct_change' in metrics
        assert 'stop_distance_pct' in metrics
        assert 'take_distance_pct' in metrics
        assert 'risk_reward_ratio' in metrics
        
        # Check P&L calculation
        expected_pnl = (current_price - position.entry_price) * position.quantity
        assert metrics['current_pnl'] == expected_pnl
    
    def test_position_to_dict(self):
        """Test position serialization"""
        timestamp = datetime.now()
        position = Position(
            position_id="test_dict_001",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0'),
            confidence=Decimal('0.75'),
            tags=['test', 'sample']
        )
        
        position_dict = position.to_dict()
        
        assert position_dict['position_id'] == "test_dict_001"
        assert position_dict['symbol'] == "XAUUSD"
        assert position_dict['position_type'] == "BUY"
        assert position_dict['entry_price'] == 2050.0
        assert position_dict['quantity'] == 1.0
        assert position_dict['confidence'] == 0.75
        assert position_dict['tags'] == ['test', 'sample']
    
    def test_position_str_representation(self):
        """Test position string representation"""
        position = Position(
            position_id="test_str_001",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0')
        )
        
        str_repr = str(position)
        assert "🟢" in str_repr  # Open position emoji
        assert "📈" in str_repr  # Long position emoji
        assert "BUY" in str_repr
        assert "XAUUSD" in str_repr
        assert "2050.00" in str_repr
    
    def test_position_repr(self):
        """Test position detailed representation"""
        position = Position(
            position_id="test_repr_001",
            symbol="XAUUSD",
            position_type=PositionType.LONG,
            entry_price=Decimal('2050.00'),
            quantity=Decimal('1.0')
        )
        
        repr_str = repr(position)
        assert "Position" in repr_str
        assert "test_repr_001" in repr_str
        assert "XAUUSD" in repr_str
        assert "BUY" in repr_str
        assert "OPEN" in repr_str

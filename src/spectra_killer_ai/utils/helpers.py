"""
Helper utility functions with security integrations
"""

import uuid
from decimal import Decimal
from typing import Union, Tuple, Dict, List

# Import security modules
from ..security.credential_manager import get_credential_manager, secure_get_env
from ..security.secure_logging import setup_secure_logging, log_security_event
from ..security.validators import TradingValidator, ValidationError


def format_currency(amount: Union[float, Decimal], currency: str = "$", decimals: int = 2) -> str:
    """
    Format amount as currency
    
    Args:
        amount: Amount to format
        currency: Currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    if amount is None:
        return f"{currency}0.00"
    
    amount = float(amount)
    return f"{currency}{amount:,.{decimals}f}"


def format_percentage(value: Union[float, Decimal], decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format (as decimal, e.g., 0.05 for 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "0.00%"
    
    value = float(value) * 100
    return f"{value:.{decimals}f}%"


def generate_id(prefix: str = "") -> str:
    """
    Generate unique ID with optional prefix
    
    Args:
        prefix: Optional prefix
        
    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{unique_id}" if prefix else unique_id


def calculate_position_risk(position_size: Decimal, entry_price: Decimal, 
                          stop_loss: Decimal) -> Decimal:
    """
    Calculate position risk in currency units
    
    Args:
        position_size: Position size
        entry_price: Entry price
        stop_loss: Stop loss price
        
    Returns:
        Risk amount
    """
    if position_size <= 0 or entry_price <= 0:
        return Decimal('0')
    
    risk_per_unit = abs(entry_price - stop_loss)
    return risk_per_unit * position_size


def calculate_required_margin(position_value: Decimal, margin_requirement: Decimal = Decimal('0.1')) -> Decimal:
    """
    Calculate required margin for position
    
    Args:
        position_value: Total position value
        margin_requirement: Margin requirement as decimal (0.1 = 10%)
        
    Returns:
        Required margin
    """
    return position_value * margin_requirement


def round_to_tick_size(price: Decimal, tick_size: Decimal) -> Decimal:
    """
    Round price to tick size
    
    Args:
        price: Price to round
        tick_size: Minimum tick size
        
    Returns:
        Rounded price
    """
    return (price / tick_size).quantize(Decimal('1')) * tick_size

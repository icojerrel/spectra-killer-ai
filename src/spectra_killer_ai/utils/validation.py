"""
Validation utilities for trading system
"""

from decimal import Decimal
from typing import Dict, List, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


def validate_trading_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate trading configuration
    
    Args:
        config: Trading configuration dictionary
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required sections
    required_sections = ['trading', 'risk_management']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")
    
    # Validate trading section
    if 'trading' in config:
        trading = config['trading']
        
        # Check required fields
        required_trading_fields = ['symbol', 'timeframe', 'initial_balance', 'mode']
        for field in required_trading_fields:
            if field not in trading:
                issues.append(f"Missing trading field: {field}")
        
        # Validate values
        if 'initial_balance' in trading:
            try:
                balance = float(trading['initial_balance'])
                if balance <= 0:
                    issues.append("Initial balance must be positive")
                if balance > 1000000:
                    issues.append("Initial balance seems too high (> 1M)")
            except (ValueError, TypeError):
                issues.append("Initial balance must be a valid number")
        
        if 'mode' in trading:
            valid_modes = ['paper', 'live', 'backtest']
            if trading['mode'] not in valid_modes:
                issues.append(f"Invalid trading mode: {trading['mode']}")
        
        if 'timeframe' in trading:
            valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
            if trading['timeframe'] not in valid_timeframes:
                issues.append(f"Invalid timeframe: {trading['timeframe']}")
    
    # Validate risk management section
    if 'risk_management' in config:
        risk = config['risk_management']
        
        # Check risk percentages
        percentage_fields = ['max_risk_per_trade', 'max_daily_loss', 'max_portfolio_risk']
        for field in percentage_fields:
            if field in risk:
                try:
                    value = float(risk[field])
                    if not (0 < value <= 1):
                        issues.append(f"{field} should be between 0 and 1 (as decimal)")
                except (ValueError, TypeError):
                    issues.append(f"{field} must be a valid number")
        
        # Check position limits
        if 'max_position_size' in risk:
            try:
                size = float(risk['max_position_size'])
                if size <= 0:
                    issues.append("Max position size must be positive")
            except (ValueError, TypeError):
                issues.append("Max position size must be a valid number")
        
        if 'max_positions' in risk:
            try:
                positions = int(risk['max_positions'])
                if not (1 <= positions <= 20):
                    issues.append("Max positions should be between 1 and 20")
            except (ValueError, TypeError):
                issues.append("Max positions must be a valid integer")
    
    return len(issues) == 0, issues


def validate_position_size(size: Union[float, Decimal], account_balance: Union[float, Decimal],
                          max_position_size: Union[float, Decimal]) -> Tuple[bool, List[str]]:
    """
    Validate position size
    
    Args:
        size: Proposed position size
        account_balance: Current account balance
        max_position_size: Maximum allowed position size
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        size = float(size)
        balance = float(account_balance)
        max_size = float(max_position_size)
        
        if size <= 0:
            issues.append("Position size must be positive")
        
        if size > balance:
            issues.append("Position size cannot exceed account balance")
        
        if size > max_size:
            issues.append(f"Position size exceeds maximum: {max_size}")
        
        # Check for reasonable minimum
        if size < 0.01:
            issues.append("Position size too small (minimum 0.01)")
            
    except (ValueError, TypeError) as e:
        issues.append(f"Invalid position size format: {e}")
    
    return len(issues) == 0, issues


def validate_price(price: Union[float, Decimal], symbol: str = "") -> Tuple[bool, List[str]]:
    """
    Validate price value
    
    Args:
        price: Price to validate
        symbol: Symbol for context
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        price = float(price)
        
        if price <= 0:
            issues.append("Price must be positive")
        
        # Symbol-specific validation
        if symbol == 'XAUUSD':
            if price < 1000 or price > 5000:
                issues.append("XAUUSD price seems unreasonable (should be 1000-5000)")
        elif symbol == 'EURUSD':
            if price < 0.5 or price > 2.0:
                issues.append("EURUSD price seems unreasonable (should be 0.5-2.0)")
                
    except (ValueError, TypeError):
        issues.append("Price must be a valid number")
    
    return len(issues) == 0, issues


def validate_stop_loss_take_profit(entry_price: Union[float, Decimal],
                                 stop_loss: Union[float, Decimal],
                                 take_profit: Union[float, Decimal],
                                 position_type: str) -> Tuple[bool, List[str]]:
    """
    Validate stop loss and take profit levels
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        position_type: 'BUY' or 'SELL'
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        entry = float(entry_price)
        sl = float(stop_loss)
        tp = float(take_profit)
        
        if position_type.upper() == 'BUY':
            if sl >= entry:
                issues.append("Buy position stop loss must be below entry price")
            if tp <= entry:
                issues.append("Buy position take profit must be above entry price")
        elif position_type.upper() == 'SELL':
            if sl <= entry:
                issues.append("Sell position stop loss must be above entry price")
            if tp >= entry:
                issues.append("Sell position take profit must be below entry price")
        else:
            issues.append("Position type must be 'BUY' or 'SELL'")
        
        # Check risk/reward ratio
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        
        if reward <= 0:
            issues.append("Take profit must be different from entry price")
        
        if risk <= 0:
            issues.append("Stop loss must be different from entry price")
        
        if risk > 0 and reward > 0:
            rr_ratio = reward / risk
            if rr_ratio < 0.5:
                issues.append("Risk/reward ratio too low (< 0.5)")
        
    except (ValueError, TypeError):
        issues.append("All price values must be valid numbers")
    
    return len(issues) == 0, issues


def validate_symbol(symbol: str) -> Tuple[bool, List[str]]:
    """
    Validate trading symbol
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not symbol:
        issues.append("Symbol cannot be empty")
        return False, issues
    
    if not isinstance(symbol, str):
        issues.append("Symbol must be a string")
        return False, issues
    
    # Common valid symbols
    valid_symbols = [
        'XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD',
        'AUDUSD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY'
    ]
    
    if symbol not in valid_symbols:
        issues.append(f"Unknown symbol: {symbol}. Valid symbols: {', '.join(valid_symbols)}")
    
    # Check format
    if len(symbol) != 6 or not symbol.isalnum():
        issues.append("Symbol format invalid (should be 6 alphanumeric characters)")
    
    return len(issues) == 0, issues


def validate_timeframe(timeframe: str) -> Tuple[bool, List[str]]:
    """
    Validate timeframe
    
    Args:
        timeframe: Timeframe to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not timeframe:
        issues.append("Timeframe cannot be empty")
        return False, issues
    
    valid_timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']
    
    if timeframe not in valid_timeframes:
        issues.append(f"Invalid timeframe: {timeframe}. Valid: {', '.join(valid_timeframes)}")
    
    return len(issues) == 0, issues


def validate_api_credential(credential_type: str, credential: str) -> Tuple[bool, List[str]]:
    """
    Validate API credentials
    
    Args:
        credential_type: Type of credential
        credential: Credential value
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not credential:
        issues.append(f"{credential_type} cannot be empty")
        return False, issues
    
    if not isinstance(credential, str):
        issues.append(f"{credential_type} must be a string")
        return False, issues
    
    length = len(credential)
    
    if credential_type.lower() == 'password':
        if length < 8:
            issues.append("Password must be at least 8 characters")
        if not any(c.isdigit() for c in credential):
            issues.append("Password should contain at least one number")
    
    elif credential_type.lower() == 'api_key':
        if length < 16:
            issues.append("API key seems too short")
        if ' ' in credential:
            issues.append("API key should not contain spaces")
    
    return len(issues) == 0, issues


def validate_email(email: str) -> Tuple[bool, List[str]]:
    """
    Validate email address
    
    Args:
        email: Email to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not email:
        issues.append("Email cannot be empty")
        return False, issues
    
    if '@' not in email or '.' not in email:
        issues.append("Invalid email format")
    
    if len(email) < 5:
        issues.append("Email too short")
    
    if len(email) > 254:
        issues.append("Email too long")
    
    return len(issues) == 0, issues


def validate_phone_number(phone: str) -> Tuple[bool, List[str]]:
    """
    Validate phone number
    
    Args:
        phone: Phone number to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not phone:
        issues.append("Phone number cannot be empty")
        return False, issues
    
    # Remove common characters
    clean_phone = phone.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
    
    if not clean_phone.isdigit():
        issues.append("Phone number should contain only digits")
    
    if len(clean_phone) < 10:
        issues.append("Phone number too short")
    
    if len(clean_phone) > 15:
        issues.append("Phone number too long")
    
    return len(issues) == 0, issues

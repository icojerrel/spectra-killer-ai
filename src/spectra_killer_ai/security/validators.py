"""
Input Validation and Security Validators
Comprehensive validation for trading parameters and API inputs
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, validator, Field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass


class TradingSymbol(str, Enum):
    """Valid trading symbols"""
    XAUUSD = "XAUUSD"
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"
    USDJPY = "USDJPY"


class TimeFrame(str, Enum):
    """Valid timeframes"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class OrderType(str, Enum):
    """Valid order types"""
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class RiskLevel(str, Enum):
    """Risk levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TradingParameters(BaseModel):
    """Validated trading parameters"""
    
    symbol: TradingSymbol = Field(..., description="Trading symbol")
    timeframe: TimeFrame = Field(..., description="Timeframe")
    order_type: OrderType = Field(..., description="Order type")
    
    volume: Decimal = Field(..., gt=0, le=100, description="Position volume (0.01-100)")
    price: Optional[Decimal] = Field(None, gt=0, description="Entry price (for limit/stop orders)")
    
    stop_loss: Optional[Decimal] = Field(None, gt=0, description="Stop loss price")
    take_profit: Optional[Decimal] = Field(None, gt=0, description="Take profit price")
    
    confidence: Decimal = Field(..., ge=0, le=1, description="Signal confidence (0-1)")
    risk_level: RiskLevel = Field(RiskLevel.MEDIUM, description="Risk level")
    
    max_slippage: Decimal = Field(Decimal('5'), ge=0, le=50, description="Max slippage in pips")
    
    @validator('stop_loss')
    def validate_stop_loss(cls, v, values):
        """Validate stop loss logic"""
        if v is None:
            return v
            
        price = values.get('price')
        if price is None:
            raise ValidationError("Price required when setting stop loss")
        
        order_type = values.get('order_type')
        if order_type in [OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP]:
            if v >= price:
                raise ValidationError("Stop loss must be below entry price for BUY orders")
        else:  # SELL orders
            if v <= price:
                raise ValidationError("Stop loss must be above entry price for SELL orders")
        
        # Check stop loss distance
        distance = abs(price - v)
        if distance > Decimal('100'):  # Max 100 pips
            raise ValidationError("Stop loss too far from entry price")
        if distance < Decimal('5'):  # Min 5 pips
            raise ValidationError("Stop loss too close to entry price")
        
        return v
    
    @validator('take_profit')
    def validate_take_profit(cls, v, values):
        """Validate take profit logic"""
        if v is None:
            return v
            
        price = values.get('price')
        if price is None:
            raise ValidationError("Price required when setting take profit")
        
        order_type = values.get('order_type')
        if order_type in [OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP]:
            if v <= price:
                raise ValidationError("Take profit must be above entry price for BUY orders")
        else:  # SELL orders
            if v >= price:
                raise ValidationError("Take profit must be below entry price for SELL orders")
        
        # Check take profit distance
        distance = abs(v - price)
        if distance > Decimal('500'):  # Max 500 pips
            raise ValidationError("Take profit too far from entry price")
        if distance < Decimal('5'):  # Min 5 pips
            raise ValidationError("Take profit too close to entry price")
        
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v, values):
        """Validate confidence level"""
        risk_level = values.get('risk_level', RiskLevel.MEDIUM)
        
        # High confidence required for high-risk trades
        if risk_level == RiskLevel.HIGH and v < Decimal('0.7'):
            raise ValidationError("High confidence (0.7+) required for high-risk trades")
        elif risk_level == RiskLevel.CRITICAL and v < Decimal('0.8'):
            raise ValidationError("Very high confidence (0.8+) required for critical risk trades")
        
        return v


class APIRequest(BaseModel):
    """Validated API request parameters"""
    
    api_key: str = Field(..., min_length=10, max_length=256)
    request_id: str = Field(..., min_length=1, max_length=100)
    timestamp: datetime = Field(...)
    
    @validator('api_key')
    def validate_api_key(cls, v):
        """Validate API key format"""
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValidationError("Invalid API key format")
        return v
    
    @validator('request_id')
    def validate_request_id(cls, v):
        """Validate request ID format"""
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValidationError("Invalid request ID format")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp (not too old or in future)"""
        now = datetime.utcnow()
        if v > now + timedelta(minutes=5):
            raise ValidationError("Timestamp too far in future")
        if v < now - timedelta(hours=1):
            raise ValidationError("Timestamp too old")
        return v


class TradingValidator:
    """Comprehensive trading parameter validator"""
    
    MAX_DAILY_TRADES = 50
    MAX_POSITION_SIZE = Decimal('100')
    MAX_RISK_PER_TRADE = Decimal('0.05')  # 5%
    MIN_RISK_REWARD_RATIO = Decimal('1.5')
    
    @staticmethod
    def validate_trading_parameters(params: Dict[str, Any]) -> TradingParameters:
        """
        Validate and sanitize trading parameters
        
        Args:
            params: Raw trading parameters dictionary
            
        Returns:
            Validated TradingParameters object
            
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            # Convert decimal strings
            if 'volume' in params and isinstance(params['volume'], str):
                params['volume'] = Decimal(params['volume'])
            
            if 'price' in params and isinstance(params['price'], str):
                params['price'] = Decimal(params['price'])
            
            if 'stop_loss' in params and isinstance(params['stop_loss'], str):
                params['stop_loss'] = Decimal(params['stop_loss'])
            
            if 'take_profit' in params and isinstance(params['take_profit'], str):
                params['take_profit'] = Decimal(params['take_profit'])
            
            if 'confidence' in params and isinstance(params['confidence'], str):
                params['confidence'] = Decimal(params['confidence'])
            
            # Validate with Pydantic model
            return TradingParameters(**params)
            
        except Exception as e:
            raise ValidationError(f"Parameter validation failed: {str(e)}")
    
    @staticmethod
    def validate_risk_reward_ratio(entry_price: Decimal, stop_loss: Decimal, 
                                  take_profit: Decimal) -> bool:
        """
        Validate risk/reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if ratio is acceptable
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return False
        
        ratio = reward / risk
        return ratio >= TradingValidator.MIN_RISK_REWARD_RATIO
    
    @staticmethod
    def validate_account_balance(balance: Decimal, risk_amount: Decimal) -> bool:
        """
        Validate if risk amount is acceptable for account balance
        
        Args:
            balance: Account balance
            risk_amount: Risk amount
            
        Returns:
            True if acceptable
        """
        if balance <= 0:
            return False
        
        risk_percentage = risk_amount / balance
        return risk_percentage <= TradingValidator.MAX_RISK_PER_TRADE
    
    @staticmethod
    def validate_trading_time() -> bool:
        """
        Validate if current time is acceptable for trading
        
        Returns:
            True if trading is allowed
        """
        now = datetime.utcnow()
        
        # No trading on weekends
        if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
            return False
        
        # No trading during major news events (simplified)
        hour = now.hour
        if hour in [13, 14]:  # During major US news releases (UTC)
            return False
        
        return True
    
    @staticmethod
    def sanitize_string_input(text: str, max_length: int = 1000) -> str:
        """
        Sanitize string input to prevent injection attacks
        
        Args:
            text: Input string
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not isinstance(text, str):
            raise ValidationError("Input must be a string")
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '`', '$', '|']
        sanitized = text
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        sanitized = sanitized[:max_length]
        
        # Remove extra whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized.strip()
    
    @staticmethod
    def validate_sql_input(sql: str) -> bool:
        """
        Validate SQL input for dangerous patterns
        
        Args:
            sql: SQL string to validate
            
        Returns:
            True if safe
        """
        dangerous_patterns = [
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'INSERT\s+INTO',
            r'UPDATE\s+.*\s+SET',
            r'UNION\s+SELECT',
            r'--',
            r';',
            r'xp_',
            r'sp_',
        ]
        
        sql_upper = sql.upper()
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_upper):
                return False
        
        return True
    
    @staticmethod
    def validate_api_request(request_data: Dict[str, Any]) -> APIRequest:
        """
        Validate API request parameters
        
        Args:
            request_data: API request data
            
        Returns:
            Validated APIRequest object
            
        Raises:
            ValidationError: If request is invalid
        """
        try:
            return APIRequest(**request_data)
        except Exception as e:
            raise ValidationError(f"API request validation failed: {str(e)}")
    
    @staticmethod
    def rate_limit_check(client_id: str, max_requests: int = 100, 
                        window_minutes: int = 60) -> bool:
        """
        Simple rate limiting check (in production, use Redis or similar)
        
        Args:
            client_id: Client identifier
            max_requests: Maximum requests allowed
            window_minutes: Time window in minutes
            
        Returns:
            True if request is allowed
        """
        # This is a simplified version - in production use proper rate limit storage
        from collections import defaultdict
        import time
        
        if not hasattr(TradingValidator, '_rate_limit_store'):
            TradingValidator._rate_limit_store = defaultdict(list)
        
        now = time.time()
        window_start = now - (window_minutes * 60)
        
        # Clean old requests
        client_requests = TradingValidator._rate_limit_store[client_id]
        client_requests[:] = [req_time for req_time in client_requests if req_time > window_start]
        
        # Check if under limit
        if len(client_requests) >= max_requests:
            return False
        
        # Add current request
        client_requests.append(now)
        return True


def validate_json_payload(payload: str, schema: Optional[Dict] = None) -> Dict:
    """
    Validate JSON payload against schema
    
    Args:
        payload: JSON string
        schema: Optional JSON schema
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValidationError: If JSON is invalid
    """
    import json
    
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {str(e)}")
    
    # Basic schema validation would go here
    if schema:
        # Implement JSON schema validation
        pass
    
    return data


class SecurityHeaders:
    """Security headers for API responses"""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()'
        }

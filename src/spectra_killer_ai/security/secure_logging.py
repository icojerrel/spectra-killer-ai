"""
Secure Logging Configuration
Redacts sensitive information and implements secure logging practices
"""

import logging
import logging.handlers
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive information from logs"""
    
    # Patterns to detect and redact
    SENSITIVE_PATTERNS = [
        # API keys
        r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-]{20,})(["\']?)',
        r'(token["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-]{20,})(["\']?)',
        r'(secret["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-]{20,})(["\']?)',
        r'(password["\']?\s*[:=]\s*["\']?)([^\s"\']+)',
        
        # Credit card numbers
        r'\b(\d{4}[\s-]?){3}\d{4}\b',
        
        # Email addresses
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        # SQL injection patterns
        r'(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)',
        
        # Financial account numbers
        r'\b\d{10,12}\b',
    ]
    
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.SENSITIVE_PATTERNS]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact sensitive information"""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._redact_sensitive_data(record.msg)
        
        if hasattr(record, 'args') and record.args:
            record.args = tuple(
                self._redact_sensitive_data(str(arg)) if isinstance(arg, str) else arg 
                for arg in record.args
            )
        
        return True
    
    def _redact_sensitive_data(self, text: str) -> str:
        """Redact sensitive information from text"""
        redacted_text = text
        
        for pattern in self.compiled_patterns:
            def replace_match(match):
                if len(match.groups()) >= 3:
                    return f"{match.group(1)}{'*' * 8}{match.group(3)}"
                else:
                    return '*' * len(match.group(0))
            
            redacted_text = pattern.sub(replace_match, redacted_text)
        
        return redacted_text


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for better log analysis"""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add thread and process information for debugging
        if record.thread:
            log_data['thread_id'] = record.thread
            log_data['thread_name'] = record.threadName
        
        if record.process:
            log_data['process_id'] = record.process
            log_data['process_name'] = record.processName
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add stack trace if present
        if record.stack_info:
            log_data['stack_trace'] = self.formatStack(record.stack_info)
        
        # Add extra fields
        if self.include_extra and hasattr(record, '__dict__'):
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                             'pathname', 'filename', 'module', 'lineno', 
                             'funcName', 'created', 'msecs', 'relativeCreated',
                             'thread', 'threadName', 'processName', 'process',
                             'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                    extra_fields[key] = value
            
            if extra_fields:
                log_data['extra'] = extra_fields
        
        return json.dumps(log_data, default=str)


class EncryptedFileHandler(logging.Handler):
    """Handler that writes encrypted log files"""
    
    def __init__(self, filename: str, encryption_key: str, 
                 max_bytes: int = 10*1024*1024, backup_count: int = 5):
        super().__init__()
        self.filename = Path(filename)
        self.encryption_key = encryption_key
        self.cipher = Fernet(encryption_key.encode())
        
        # Create rotating file handler for backup management
        self.file_handler = logging.handlers.RotatingFileHandler(
            filename=str(self.filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
    
    def emit(self, record: logging.LogRecord):
        """Emit log record with encryption"""
        try:
            # Format the record
            msg = self.format(record)
            
            # Encrypt the message
            encrypted_msg = self.cipher.encrypt(msg.encode('utf-8'))
            
            # Write to file with encryption indicator
            log_entry = f"ENCRYPTED:{encrypted_msg.decode('utf-8')}\n"
            self.file_handler.stream.write(log_entry)
            self.file_handler.stream.flush()
            
        except Exception:
            self.handleError(record)


class TradingLogFilter(logging.Filter):
    """Special filter for trading logs to prevent sensitive financial data exposure"""
    
    TRADING_SENSITIVE_PATTERNS = [
        r'(account[_-]?(balance|equity))["\']?\s*[:=]\s*["\']?(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'(position[_-]?size)["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'(stop[_-]?loss|take[_-]?profit)["\']?\s*[:=]\s*["\']?(\$?\s*\d+\.?\d*)',
        r'(order[_-]?id)["\']?\s*[:=]\s*["\']?(\d+)',
        r'(trade[_-]?id)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{10,})',
    ]
    
    def __init__(self):
        super().__init__()
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.TRADING_SENSITIVE_PATTERNS]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter trading logs for sensitive financial data"""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._redact_trading_data(record.msg)
        
        # Additional check for trading-related loggers
        if any(word in record.name.lower() for word in ['trade', 'order', 'position', 'risk']):
            record.msg = self._redact_trading_data(record.msg)
        
        return True
    
    def _redact_trading_data(self, text: str) -> str:
        """Redact sensitive trading data"""
        redacted_text = text
        
        for pattern in self.compiled_patterns:
            def replace_trading_match(match):
                if len(match.groups()) >= 2:
                    # Financial data - show first few characters only
                    value = match.group(2)
                    if value.isdigit() or ('$' in value or any(c.isdigit() for c in value)):
                        if len(value) > 4:
                            return f"{match.group(1)}{value[:2]}{'*' * (len(value) - 2)}"
                        else:
                            return f"{match.group(1)}{'*' * len(value)}"
                    else:
                        return f"{match.group(1)}{'*' * min(8, len(value))}"
                else:
                    return '*' * len(match.group(0))
            
            redacted_text = pattern.sub(replace_trading_match, redacted_text)
        
        return redacted_text


class SecureLoggerManager:
    """Manager for secure logging configuration"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = getattr(logging, log_level.upper())
        self.log_directory = Path("logs")
        self.log_directory.mkdir(exist_ok=True)
        
        # Encryption key for sensitive logs
        self.encryption_key = os.getenv('LOG_ENCRYPTION_KEY', 
                                       Fernet.generate_key().decode())
    
    def setup_logging(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Setup secure logging configuration"""
        config = config or {}
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with sensitive data filter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formater = StructuredFormatter()
        console_handler.setFormatter(console_formater)
        console_handler.addFilter(SensitiveDataFilter())
        console_handler.addFilter(TradingLogFilter())
        
        # File handler for general logs
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_directory / "spectra_killer.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_formater = StructuredFormatter()
        file_handler.setFormatter(file_formater)
        file_handler.addFilter(SensitiveDataFilter())
        file_handler.addFilter(TradingLogFilter())
        
        # Encrypted file handler for sensitive logs
        encrypted_handler = EncryptedFileHandler(
            filename=self.log_directory / "sensitive.log.encrypted",
            encryption_key=self.encryption_key
        )
        encrypted_handler.setLevel(logging.WARNING)
        encrypted_handler.setFormatter(StructuredFormatter())
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(encrypted_handler)
        
        # Specialized loggers
        self._setup_trading_logger()
        self._setup_security_logger()
        self._setup_audit_logger()
        
        logger.info("Secure logging configuration completed")
    
    def _setup_trading_logger(self) -> None:
        """Setup specialized trading logger"""
        trading_logger = logging.getLogger('spectra.trading')
        trading_logger.setLevel(logging.INFO)
        
        # Separate trading log file
        trading_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_directory / "trading.log",
            maxBytes=50*1024*1024,  # 50MB for trading logs
            backupCount=10,
            encoding='utf-8'
        )
        trading_handler.setLevel(logging.INFO)
        trading_handler.setFormatter(StructuredFormatter())
        trading_handler.addFilter(TradingLogFilter())
        
        trading_logger.addHandler(trading_handler)
        trading_logger.propagate = False
    
    def _setup_security_logger(self) -> None:
        """Setup security-specific logger"""
        security_logger = logging.getLogger('spectra.security')
        security_logger.setLevel(logging.INFO)
        
        # Security events log
        security_handler = EncryptedFileHandler(
            filename=self.log_directory / "security.log.encrypted",
            encryption_key=self.encryption_key
        )
        security_handler.setLevel(logging.INFO)
        security_handler.setFormatter(StructuredFormatter())
        
        security_logger.addHandler(security_handler)
        security_logger.propagate = False
    
    def _setup_audit_logger(self) -> None:
        """Setup audit logger for compliance"""
        audit_logger = logging.getLogger('spectra.audit')
        audit_logger.setLevel(logging.INFO)
        
        # Audit trail log
        audit_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_directory / "audit.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=20,  # Longer retention for audit logs
            encoding='utf-8'
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(StructuredFormatter())
        
        audit_logger.addHandler(audit_handler)
        audit_logger.propagate = False
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event with detailed information"""
        security_logger = logging.getLogger('spectra.security')
        
        security_logger.info(
            "Security Event",
            extra={
                'event_type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'details': details,
                'severity': details.get('severity', 'medium')
            }
        )
    
    def log_trading_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log trading operation with sensitive data protection"""
        trading_logger = logging.getLogger('spectra.trading')
        
        trading_logger.info(
            f"Trading Operation: {operation}",
            extra={
                'operation': operation,
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': details.get('symbol'),
                'order_type': details.get('order_type'),
                'risk_level': details.get('risk_level')
            }
        )
    
    def log_audit_event(self, user_id: str, action: str, resource: str, 
                       success: bool, details: Optional[Dict] = None) -> None:
        """Log audit event for compliance"""
        audit_logger = logging.getLogger('spectra.audit')
        
        audit_logger.info(
            "Audit Event",
            extra={
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'success': success,
                'timestamp': datetime.utcnow().isoformat(),
                'details': details or {}
            }
        )


# Global secure logger manager
_secure_logger: Optional[SecureLoggerManager] = None


def get_secure_logger_manager() -> SecureLoggerManager:
    """Get global secure logger manager instance"""
    global _secure_logger
    if _secure_logger is None:
        _secure_logger = SecureLoggerManager()
        _secure_logger.setup_logging()
    return _secure_logger


def setup_secure_logging(log_level: str = "INFO", 
                        config: Optional[Dict[str, Any]] = None) -> None:
    """Setup secure logging (call this at application start)"""
    manager = get_secure_logger_manager()
    manager.log_level = getattr(logging, log_level.upper())
    manager.setup_logging(config)


# Convenience functions for specific logging needs
def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log security event"""
    manager = get_secure_logger_manager()
    manager.log_security_event(event_type, details)


def log_trading_operation(operation: str, details: Dict[str, Any]) -> None:
    """Log trading operation"""
    manager = get_secure_logger_manager()
    manager.log_trading_operation(operation, details)


def log_audit_event(user_id: str, action: str, resource: str, 
                   success: bool, details: Optional[Dict] = None) -> None:
    """Log audit event"""
    manager = get_secure_logger_manager()
    manager.log_audit_event(user_id, action, resource, success, details)

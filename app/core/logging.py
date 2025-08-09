"""Structured logging configuration for WonyBot"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import traceback
from logging.handlers import RotatingFileHandler
from functools import wraps
import time

from app.config import settings


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_data['session_id'] = record.session_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra data if present
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console"""
        # Get color for level
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Format timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Build message
        message = f"{color}[{timestamp}] {record.levelname:8s}{self.RESET} "
        message += f"{record.name:20s} | {record.getMessage()}"
        
        # Add exception if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


class SecurityFilter(logging.Filter):
    """Filter to prevent logging sensitive information"""
    
    SENSITIVE_PATTERNS = [
        'password',
        'token',
        'api_key',
        'secret',
        'DATABASE_URL',
        'OPENAI_API_KEY',
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out sensitive information"""
        message = record.getMessage().lower()
        
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern.lower() in message:
                record.msg = self._mask_sensitive(record.msg, pattern)
        
        return True
    
    def _mask_sensitive(self, message: str, pattern: str) -> str:
        """Mask sensitive data in message"""
        import re
        # Simple masking - can be improved
        return re.sub(
            f"{pattern}['\"]?\\s*[:=]\\s*['\"]?[^'\"\\s]+",
            f"{pattern}=***REDACTED***",
            message,
            flags=re.IGNORECASE
        )


def setup_logging(
    log_level: str = None,
    log_file: Optional[Path] = None,
    structured: bool = False
) -> None:
    """Setup logging configuration"""
    
    # Determine log level
    level_str = log_level or settings.log_level or "INFO"
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    # Create logs directory if needed
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if structured:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())
    
    # Add security filter
    console_handler.addFilter(SecurityFilter())
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(StructuredFormatter())
        file_handler.addFilter(SecurityFilter())
        root_logger.addHandler(file_handler)
    
    # Set levels for specific loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Log startup
    root_logger.info(
        "Logging initialized",
        extra={
            "extra_data": {
                "level": level_str,
                "structured": structured,
                "log_file": str(log_file) if log_file else None
            }
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(name)


def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            _logger = logger or logging.getLogger(func.__module__)
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                _logger.debug(
                    f"{func.__name__} completed",
                    extra={"duration_ms": duration_ms}
                )
                
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                _logger.error(
                    f"{func.__name__} failed",
                    exc_info=True,
                    extra={"duration_ms": duration_ms}
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            _logger = logger or logging.getLogger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                _logger.debug(
                    f"{func.__name__} completed",
                    extra={"duration_ms": duration_ms}
                )
                
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                _logger.error(
                    f"{func.__name__} failed",
                    exc_info=True,
                    extra={"duration_ms": duration_ms}
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None):
    """Log an error with context"""
    logger.error(
        f"{error.__class__.__name__}: {str(error)}",
        exc_info=True,
        extra={"extra_data": context or {}}
    )


# Create module logger
logger = get_logger(__name__)
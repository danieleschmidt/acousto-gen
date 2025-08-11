"""
Centralized logging configuration for Acousto-Gen.
Provides structured logging with different levels and outputs for different components.
"""

import logging
import logging.config
import logging.handlers
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import traceback


class AcoustoFormatter(logging.Formatter):
    """Custom formatter for Acousto-Gen logs with structured output."""
    
    def __init__(self):
        super().__init__()
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'hostname': self.hostname
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
        if hasattr(record, 'hardware_id'):
            log_entry['hardware_id'] = record.hardware_id
        
        return json.dumps(log_entry)


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with color coding for different log levels."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        
        # Standard format with colors
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        return formatter.format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    structured_logs: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> Dict[str, logging.Logger]:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Default logging level
        log_dir: Directory for log files (None = no file logging)
        console_output: Enable console output
        structured_logs: Use structured JSON logging for files
        max_file_size: Maximum size per log file
        backup_count: Number of backup files to keep
        
    Returns:
        Dictionary of configured loggers
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create log directory if specified
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'structured': {
                '()': AcoustoFormatter,
            },
            'console': {
                '()': ColoredConsoleFormatter,
            },
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {},
        'loggers': {
            'acousto_gen': {
                'level': numeric_level,
                'handlers': [],
                'propagate': False
            },
            'acousto_gen.core': {
                'level': numeric_level,
                'handlers': [],
                'propagate': True
            },
            'acousto_gen.hardware': {
                'level': numeric_level,
                'handlers': [],
                'propagate': True
            },
            'acousto_gen.optimization': {
                'level': numeric_level,
                'handlers': [],
                'propagate': True
            },
            'acousto_gen.api': {
                'level': numeric_level,
                'handlers': [],
                'propagate': True
            },
            'acousto_gen.security': {
                'level': numeric_level,
                'handlers': [],
                'propagate': True
            },
            'acousto_gen.monitoring': {
                'level': numeric_level,
                'handlers': [],
                'propagate': True
            },
            'audit': {
                'level': logging.INFO,
                'handlers': [],
                'propagate': False
            }
        },
        'root': {
            'level': logging.WARNING,
            'handlers': []
        }
    }
    
    # Console handler
    if console_output:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': numeric_level,
            'formatter': 'console',
            'stream': 'ext://sys.stdout'
        }
        
        # Add console handler to loggers
        for logger_name in config['loggers']:
            config['loggers'][logger_name]['handlers'].append('console')
    
    # File handlers
    if log_dir:
        # Application log
        config['handlers']['app_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': numeric_level,
            'formatter': 'structured' if structured_logs else 'simple',
            'filename': os.path.join(log_dir, 'acousto-gen.log'),
            'maxBytes': max_file_size,
            'backupCount': backup_count
        }
        
        # Error log
        config['handlers']['error_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': logging.ERROR,
            'formatter': 'structured' if structured_logs else 'simple',
            'filename': os.path.join(log_dir, 'acousto-gen-errors.log'),
            'maxBytes': max_file_size,
            'backupCount': backup_count
        }
        
        # Hardware log
        config['handlers']['hardware_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': numeric_level,
            'formatter': 'structured' if structured_logs else 'simple',
            'filename': os.path.join(log_dir, 'hardware.log'),
            'maxBytes': max_file_size,
            'backupCount': backup_count
        }
        
        # Audit log
        config['handlers']['audit_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': logging.INFO,
            'formatter': 'structured' if structured_logs else 'simple',
            'filename': os.path.join(log_dir, 'audit.log'),
            'maxBytes': max_file_size,
            'backupCount': backup_count
        }
        
        # Add file handlers to appropriate loggers
        for logger_name in config['loggers']:
            if logger_name != 'audit':
                config['loggers'][logger_name]['handlers'].extend(['app_file', 'error_file'])
            if logger_name == 'acousto_gen.hardware':
                config['loggers'][logger_name]['handlers'].append('hardware_file')
        
        # Audit logger gets its own handler
        config['loggers']['audit']['handlers'].append('audit_file')
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Get configured loggers
    loggers = {}
    for logger_name in config['loggers']:
        loggers[logger_name] = logging.getLogger(logger_name)
    
    # Log startup message
    app_logger = loggers['acousto_gen']
    app_logger.info("Logging system initialized", extra={
        'operation': 'logging_init',
        'log_level': log_level,
        'log_dir': log_dir,
        'console_output': console_output,
        'structured_logs': structured_logs
    })
    
    return loggers


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
    
    def log_operation(self, operation: str, **kwargs):
        """Log an operation with structured data."""
        self.logger.info(f"Operation: {operation}", extra={
            'operation': operation,
            **kwargs
        })
    
    def log_error(self, error: Exception, operation: str = "", **kwargs):
        """Log an error with context."""
        self.logger.error(f"Error in {operation}: {error}", extra={
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            **kwargs
        }, exc_info=True)
    
    def log_warning(self, message: str, operation: str = "", **kwargs):
        """Log a warning with context."""
        self.logger.warning(message, extra={
            'operation': operation,
            **kwargs
        })
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.logger.info(f"Performance: {operation} took {duration:.3f}s", extra={
            'operation': operation,
            'duration': duration,
            **kwargs
        })


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, operation: str = "", **kwargs):
    """Decorator to log exceptions from functions."""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            try:
                return func(*args, **func_kwargs)
            except Exception as e:
                logger.error(f"Exception in {operation or func.__name__}: {e}", extra={
                    'operation': operation or func.__name__,
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(func_kwargs),
                    **kwargs
                }, exc_info=True)
                raise
        return wrapper
    return decorator


def log_function_call(logger: logging.Logger, log_args: bool = False):
    """Decorator to log function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            extra_data = {'function': func.__name__}
            
            if log_args:
                extra_data.update({
                    'args': str(args),
                    'kwargs': str(kwargs)
                })
            
            logger.debug(f"Calling {func.__name__}", extra=extra_data)
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func.__name__}", extra=extra_data)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", extra={
                    **extra_data,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }, exc_info=True)
                raise
        
        return wrapper
    return decorator


# Context manager for operation logging
class LogOperation:
    """Context manager for logging operations with timing."""
    
    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation}", extra={
            'operation': self.operation,
            'phase': 'start',
            **self.kwargs
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.operation}", extra={
                'operation': self.operation,
                'phase': 'complete',
                'duration': duration,
                **self.kwargs
            })
        else:
            self.logger.error(f"Failed operation: {self.operation}", extra={
                'operation': self.operation,
                'phase': 'error',
                'duration': duration,
                'error_type': exc_type.__name__,
                'error_message': str(exc_val),
                **self.kwargs
            }, exc_info=True)


# Global logging setup - can be called once at application startup
def initialize_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = "logs",
    console_output: bool = True
) -> Dict[str, logging.Logger]:
    """
    Initialize logging for the entire application.
    
    Args:
        log_level: Default logging level
        log_dir: Directory for log files
        console_output: Enable console output
        
    Returns:
        Dictionary of configured loggers
    """
    return setup_logging(
        log_level=log_level,
        log_dir=log_dir,
        console_output=console_output,
        structured_logs=True
    )


# Default loggers for common use
def get_default_loggers() -> Dict[str, logging.Logger]:
    """Get default loggers for common components."""
    return {
        'app': get_logger('acousto_gen'),
        'core': get_logger('acousto_gen.core'),
        'hardware': get_logger('acousto_gen.hardware'),
        'optimization': get_logger('acousto_gen.optimization'),
        'api': get_logger('acousto_gen.api'),
        'security': get_logger('acousto_gen.security'),
        'monitoring': get_logger('acousto_gen.monitoring'),
        'audit': get_logger('audit')
    }
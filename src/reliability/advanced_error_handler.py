"""
Advanced error handling and recovery system for Acousto-Gen Generation 2.
Provides comprehensive exception handling, automatic recovery, and detailed logging.
"""

import sys
import traceback
import functools
import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from pathlib import Path
import json
from threading import Lock
from collections import defaultdict

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    REDUCE_PRECISION = "reduce_precision"
    SIMPLIFY_TARGET = "simplify_target"
    RESTART_COMPONENT = "restart_component"
    FAIL_SAFE = "fail_safe"


@dataclass
class ErrorEvent:
    """Represents an error event with context."""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False


class ErrorStatistics:
    """Tracks error statistics and patterns."""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_history = []
        self.component_errors = defaultdict(list)
        self._lock = Lock()
    
    def record_error(self, event: ErrorEvent):
        """Record an error event."""
        with self._lock:
            self.error_counts[event.error_type] += 1
            self.error_history.append(event)
            self.component_errors[event.component].append(event)
            
            # Keep only last 1000 errors to prevent memory growth
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]
    
    def get_error_rate(self, time_window: float = 3600) -> Dict[str, int]:
        """Get error rate within time window (default 1 hour)."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        error_counts = defaultdict(int)
        
        for error in recent_errors:
            error_counts[error.error_type] += 1
        
        return dict(error_counts)
    
    def get_most_frequent_errors(self, limit: int = 10) -> List[tuple]:
        """Get most frequent error types."""
        return sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:limit]


class RecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies = {
            "torch.cuda.OutOfMemoryError": RecoveryStrategy.REDUCE_PRECISION,
            "numpy.linalg.LinAlgError": RecoveryStrategy.FALLBACK,
            "ConnectionError": RecoveryStrategy.RETRY,
            "TimeoutError": RecoveryStrategy.RETRY,
            "RuntimeError": RecoveryStrategy.SIMPLIFY_TARGET,
            "ValueError": RecoveryStrategy.FALLBACK,
            "ImportError": RecoveryStrategy.FAIL_SAFE,
        }
        
        self.max_retries = {
            RecoveryStrategy.RETRY: 3,
            RecoveryStrategy.REDUCE_PRECISION: 2,
            RecoveryStrategy.SIMPLIFY_TARGET: 1,
            RecoveryStrategy.RESTART_COMPONENT: 1,
        }
    
    def get_recovery_strategy(self, error_type: str, context: Dict[str, Any]) -> Optional[RecoveryStrategy]:
        """Get appropriate recovery strategy for error type."""
        # Direct mapping
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]
        
        # Pattern-based matching
        if "memory" in error_type.lower() or "cuda" in error_type.lower():
            return RecoveryStrategy.REDUCE_PRECISION
        
        if "timeout" in error_type.lower() or "connection" in error_type.lower():
            return RecoveryStrategy.RETRY
        
        if "convergence" in error_type.lower() or "optimization" in error_type.lower():
            return RecoveryStrategy.SIMPLIFY_TARGET
        
        # Default fallback
        return RecoveryStrategy.FALLBACK
    
    def apply_recovery_strategy(self, strategy: RecoveryStrategy, 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply recovery strategy and return modified parameters."""
        recovery_params = context.copy()
        
        if strategy == RecoveryStrategy.REDUCE_PRECISION:
            # Reduce precision and resolution
            if "resolution" in recovery_params:
                recovery_params["resolution"] = recovery_params["resolution"] * 2  # Double voxel size
            if "dtype" in recovery_params:
                recovery_params["dtype"] = torch.float16 if recovery_params["dtype"] == torch.float32 else torch.float32
            if "device" in recovery_params and "cuda" in str(recovery_params["device"]):
                recovery_params["device"] = "cpu"  # Fall back to CPU
        
        elif strategy == RecoveryStrategy.SIMPLIFY_TARGET:
            # Simplify optimization target
            if "iterations" in recovery_params:
                recovery_params["iterations"] = min(recovery_params["iterations"], 500)
            if "learning_rate" in recovery_params:
                recovery_params["learning_rate"] = recovery_params["learning_rate"] * 0.5
            if "target_complexity" in recovery_params:
                recovery_params["target_complexity"] = "simple"
        
        elif strategy == RecoveryStrategy.FALLBACK:
            # Use safe defaults
            recovery_params.update({
                "device": "cpu",
                "dtype": torch.float32,
                "method": "adam",
                "iterations": 100
            })
        
        return recovery_params


class AcoustoGenErrorHandler:
    """Main error handling system for Acousto-Gen."""
    
    def __init__(self, enable_recovery: bool = True, log_level: int = logging.INFO):
        """Initialize error handler."""
        self.enable_recovery = enable_recovery
        self.statistics = ErrorStatistics()
        self.recovery_manager = RecoveryManager()
        self.retry_counts = defaultdict(int)
        
        # Setup logging
        self.logger = logging.getLogger("acousto_gen.error_handler")
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def handle_error(self, error: Exception, component: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle error with automatic recovery if enabled."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Determine severity
        severity = self._determine_severity(error_type, error_message)
        
        # Create error event
        event = ErrorEvent(
            timestamp=time.time(),
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            component=component,
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        # Record error
        self.statistics.record_error(event)
        
        # Log error
        self._log_error(event)
        
        # Attempt recovery if enabled and appropriate
        if self.enable_recovery and severity != ErrorSeverity.CRITICAL:
            return self._attempt_recovery(event)
        
        return None
    
    def _determine_severity(self, error_type: str, error_message: str) -> ErrorSeverity:
        """Determine error severity."""
        critical_patterns = [
            "system", "hardware", "safety", "corruption", "segmentation fault"
        ]
        
        high_patterns = [
            "memory", "cuda", "convergence failure", "timeout"
        ]
        
        medium_patterns = [
            "connection", "import", "invalid", "not found"
        ]
        
        error_lower = f"{error_type} {error_message}".lower()
        
        if any(pattern in error_lower for pattern in critical_patterns):
            return ErrorSeverity.CRITICAL
        elif any(pattern in error_lower for pattern in high_patterns):
            return ErrorSeverity.HIGH
        elif any(pattern in error_lower for pattern in medium_patterns):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _log_error(self, event: ErrorEvent):
        """Log error event."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[event.severity]
        
        self.logger.log(
            log_level,
            f"{event.component}: {event.error_type} - {event.error_message}"
        )
        
        if event.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.log(log_level, f"Stack trace:\n{event.stack_trace}")
    
    def _attempt_recovery(self, event: ErrorEvent) -> Optional[Dict[str, Any]]:
        """Attempt error recovery."""
        # Check retry count
        retry_key = f"{event.component}:{event.error_type}"
        if self.retry_counts[retry_key] >= 5:  # Max 5 retries per error type per component
            self.logger.warning(f"Max retries exceeded for {retry_key}")
            return None
        
        # Get recovery strategy
        strategy = self.recovery_manager.get_recovery_strategy(event.error_type, event.context)
        if not strategy:
            return None
        
        # Apply recovery
        try:
            recovery_params = self.recovery_manager.apply_recovery_strategy(strategy, event.context)
            
            # Update event
            event.recovery_attempted = True
            event.recovery_strategy = strategy
            event.recovery_successful = True
            
            self.logger.info(f"Applied recovery strategy {strategy.value} for {event.error_type}")
            self.retry_counts[retry_key] += 1
            
            return recovery_params
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            event.recovery_successful = False
            return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        recent_errors = self.statistics.get_error_rate(3600)  # Last hour
        frequent_errors = self.statistics.get_most_frequent_errors()
        
        return {
            "total_errors": len(self.statistics.error_history),
            "recent_errors_1h": sum(recent_errors.values()),
            "most_frequent_errors": frequent_errors,
            "error_rate_by_type": recent_errors,
            "components_with_errors": list(self.statistics.component_errors.keys()),
            "recovery_enabled": self.enable_recovery
        }


def robust_execute(component: str = "unknown", 
                  max_retries: int = 3,
                  recovery_enabled: bool = True,
                  error_handler: Optional[AcoustoGenErrorHandler] = None):
    """Decorator for robust execution with error handling and recovery."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if error_handler is None:
                handler = AcoustoGenErrorHandler(enable_recovery=recovery_enabled)
            else:
                handler = error_handler
            
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_error = e
                    
                    # Prepare context
                    context = {
                        "function": func.__name__,
                        "attempt": attempt,
                        "args": str(args)[:200],  # Truncate long args
                        "kwargs": str(kwargs)[:200]
                    }
                    
                    # Handle error
                    recovery_params = handler.handle_error(e, component, context)
                    
                    # If this is the last attempt or no recovery possible, re-raise
                    if attempt == max_retries or not recovery_params:
                        break
                    
                    # Apply recovery parameters if possible
                    if recovery_params and "kwargs_update" in recovery_params:
                        kwargs.update(recovery_params["kwargs_update"])
                    
                    # Brief delay before retry
                    time.sleep(0.1 * (attempt + 1))
            
            # If we get here, all retries failed
            if last_error:
                raise last_error
        
        return wrapper
    return decorator


class SafetyInterlock:
    """Safety interlock system for critical operations."""
    
    def __init__(self):
        self.interlocks_active = True
        self.safety_violations = []
        self.emergency_stop_triggered = False
    
    def check_safety_conditions(self, operation: str, parameters: Dict[str, Any]) -> bool:
        """Check if operation is safe to proceed."""
        if not self.interlocks_active:
            return True
        
        violations = []
        
        # Pressure safety check
        if "max_pressure" in parameters:
            if parameters["max_pressure"] > 10000:  # 10 kPa absolute limit
                violations.append(f"Pressure {parameters['max_pressure']:.0f} Pa exceeds absolute limit 10000 Pa")
        
        # Temperature safety check  
        if "temperature" in parameters:
            if parameters["temperature"] > 60:  # 60°C absolute limit
                violations.append(f"Temperature {parameters['temperature']:.1f}°C exceeds absolute limit 60°C")
        
        # Power safety check
        if "total_power" in parameters:
            if parameters["total_power"] > 100:  # 100W absolute limit
                violations.append(f"Power {parameters['total_power']:.1f}W exceeds absolute limit 100W")
        
        if violations:
            self.safety_violations.extend(violations)
            logger.critical(f"Safety interlock triggered for {operation}: {violations}")
            return False
        
        return True
    
    def emergency_stop(self, reason: str):
        """Trigger emergency stop."""
        self.emergency_stop_triggered = True
        logger.critical(f"EMERGENCY STOP: {reason}")
        # In real implementation, this would shut down hardware
    
    def reset_interlocks(self):
        """Reset safety interlocks after manual verification."""
        self.emergency_stop_triggered = False
        self.safety_violations = []
        logger.info("Safety interlocks reset")


# Global error handler instance
global_error_handler = AcoustoGenErrorHandler()
safety_interlock = SafetyInterlock()


# Convenience function for error context
def create_error_context(**kwargs) -> Dict[str, Any]:
    """Create standardized error context."""
    return {
        "timestamp": time.time(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        **kwargs
    }
"""
Error Recovery and Fault Tolerance System for Acoustic Holography.
Implements comprehensive error handling, recovery strategies, and system resilience.
"""

import logging
import time
import threading
import traceback
import functools
import queue
from typing import Dict, List, Optional, Any, Callable, Type, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FAILOVER = "failover"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL_INTERVENTION = "manual_intervention"
    SYSTEM_RESTART = "system_restart"


@dataclass
class ErrorInfo:
    """Information about an error occurrence."""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime = field(default_factory=datetime.now)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    component: Optional[str] = None
    operation: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    resolved: bool = False


@dataclass
class RecoveryAction:
    """Recovery action configuration."""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    condition_check: Optional[Callable[[], bool]] = None
    recovery_function: Optional[Callable[[], bool]] = None


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
        logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    @staticmethod
    def retry_with_backoff(max_attempts: int = 3, 
                          initial_delay: float = 1.0,
                          backoff_multiplier: float = 2.0,
                          max_delay: float = 60.0,
                          exceptions: Tuple[Type[Exception], ...] = (Exception,)):
        """
        Decorator for retry logic with exponential backoff.
        
        Args:
            max_attempts: Maximum retry attempts
            initial_delay: Initial delay between retries
            backoff_multiplier: Multiplier for delay increase
            max_delay: Maximum delay between retries
            exceptions: Exception types to retry on
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_attempts - 1:
                            logger.error(f"Final retry attempt failed for {func.__name__}: {e}")
                            break
                        
                        # Calculate delay with backoff
                        delay = min(initial_delay * (backoff_multiplier ** attempt), max_delay)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s")
                        time.sleep(delay)
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    def retry_async(func: Callable, 
                   max_attempts: int = 3,
                   delay: float = 1.0,
                   backoff_multiplier: float = 2.0) -> Any:
        """
        Execute function with retry logic asynchronously.
        
        Args:
            func: Function to execute
            max_attempts: Maximum retry attempts
            delay: Initial delay between retries
            backoff_multiplier: Multiplier for delay increase
            
        Returns:
            Function result if successful
        """
        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                
                current_delay = delay * (backoff_multiplier ** attempt)
                logger.warning(f"Async retry attempt {attempt + 1} failed: {e}. Retrying in {current_delay:.1f}s")
                time.sleep(current_delay)


class ErrorRecoveryManager:
    """
    Comprehensive error recovery and fault tolerance manager.
    Provides automatic error handling, recovery strategies, and system resilience.
    """
    
    def __init__(self):
        """Initialize error recovery manager."""
        self.error_history: List[ErrorInfo] = []
        self.recovery_strategies: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        
        # Recovery queue for background processing
        self.recovery_queue = queue.Queue()
        self.recovery_thread: Optional[threading.Thread] = None
        self.stop_recovery = False
        
        # Error statistics
        self.error_counts: Dict[str, int] = {}
        self.recovery_success_rates: Dict[str, List[bool]] = {}
        
        # System state management
        self.degraded_components: Set[str] = set()
        self.offline_components: Set[str] = set()
        
        # Recovery callbacks
        self.recovery_callbacks: List[Callable[[ErrorInfo, bool], None]] = []\
        self.degradation_callbacks: List[Callable[[str, bool], None]] = []\
        \n        self._lock = threading.Lock()\n        \n        # Start background recovery processing\n        self._start_recovery_thread()\n        \n        logger.info(\"Error Recovery Manager initialized\")\n    \n    def register_recovery_strategy(self, error_pattern: str, recovery_action: RecoveryAction):\n        \"\"\"\n        Register recovery strategy for specific error patterns.\n        \n        Args:\n            error_pattern: Error pattern to match (regex or exact match)\n            recovery_action: Recovery action configuration\n        \"\"\"\n        self.recovery_strategies[error_pattern] = recovery_action\n        logger.info(f\"Registered recovery strategy for pattern: {error_pattern}\")\n    \n    def register_circuit_breaker(self, component: str, failure_threshold: int = 5, \n                               recovery_timeout: float = 60.0):\n        \"\"\"\n        Register circuit breaker for component.\n        \n        Args:\n            component: Component name\n            failure_threshold: Number of failures before opening circuit\n            recovery_timeout: Time to wait before attempting recovery\n        \"\"\"\n        self.circuit_breakers[component] = CircuitBreaker(failure_threshold, recovery_timeout)\n        logger.info(f\"Registered circuit breaker for component: {component}\")\n    \n    def add_recovery_callback(self, callback: Callable[[ErrorInfo, bool], None]):\n        \"\"\"Add callback for recovery completion notifications.\"\"\"\n        self.recovery_callbacks.append(callback)\n    \n    def add_degradation_callback(self, callback: Callable[[str, bool], None]):\n        \"\"\"Add callback for component degradation notifications.\"\"\"\n        self.degradation_callbacks.append(callback)\n    \n    def handle_error(self, error: Exception, \n                    component: Optional[str] = None,\n                    operation: Optional[str] = None,\n                    context: Optional[Dict[str, Any]] = None,\n                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> bool:\n        \"\"\"\n        Handle error with automatic recovery attempts.\n        \n        Args:\n            error: Exception that occurred\n            component: Component where error occurred\n            operation: Operation that failed\n            context: Additional context information\n            severity: Error severity level\n            \n        Returns:\n            True if recovery was successful, False otherwise\n        \"\"\"\n        error_info = ErrorInfo(\n            error_type=type(error).__name__,\n            error_message=str(error),\n            stack_trace=traceback.format_exc(),\n            severity=severity,\n            component=component,\n            operation=operation,\n            context=context or {}\n        )\n        \n        with self._lock:\n            self.error_history.append(error_info)\n            self.error_counts[error_info.error_type] = self.error_counts.get(error_info.error_type, 0) + 1\n        \n        # Log error\n        log_msg = f\"Error in {component or 'unknown'}.{operation or 'unknown'}: {error}\"\n        if severity == ErrorSeverity.CRITICAL:\n            logger.critical(log_msg)\n        elif severity == ErrorSeverity.HIGH:\n            logger.error(log_msg)\n        elif severity == ErrorSeverity.MEDIUM:\n            logger.warning(log_msg)\n        else:\n            logger.info(log_msg)\n        \n        # Attempt immediate recovery for critical errors\n        if severity == ErrorSeverity.CRITICAL:\n            return self._attempt_recovery(error_info)\n        else:\n            # Queue for background recovery\n            self.recovery_queue.put(error_info)\n            return False\n    \n    def _attempt_recovery(self, error_info: ErrorInfo) -> bool:\n        \"\"\"\n        Attempt recovery for error.\n        \n        Args:\n            error_info: Error information\n            \n        Returns:\n            True if recovery successful, False otherwise\n        \"\"\"\n        # Find matching recovery strategy\n        recovery_action = self._find_recovery_strategy(error_info)\n        \n        if not recovery_action:\n            logger.warning(f\"No recovery strategy found for error: {error_info.error_type}\")\n            return False\n        \n        success = False\n        \n        try:\n            if recovery_action.strategy == RecoveryStrategy.RETRY:\n                success = self._execute_retry_recovery(error_info, recovery_action)\n            elif recovery_action.strategy == RecoveryStrategy.FAILOVER:\n                success = self._execute_failover_recovery(error_info, recovery_action)\n            elif recovery_action.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:\n                success = self._execute_degradation_recovery(error_info, recovery_action)\n            elif recovery_action.strategy == RecoveryStrategy.CIRCUIT_BREAKER:\n                success = self._handle_circuit_breaker(error_info, recovery_action)\n            elif recovery_action.strategy == RecoveryStrategy.SYSTEM_RESTART:\n                success = self._execute_system_restart(error_info, recovery_action)\n            else:\n                logger.warning(f\"Unknown recovery strategy: {recovery_action.strategy}\")\n        \n        except Exception as recovery_error:\n            logger.error(f\"Recovery attempt failed: {recovery_error}\")\n            success = False\n        \n        # Update recovery statistics\n        error_type = error_info.error_type\n        if error_type not in self.recovery_success_rates:\n            self.recovery_success_rates[error_type] = []\n        self.recovery_success_rates[error_type].append(success)\n        \n        # Keep only recent success/failure history\n        if len(self.recovery_success_rates[error_type]) > 100:\n            self.recovery_success_rates[error_type] = self.recovery_success_rates[error_type][-100:]\n        \n        # Mark error as resolved if recovery successful\n        if success:\n            error_info.resolved = True\n            logger.info(f\"Successfully recovered from error: {error_info.error_type}\")\n        else:\n            error_info.recovery_attempts += 1\n            logger.warning(f\"Recovery failed for error: {error_info.error_type}\")\n        \n        # Notify callbacks\n        for callback in self.recovery_callbacks:\n            try:\n                callback(error_info, success)\n            except Exception as e:\n                logger.error(f\"Error in recovery callback: {e}\")\n        \n        return success\n    \n    def _find_recovery_strategy(self, error_info: ErrorInfo) -> Optional[RecoveryAction]:\n        \"\"\"Find matching recovery strategy for error.\"\"\"\n        import re\n        \n        for pattern, action in self.recovery_strategies.items():\n            # Try exact match first\n            if pattern == error_info.error_type:\n                return action\n            \n            # Try regex match\n            try:\n                if re.search(pattern, error_info.error_message, re.IGNORECASE):\n                    return action\n            except re.error:\n                pass  # Invalid regex, continue\n        \n        return None\n    \n    def _execute_retry_recovery(self, error_info: ErrorInfo, recovery_action: RecoveryAction) -> bool:\n        \"\"\"Execute retry-based recovery.\"\"\"\n        if not recovery_action.recovery_function:\n            logger.warning(\"No recovery function provided for retry strategy\")\n            return False\n        \n        for attempt in range(recovery_action.max_attempts):\n            try:\n                # Check precondition if provided\n                if recovery_action.condition_check and not recovery_action.condition_check():\n                    logger.info(f\"Recovery precondition not met for attempt {attempt + 1}\")\n                    continue\n                \n                # Execute recovery function\n                if recovery_action.recovery_function():\n                    logger.info(f\"Retry recovery successful on attempt {attempt + 1}\")\n                    return True\n                \n            except Exception as e:\n                logger.warning(f\"Retry attempt {attempt + 1} failed: {e}\")\n            \n            # Wait before next attempt (with backoff)\n            if attempt < recovery_action.max_attempts - 1:\n                delay = min(\n                    recovery_action.delay_seconds * (recovery_action.backoff_multiplier ** attempt),\n                    recovery_action.max_delay\n                )\n                time.sleep(delay)\n        \n        return False\n    \n    def _execute_failover_recovery(self, error_info: ErrorInfo, recovery_action: RecoveryAction) -> bool:\n        \"\"\"Execute failover-based recovery.\"\"\"\n        if not error_info.component:\n            logger.warning(\"Cannot execute failover: no component specified\")\n            return False\n        \n        # Mark component as offline\n        self.offline_components.add(error_info.component)\n        \n        # Execute failover function if provided\n        if recovery_action.recovery_function:\n            try:\n                success = recovery_action.recovery_function()\n                if success:\n                    logger.info(f\"Failover recovery successful for component: {error_info.component}\")\n                    return True\n            except Exception as e:\n                logger.error(f\"Failover recovery failed: {e}\")\n        \n        logger.warning(f\"Failover recovery failed for component: {error_info.component}\")\n        return False\n    \n    def _execute_degradation_recovery(self, error_info: ErrorInfo, recovery_action: RecoveryAction) -> bool:\n        \"\"\"Execute graceful degradation recovery.\"\"\"\n        if not error_info.component:\n            logger.warning(\"Cannot execute degradation: no component specified\")\n            return False\n        \n        # Mark component as degraded\n        self.degraded_components.add(error_info.component)\n        \n        # Notify degradation callbacks\n        for callback in self.degradation_callbacks:\n            try:\n                callback(error_info.component, True)\n            except Exception as e:\n                logger.error(f\"Error in degradation callback: {e}\")\n        \n        logger.info(f\"Component {error_info.component} degraded gracefully\")\n        return True\n    \n    def _handle_circuit_breaker(self, error_info: ErrorInfo, recovery_action: RecoveryAction) -> bool:\n        \"\"\"Handle circuit breaker recovery.\"\"\"\n        component = error_info.component or \"default\"\n        \n        if component not in self.circuit_breakers:\n            self.register_circuit_breaker(component)\n        \n        circuit_breaker = self.circuit_breakers[component]\n        circuit_breaker._on_failure()  # Register failure\n        \n        # Circuit breaker will handle future calls\n        logger.info(f\"Circuit breaker updated for component: {component}\")\n        return True\n    \n    def _execute_system_restart(self, error_info: ErrorInfo, recovery_action: RecoveryAction) -> bool:\n        \"\"\"Execute system restart recovery.\"\"\"\n        logger.critical(\"System restart recovery triggered - this would restart the system\")\n        \n        # In a real implementation, this might:\n        # 1. Save current state\n        # 2. Gracefully shutdown components\n        # 3. Restart the application\n        \n        # For now, just execute recovery function if provided\n        if recovery_action.recovery_function:\n            try:\n                return recovery_action.recovery_function()\n            except Exception as e:\n                logger.error(f\"System restart recovery failed: {e}\")\n                return False\n        \n        return False\n    \n    def _start_recovery_thread(self):\n        \"\"\"Start background recovery processing thread.\"\"\"\n        def recovery_loop():\n            while not self.stop_recovery:\n                try:\n                    # Get error from queue with timeout\n                    error_info = self.recovery_queue.get(timeout=1.0)\n                    self._attempt_recovery(error_info)\n                    self.recovery_queue.task_done()\n                except queue.Empty:\n                    continue\n                except Exception as e:\n                    logger.error(f\"Error in recovery thread: {e}\")\n        \n        self.recovery_thread = threading.Thread(target=recovery_loop, daemon=True)\n        self.recovery_thread.start()\n        logger.info(\"Recovery processing thread started\")\n    \n    def stop_recovery_processing(self):\n        \"\"\"Stop background recovery processing.\"\"\"\n        self.stop_recovery = True\n        \n        if self.recovery_thread:\n            self.recovery_thread.join(timeout=2.0)\n            self.recovery_thread = None\n        \n        logger.info(\"Recovery processing stopped\")\n    \n    def get_component_health(self, component: str) -> str:\n        \"\"\"\n        Get health status of component.\n        \n        Args:\n            component: Component name\n            \n        Returns:\n            Health status: 'healthy', 'degraded', 'offline'\n        \"\"\"\n        if component in self.offline_components:\n            return 'offline'\n        elif component in self.degraded_components:\n            return 'degraded'\n        else:\n            return 'healthy'\n    \n    def restore_component(self, component: str) -> bool:\n        \"\"\"\n        Attempt to restore degraded or offline component.\n        \n        Args:\n            component: Component name\n            \n        Returns:\n            True if restoration successful, False otherwise\n        \"\"\"\n        was_degraded = component in self.degraded_components\n        was_offline = component in self.offline_components\n        \n        # Remove from degraded/offline sets\n        self.degraded_components.discard(component)\n        self.offline_components.discard(component)\n        \n        # Reset circuit breaker if exists\n        if component in self.circuit_breakers:\n            circuit_breaker = self.circuit_breakers[component]\n            circuit_breaker.failure_count = 0\n            circuit_breaker.state = \"CLOSED\"\n        \n        # Notify callbacks\n        if was_degraded:\n            for callback in self.degradation_callbacks:\n                try:\n                    callback(component, False)  # Not degraded anymore\n                except Exception as e:\n                    logger.error(f\"Error in degradation callback: {e}\")\n        \n        logger.info(f\"Component {component} restored to healthy state\")\n        return True\n    \n    def get_error_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get error and recovery statistics.\"\"\"\n        total_errors = len(self.error_history)\n        recent_errors = [e for e in self.error_history \n                        if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour\n        \n        resolved_errors = sum(1 for e in self.error_history if e.resolved)\n        \n        # Calculate recovery success rates\n        recovery_rates = {}\n        for error_type, success_history in self.recovery_success_rates.items():\n            if success_history:\n                success_rate = sum(success_history) / len(success_history)\n                recovery_rates[error_type] = {\n                    \"success_rate\": success_rate,\n                    \"total_attempts\": len(success_history)\n                }\n        \n        return {\n            \"total_errors\": total_errors,\n            \"recent_errors_1h\": len(recent_errors),\n            \"resolved_errors\": resolved_errors,\n            \"resolution_rate\": resolved_errors / total_errors if total_errors > 0 else 0.0,\n            \"error_counts_by_type\": dict(self.error_counts),\n            \"recovery_success_rates\": recovery_rates,\n            \"degraded_components\": list(self.degraded_components),\n            \"offline_components\": list(self.offline_components),\n            \"active_circuit_breakers\": len([cb for cb in self.circuit_breakers.values() \n                                           if cb.state != \"CLOSED\"])\n        }\n    \n    def get_system_health(self) -> Dict[str, Any]:\n        \"\"\"Get overall system health status.\"\"\"\n        stats = self.get_error_statistics()\n        \n        # Determine overall health\n        if len(self.offline_components) > 0:\n            health_status = \"critical\"\n        elif len(self.degraded_components) > 0:\n            health_status = \"degraded\"\n        elif stats[\"recent_errors_1h\"] > 10:\n            health_status = \"warning\"\n        else:\n            health_status = \"healthy\"\n        \n        return {\n            \"status\": health_status,\n            \"timestamp\": datetime.now().isoformat(),\n            \"components\": {\n                \"healthy\": [],  # Would list healthy components\n                \"degraded\": list(self.degraded_components),\n                \"offline\": list(self.offline_components)\n            },\n            \"statistics\": stats\n        }\n\n\n# Recovery decorators\n\ndef with_error_recovery(component: str = None, \n                       operation: str = None,\n                       severity: ErrorSeverity = ErrorSeverity.MEDIUM):\n    \"\"\"\n    Decorator to add error recovery to functions.\n    \n    Args:\n        component: Component name\n        operation: Operation name  \n        severity: Error severity level\n    \"\"\"\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            recovery_manager = get_error_recovery_manager()\n            \n            try:\n                return func(*args, **kwargs)\n            except Exception as e:\n                recovery_manager.handle_error(\n                    e, \n                    component=component or func.__module__,\n                    operation=operation or func.__name__,\n                    severity=severity\n                )\n                raise  # Re-raise after handling\n        \n        return wrapper\n    return decorator\n\n\ndef with_circuit_breaker(component: str, \n                        failure_threshold: int = 5,\n                        recovery_timeout: float = 60.0):\n    \"\"\"\n    Decorator to add circuit breaker protection.\n    \n    Args:\n        component: Component name\n        failure_threshold: Failures before opening circuit\n        recovery_timeout: Recovery timeout in seconds\n    \"\"\"\n    def decorator(func):\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):\n            recovery_manager = get_error_recovery_manager()\n            \n            if component not in recovery_manager.circuit_breakers:\n                recovery_manager.register_circuit_breaker(\n                    component, failure_threshold, recovery_timeout\n                )\n            \n            circuit_breaker = recovery_manager.circuit_breakers[component]\n            return circuit_breaker.call(func, *args, **kwargs)\n        \n        return wrapper\n    return decorator\n\n\n# Global recovery manager instance\n_error_recovery_manager: Optional[ErrorRecoveryManager] = None\n\n\ndef get_error_recovery_manager() -> ErrorRecoveryManager:\n    \"\"\"Get global error recovery manager instance.\"\"\"\n    global _error_recovery_manager\n    if _error_recovery_manager is None:\n        _error_recovery_manager = ErrorRecoveryManager()\n    return _error_recovery_manager\n\n\ndef initialize_error_recovery():\n    \"\"\"Initialize global error recovery manager.\"\"\"\n    global _error_recovery_manager\n    _error_recovery_manager = ErrorRecoveryManager()\n    logger.info(\"Global error recovery manager initialized\")\n"
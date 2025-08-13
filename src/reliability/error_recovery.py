"""
Advanced Error Recovery and Fault Tolerance System
Implements robust error handling with automatic recovery capabilities.
"""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from contextlib import asynccontextmanager
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    IMMEDIATE_SHUTDOWN = "immediate_shutdown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    operation: str
    parameters: Dict[str, Any]
    stack_trace: str
    recovery_attempts: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action specification."""
    strategy: RecoveryStrategy
    parameters: Dict[str, Any]
    timeout_seconds: float
    max_attempts: int
    backoff_multiplier: float = 1.5
    jitter: bool = True


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        now = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if now >= self.next_attempt_time:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.next_attempt_time = time.time() + self.config.recovery_timeout
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout


class ErrorRecoverySystem:
    """Advanced error recovery and fault tolerance system."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.error_log: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_patterns: Dict[str, RecoveryStrategy] = {}
        
        # Statistics
        self.stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "critical_errors": 0,
            "recovery_success_rate": 0.0,
            "mean_recovery_time": 0.0
        }
        
        if config_file:
            self.load_configuration(config_file)
        else:
            self.setup_default_configuration()
    
    def setup_default_configuration(self):
        """Setup default error recovery configuration."""
        
        # Default recovery strategies
        self.recovery_strategies.update({
            "optimization_failure": RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                parameters={"max_retries": 3, "exponential_backoff": True},
                timeout_seconds=300.0,
                max_attempts=3,
                backoff_multiplier=2.0
            ),
            "hardware_communication_error": RecoveryAction(
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                parameters={"failure_threshold": 5, "recovery_timeout": 60.0},
                timeout_seconds=30.0,
                max_attempts=5
            ),
            "memory_error": RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                parameters={"reduced_batch_size": 0.5, "simplified_computation": True},
                timeout_seconds=60.0,
                max_attempts=2
            ),
            "critical_system_failure": RecoveryAction(
                strategy=RecoveryStrategy.IMMEDIATE_SHUTDOWN,
                parameters={"save_state": True, "notify_admin": True},
                timeout_seconds=10.0,
                max_attempts=1
            )
        })
        
        # Default circuit breaker configurations
        self.setup_circuit_breaker("hardware_interface", CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2
        ))
        
        self.setup_circuit_breaker("optimization_engine", CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=3
        ))
        
        # Error pattern mapping
        self.error_patterns.update({
            "ConnectionError": RecoveryStrategy.CIRCUIT_BREAKER,
            "TimeoutError": RecoveryStrategy.RETRY,
            "MemoryError": RecoveryStrategy.GRACEFUL_DEGRADATION,
            "ValueError": RecoveryStrategy.FALLBACK,
            "RuntimeError": RecoveryStrategy.RETRY,
            "AssertionError": RecoveryStrategy.IMMEDIATE_SHUTDOWN
        })
    
    def setup_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Setup a circuit breaker for a specific component."""
        self.circuit_breakers[name] = CircuitBreaker(name, config)
        logger.info(f"Circuit breaker configured for {name}")
    
    def register_fallback_handler(self, error_type: str, handler: Callable):
        """Register a fallback handler for specific error types."""
        self.fallback_handlers[error_type] = handler
        logger.info(f"Fallback handler registered for {error_type}")
    
    async def execute_with_recovery(
        self,
        operation: Callable,
        component: str,
        operation_name: str,
        parameters: Dict[str, Any] = None,
        recovery_strategy: Optional[str] = None
    ) -> Any:
        """
        Execute an operation with automatic error recovery.
        
        Args:
            operation: The operation to execute
            component: Component name for circuit breaker
            operation_name: Operation name for logging
            parameters: Operation parameters
            recovery_strategy: Override recovery strategy
            
        Returns:
            Operation result or fallback value
        """
        if parameters is None:
            parameters = {}
        
        error_context = None
        
        # Check circuit breaker
        if component in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[component]
            if not circuit_breaker.can_execute():
                raise RuntimeError(f"Circuit breaker open for {component}")
        
        try:
            # Execute operation with timeout
            result = await asyncio.wait_for(operation(**parameters), timeout=30.0)
            
            # Record success in circuit breaker
            if component in self.circuit_breakers:
                self.circuit_breakers[component].record_success()
            
            return result
            
        except Exception as e:
            # Create error context
            error_context = ErrorContext(
                error_id=f"err_{int(time.time())}_{len(self.error_log)}",
                timestamp=datetime.now(timezone.utc),
                error_type=type(e).__name__,
                error_message=str(e),
                severity=self._determine_severity(e),
                component=component,
                operation=operation_name,
                parameters=parameters,
                stack_trace=traceback.format_exc(),
                metadata={"recovery_strategy": recovery_strategy}
            )
            
            # Record error in circuit breaker
            if component in self.circuit_breakers:
                self.circuit_breakers[component].record_failure()
            
            # Log and store error
            self.error_log.append(error_context)
            self.stats["total_errors"] += 1
            
            logger.error(f"Error in {component}.{operation_name}: {e}")
            
            # Attempt recovery
            recovery_result = await self._attempt_recovery(error_context, operation, parameters)
            
            if recovery_result is not None:
                self.stats["recovered_errors"] += 1
                error_context.resolution_time = datetime.now(timezone.utc)
                return recovery_result
            else:
                # Recovery failed, re-raise original exception
                if error_context.severity == ErrorSeverity.CRITICAL:
                    self.stats["critical_errors"] += 1
                raise
    
    async def _attempt_recovery(
        self,
        error_context: ErrorContext,
        operation: Callable,
        parameters: Dict[str, Any]
    ) -> Any:
        """Attempt to recover from an error."""
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(error_context)
        error_context.recovery_strategy = strategy
        
        logger.info(f"Attempting recovery for {error_context.error_id} using {strategy.value}")
        
        if strategy == RecoveryStrategy.RETRY:
            return await self._retry_with_backoff(error_context, operation, parameters)
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._execute_fallback(error_context, operation, parameters)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation(error_context, operation, parameters)
        elif strategy == RecoveryStrategy.IMMEDIATE_SHUTDOWN:
            await self._immediate_shutdown(error_context)
            return None
        
        return None
    
    def _determine_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Determine the appropriate recovery strategy for an error."""
        
        # Check if there's a specific strategy configured for this error type
        if error_context.error_type in self.error_patterns:
            return self.error_patterns[error_context.error_type]
        
        # Check if there's a component-specific strategy
        strategy_key = f"{error_context.component}_{error_context.error_type.lower()}"
        if strategy_key in self.recovery_strategies:
            return self.recovery_strategies[strategy_key].strategy
        
        # Default strategy based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.IMMEDIATE_SHUTDOWN
        elif error_context.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        elif error_context.severity == ErrorSeverity.MEDIUM:
            return RecoveryStrategy.FALLBACK
        else:
            return RecoveryStrategy.RETRY
    
    def _determine_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type and context."""
        
        critical_errors = [SystemExit, KeyboardInterrupt, MemoryError]
        high_errors = [ConnectionError, TimeoutError, RuntimeError]
        medium_errors = [ValueError, TypeError, AttributeError]
        
        if any(isinstance(exception, err_type) for err_type in critical_errors):
            return ErrorSeverity.CRITICAL
        elif any(isinstance(exception, err_type) for err_type in high_errors):
            return ErrorSeverity.HIGH
        elif any(isinstance(exception, err_type) for err_type in medium_errors):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    async def _retry_with_backoff(
        self,
        error_context: ErrorContext,
        operation: Callable,
        parameters: Dict[str, Any]
    ) -> Any:
        """Retry operation with exponential backoff."""
        
        strategy_config = self.recovery_strategies.get(
            f"{error_context.component}_retry",
            self.recovery_strategies.get("optimization_failure")  # Default
        )
        
        max_attempts = strategy_config.max_attempts
        backoff_multiplier = strategy_config.backoff_multiplier
        base_delay = 1.0
        
        for attempt in range(max_attempts):
            error_context.recovery_attempts += 1
            
            if attempt > 0:
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (backoff_multiplier ** (attempt - 1))
                if strategy_config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)  # Add jitter
                
                logger.info(f"Retrying {error_context.operation} in {delay:.2f}s (attempt {attempt + 1}/{max_attempts})")
                await asyncio.sleep(delay)
            
            try:
                result = await asyncio.wait_for(
                    operation(**parameters),
                    timeout=strategy_config.timeout_seconds
                )
                logger.info(f"Recovery successful for {error_context.error_id} after {attempt + 1} attempts")
                return result
                
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    logger.error(f"All retry attempts exhausted for {error_context.error_id}")
                    break
        
        return None
    
    async def _execute_fallback(
        self,
        error_context: ErrorContext,
        operation: Callable,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute fallback handler for the error."""
        
        fallback_key = f"{error_context.component}_{error_context.error_type}"
        if fallback_key in self.fallback_handlers:
            try:
                logger.info(f"Executing fallback for {error_context.error_id}")
                result = await self.fallback_handlers[fallback_key](error_context, parameters)
                logger.info(f"Fallback successful for {error_context.error_id}")
                return result
            except Exception as e:
                logger.error(f"Fallback failed for {error_context.error_id}: {e}")
        
        # Default fallback: return safe default value
        return self._get_safe_default(error_context)
    
    async def _graceful_degradation(
        self,
        error_context: ErrorContext,
        operation: Callable,
        parameters: Dict[str, Any]
    ) -> Any:
        """Implement graceful degradation for the failed operation."""
        
        logger.info(f"Implementing graceful degradation for {error_context.error_id}")
        
        # Modify parameters for degraded operation
        degraded_params = parameters.copy()
        
        if error_context.error_type == "MemoryError":
            # Reduce computational complexity
            if "batch_size" in degraded_params:
                degraded_params["batch_size"] = max(1, degraded_params["batch_size"] // 2)
            if "resolution" in degraded_params:
                degraded_params["resolution"] = max(0.001, degraded_params["resolution"] * 2)
            if "iterations" in degraded_params:
                degraded_params["iterations"] = max(10, degraded_params["iterations"] // 2)
        
        elif error_context.error_type == "TimeoutError":
            # Reduce timeout requirements
            if "timeout" in degraded_params:
                degraded_params["timeout"] = degraded_params["timeout"] * 2
            if "max_iterations" in degraded_params:
                degraded_params["max_iterations"] = max(10, degraded_params["max_iterations"] // 4)
        
        try:
            result = await asyncio.wait_for(operation(**degraded_params), timeout=60.0)
            logger.info(f"Graceful degradation successful for {error_context.error_id}")
            return result
        except Exception as e:
            logger.error(f"Graceful degradation failed for {error_context.error_id}: {e}")
            return self._get_safe_default(error_context)
    
    async def _immediate_shutdown(self, error_context: ErrorContext):
        """Handle immediate shutdown for critical errors."""
        
        logger.critical(f"Initiating immediate shutdown due to critical error: {error_context.error_id}")
        
        # Save current state
        await self._save_system_state(error_context)
        
        # Notify administrators
        await self._notify_administrators(error_context)
        
        # Graceful shutdown sequence
        logger.critical("System shutdown initiated due to critical error")
        
        # In a real system, this would trigger actual shutdown procedures
        # For now, we'll just log the critical error
        
    def _get_safe_default(self, error_context: ErrorContext) -> Any:
        """Get a safe default value based on the operation context."""
        
        operation_defaults = {
            "optimize": {"phases": [], "loss": float('inf'), "success": False},
            "calculate_field": {"field_data": [], "metrics": {}},
            "hardware_control": {"status": "disconnected", "success": False},
            "generate_pattern": {"pattern": [], "confidence": 0.0}
        }
        
        for op_name, default_value in operation_defaults.items():
            if op_name in error_context.operation.lower():
                return default_value
        
        # Generic safe default
        return {"error": True, "error_id": error_context.error_id, "message": "Operation failed with fallback"}
    
    async def _save_system_state(self, error_context: ErrorContext):
        """Save current system state for recovery."""
        
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_context": {
                "error_id": error_context.error_id,
                "error_type": error_context.error_type,
                "error_message": error_context.error_message,
                "component": error_context.component,
                "operation": error_context.operation
            },
            "system_metrics": await self._collect_system_metrics(),
            "active_operations": self._get_active_operations(),
            "configuration": self._get_system_configuration()
        }
        
        state_file = Path("system_state_emergency.json")
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"System state saved to {state_file}")
    
    async def _notify_administrators(self, error_context: ErrorContext):
        """Notify system administrators of critical errors."""
        
        notification = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "CRITICAL",
            "error_id": error_context.error_id,
            "component": error_context.component,
            "operation": error_context.operation,
            "error_message": error_context.error_message,
            "system_status": "SHUTDOWN_INITIATED"
        }
        
        # In a real system, this would send notifications via email, Slack, etc.
        logger.critical(f"ADMINISTRATOR NOTIFICATION: {json.dumps(notification, indent=2)}")
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        
        try:
            import psutil
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_processes": len(psutil.pids()),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except ImportError:
            return {"error": "psutil not available for system metrics"}
    
    def _get_active_operations(self) -> List[str]:
        """Get list of currently active operations."""
        # This would track actual active operations in a real system
        return ["monitoring", "background_optimization", "health_checks"]
    
    def _get_system_configuration(self) -> Dict[str, Any]:
        """Get current system configuration."""
        return {
            "recovery_strategies": len(self.recovery_strategies),
            "circuit_breakers": len(self.circuit_breakers),
            "fallback_handlers": len(self.fallback_handlers),
            "error_patterns": len(self.error_patterns)
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        
        if self.stats["total_errors"] > 0:
            self.stats["recovery_success_rate"] = (
                self.stats["recovered_errors"] / self.stats["total_errors"] * 100
            )
        
        # Calculate mean recovery time
        recovery_times = []
        for error in self.error_log:
            if error.resolution_time and error.timestamp:
                recovery_time = (error.resolution_time - error.timestamp).total_seconds()
                recovery_times.append(recovery_time)
        
        if recovery_times:
            self.stats["mean_recovery_time"] = sum(recovery_times) / len(recovery_times)
        
        # Error distribution by severity
        severity_distribution = {}
        for severity in ErrorSeverity:
            count = sum(1 for error in self.error_log if error.severity == severity)
            severity_distribution[severity.value] = count
        
        # Error distribution by component
        component_distribution = {}
        for error in self.error_log:
            component_distribution[error.component] = component_distribution.get(error.component, 0) + 1
        
        return {
            **self.stats,
            "severity_distribution": severity_distribution,
            "component_distribution": component_distribution,
            "total_errors_logged": len(self.error_log),
            "circuit_breaker_states": {
                name: breaker.state.value
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    def load_configuration(self, config_file: str):
        """Load error recovery configuration from file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load recovery strategies
            if "recovery_strategies" in config:
                for name, strategy_config in config["recovery_strategies"].items():
                    self.recovery_strategies[name] = RecoveryAction(
                        strategy=RecoveryStrategy(strategy_config["strategy"]),
                        parameters=strategy_config["parameters"],
                        timeout_seconds=strategy_config["timeout_seconds"],
                        max_attempts=strategy_config["max_attempts"],
                        backoff_multiplier=strategy_config.get("backoff_multiplier", 1.5),
                        jitter=strategy_config.get("jitter", True)
                    )
            
            # Load circuit breaker configurations
            if "circuit_breakers" in config:
                for name, cb_config in config["circuit_breakers"].items():
                    self.setup_circuit_breaker(name, CircuitBreakerConfig(
                        failure_threshold=cb_config["failure_threshold"],
                        recovery_timeout=cb_config["recovery_timeout"],
                        success_threshold=cb_config["success_threshold"],
                        timeout=cb_config.get("timeout", 30.0)
                    ))
            
            # Load error patterns
            if "error_patterns" in config:
                for error_type, strategy in config["error_patterns"].items():
                    self.error_patterns[error_type] = RecoveryStrategy(strategy)
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            self.setup_default_configuration()
    
    def save_configuration(self, config_file: str):
        """Save current error recovery configuration to file."""
        
        config = {
            "recovery_strategies": {
                name: {
                    "strategy": action.strategy.value,
                    "parameters": action.parameters,
                    "timeout_seconds": action.timeout_seconds,
                    "max_attempts": action.max_attempts,
                    "backoff_multiplier": action.backoff_multiplier,
                    "jitter": action.jitter
                }
                for name, action in self.recovery_strategies.items()
            },
            "circuit_breakers": {
                name: {
                    "failure_threshold": breaker.config.failure_threshold,
                    "recovery_timeout": breaker.config.recovery_timeout,
                    "success_threshold": breaker.config.success_threshold,
                    "timeout": breaker.config.timeout
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "error_patterns": {
                error_type: strategy.value
                for error_type, strategy in self.error_patterns.items()
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_file}")


# Global error recovery system instance
error_recovery = ErrorRecoverySystem()


@asynccontextmanager
async def error_recovery_context(
    component: str,
    operation: str,
    parameters: Dict[str, Any] = None,
    recovery_strategy: Optional[str] = None
):
    """Context manager for error recovery."""
    
    if parameters is None:
        parameters = {}
    
    try:
        yield
    except Exception as e:
        # Handle error through recovery system
        await error_recovery._attempt_recovery(
            ErrorContext(
                error_id=f"ctx_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                error_type=type(e).__name__,
                error_message=str(e),
                severity=error_recovery._determine_severity(e),
                component=component,
                operation=operation,
                parameters=parameters,
                stack_trace=traceback.format_exc(),
                recovery_strategy=RecoveryStrategy(recovery_strategy) if recovery_strategy else None
            ),
            lambda **kwargs: None,  # Dummy operation for context
            parameters
        )
        raise


def resilient_operation(
    component: str,
    operation_name: str = None,
    recovery_strategy: str = None,
    timeout: float = 30.0
):
    """Decorator for making operations resilient with automatic error recovery."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            return await error_recovery.execute_with_recovery(
                operation=lambda **params: func(*args, **kwargs),
                component=component,
                operation_name=op_name,
                parameters=kwargs,
                recovery_strategy=recovery_strategy
            )
        
        return wrapper
    return decorator


# Example usage demonstration
async def demonstrate_error_recovery():
    """Demonstrate error recovery capabilities."""
    
    print("🛡️ Error Recovery System Demonstration")
    print("=" * 50)
    
    # Setup error recovery
    recovery_system = ErrorRecoverySystem()
    
    # Define a problematic operation
    @resilient_operation(component="optimization", operation_name="test_optimization")
    async def problematic_operation(fail_count: int = 0):
        if fail_count > 0:
            raise ConnectionError("Simulated connection failure")
        return {"result": "success", "value": 42}
    
    # Register a fallback handler
    async def optimization_fallback(error_context, parameters):
        print(f"  🔄 Executing fallback for {error_context.error_id}")
        return {"result": "fallback", "value": 0, "error_handled": True}
    
    recovery_system.register_fallback_handler("optimization_ConnectionError", optimization_fallback)
    
    try:
        # Test successful operation
        print("\n1. Testing successful operation:")
        result = await problematic_operation()
        print(f"   ✅ Result: {result}")
        
        # Test operation with recovery
        print("\n2. Testing operation with error and recovery:")
        result = await problematic_operation(fail_count=1)
        print(f"   ✅ Recovered result: {result}")
        
    except Exception as e:
        print(f"   ❌ Unrecoverable error: {e}")
    
    # Display statistics
    print("\n📊 Error Recovery Statistics:")
    stats = recovery_system.get_error_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 50)
    return recovery_system


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_error_recovery())
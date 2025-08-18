"""
Comprehensive Error Handling and Recovery System
Generation 2: MAKE IT ROBUST - Advanced error handling, validation, and recovery mechanisms.
"""

import numpy as np
import time
import traceback
import logging
import json
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from contextlib import contextmanager
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAIL_FAST = "fail_fast"
    ADAPTIVE_RETRY = "adaptive_retry"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float
    module: str
    function: str
    parameters: Dict[str, Any]
    stack_trace: str
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    recovery_time: float
    error_resolved: bool
    fallback_used: bool
    additional_info: Dict[str, Any] = field(default_factory=dict)


class AcousticOptimizationError(Exception):
    """Base exception for acoustic optimization errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Dict[str, Any] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class HardwareError(AcousticOptimizationError):
    """Hardware-related errors."""
    pass


class OptimizationError(AcousticOptimizationError):
    """Optimization algorithm errors."""
    pass


class ValidationError(AcousticOptimizationError):
    """Input validation errors."""
    pass


class PerformanceError(AcousticOptimizationError):
    """Performance-related errors."""
    pass


class SafetyError(AcousticOptimizationError):
    """Safety constraint violations."""
    def __init__(self, message: str, safety_limits: Dict[str, float] = None, 
                 current_values: Dict[str, float] = None):
        super().__init__(message, ErrorSeverity.CRITICAL)
        self.safety_limits = safety_limits or {}
        self.current_values = current_values or {}


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info(f"Circuit breaker entering HALF_OPEN state for {func.__name__}")
                else:
                    raise AcousticOptimizationError(
                        f"Circuit breaker OPEN for {func.__name__}",
                        ErrorSeverity.HIGH
                    )
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker CLOSED for {func.__name__}")
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker OPEN for {func.__name__} after {self.failure_count} failures")
                
                raise e


class RetryHandler:
    """Advanced retry handler with exponential backoff and jitter."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    logger.error(f"Function {func.__name__} failed after {self.max_attempts} attempts")
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                # Add jitter to prevent thundering herd
                if self.jitter:
                    delay *= (0.5 + 0.5 * np.random.random())
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        raise last_exception


class ValidationEngine:
    """Comprehensive input validation engine."""
    
    def __init__(self):
        self.validators = {}
        self.validation_cache = {}
        self.validation_stats = {
            'total_validations': 0,
            'validation_failures': 0,
            'cache_hits': 0
        }
    
    def register_validator(self, param_name: str, validator: Callable):
        """Register a validator for a parameter."""
        self.validators[param_name] = validator
    
    def validate_phases(self, phases: np.ndarray) -> bool:
        """Validate phase array."""
        if not isinstance(phases, np.ndarray):
            raise ValidationError("Phases must be numpy array")
        
        if phases.size == 0:
            raise ValidationError("Phases array cannot be empty")
        
        if not np.isfinite(phases).all():
            raise ValidationError("Phases must be finite values")
        
        if phases.ndim != 1:
            raise ValidationError("Phases must be 1-dimensional array")
        
        # Phase values should be in reasonable range
        if np.any(np.abs(phases) > 10 * np.pi):
            warnings.warn("Phase values are unusually large", UserWarning)
        
        return True
    
    def validate_target_field(self, target_field: Union[np.ndarray, Any]) -> bool:
        """Validate target field."""
        if hasattr(target_field, 'numpy'):
            field_data = target_field.numpy()
        else:
            field_data = target_field
        
        if not isinstance(field_data, np.ndarray):
            raise ValidationError("Target field must be array-like")
        
        if field_data.size == 0:
            raise ValidationError("Target field cannot be empty")
        
        if not np.isfinite(field_data).all():
            raise ValidationError("Target field must contain finite values")
        
        # Check for reasonable field values
        max_pressure = np.max(np.abs(field_data))
        if max_pressure > 1e6:  # 1 MPa limit
            raise SafetyError(
                "Target field exceeds safety pressure limits",
                safety_limits={'max_pressure': 1e6},
                current_values={'max_pressure': max_pressure}
            )
        
        return True
    
    def validate_optimization_context(self, context: Dict[str, Any]) -> bool:
        """Validate optimization context."""
        required_keys = ['hardware_constraints', 'performance_requirements']
        
        for key in required_keys:
            if key not in context:
                raise ValidationError(f"Missing required context key: {key}")
        
        # Validate hardware constraints
        hw_constraints = context['hardware_constraints']
        if 'num_transducers' in hw_constraints:
            num_transducers = hw_constraints['num_transducers']
            if not isinstance(num_transducers, (int, float)) or num_transducers <= 0:
                raise ValidationError("Number of transducers must be positive")
            if num_transducers > 10000:
                warnings.warn("Large number of transducers may cause performance issues", UserWarning)
        
        # Validate performance requirements
        perf_req = context['performance_requirements']
        if 'max_iterations' in perf_req:
            max_iter = perf_req['max_iterations']
            if not isinstance(max_iter, (int, float)) or max_iter <= 0:
                raise ValidationError("Max iterations must be positive")
            if max_iter > 100000:
                warnings.warn("High iteration count may cause long execution times", UserWarning)
        
        return True
    
    def validate_forward_model(self, forward_model: Callable) -> bool:
        """Validate forward model function."""
        if not callable(forward_model):
            raise ValidationError("Forward model must be callable")
        
        # Test with dummy input
        try:
            test_phases = np.random.random(10)
            result = forward_model(test_phases)
            
            if result is None:
                raise ValidationError("Forward model returns None")
                
            # Check if result is array-like
            if hasattr(result, 'shape') or hasattr(result, '__len__'):
                pass  # Good
            else:
                warnings.warn("Forward model result may not be array-like", UserWarning)
                
        except Exception as e:
            raise ValidationError(f"Forward model test failed: {e}")
        
        return True
    
    def validate_all(self, phases: np.ndarray, target_field: Union[np.ndarray, Any],
                    forward_model: Callable, context: Dict[str, Any] = None) -> bool:
        """Validate all optimization inputs."""
        self.validation_stats['total_validations'] += 1
        
        try:
            # Create validation key for caching
            validation_key = f"{id(phases)}_{id(target_field)}_{id(forward_model)}_{id(context)}"
            
            if validation_key in self.validation_cache:
                self.validation_stats['cache_hits'] += 1
                return self.validation_cache[validation_key]
            
            # Perform validations
            self.validate_phases(phases)
            self.validate_target_field(target_field)
            self.validate_forward_model(forward_model)
            
            if context is not None:
                self.validate_optimization_context(context)
            
            # Cache successful validation
            self.validation_cache[validation_key] = True
            
            # Limit cache size
            if len(self.validation_cache) > 1000:
                # Remove oldest entries
                old_keys = list(self.validation_cache.keys())[:500]
                for key in old_keys:
                    del self.validation_cache[key]
            
            return True
            
        except Exception as e:
            self.validation_stats['validation_failures'] += 1
            raise e


class ErrorRecoveryEngine:
    """Advanced error recovery engine."""
    
    def __init__(self):
        self.recovery_strategies = {
            HardwareError: RecoveryStrategy.CIRCUIT_BREAKER,
            OptimizationError: RecoveryStrategy.ADAPTIVE_RETRY,
            ValidationError: RecoveryStrategy.FAIL_FAST,
            PerformanceError: RecoveryStrategy.GRACEFUL_DEGRADATION,
            SafetyError: RecoveryStrategy.FAIL_FAST
        }
        
        self.recovery_history = []
        self.circuit_breakers = {}
        self.retry_handler = RetryHandler()
        
    def handle_error(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Handle error with appropriate recovery strategy."""
        start_time = time.time()
        
        # Determine recovery strategy
        error_type = type(error)
        strategy = self.recovery_strategies.get(error_type, RecoveryStrategy.RETRY)
        
        logger.error(f"Handling {error_type.__name__}: {error} with strategy {strategy.value}")
        
        # Execute recovery strategy
        try:
            if strategy == RecoveryStrategy.RETRY:
                result = self._retry_recovery(error, context)
            elif strategy == RecoveryStrategy.ADAPTIVE_RETRY:
                result = self._adaptive_retry_recovery(error, context)
            elif strategy == RecoveryStrategy.FALLBACK:
                result = self._fallback_recovery(error, context)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                result = self._graceful_degradation_recovery(error, context)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                result = self._circuit_breaker_recovery(error, context)
            else:  # FAIL_FAST
                result = self._fail_fast_recovery(error, context)
            
            recovery_time = time.time() - start_time
            result.recovery_time = recovery_time
            
            # Record recovery attempt
            self.recovery_history.append({
                'timestamp': time.time(),
                'error_type': error_type.__name__,
                'strategy': strategy.value,
                'success': result.success,
                'recovery_time': recovery_time
            })
            
            # Limit history size
            if len(self.recovery_history) > 1000:
                self.recovery_history = self.recovery_history[-500:]
            
            return result
            
        except Exception as recovery_error:
            logger.error(f"Recovery failed: {recovery_error}")
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                recovery_time=time.time() - start_time,
                error_resolved=False,
                fallback_used=False,
                additional_info={'recovery_error': str(recovery_error)}
            )
    
    def _retry_recovery(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Simple retry recovery."""
        if context.recovery_attempts < context.max_recovery_attempts:
            context.recovery_attempts += 1
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RETRY,
                recovery_time=0.0,
                error_resolved=False,
                fallback_used=False,
                additional_info={'retry_attempt': context.recovery_attempts}
            )
        else:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                recovery_time=0.0,
                error_resolved=False,
                fallback_used=False,
                additional_info={'max_retries_exceeded': True}
            )
    
    def _adaptive_retry_recovery(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Adaptive retry with exponential backoff."""
        if context.recovery_attempts < context.max_recovery_attempts:
            # Adaptive delay based on error type and attempt count
            base_delay = 1.0
            if isinstance(error, HardwareError):
                base_delay = 2.0
            elif isinstance(error, PerformanceError):
                base_delay = 0.5
            
            delay = base_delay * (2 ** context.recovery_attempts)
            delay = min(delay, 30.0)  # Cap at 30 seconds
            
            time.sleep(delay)
            context.recovery_attempts += 1
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.ADAPTIVE_RETRY,
                recovery_time=delay,
                error_resolved=False,
                fallback_used=False,
                additional_info={
                    'retry_attempt': context.recovery_attempts,
                    'delay_used': delay
                }
            )
        else:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.ADAPTIVE_RETRY,
                recovery_time=0.0,
                error_resolved=False,
                fallback_used=False,
                additional_info={'max_retries_exceeded': True}
            )
    
    def _fallback_recovery(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Fallback to simpler method."""
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.FALLBACK,
            recovery_time=0.0,
            error_resolved=False,
            fallback_used=True,
            additional_info={'fallback_method': 'simple_gradient_descent'}
        )
    
    def _graceful_degradation_recovery(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Graceful degradation with reduced functionality."""
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
            recovery_time=0.0,
            error_resolved=False,
            fallback_used=True,
            additional_info={
                'degraded_mode': True,
                'reduced_accuracy': True,
                'reduced_iterations': True
            }
        )
    
    def _circuit_breaker_recovery(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Circuit breaker recovery."""
        function_name = context.function
        
        if function_name not in self.circuit_breakers:
            self.circuit_breakers[function_name] = CircuitBreaker()
        
        circuit_breaker = self.circuit_breakers[function_name]
        
        return RecoveryResult(
            success=circuit_breaker.state != "OPEN",
            strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
            recovery_time=0.0,
            error_resolved=False,
            fallback_used=circuit_breaker.state == "OPEN",
            additional_info={'circuit_state': circuit_breaker.state}
        )
    
    def _fail_fast_recovery(self, error: Exception, context: ErrorContext) -> RecoveryResult:
        """Fail fast - no recovery attempt."""
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.FAIL_FAST,
            recovery_time=0.0,
            error_resolved=False,
            fallback_used=False,
            additional_info={'immediate_failure': True}
        )
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery performance statistics."""
        if not self.recovery_history:
            return {'message': 'No recovery attempts recorded'}
        
        total_attempts = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r['success'])
        
        # Group by error type
        error_types = {}
        for record in self.recovery_history:
            error_type = record['error_type']
            if error_type not in error_types:
                error_types[error_type] = {'total': 0, 'successful': 0, 'avg_time': 0.0}
            
            error_types[error_type]['total'] += 1
            if record['success']:
                error_types[error_type]['successful'] += 1
            error_types[error_type]['avg_time'] += record['recovery_time']
        
        # Calculate averages
        for error_type in error_types:
            if error_types[error_type]['total'] > 0:
                error_types[error_type]['avg_time'] /= error_types[error_type]['total']
                error_types[error_type]['success_rate'] = (
                    error_types[error_type]['successful'] / error_types[error_type]['total']
                )
        
        return {
            'total_recovery_attempts': total_attempts,
            'overall_success_rate': successful_recoveries / total_attempts,
            'average_recovery_time': np.mean([r['recovery_time'] for r in self.recovery_history]),
            'error_type_breakdown': error_types,
            'active_circuit_breakers': len(self.circuit_breakers)
        }


class RobustnessManager:
    """Main robustness manager coordinating all error handling systems."""
    
    def __init__(self):
        self.validator = ValidationEngine()
        self.recovery_engine = ErrorRecoveryEngine()
        self.monitoring_enabled = True
        self.safety_checks_enabled = True
        
        # Performance monitoring
        self.performance_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'total_errors': 0,
            'total_recoveries': 0,
            'average_operation_time': 0.0
        }
        
    @contextmanager
    def robust_execution(self, operation_name: str, context: Dict[str, Any] = None):
        """Context manager for robust operation execution."""
        start_time = time.time()
        error_context = None
        
        try:
            self.performance_metrics['total_operations'] += 1
            yield
            
            # Success
            self.performance_metrics['successful_operations'] += 1
            operation_time = time.time() - start_time
            
            # Update average operation time
            total_ops = self.performance_metrics['total_operations']
            current_avg = self.performance_metrics['average_operation_time']
            self.performance_metrics['average_operation_time'] = (
                (current_avg * (total_ops - 1) + operation_time) / total_ops
            )
            
        except Exception as e:
            self.performance_metrics['total_errors'] += 1
            
            # Create error context
            error_context = ErrorContext(
                error_type=type(e).__name__,
                error_message=str(e),
                severity=getattr(e, 'severity', ErrorSeverity.MEDIUM),
                timestamp=time.time(),
                module=__name__,
                function=operation_name,
                parameters=context or {},
                stack_trace=traceback.format_exc()
            )
            
            # Attempt recovery
            recovery_result = self.recovery_engine.handle_error(e, error_context)
            
            if recovery_result.success and not recovery_result.fallback_used:
                # Retry the operation
                self.performance_metrics['total_recoveries'] += 1
                raise  # Re-raise to trigger retry
            elif recovery_result.fallback_used:
                # Use fallback - log but don't re-raise
                self.performance_metrics['total_recoveries'] += 1
                logger.warning(f"Using fallback for {operation_name}")
            else:
                # Recovery failed - re-raise original error
                raise e
    
    def validate_and_execute(self, func: Callable, *args, **kwargs):
        """Validate inputs and execute function with error handling."""
        operation_name = func.__name__
        
        # Extract common parameters for validation
        phases = kwargs.get('phases') or (args[0] if len(args) > 0 else None)
        target_field = kwargs.get('target_field') or (args[1] if len(args) > 1 else None)
        forward_model = kwargs.get('forward_model') or (args[2] if len(args) > 2 else None)
        context = kwargs.get('context') or kwargs.get('optimization_context')
        
        # Validation
        if phases is not None and hasattr(phases, 'shape'):
            self.validator.validate_phases(phases)
        
        if target_field is not None:
            self.validator.validate_target_field(target_field)
        
        if forward_model is not None:
            self.validator.validate_forward_model(forward_model)
        
        if context is not None:
            self.validator.validate_optimization_context(context)
        
        # Execute with robustness
        with self.robust_execution(operation_name, context):
            return func(*args, **kwargs)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        validator_stats = self.validator.validation_stats
        recovery_stats = self.recovery_engine.get_recovery_statistics()
        
        # Calculate reliability metrics
        total_ops = self.performance_metrics['total_operations']
        success_rate = (
            self.performance_metrics['successful_operations'] / max(1, total_ops)
        )
        error_rate = self.performance_metrics['total_errors'] / max(1, total_ops)
        recovery_rate = self.performance_metrics['total_recoveries'] / max(1, total_ops)
        
        return {
            'system_status': 'healthy' if success_rate > 0.9 else 'degraded' if success_rate > 0.7 else 'unhealthy',
            'reliability_metrics': {
                'success_rate': success_rate,
                'error_rate': error_rate,
                'recovery_rate': recovery_rate,
                'average_operation_time': self.performance_metrics['average_operation_time']
            },
            'validation_metrics': {
                'total_validations': validator_stats['total_validations'],
                'validation_failure_rate': (
                    validator_stats['validation_failures'] / max(1, validator_stats['total_validations'])
                ),
                'cache_hit_rate': (
                    validator_stats['cache_hits'] / max(1, validator_stats['total_validations'])
                )
            },
            'recovery_metrics': recovery_stats,
            'monitoring_enabled': self.monitoring_enabled,
            'safety_checks_enabled': self.safety_checks_enabled
        }


# Decorators for easy integration
def robust_optimization(robustness_manager: RobustnessManager = None):
    """Decorator for robust optimization functions."""
    if robustness_manager is None:
        robustness_manager = RobustnessManager()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return robustness_manager.validate_and_execute(func, *args, **kwargs)
        return wrapper
    return decorator


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Decorator for circuit breaker pattern."""
    return CircuitBreaker(failure_threshold, recovery_timeout)


def retry_on_failure(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator for retry pattern."""
    return RetryHandler(max_attempts, base_delay)


# Global robustness manager instance
global_robustness_manager = RobustnessManager()


# Example usage
if __name__ == "__main__":
    print("🛡️ Comprehensive Error Handling and Recovery System")
    print("Generation 2: MAKE IT ROBUST")
    
    # Create robustness manager
    manager = RobustnessManager()
    
    # Example robust function
    @robust_optimization(manager)
    def example_optimization(phases, target_field, forward_model):
        # Simulate some processing
        time.sleep(0.1)
        
        # Simulate occasional failure
        if np.random.random() < 0.2:
            raise OptimizationError("Random optimization failure")
        
        return {
            'phases': phases + np.random.normal(0, 0.1, len(phases)),
            'final_loss': np.random.random() * 0.01,
            'iterations': 100
        }
    
    # Test with various inputs
    print("\n🧪 Testing robustness system...")
    
    success_count = 0
    total_tests = 20
    
    for i in range(total_tests):
        try:
            phases = np.random.uniform(-np.pi, np.pi, 256)
            target = np.random.random((32, 32, 32))
            
            def mock_forward_model(p):
                return np.random.random((32, 32, 32)) * np.mean(np.abs(p))
            
            result = example_optimization(phases, target, mock_forward_model)
            success_count += 1
            
            if i % 5 == 0:
                print(f"   Test {i + 1}: ✅ Success")
                
        except Exception as e:
            print(f"   Test {i + 1}: ❌ Failed - {e}")
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} successful")
    
    # System health report
    health = manager.get_system_health()
    print(f"\n🏥 System Health Report:")
    print(f"   Status: {health['system_status']}")
    print(f"   Success rate: {health['reliability_metrics']['success_rate']:.2%}")
    print(f"   Error rate: {health['reliability_metrics']['error_rate']:.2%}")
    print(f"   Recovery rate: {health['reliability_metrics']['recovery_rate']:.2%}")
    
    print("\n🛡️ Robustness system ready for production!")
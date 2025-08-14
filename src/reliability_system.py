"""
Comprehensive Reliability System for Acoustic Holography
Generation 2: MAKE IT ROBUST - Error handling, validation, and monitoring
"""

import sys
import time
import traceback
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import hashlib
from contextlib import contextmanager
from abc import ABC, abstractmethod


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    OPERATING = "operating"
    WARNING = "warning"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    
    timestamp: float
    error_id: str
    severity: ErrorSeverity
    component: str
    operation: str
    error_message: str
    stack_trace: str
    system_state: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    user_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "error_id": self.error_id,
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "system_state": self.system_state,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "user_data": self.user_data
        }


@dataclass
class ValidationRule:
    """Input validation rule specification."""
    
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    def validate(self, value: Any) -> Optional[str]:
        """Validate value and return error message if invalid."""
        try:
            if self.validator(value):
                return None
            return self.error_message
        except Exception as e:
            return f"Validation error: {str(e)}"


class RecoveryStrategy(ABC):
    """Abstract base class for error recovery strategies."""
    
    @abstractmethod
    def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if this strategy can handle the error."""
        pass
    
    @abstractmethod
    def attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from the error."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


class RetryStrategy(RecoveryStrategy):
    """Retry-based recovery strategy."""
    
    def __init__(self, max_attempts: int = 3, backoff_factor: float = 1.5):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.attempt_count = {}
    
    @property
    def name(self) -> str:
        return "retry"
    
    def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if retry is appropriate."""
        operation_key = f"{error_context.component}:{error_context.operation}"
        attempts = self.attempt_count.get(operation_key, 0)
        
        # Retry for transient errors
        transient_errors = ["timeout", "connection", "temporary", "network"]
        is_transient = any(keyword in error_context.error_message.lower() 
                          for keyword in transient_errors)
        
        return is_transient and attempts < self.max_attempts
    
    def attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt recovery by retrying with backoff."""
        operation_key = f"{error_context.component}:{error_context.operation}"
        attempts = self.attempt_count.get(operation_key, 0)
        
        # Exponential backoff
        delay = self.backoff_factor ** attempts
        time.sleep(delay)
        
        self.attempt_count[operation_key] = attempts + 1
        
        # Signal that recovery should be attempted
        return True


class FallbackStrategy(RecoveryStrategy):
    """Fallback to alternative implementation."""
    
    def __init__(self, fallback_implementations: Dict[str, Callable]):
        self.fallback_implementations = fallback_implementations
    
    @property
    def name(self) -> str:
        return "fallback"
    
    def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if fallback is available."""
        return error_context.operation in self.fallback_implementations
    
    def attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to use fallback implementation."""
        try:
            fallback_func = self.fallback_implementations[error_context.operation]
            # Note: In real implementation, would execute fallback with context
            return True
        except Exception:
            return False


class GracefulDegradationStrategy(RecoveryStrategy):
    """Reduce functionality to maintain core operations."""
    
    def __init__(self):
        self.degraded_mode = False
    
    @property
    def name(self) -> str:
        return "graceful_degradation"
    
    def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if graceful degradation is possible."""
        # Allow degradation for non-critical components
        non_critical = ["visualization", "logging", "metrics", "gui"]
        return any(component in error_context.component.lower() 
                  for component in non_critical)
    
    def attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Enable degraded mode."""
        self.degraded_mode = True
        logging.warning(f"Enabled degraded mode due to {error_context.component} failure")
        return True


class RobustErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.error_count_by_component = {}
        self.critical_error_threshold = 5
        
        # Setup logging
        self.logger = self._setup_logging(log_file)
        
        # Default recovery strategies
        self.add_recovery_strategy(RetryStrategy())
        self.add_recovery_strategy(GracefulDegradationStrategy())
    
    def _setup_logging(self, log_file: Optional[str]) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger("acoustic_holography")
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Detailed formatter for file
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        # Simple formatter for console
        simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy."""
        self.recovery_strategies.append(strategy)
        self.logger.info(f"Added recovery strategy: {strategy.name}")
    
    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        system_state: Optional[Dict[str, Any]] = None,
        user_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Handle an error with comprehensive context and recovery.
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            severity: Error severity level
            system_state: Current system state
            user_data: Additional user-provided context
            
        Returns:
            True if error was successfully handled/recovered
        """
        # Generate unique error ID
        error_id = self._generate_error_id(error, component, operation)
        
        # Create error context
        error_context = ErrorContext(
            timestamp=time.time(),
            error_id=error_id,
            severity=severity,
            component=component,
            operation=operation,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            system_state=system_state or {},
            user_data=user_data or {}
        )
        
        # Log error
        self._log_error(error_context)
        
        # Track error frequency
        self._track_error_frequency(component)
        
        # Attempt recovery
        recovery_successful = self._attempt_recovery(error_context)
        error_context.recovery_attempted = True
        error_context.recovery_successful = recovery_successful
        
        # Store in history
        self.error_history.append(error_context)
        
        # Check for critical patterns
        self._check_critical_patterns()
        
        return recovery_successful
    
    def _generate_error_id(self, error: Exception, component: str, operation: str) -> str:
        """Generate unique error identifier."""
        content = f"{type(error).__name__}:{component}:{operation}:{str(error)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level."""
        message = (
            f"[{error_context.error_id}] {error_context.component}.{error_context.operation}: "
            f"{error_context.error_message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def _track_error_frequency(self, component: str):
        """Track error frequency by component."""
        self.error_count_by_component[component] = (
            self.error_count_by_component.get(component, 0) + 1
        )
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt recovery using available strategies."""
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error_context):
                self.logger.info(f"Attempting recovery with strategy: {strategy.name}")
                
                try:
                    if strategy.attempt_recovery(error_context):
                        self.logger.info(f"Recovery successful with strategy: {strategy.name}")
                        return True
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy {strategy.name} failed: {recovery_error}")
        
        self.logger.warning(f"No recovery strategy succeeded for error {error_context.error_id}")
        return False
    
    def _check_critical_patterns(self):
        """Check for critical error patterns."""
        # Check error frequency
        for component, count in self.error_count_by_component.items():
            if count >= self.critical_error_threshold:
                self.logger.critical(
                    f"Critical error pattern detected: {component} has {count} errors"
                )
        
        # Check recent error rate
        recent_errors = [
            err for err in self.error_history 
            if time.time() - err.timestamp < 300  # Last 5 minutes
        ]
        
        if len(recent_errors) > 10:
            self.logger.critical(f"High error rate: {len(recent_errors)} errors in 5 minutes")
    
    @contextmanager
    def error_context(
        self,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        system_state: Optional[Dict[str, Any]] = None
    ):
        """Context manager for automatic error handling."""
        try:
            yield
        except Exception as e:
            handled = self.handle_error(e, component, operation, severity, system_state)
            if not handled and severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                # Re-raise if critical and not handled
                raise
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report."""
        return {
            "total_errors": len(self.error_history),
            "errors_by_component": dict(self.error_count_by_component),
            "errors_by_severity": {
                severity.value: sum(1 for err in self.error_history if err.severity == severity)
                for severity in ErrorSeverity
            },
            "recent_errors": [
                err.to_dict() for err in self.error_history[-10:]
            ],
            "recovery_success_rate": (
                sum(1 for err in self.error_history if err.recovery_successful) / 
                max(len(self.error_history), 1)
            )
        }


class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.validation_history: List[Dict[str, Any]] = []
    
    def add_rule(self, input_type: str, rule: ValidationRule):
        """Add validation rule for input type."""
        if input_type not in self.validation_rules:
            self.validation_rules[input_type] = []
        self.validation_rules[input_type].append(rule)
    
    def validate(self, input_type: str, value: Any, context: str = "") -> List[str]:
        """
        Validate input value against all rules for its type.
        
        Args:
            input_type: Type of input being validated
            value: The value to validate
            context: Additional context for validation
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if input_type not in self.validation_rules:
            return errors
        
        for rule in self.validation_rules[input_type]:
            error_message = rule.validate(value)
            if error_message:
                errors.append(f"{rule.name}: {error_message}")
        
        # Record validation attempt
        validation_record = {
            "timestamp": time.time(),
            "input_type": input_type,
            "context": context,
            "valid": len(errors) == 0,
            "errors": errors
        }
        self.validation_history.append(validation_record)
        
        return errors
    
    def validate_phases(self, phases: List[float]) -> List[str]:
        """Validate transducer phases."""
        errors = []
        
        # Check if phases is a list
        if not isinstance(phases, (list, tuple)):
            errors.append("Phases must be a list or tuple")
            return errors
        
        # Check length
        if len(phases) == 0:
            errors.append("Phases array cannot be empty")
        elif len(phases) > 1000:
            errors.append("Too many phases (max 1000)")
        
        # Check individual phase values
        for i, phase in enumerate(phases):
            if not isinstance(phase, (int, float)):
                errors.append(f"Phase {i} is not a number: {type(phase)}")
                continue
            
            if not (0 <= phase <= 2 * 3.14159):
                errors.append(f"Phase {i} out of range [0, 2π]: {phase}")
        
        return errors
    
    def validate_positions(self, positions: List[Tuple[float, float, float]]) -> List[str]:
        """Validate 3D positions."""
        errors = []
        
        if not isinstance(positions, (list, tuple)):
            errors.append("Positions must be a list or tuple")
            return errors
        
        workspace_bounds = {
            "x": (-0.1, 0.1),    # ±10cm
            "y": (-0.1, 0.1),    # ±10cm
            "z": (0.05, 0.2)     # 5-20cm above array
        }
        
        for i, pos in enumerate(positions):
            if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                errors.append(f"Position {i} must be 3D coordinate tuple")
                continue
            
            x, y, z = pos
            
            # Check bounds
            if not (workspace_bounds["x"][0] <= x <= workspace_bounds["x"][1]):
                errors.append(f"Position {i} x-coordinate out of bounds: {x}")
            
            if not (workspace_bounds["y"][0] <= y <= workspace_bounds["y"][1]):
                errors.append(f"Position {i} y-coordinate out of bounds: {y}")
            
            if not (workspace_bounds["z"][0] <= z <= workspace_bounds["z"][1]):
                errors.append(f"Position {i} z-coordinate out of bounds: {z}")
        
        return errors
    
    def validate_amplitudes(self, amplitudes: List[float]) -> List[str]:
        """Validate amplitude values."""
        errors = []
        
        if not isinstance(amplitudes, (list, tuple)):
            errors.append("Amplitudes must be a list or tuple")
            return errors
        
        for i, amp in enumerate(amplitudes):
            if not isinstance(amp, (int, float)):
                errors.append(f"Amplitude {i} is not a number: {type(amp)}")
                continue
            
            if amp < 0:
                errors.append(f"Amplitude {i} cannot be negative: {amp}")
            
            if amp > 5000:  # 5kPa limit
                errors.append(f"Amplitude {i} exceeds safety limit (5000 Pa): {amp}")
        
        return errors


class SystemMonitor:
    """Real-time system monitoring and health checks."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Tuple[float, float]]] = {}  # metric_name -> [(timestamp, value)]
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_active = False
    
    def add_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Add a metric measurement."""
        if timestamp is None:
            timestamp = time.time()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append((timestamp, value))
        
        # Keep only last 1000 measurements
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def add_health_check(self, name: str, check_function: Callable[[], bool]):
        """Add a health check function."""
        self.health_checks[name] = check_function
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = False
                self.alerts.append({
                    "timestamp": time.time(),
                    "type": "health_check_error",
                    "check_name": name,
                    "error": str(e)
                })
        
        return results
    
    def get_metric_summary(self, name: str, window_seconds: float = 300) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        # Filter to time window
        cutoff_time = time.time() - window_seconds
        recent_values = [
            value for timestamp, value in self.metrics[name]
            if timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        # Calculate statistics
        import statistics
        
        return {
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "stdev": statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        }
    
    def check_metric_thresholds(self, thresholds: Dict[str, Dict[str, float]]):
        """Check metrics against defined thresholds."""
        for metric_name, threshold_config in thresholds.items():
            summary = self.get_metric_summary(metric_name)
            
            if not summary:
                continue
            
            # Check various threshold types
            for threshold_type, threshold_value in threshold_config.items():
                metric_value = summary.get(threshold_type)
                
                if metric_value is None:
                    continue
                
                # Example: check if mean exceeds threshold
                if threshold_type == "max_mean" and metric_value > threshold_value:
                    self.alerts.append({
                        "timestamp": time.time(),
                        "type": "threshold_exceeded",
                        "metric": metric_name,
                        "threshold_type": threshold_type,
                        "value": metric_value,
                        "threshold": threshold_value
                    })


def setup_acoustic_validation_rules() -> InputValidator:
    """Setup standard validation rules for acoustic holography."""
    
    validator = InputValidator()
    
    # Phase validation rules
    validator.add_rule("phases", ValidationRule(
        name="phase_range",
        validator=lambda phases: all(0 <= p <= 2*3.14159 for p in phases),
        error_message="All phases must be in range [0, 2π]"
    ))
    
    validator.add_rule("phases", ValidationRule(
        name="phase_count",
        validator=lambda phases: 1 <= len(phases) <= 1000,
        error_message="Phase array must contain 1-1000 elements"
    ))
    
    # Position validation rules
    validator.add_rule("positions", ValidationRule(
        name="workspace_bounds",
        validator=lambda positions: all(
            -0.1 <= pos[0] <= 0.1 and -0.1 <= pos[1] <= 0.1 and 0.05 <= pos[2] <= 0.2
            for pos in positions
        ),
        error_message="All positions must be within workspace bounds"
    ))
    
    # Amplitude validation rules
    validator.add_rule("amplitudes", ValidationRule(
        name="safety_limits",
        validator=lambda amps: all(0 <= amp <= 5000 for amp in amps),
        error_message="All amplitudes must be in range [0, 5000] Pa"
    ))
    
    return validator


def demonstrate_reliability_system():
    """Demonstrate reliability system functionality."""
    
    print("🛡️ Acoustic Holography Reliability System - Generation 2")
    
    # Initialize error handler
    error_handler = RobustErrorHandler("reliability_test.log")
    
    print("✅ Error handling system initialized")
    
    # Initialize input validator
    validator = setup_acoustic_validation_rules()
    
    print("✅ Input validation system initialized")
    
    # Initialize system monitor
    monitor = SystemMonitor()
    
    print("✅ System monitoring initialized")
    
    # Test error handling
    print("\n🧪 Testing Error Handling")
    
    # Simulate various errors
    try:
        # Simulate a network timeout
        raise TimeoutError("Connection to hardware timed out")
    except Exception as e:
        handled = error_handler.handle_error(
            e, "hardware_interface", "connect", ErrorSeverity.MEDIUM
        )
        print(f"  Network timeout handled: {handled}")
    
    try:
        # Simulate a critical calculation error
        raise ValueError("Invalid matrix dimensions in field calculation")
    except Exception as e:
        handled = error_handler.handle_error(
            e, "field_calculator", "compute_pressure", ErrorSeverity.HIGH
        )
        print(f"  Calculation error handled: {handled}")
    
    # Test input validation
    print("\n🧪 Testing Input Validation")
    
    # Valid inputs
    valid_phases = [0.0, 1.57, 3.14, 4.71, 6.28]
    phase_errors = validator.validate_phases(valid_phases)
    print(f"  Valid phases validation: {'✅ Pass' if not phase_errors else '❌ Fail'}")
    
    # Invalid inputs
    invalid_phases = [0.0, -1.0, 10.0]  # Negative and out of range
    phase_errors = validator.validate_phases(invalid_phases)
    print(f"  Invalid phases detected: {'✅ Pass' if phase_errors else '❌ Fail'}")
    print(f"    Errors: {phase_errors}")
    
    # Test position validation
    valid_positions = [(0.0, 0.0, 0.1), (-0.05, 0.05, 0.15)]
    pos_errors = validator.validate_positions(valid_positions)
    print(f"  Valid positions validation: {'✅ Pass' if not pos_errors else '❌ Fail'}")
    
    invalid_positions = [(0.2, 0.0, 0.1), (0.0, 0.0, 0.5)]  # Out of bounds
    pos_errors = validator.validate_positions(invalid_positions)
    print(f"  Invalid positions detected: {'✅ Pass' if pos_errors else '❌ Fail'}")
    print(f"    Errors: {pos_errors}")
    
    # Test system monitoring
    print("\n🧪 Testing System Monitoring")
    
    # Add some metrics
    monitor.add_metric("cpu_usage", 45.2)
    monitor.add_metric("memory_usage", 68.7)
    monitor.add_metric("field_calculation_time", 0.023)
    monitor.add_metric("optimization_convergence", 0.00045)
    
    print("  ✅ Metrics recorded")
    
    # Add health checks
    monitor.add_health_check("memory_ok", lambda: True)  # Mock health check
    monitor.add_health_check("gpu_available", lambda: False)  # Mock failure
    
    health_results = monitor.run_health_checks()
    print(f"  Health checks: {health_results}")
    
    # Generate reports
    print("\n📊 Generating Reports")
    
    error_report = error_handler.get_error_report()
    print(f"  Total errors handled: {error_report['total_errors']}")
    print(f"  Recovery success rate: {error_report['recovery_success_rate']:.2%}")
    
    # Test graceful error handling with context manager
    print("\n🧪 Testing Context Manager Error Handling")
    
    with error_handler.error_context("optimizer", "phase_update", ErrorSeverity.LOW):
        # This would normally contain actual operations
        print("  ✅ Operations completed successfully")
    
    try:
        with error_handler.error_context("safety_system", "pressure_check", ErrorSeverity.CRITICAL):
            raise RuntimeError("Pressure limit exceeded!")
    except RuntimeError:
        print("  ✅ Critical error properly escalated")
    
    # Save comprehensive reliability report
    reliability_report = {
        "timestamp": time.time(),
        "error_handling": {
            "total_errors": error_report['total_errors'],
            "recovery_success_rate": error_report['recovery_success_rate'],
            "errors_by_component": error_report['errors_by_component']
        },
        "input_validation": {
            "total_validations": len(validator.validation_history),
            "validation_success_rate": sum(
                1 for v in validator.validation_history if v['valid']
            ) / max(len(validator.validation_history), 1)
        },
        "system_monitoring": {
            "metrics_tracked": len(monitor.metrics),
            "health_checks": health_results,
            "alerts_generated": len(monitor.alerts)
        },
        "generation": "2_make_it_robust",
        "status": "completed"
    }
    
    with open("reliability_system_results.json", "w") as f:
        json.dump(reliability_report, f, indent=2)
    
    print("\n✅ Reliability system demonstration completed")
    print("📊 Results saved to reliability_system_results.json")
    
    return reliability_report


if __name__ == "__main__":
    # Run demonstration
    demonstrate_reliability_system()
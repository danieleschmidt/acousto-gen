"""
Comprehensive safety monitoring system for acoustic holography applications.
Implements real-time safety validation, emergency controls, and regulatory compliance.
"""

import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from collections import deque
import warnings

from ..models.acoustic_field import AcousticField
from ..physics.transducers.transducer_array import TransducerArray
from ..monitoring.metrics import MetricsCollector


class SafetyLevel(Enum):
    """Safety alert levels."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ViolationType(Enum):
    """Types of safety violations."""
    PRESSURE_EXCEEDED = "pressure_exceeded"
    INTENSITY_EXCEEDED = "intensity_exceeded"
    TEMPERATURE_HIGH = "temperature_high"
    EXPOSURE_TIME_LONG = "exposure_time_long"
    SPATIAL_HOTSPOT = "spatial_hotspot"
    POWER_EXCEEDED = "power_exceeded"
    HARDWARE_FAULT = "hardware_fault"
    COMMUNICATION_LOSS = "communication_loss"


@dataclass
class SafetyLimits:
    """Comprehensive safety limits configuration."""
    # Acoustic pressure limits (Pa)
    max_pressure: float = 4000  # FDA limit for diagnostic ultrasound
    max_spatial_peak_pressure: float = 10000  # Peak pressure anywhere
    max_temporal_average_pressure: float = 2000  # Time-averaged limit
    
    # Acoustic intensity limits (W/cm²)
    max_intensity: float = 10  # Spatial peak, temporal average
    max_spatial_peak_intensity: float = 50  # Instantaneous peak
    
    # Temperature limits (°C)
    max_temperature: float = 45  # Tissue damage threshold
    max_temperature_rise: float = 10  # Maximum rise above baseline
    ambient_temperature: float = 37  # Body temperature
    
    # Exposure time limits (seconds)
    max_continuous_exposure: float = 300  # 5 minutes
    max_daily_exposure: float = 3600  # 1 hour total
    min_rest_period: float = 60  # Between exposures
    
    # Power limits (W)
    max_total_power: float = 100  # Total acoustic power
    max_element_power: float = 5  # Per transducer element
    
    # Spatial limits
    max_gradient: float = 1000  # Pa/mm - spatial pressure gradient
    min_focus_size: float = 0.001  # 1mm minimum focus diameter
    
    # Special limits for different applications
    medical_limits: Dict[str, float] = field(default_factory=lambda: {
        'max_pressure': 1000,  # More conservative for medical
        'max_intensity': 3,
        'max_temperature': 42
    })
    
    haptics_limits: Dict[str, float] = field(default_factory=lambda: {
        'max_pressure': 500,  # Perception threshold safety
        'max_intensity': 1,
        'max_temperature': 40
    })
    
    research_limits: Dict[str, float] = field(default_factory=lambda: {
        'max_pressure': 8000,  # Higher limits for research
        'max_intensity': 20,
        'max_temperature': 50
    })


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    violation_type: ViolationType
    severity: SafetyLevel
    timestamp: float
    value: float
    limit: float
    location: Optional[Tuple[float, float, float]]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyStatus:
    """Current safety status."""
    overall_level: SafetyLevel
    active_violations: List[SafetyViolation]
    recent_violations: List[SafetyViolation]
    system_health: Dict[str, Any]
    uptime: float
    last_check: float
    monitoring_enabled: bool


class SafetyMonitor:
    """
    Comprehensive safety monitoring system for acoustic holography.
    
    Provides real-time monitoring of acoustic pressure, intensity, temperature,
    and other safety-critical parameters with automatic violation detection
    and emergency response capabilities.
    """
    
    def __init__(
        self,
        limits: Optional[SafetyLimits] = None,
        check_interval: float = 0.1,  # 100ms monitoring interval
        enable_auto_shutdown: bool = True,
        enable_logging: bool = True
    ):
        """
        Initialize safety monitor.
        
        Args:
            limits: Safety limits configuration
            check_interval: Time between safety checks in seconds
            enable_auto_shutdown: Enable automatic emergency shutdown
            enable_logging: Enable safety event logging
        """
        self.limits = limits or SafetyLimits()
        self.check_interval = check_interval
        self.enable_auto_shutdown = enable_auto_shutdown
        self.enable_logging = enable_logging
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_monitoring = False
        
        # Safety status
        self.current_status = SafetyStatus(
            overall_level=SafetyLevel.SAFE,
            active_violations=[],
            recent_violations=[],
            system_health={},
            uptime=0,
            last_check=0,
            monitoring_enabled=False
        )
        
        # Violation history
        self.violation_history: deque = deque(maxlen=1000)
        self.exposure_history: deque = deque(maxlen=100)
        
        # Callbacks
        self.violation_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        
        # Hardware interfaces
        self.transducer_array: Optional[TransducerArray] = None
        self.temperature_sensors: List[Any] = []
        self.power_meters: List[Any] = []
        
        # Performance tracking
        self.check_times = deque(maxlen=100)
        self.false_alarm_count = 0
        
        # Setup logging
        if self.enable_logging:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Create safety log file
            handler = logging.FileHandler('safety_monitor.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            self.logger = None
        
        self._log_info("Safety monitor initialized")
    
    def set_application_mode(self, mode: str) -> None:
        """
        Set safety limits based on application mode.
        
        Args:
            mode: Application mode ('medical', 'haptics', 'research', 'general')
        """
        if mode == 'medical':
            for key, value in self.limits.medical_limits.items():
                setattr(self.limits, key, value)
        elif mode == 'haptics':
            for key, value in self.limits.haptics_limits.items():
                setattr(self.limits, key, value)
        elif mode == 'research':
            for key, value in self.limits.research_limits.items():
                setattr(self.limits, key, value)
        
        self._log_info(f"Safety limits set for {mode} mode")
    
    def register_hardware(self, transducer_array: TransducerArray) -> None:
        """Register transducer array for monitoring."""
        self.transducer_array = transducer_array
        self._log_info(f"Registered transducer array: {transducer_array.name}")
    
    def add_violation_callback(self, callback: Callable[[SafetyViolation], None]) -> None:
        """Register callback for safety violations."""
        self.violation_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for emergency situations."""
        self.emergency_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start real-time safety monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_monitoring = False
        self.current_status.monitoring_enabled = True
        self.current_status.uptime = time.time()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self._log_info("Safety monitoring started")
    
    def stop_monitoring_loop(self) -> None:
        """Stop real-time safety monitoring."""
        if not self.is_monitoring:
            return
        
        self.stop_monitoring = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        self.is_monitoring = False
        self.current_status.monitoring_enabled = False
        
        self._log_info("Safety monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while not self.stop_monitoring:
            start_time = time.time()
            
            try:
                # Perform safety checks
                self._perform_safety_check()
                
                # Update status
                self.current_status.last_check = time.time()
                self.current_status.uptime = time.time() - self.current_status.uptime
                
                # Record check time
                check_duration = time.time() - start_time
                self.check_times.append(check_duration)
                
                # Sleep until next check
                sleep_time = self.check_interval - check_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self._log_error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _perform_safety_check(self) -> None:
        """Perform comprehensive safety checks."""
        violations = []
        
        # Check acoustic field if available
        if hasattr(self, '_current_field') and self._current_field is not None:
            violations.extend(self._check_acoustic_safety(self._current_field))
        
        # Check hardware status
        violations.extend(self._check_hardware_safety())
        
        # Check exposure history
        violations.extend(self._check_exposure_safety())
        
        # Update active violations
        self.current_status.active_violations = violations
        
        # Determine overall safety level
        if any(v.severity == SafetyLevel.EMERGENCY for v in violations):
            self.current_status.overall_level = SafetyLevel.EMERGENCY
        elif any(v.severity == SafetyLevel.CRITICAL for v in violations):
            self.current_status.overall_level = SafetyLevel.CRITICAL
        elif any(v.severity == SafetyLevel.WARNING for v in violations):
            self.current_status.overall_level = SafetyLevel.WARNING
        else:
            self.current_status.overall_level = SafetyLevel.SAFE
        
        # Handle violations
        for violation in violations:
            self._handle_violation(violation)
    
    def _check_acoustic_safety(self, field: AcousticField) -> List[SafetyViolation]:
        """Check acoustic field for safety violations."""
        violations = []
        
        # Get field properties
        amplitude = field.get_amplitude_field()
        intensity = field.get_intensity_field()
        
        # Check maximum pressure
        max_pressure = np.max(amplitude)
        if max_pressure > self.limits.max_pressure:
            violations.append(SafetyViolation(
                violation_type=ViolationType.PRESSURE_EXCEEDED,
                severity=SafetyLevel.CRITICAL if max_pressure > 2 * self.limits.max_pressure else SafetyLevel.WARNING,
                timestamp=time.time(),
                value=max_pressure,
                limit=self.limits.max_pressure,
                location=self._find_max_location(amplitude, field),
                description=f"Maximum pressure {max_pressure:.1f} Pa exceeds limit {self.limits.max_pressure:.1f} Pa"
            ))
        
        # Check maximum intensity
        max_intensity = np.max(intensity)
        if max_intensity > self.limits.max_intensity:
            violations.append(SafetyViolation(
                violation_type=ViolationType.INTENSITY_EXCEEDED,
                severity=SafetyLevel.CRITICAL if max_intensity > 2 * self.limits.max_intensity else SafetyLevel.WARNING,
                timestamp=time.time(),
                value=max_intensity,
                limit=self.limits.max_intensity,
                location=self._find_max_location(intensity, field),
                description=f"Maximum intensity {max_intensity:.2f} W/cm² exceeds limit {self.limits.max_intensity:.2f} W/cm²"
            ))
        
        # Check spatial pressure gradient
        gradient_magnitude = self._calculate_pressure_gradient(amplitude, field)
        max_gradient = np.max(gradient_magnitude) / field.resolution  # Pa/m to Pa/mm
        
        if max_gradient > self.limits.max_gradient:
            violations.append(SafetyViolation(
                violation_type=ViolationType.SPATIAL_HOTSPOT,
                severity=SafetyLevel.WARNING,
                timestamp=time.time(),
                value=max_gradient,
                limit=self.limits.max_gradient,
                location=self._find_max_location(gradient_magnitude, field),
                description=f"Spatial gradient {max_gradient:.1f} Pa/mm exceeds limit {self.limits.max_gradient:.1f} Pa/mm"
            ))
        
        # Check total acoustic power
        total_power = np.sum(intensity) * field.resolution**3  # Approximate
        if total_power > self.limits.max_total_power:
            violations.append(SafetyViolation(
                violation_type=ViolationType.POWER_EXCEEDED,
                severity=SafetyLevel.WARNING,
                timestamp=time.time(),
                value=total_power,
                limit=self.limits.max_total_power,
                location=None,
                description=f"Total acoustic power {total_power:.1f} W exceeds limit {self.limits.max_total_power:.1f} W"
            ))
        
        return violations
    
    def _check_hardware_safety(self) -> List[SafetyViolation]:
        """Check hardware components for safety violations."""
        violations = []
        
        # Check transducer array temperature (if available)
        if self.transducer_array and hasattr(self.transducer_array, 'temperature'):
            temp = self.transducer_array.temperature
            if temp > self.limits.max_temperature:
                violations.append(SafetyViolation(
                    violation_type=ViolationType.TEMPERATURE_HIGH,
                    severity=SafetyLevel.CRITICAL if temp > self.limits.max_temperature + 5 else SafetyLevel.WARNING,
                    timestamp=time.time(),
                    value=temp,
                    limit=self.limits.max_temperature,
                    location=None,
                    description=f"Hardware temperature {temp:.1f}°C exceeds limit {self.limits.max_temperature:.1f}°C"
                ))
        
        # Check power consumption per element
        if self.transducer_array and hasattr(self.transducer_array, 'element_powers'):
            powers = self.transducer_array.element_powers
            max_element_power = np.max(powers)
            
            if max_element_power > self.limits.max_element_power:
                violations.append(SafetyViolation(
                    violation_type=ViolationType.POWER_EXCEEDED,
                    severity=SafetyLevel.WARNING,
                    timestamp=time.time(),
                    value=max_element_power,
                    limit=self.limits.max_element_power,
                    location=None,
                    description=f"Element power {max_element_power:.2f} W exceeds limit {self.limits.max_element_power:.2f} W"
                ))
        
        # Check communication with hardware
        if self.transducer_array and hasattr(self.transducer_array, 'is_connected'):
            if not self.transducer_array.is_connected():
                violations.append(SafetyViolation(
                    violation_type=ViolationType.COMMUNICATION_LOSS,
                    severity=SafetyLevel.CRITICAL,
                    timestamp=time.time(),
                    value=0,
                    limit=1,
                    location=None,
                    description="Lost communication with transducer array"
                ))
        
        return violations
    
    def _check_exposure_safety(self) -> List[SafetyViolation]:
        """Check exposure time limits."""
        violations = []
        current_time = time.time()
        
        # Check continuous exposure time
        if self.exposure_history:
            # Find current continuous exposure period
            continuous_exposure = 0
            for i in range(len(self.exposure_history) - 1, -1, -1):
                exposure = self.exposure_history[i]
                if current_time - exposure['end_time'] < self.limits.min_rest_period:
                    continuous_exposure += exposure['duration']
                else:
                    break
            
            if continuous_exposure > self.limits.max_continuous_exposure:
                violations.append(SafetyViolation(
                    violation_type=ViolationType.EXPOSURE_TIME_LONG,
                    severity=SafetyLevel.WARNING,
                    timestamp=current_time,
                    value=continuous_exposure,
                    limit=self.limits.max_continuous_exposure,
                    location=None,
                    description=f"Continuous exposure {continuous_exposure:.1f}s exceeds limit {self.limits.max_continuous_exposure:.1f}s"
                ))
        
        # Check daily exposure
        daily_exposure = sum(
            exp['duration'] for exp in self.exposure_history
            if current_time - exp['end_time'] < 86400  # 24 hours
        )
        
        if daily_exposure > self.limits.max_daily_exposure:
            violations.append(SafetyViolation(
                violation_type=ViolationType.EXPOSURE_TIME_LONG,
                severity=SafetyLevel.CRITICAL,
                timestamp=current_time,
                value=daily_exposure,
                limit=self.limits.max_daily_exposure,
                location=None,
                description=f"Daily exposure {daily_exposure:.1f}s exceeds limit {self.limits.max_daily_exposure:.1f}s"
            ))
        
        return violations
    
    def _calculate_pressure_gradient(
        self,
        pressure_field: np.ndarray,
        field: AcousticField
    ) -> np.ndarray:
        """Calculate spatial pressure gradient magnitude."""
        # Use numpy gradient
        grad_x = np.gradient(pressure_field, field.resolution, axis=0)
        grad_y = np.gradient(pressure_field, field.resolution, axis=1)
        grad_z = np.gradient(pressure_field, field.resolution, axis=2)
        
        # Calculate magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        return gradient_magnitude
    
    def _find_max_location(
        self,
        field: np.ndarray,
        acoustic_field: AcousticField
    ) -> Tuple[float, float, float]:
        """Find 3D location of maximum value in field."""
        max_idx = np.unravel_index(np.argmax(field), field.shape)
        
        # Convert indices to physical coordinates
        x = acoustic_field.x_coords[max_idx[0]]
        y = acoustic_field.y_coords[max_idx[1]]
        z = acoustic_field.z_coords[max_idx[2]]
        
        return (x, y, z)
    
    def _handle_violation(self, violation: SafetyViolation) -> None:
        """Handle detected safety violation."""
        # Record violation
        self.violation_history.append(violation)
        self.current_status.recent_violations.append(violation)
        
        # Log violation
        self._log_warning(f"Safety violation: {violation.description}")
        
        # Trigger callbacks
        for callback in self.violation_callbacks:
            try:
                callback(violation)
            except Exception as e:
                self._log_error(f"Error in violation callback: {e}")
        
        # Emergency response
        if violation.severity == SafetyLevel.EMERGENCY:
            self._trigger_emergency_response(violation)
        elif violation.severity == SafetyLevel.CRITICAL and self.enable_auto_shutdown:
            self._trigger_emergency_response(violation)
    
    def _trigger_emergency_response(self, violation: SafetyViolation) -> None:
        """Trigger emergency response procedures."""
        self._log_error(f"EMERGENCY: {violation.description}")
        
        # Emergency shutdown
        if self.transducer_array and hasattr(self.transducer_array, 'emergency_stop'):
            try:
                self.transducer_array.emergency_stop()
                self._log_info("Emergency stop activated")
            except Exception as e:
                self._log_error(f"Failed to execute emergency stop: {e}")
        
        # Trigger emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                self._log_error(f"Error in emergency callback: {e}")
    
    def validate_field(
        self,
        field: AcousticField,
        duration: float = 0,
        allow_warnings: bool = True
    ) -> Tuple[bool, List[SafetyViolation]]:
        """
        Validate acoustic field against safety limits.
        
        Args:
            field: Acoustic field to validate
            duration: Expected exposure duration
            allow_warnings: Whether to allow warnings (only fail on critical)
            
        Returns:
            Tuple of (is_safe, violations_list)
        """
        # Store field for checking
        self._current_field = field
        
        # Perform safety check
        violations = self._check_acoustic_safety(field)
        
        # Check exposure duration
        if duration > self.limits.max_continuous_exposure:
            violations.append(SafetyViolation(
                violation_type=ViolationType.EXPOSURE_TIME_LONG,
                severity=SafetyLevel.WARNING,
                timestamp=time.time(),
                value=duration,
                limit=self.limits.max_continuous_exposure,
                location=None,
                description=f"Planned exposure {duration:.1f}s exceeds limit {self.limits.max_continuous_exposure:.1f}s"
            ))
        
        # Determine if safe
        if allow_warnings:
            is_safe = not any(v.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY] for v in violations)
        else:
            is_safe = len(violations) == 0
        
        return is_safe, violations
    
    def validate_pattern(
        self,
        phases: np.ndarray,
        amplitudes: Optional[np.ndarray] = None,
        duration: float = 0
    ) -> Tuple[bool, List[SafetyViolation]]:
        """
        Validate phase/amplitude pattern before execution.
        
        Args:
            phases: Phase array for transducers
            amplitudes: Amplitude array (optional)
            duration: Expected exposure duration
            
        Returns:
            Tuple of (is_safe, violations_list)
        """
        if not self.transducer_array:
            warnings.warn("No transducer array registered for validation")
            return True, []
        
        # Estimate field from phases (simplified)
        if amplitudes is None:
            amplitudes = np.ones(len(phases))
        
        # Create mock field for validation (would use actual propagation in practice)
        # This is a simplified version
        max_estimated_pressure = np.sum(amplitudes) * 100  # Rough estimate
        
        violations = []
        
        if max_estimated_pressure > self.limits.max_pressure:
            violations.append(SafetyViolation(
                violation_type=ViolationType.PRESSURE_EXCEEDED,
                severity=SafetyLevel.WARNING,
                timestamp=time.time(),
                value=max_estimated_pressure,
                limit=self.limits.max_pressure,
                location=None,
                description=f"Estimated pressure {max_estimated_pressure:.1f} Pa may exceed limit"
            ))
        
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def record_exposure(self, duration: float, field_type: str = "unknown") -> None:
        """Record completed exposure for tracking."""
        exposure = {
            'start_time': time.time() - duration,
            'end_time': time.time(),
            'duration': duration,
            'field_type': field_type
        }
        
        self.exposure_history.append(exposure)
        self._log_info(f"Recorded exposure: {duration:.1f}s of {field_type}")
    
    def get_safety_status(self) -> SafetyStatus:
        """Get current safety status."""
        return self.current_status
    
    def get_violation_history(self, hours: int = 24) -> List[SafetyViolation]:
        """Get violation history for specified time period."""
        cutoff_time = time.time() - hours * 3600
        
        return [
            v for v in self.violation_history
            if v.timestamp > cutoff_time
        ]
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        current_time = time.time()
        
        # Calculate statistics
        total_violations = len(self.violation_history)
        recent_violations = len(self.get_violation_history(24))
        
        violation_by_type = {}
        for violation in self.violation_history:
            vtype = violation.violation_type.value
            violation_by_type[vtype] = violation_by_type.get(vtype, 0) + 1
        
        # Performance metrics
        avg_check_time = np.mean(self.check_times) if self.check_times else 0
        
        return {
            'timestamp': current_time,
            'monitoring_status': {
                'enabled': self.current_status.monitoring_enabled,
                'uptime': self.current_status.uptime,
                'check_interval': self.check_interval,
                'avg_check_time': avg_check_time
            },
            'current_status': {
                'level': self.current_status.overall_level.value,
                'active_violations': len(self.current_status.active_violations),
                'last_check': self.current_status.last_check
            },
            'violation_statistics': {
                'total_violations': total_violations,
                'recent_violations_24h': recent_violations,
                'violations_by_type': violation_by_type,
                'false_alarm_rate': self.false_alarm_count / max(1, total_violations)
            },
            'exposure_tracking': {
                'total_exposures': len(self.exposure_history),
                'total_exposure_time': sum(exp['duration'] for exp in self.exposure_history),
                'daily_exposure_time': sum(
                    exp['duration'] for exp in self.exposure_history
                    if current_time - exp['end_time'] < 86400
                )
            },
            'safety_limits': {
                'max_pressure': self.limits.max_pressure,
                'max_intensity': self.limits.max_intensity,
                'max_temperature': self.limits.max_temperature,
                'max_continuous_exposure': self.limits.max_continuous_exposure
            }
        }
    
    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self.logger:
            self.logger.info(message)
    
    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        if self.logger:
            self.logger.warning(message)
        print(f"WARNING: {message}")
    
    def _log_error(self, message: str) -> None:
        """Log error message."""
        if self.logger:
            self.logger.error(message)
        print(f"ERROR: {message}")


def create_safety_monitor_for_application(application_type: str) -> SafetyMonitor:
    """
    Create appropriately configured safety monitor for application.
    
    Args:
        application_type: Type of application ('medical', 'haptics', 'research')
        
    Returns:
        Configured SafetyMonitor instance
    """
    monitor = SafetyMonitor()
    monitor.set_application_mode(application_type)
    
    return monitor
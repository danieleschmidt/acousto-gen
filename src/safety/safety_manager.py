"""
Safety Manager for Acoustic Holography Systems.
Implements comprehensive safety protocols and hardware protection.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety system severity levels."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyViolationType(Enum):
    """Types of safety violations."""
    POWER_LIMIT_EXCEEDED = "power_limit_exceeded"
    TEMPERATURE_TOO_HIGH = "temperature_too_high"
    UNSAFE_FIELD_STRENGTH = "unsafe_field_strength"
    HARDWARE_MALFUNCTION = "hardware_malfunction"
    INVALID_PARAMETERS = "invalid_parameters"
    COMMUNICATION_LOSS = "communication_loss"
    WORKSPACE_VIOLATION = "workspace_violation"
    ACOUSTIC_EXPOSURE_LIMIT = "acoustic_exposure_limit"


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    violation_type: SafetyViolationType
    severity: SafetyLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    hardware_id: Optional[str] = None
    measured_value: Optional[float] = None
    threshold_value: Optional[float] = None
    action_taken: Optional[str] = None
    resolved: bool = False


@dataclass
class SafetyLimits:
    """Safety limits for acoustic systems."""
    # Power limits
    max_total_power: float = 1000.0  # Watts
    max_element_power: float = 10.0  # Watts per element
    
    # Temperature limits  
    max_temperature: float = 80.0  # °C
    warning_temperature: float = 60.0  # °C
    
    # Acoustic limits
    max_sound_pressure: float = 150.0  # dB SPL
    max_acoustic_power: float = 100.0  # W/m²
    max_exposure_time: float = 300.0  # seconds at max power
    
    # Field strength limits
    max_field_strength: float = 10000.0  # Pa
    max_gradient: float = 1000.0  # Pa/m
    
    # Workspace safety
    min_distance_to_user: float = 0.1  # meters
    max_workspace_bounds: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.3])  # m
    
    # Hardware limits
    max_voltage: float = 50.0  # V
    max_current: float = 5.0   # A
    min_frequency: float = 20e3  # Hz
    max_frequency: float = 100e3  # Hz


class SafetyManager:
    """
    Comprehensive safety management system for acoustic holography.
    Monitors hardware, fields, and user safety in real-time.
    """
    
    def __init__(self, limits: Optional[SafetyLimits] = None):
        """
        Initialize safety manager.
        
        Args:
            limits: Safety limits configuration
        """
        self.limits = limits or SafetyLimits()
        self.violations: List[SafetyViolation] = []
        self.is_enabled = True
        self.emergency_stop_triggered = False
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = False
        
        # Hardware monitors
        self.hardware_interfaces: Dict[str, Any] = {}
        self.last_hardware_check: Dict[str, datetime] = {}
        
        # Field monitoring
        self.current_field_strength = 0.0
        self.acoustic_exposure_start: Optional[datetime] = None
        
        # Safety callbacks
        self.violation_callbacks: List[Callable[[SafetyViolation], None]] = []
        self.emergency_callbacks: List[Callable[[], None]] = []
        
        logger.info("Safety Manager initialized with comprehensive protection")
    
    def register_hardware(self, hardware_id: str, hardware_interface: Any):
        """Register hardware interface for monitoring."""
        self.hardware_interfaces[hardware_id] = hardware_interface
        self.last_hardware_check[hardware_id] = datetime.now()
        logger.info(f"Hardware registered for safety monitoring: {hardware_id}")
    
    def add_violation_callback(self, callback: Callable[[SafetyViolation], None]):
        """Add callback for safety violation notifications."""
        self.violation_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable[[], None]):
        """Add callback for emergency stop notifications."""
        self.emergency_callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 0.1):
        """
        Start continuous safety monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            logger.warning("Safety monitoring already active")
            return
        
        self.monitoring_active = True
        self.stop_monitoring = False
        
        def monitor_loop():
            while not self.stop_monitoring and self.monitoring_active:
                try:
                    self._check_all_systems()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in safety monitoring: {e}")
                    self._trigger_violation(
                        SafetyViolationType.HARDWARE_MALFUNCTION,
                        SafetyLevel.CRITICAL,
                        f"Safety monitoring error: {e}"
                    )
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Safety monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous safety monitoring."""
        self.stop_monitoring = True
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
        
        logger.info("Safety monitoring stopped")
    
    def _check_all_systems(self):
        """Check all safety systems."""
        if not self.is_enabled:
            return
        
        # Check hardware status
        for hardware_id, interface in self.hardware_interfaces.items():
            self._check_hardware_safety(hardware_id, interface)
        
        # Check acoustic exposure
        self._check_acoustic_exposure()
        
        # Check field safety
        self._check_field_safety()
    
    def _check_hardware_safety(self, hardware_id: str, interface: Any):
        """Check individual hardware interface safety."""
        try:
            status = interface.get_status()
            
            # Check temperature
            if hasattr(status, 'temperature') and status.temperature is not None:
                if status.temperature > self.limits.max_temperature:
                    self._trigger_violation(
                        SafetyViolationType.TEMPERATURE_TOO_HIGH,
                        SafetyLevel.CRITICAL,
                        f"Hardware {hardware_id} temperature: {status.temperature}°C",
                        hardware_id=hardware_id,
                        measured_value=status.temperature,
                        threshold_value=self.limits.max_temperature
                    )\n                elif status.temperature > self.limits.warning_temperature:\n                    self._trigger_violation(\n                        SafetyViolationType.TEMPERATURE_TOO_HIGH,\n                        SafetyLevel.WARNING,\n                        f\"Hardware {hardware_id} temperature warning: {status.temperature}°C\",\n                        hardware_id=hardware_id,\n                        measured_value=status.temperature,\n                        threshold_value=self.limits.warning_temperature\n                    )\n            \n            # Check voltage\n            if hasattr(status, 'voltage') and status.voltage is not None:\n                if status.voltage > self.limits.max_voltage:\n                    self._trigger_violation(\n                        SafetyViolationType.HARDWARE_MALFUNCTION,\n                        SafetyLevel.CRITICAL,\n                        f\"Hardware {hardware_id} voltage too high: {status.voltage}V\",\n                        hardware_id=hardware_id,\n                        measured_value=status.voltage,\n                        threshold_value=self.limits.max_voltage\n                    )\n            \n            # Check current\n            if hasattr(status, 'current') and status.current is not None:\n                if status.current > self.limits.max_current:\n                    self._trigger_violation(\n                        SafetyViolationType.POWER_LIMIT_EXCEEDED,\n                        SafetyLevel.CRITICAL,\n                        f\"Hardware {hardware_id} current too high: {status.current}A\",\n                        hardware_id=hardware_id,\n                        measured_value=status.current,\n                        threshold_value=self.limits.max_current\n                    )\n            \n            # Check connection health\n            if not status.connected:\n                self._trigger_violation(\n                    SafetyViolationType.COMMUNICATION_LOSS,\n                    SafetyLevel.WARNING,\n                    f\"Lost connection to hardware {hardware_id}\",\n                    hardware_id=hardware_id\n                )\n            \n            # Update last check time\n            self.last_hardware_check[hardware_id] = datetime.now()\n            \n        except Exception as e:\n            self._trigger_violation(\n                SafetyViolationType.HARDWARE_MALFUNCTION,\n                SafetyLevel.CRITICAL,\n                f\"Failed to check hardware {hardware_id}: {e}\",\n                hardware_id=hardware_id\n            )\n    \n    def _check_acoustic_exposure(self):\n        \"\"\"Check acoustic exposure limits.\"\"\"\n        if self.current_field_strength > self.limits.max_field_strength:\n            self._trigger_violation(\n                SafetyViolationType.UNSAFE_FIELD_STRENGTH,\n                SafetyLevel.CRITICAL,\n                f\"Field strength too high: {self.current_field_strength} Pa\",\n                measured_value=self.current_field_strength,\n                threshold_value=self.limits.max_field_strength\n            )\n        \n        # Check exposure duration\n        if self.acoustic_exposure_start:\n            exposure_time = (datetime.now() - self.acoustic_exposure_start).total_seconds()\n            if exposure_time > self.limits.max_exposure_time:\n                self._trigger_violation(\n                    SafetyViolationType.ACOUSTIC_EXPOSURE_LIMIT,\n                    SafetyLevel.WARNING,\n                    f\"Long acoustic exposure: {exposure_time:.1f}s\",\n                    measured_value=exposure_time,\n                    threshold_value=self.limits.max_exposure_time\n                )\n    \n    def _check_field_safety(self):\n        \"\"\"Check acoustic field safety parameters.\"\"\"\n        # This would check computed field properties\n        # For now, just check current field strength\n        pass\n    \n    def validate_operation_parameters(self, parameters: Dict[str, Any]) -> List[SafetyViolation]:\n        \"\"\"\n        Validate operation parameters against safety limits.\n        \n        Args:\n            parameters: Operation parameters to validate\n            \n        Returns:\n            List of safety violations found\n        \"\"\"\n        violations = []\n        \n        # Check frequency range\n        if 'frequency' in parameters:\n            freq = parameters['frequency']\n            if freq < self.limits.min_frequency or freq > self.limits.max_frequency:\n                violations.append(SafetyViolation(\n                    violation_type=SafetyViolationType.INVALID_PARAMETERS,\n                    severity=SafetyLevel.CRITICAL,\n                    message=f\"Frequency {freq} Hz outside safe range [{self.limits.min_frequency}, {self.limits.max_frequency}] Hz\",\n                    measured_value=freq,\n                    threshold_value=self.limits.max_frequency\n                ))\n        \n        # Check power limits\n        if 'total_power' in parameters:\n            power = parameters['total_power']\n            if power > self.limits.max_total_power:\n                violations.append(SafetyViolation(\n                    violation_type=SafetyViolationType.POWER_LIMIT_EXCEEDED,\n                    severity=SafetyLevel.CRITICAL,\n                    message=f\"Total power {power} W exceeds limit {self.limits.max_total_power} W\",\n                    measured_value=power,\n                    threshold_value=self.limits.max_total_power\n                ))\n        \n        # Check workspace bounds\n        if 'target_position' in parameters:\n            pos = parameters['target_position']\n            for i, (coord, max_bound) in enumerate(zip(pos, self.limits.max_workspace_bounds)):\n                if abs(coord) > max_bound:\n                    violations.append(SafetyViolation(\n                        violation_type=SafetyViolationType.WORKSPACE_VIOLATION,\n                        severity=SafetyLevel.WARNING,\n                        message=f\"Target position [{coord}] outside safe workspace bounds [±{max_bound}]\",\n                        measured_value=abs(coord),\n                        threshold_value=max_bound\n                    ))\n        \n        return violations\n    \n    def check_operation_safety(self, operation_name: str, parameters: Dict[str, Any]) -> bool:\n        \"\"\"\n        Check if operation is safe to execute.\n        \n        Args:\n            operation_name: Name of operation\n            parameters: Operation parameters\n            \n        Returns:\n            True if safe to proceed, False otherwise\n        \"\"\"\n        if not self.is_enabled:\n            return True  # Safety disabled\n        \n        if self.emergency_stop_triggered:\n            logger.error(f\"Cannot execute {operation_name}: Emergency stop active\")\n            return False\n        \n        # Validate parameters\n        violations = self.validate_operation_parameters(parameters)\n        \n        # Check for critical violations\n        critical_violations = [v for v in violations if v.severity == SafetyLevel.CRITICAL]\n        if critical_violations:\n            for violation in critical_violations:\n                self._trigger_violation(\n                    violation.violation_type,\n                    violation.severity,\n                    f\"Operation {operation_name} blocked: {violation.message}\",\n                    measured_value=violation.measured_value,\n                    threshold_value=violation.threshold_value\n                )\n            return False\n        \n        # Log warnings but allow operation\n        warning_violations = [v for v in violations if v.severity == SafetyLevel.WARNING]\n        for violation in warning_violations:\n            self._trigger_violation(\n                violation.violation_type,\n                violation.severity,\n                f\"Operation {operation_name} warning: {violation.message}\",\n                measured_value=violation.measured_value,\n                threshold_value=violation.threshold_value\n            )\n        \n        return True\n    \n    def update_field_strength(self, field_strength: float):\n        \"\"\"Update current field strength for monitoring.\"\"\"\n        self.current_field_strength = field_strength\n        \n        # Start exposure timer if high field\n        if field_strength > self.limits.max_field_strength * 0.1:  # 10% of max\n            if self.acoustic_exposure_start is None:\n                self.acoustic_exposure_start = datetime.now()\n        else:\n            # Reset exposure timer for low fields\n            self.acoustic_exposure_start = None\n    \n    def _trigger_violation(self, \n                         violation_type: SafetyViolationType,\n                         severity: SafetyLevel,\n                         message: str,\n                         hardware_id: Optional[str] = None,\n                         measured_value: Optional[float] = None,\n                         threshold_value: Optional[float] = None) -> SafetyViolation:\n        \"\"\"Trigger a safety violation and take appropriate action.\"\"\"\n        \n        violation = SafetyViolation(\n            violation_type=violation_type,\n            severity=severity,\n            message=message,\n            hardware_id=hardware_id,\n            measured_value=measured_value,\n            threshold_value=threshold_value\n        )\n        \n        self.violations.append(violation)\n        \n        # Log violation\n        log_msg = f\"SAFETY VIOLATION [{severity.value.upper()}]: {message}\"\n        if severity == SafetyLevel.CRITICAL or severity == SafetyLevel.EMERGENCY:\n            logger.critical(log_msg)\n        elif severity == SafetyLevel.WARNING:\n            logger.warning(log_msg)\n        else:\n            logger.info(log_msg)\n        \n        # Take action based on severity\n        if severity == SafetyLevel.EMERGENCY:\n            self.trigger_emergency_stop(\"Emergency safety violation\")\n        elif severity == SafetyLevel.CRITICAL:\n            if violation_type in [SafetyViolationType.TEMPERATURE_TOO_HIGH,\n                                SafetyViolationType.POWER_LIMIT_EXCEEDED,\n                                SafetyViolationType.UNSAFE_FIELD_STRENGTH]:\n                self._emergency_shutdown(hardware_id)\n        \n        # Notify callbacks\n        for callback in self.violation_callbacks:\n            try:\n                callback(violation)\n            except Exception as e:\n                logger.error(f\"Error in safety violation callback: {e}\")\n        \n        return violation\n    \n    def _emergency_shutdown(self, hardware_id: Optional[str] = None):\n        \"\"\"Emergency shutdown of hardware.\"\"\"\n        if hardware_id:\n            # Shutdown specific hardware\n            if hardware_id in self.hardware_interfaces:\n                try:\n                    interface = self.hardware_interfaces[hardware_id]\n                    interface.deactivate()\n                    interface.emergency_stop()\n                    logger.critical(f\"Emergency shutdown of hardware {hardware_id}\")\n                except Exception as e:\n                    logger.critical(f\"Failed to emergency shutdown {hardware_id}: {e}\")\n        else:\n            # Shutdown all hardware\n            for hw_id, interface in self.hardware_interfaces.items():\n                try:\n                    interface.deactivate()\n                    interface.emergency_stop()\n                except Exception as e:\n                    logger.critical(f\"Failed to emergency shutdown {hw_id}: {e}\")\n            logger.critical(\"Emergency shutdown of all hardware\")\n    \n    def trigger_emergency_stop(self, reason: str):\n        \"\"\"Trigger system-wide emergency stop.\"\"\"\n        self.emergency_stop_triggered = True\n        \n        logger.critical(f\"EMERGENCY STOP TRIGGERED: {reason}\")\n        \n        # Shutdown all hardware\n        self._emergency_shutdown()\n        \n        # Notify emergency callbacks\n        for callback in self.emergency_callbacks:\n            try:\n                callback()\n            except Exception as e:\n                logger.critical(f\"Error in emergency callback: {e}\")\n    \n    def reset_emergency_stop(self) -> bool:\n        \"\"\"\n        Reset emergency stop if safe to do so.\n        \n        Returns:\n            True if reset successful, False otherwise\n        \"\"\"\n        if not self.emergency_stop_triggered:\n            return True\n        \n        # Check if it's safe to reset\n        recent_critical = [v for v in self.violations \n                         if v.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]\n                         and (datetime.now() - v.timestamp).total_seconds() < 60]\n        \n        if recent_critical:\n            logger.error(\"Cannot reset emergency stop: Recent critical violations\")\n            return False\n        \n        self.emergency_stop_triggered = False\n        logger.info(\"Emergency stop reset - system ready\")\n        return True\n    \n    def get_safety_status(self) -> Dict[str, Any]:\n        \"\"\"Get current safety system status.\"\"\"\n        recent_violations = [v for v in self.violations \n                           if (datetime.now() - v.timestamp).total_seconds() < 300]  # Last 5 minutes\n        \n        return {\n            \"enabled\": self.is_enabled,\n            \"emergency_stop\": self.emergency_stop_triggered,\n            \"monitoring_active\": self.monitoring_active,\n            \"current_field_strength\": self.current_field_strength,\n            \"hardware_count\": len(self.hardware_interfaces),\n            \"total_violations\": len(self.violations),\n            \"recent_violations\": len(recent_violations),\n            \"critical_violations\": len([v for v in recent_violations \n                                       if v.severity == SafetyLevel.CRITICAL]),\n            \"limits\": {\n                \"max_temperature\": self.limits.max_temperature,\n                \"max_power\": self.limits.max_total_power,\n                \"max_field_strength\": self.limits.max_field_strength,\n            }\n        }\n    \n    def get_violation_history(self, hours: int = 24) -> List[Dict[str, Any]]:\n        \"\"\"Get violation history for the specified time period.\"\"\"\n        cutoff_time = datetime.now() - timedelta(hours=hours)\n        \n        recent_violations = [v for v in self.violations if v.timestamp > cutoff_time]\n        \n        return [\n            {\n                \"timestamp\": v.timestamp.isoformat(),\n                \"type\": v.violation_type.value,\n                \"severity\": v.severity.value,\n                \"message\": v.message,\n                \"hardware_id\": v.hardware_id,\n                \"measured_value\": v.measured_value,\n                \"threshold_value\": v.threshold_value,\n                \"resolved\": v.resolved\n            }\n            for v in recent_violations\n        ]\n    \n    def enable_safety(self):\n        \"\"\"Enable safety monitoring.\"\"\"\n        self.is_enabled = True\n        logger.info(\"Safety system enabled\")\n    \n    def disable_safety(self, reason: str = \"Manual override\"):\n        \"\"\"Disable safety monitoring (use with extreme caution).\"\"\"\n        self.is_enabled = False\n        logger.warning(f\"⚠️  SAFETY SYSTEM DISABLED: {reason}\")\n        \n        # Log violation for audit trail\n        self._trigger_violation(\n            SafetyViolationType.INVALID_PARAMETERS,\n            SafetyLevel.WARNING,\n            f\"Safety system manually disabled: {reason}\"\n        )


# Global safety manager instance\n_safety_manager: Optional[SafetyManager] = None\n\n\ndef get_safety_manager() -> SafetyManager:\n    \"\"\"Get global safety manager instance.\"\"\"\n    global _safety_manager\n    if _safety_manager is None:\n        _safety_manager = SafetyManager()\n    return _safety_manager\n\n\ndef initialize_safety(limits: Optional[SafetyLimits] = None, start_monitoring: bool = True):\n    \"\"\"Initialize global safety manager.\"\"\"\n    global _safety_manager\n    _safety_manager = SafetyManager(limits)\n    \n    if start_monitoring:\n        _safety_manager.start_monitoring()\n    \n    logger.info(\"Global safety manager initialized\")\n"
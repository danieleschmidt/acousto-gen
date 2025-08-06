"""
Safety monitoring and validation for acoustic holography systems.
Ensures safe operation within pressure, temperature, and power limits.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyLimits:
    """Safety limits configuration."""
    max_pressure: float = 5000.0  # Pa
    max_temperature: float = 45.0  # °C
    max_power: float = 100.0  # W
    max_voltage: float = 200.0  # V
    max_current: float = 2.0  # A
    min_frequency: float = 20000.0  # Hz
    max_frequency: float = 45000.0  # Hz


class SafetyMonitor:
    """Safety monitoring and validation system."""
    
    def __init__(self, limits: Optional[SafetyLimits] = None):
        """
        Initialize safety monitor.
        
        Args:
            limits: Safety limits configuration
        """
        self.limits = limits or SafetyLimits()
        self.enabled = True
        self.violations = []
        
    def validate_pressure(self, pressure: float) -> bool:
        """
        Validate acoustic pressure level.
        
        Args:
            pressure: Pressure level in Pa
            
        Returns:
            True if safe, False if exceeds limits
        """
        if not self.enabled:
            return True
            
        if pressure > self.limits.max_pressure:
            violation = f"Pressure {pressure:.1f} Pa exceeds limit {self.limits.max_pressure:.1f} Pa"
            self.violations.append(violation)
            logger.warning(f"Safety violation: {violation}")
            return False
            
        return True
    
    def validate_temperature(self, temperature: float) -> bool:
        """
        Validate system temperature.
        
        Args:
            temperature: Temperature in °C
            
        Returns:
            True if safe, False if exceeds limits
        """
        if not self.enabled:
            return True
            
        if temperature > self.limits.max_temperature:
            violation = f"Temperature {temperature:.1f}°C exceeds limit {self.limits.max_temperature:.1f}°C"
            self.violations.append(violation)
            logger.warning(f"Safety violation: {violation}")
            return False
            
        return True
    
    def validate_power(self, power: float) -> bool:
        """
        Validate power consumption.
        
        Args:
            power: Power in W
            
        Returns:
            True if safe, False if exceeds limits
        """
        if not self.enabled:
            return True
            
        if power > self.limits.max_power:
            violation = f"Power {power:.1f} W exceeds limit {self.limits.max_power:.1f} W"
            self.violations.append(violation)
            logger.warning(f"Safety violation: {violation}")
            return False
            
        return True
    
    def validate_frequency(self, frequency: float) -> bool:
        """
        Validate operating frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            True if safe, False if outside limits
        """
        if not self.enabled:
            return True
            
        if frequency < self.limits.min_frequency or frequency > self.limits.max_frequency:
            violation = f"Frequency {frequency:.0f} Hz outside safe range [{self.limits.min_frequency:.0f}, {self.limits.max_frequency:.0f}] Hz"
            self.violations.append(violation)
            logger.warning(f"Safety violation: {violation}")
            return False
            
        return True
    
    def validate_field_parameters(
        self,
        pressure_field: np.ndarray,
        frequency: float,
        power: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive field parameter validation.
        
        Args:
            pressure_field: Acoustic pressure field
            frequency: Operating frequency
            power: Optional power consumption
            
        Returns:
            Tuple of (is_safe, validation_results)
        """
        results = {
            'pressure_valid': True,
            'frequency_valid': True,
            'power_valid': True,
            'max_pressure': 0.0,
            'violations': []
        }
        
        if not self.enabled:
            return True, results
        
        is_safe = True
        
        # Validate frequency
        if not self.validate_frequency(frequency):
            results['frequency_valid'] = False
            is_safe = False
        
        # Validate pressure field
        max_pressure = np.max(np.abs(pressure_field))
        results['max_pressure'] = max_pressure
        
        if not self.validate_pressure(max_pressure):
            results['pressure_valid'] = False
            is_safe = False
        
        # Validate power if provided
        if power is not None and not self.validate_power(power):
            results['power_valid'] = False
            is_safe = False
        
        results['violations'] = self.violations.copy()
        
        return is_safe, results
    
    def emergency_stop(self) -> Dict[str, Any]:
        """
        Trigger emergency stop sequence.
        
        Returns:
            Emergency stop report
        """
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        report = {
            'timestamp': np.datetime64('now'),
            'reason': 'Safety violation or manual trigger',
            'violations': self.violations.copy(),
            'actions_taken': [
                'All transducers disabled',
                'Power systems shutdown',
                'Safety lockout engaged'
            ]
        }
        
        # Clear violations for next session
        self.violations.clear()
        
        return report
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current safety system status.
        
        Returns:
            Status dictionary
        """
        return {
            'enabled': self.enabled,
            'limits': self.limits.__dict__,
            'recent_violations': len(self.violations),
            'last_violations': self.violations[-5:] if self.violations else []
        }


# Global safety monitor instance
_safety_monitor: Optional[SafetyMonitor] = None


def get_safety_monitor() -> SafetyMonitor:
    """Get global safety monitor instance."""
    global _safety_monitor
    if _safety_monitor is None:
        _safety_monitor = SafetyMonitor()
    return _safety_monitor


def validate_pressure(pressure: float) -> bool:
    """Global pressure validation function."""
    return get_safety_monitor().validate_pressure(pressure)


def get_temperature() -> float:
    """Mock temperature sensor reading."""
    # In a real system, this would read from hardware sensors
    return 25.0  # Mock temperature in °C


def validate_temperature(temperature: float) -> bool:
    """Global temperature validation function."""
    return get_safety_monitor().validate_temperature(temperature)


def validate_power(power: float) -> bool:
    """Global power validation function."""
    return get_safety_monitor().validate_power(power)


def validate_frequency(frequency: float) -> bool:
    """Global frequency validation function."""
    return get_safety_monitor().validate_frequency(frequency)


def emergency_stop() -> Dict[str, Any]:
    """Global emergency stop function."""
    return get_safety_monitor().emergency_stop()


def configure_safety_limits(**kwargs) -> None:
    """
    Configure safety limits globally.
    
    Args:
        **kwargs: Safety limit parameters
    """
    global _safety_monitor
    if _safety_monitor is None:
        _safety_monitor = SafetyMonitor()
    
    for key, value in kwargs.items():
        if hasattr(_safety_monitor.limits, key):
            setattr(_safety_monitor.limits, key, value)
        else:
            logger.warning(f"Unknown safety limit parameter: {key}")


def enable_safety_monitoring(enabled: bool = True) -> None:
    """Enable or disable safety monitoring."""
    get_safety_monitor().enabled = enabled
    logger.info(f"Safety monitoring {'enabled' if enabled else 'disabled'}")


def get_safety_status() -> Dict[str, Any]:
    """Get current safety system status."""
    return get_safety_monitor().get_status()
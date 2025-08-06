"""
Comprehensive input validation and sanitization for acoustic holography systems.
Provides validation for acoustic parameters, field configurations, and API inputs.
"""

import numpy as np
import re
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import math


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ValidationType(Enum):
    """Types of validation checks."""
    REQUIRED = "required"
    TYPE = "type"
    RANGE = "range"
    FORMAT = "format"
    LENGTH = "length"
    PATTERN = "pattern"
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """Validation rule specification."""
    field: str
    validation_type: ValidationType
    parameters: Dict[str, Any]
    message: str
    required: bool = True


class AcousticParameterValidator:
    """Validator for acoustic holography parameters."""
    
    def __init__(self):
        """Initialize with physical constants and limits."""
        # Physical limits based on safety standards and physics
        self.LIMITS = {
            # Frequency limits (Hz)
            'frequency_min': 1000,      # 1 kHz minimum
            'frequency_max': 10000000,  # 10 MHz maximum
            
            # Pressure limits (Pa)
            'pressure_min': 0,
            'pressure_max': 50000,      # 50 kPa safety limit
            
            # Intensity limits (W/cm²)
            'intensity_min': 0,
            'intensity_max': 100,       # 100 W/cm² safety limit
            
            # Temperature limits (°C)
            'temperature_min': -50,
            'temperature_max': 100,
            
            # Position limits (m)
            'position_min': -1.0,       # 1m workspace
            'position_max': 1.0,
            
            # Phase limits (radians)
            'phase_min': -np.pi,
            'phase_max': np.pi,
            
            # Amplitude limits (normalized)
            'amplitude_min': 0,
            'amplitude_max': 1,
            
            # Array size limits
            'array_size_min': 1,
            'array_size_max': 1000,
            
            # Field grid limits
            'grid_size_min': 1,
            'grid_size_max': 512,       # Memory considerations
            
            # Time limits (seconds)
            'duration_min': 0,
            'duration_max': 3600,       # 1 hour maximum
            
            # Power limits (W)
            'power_min': 0,
            'power_max': 1000,          # 1 kW safety limit
        }
    
    def validate_frequency(self, frequency: float) -> float:
        """
        Validate and sanitize frequency parameter.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Validated frequency
            
        Raises:
            ValidationError: If frequency is invalid
        """
        if not isinstance(frequency, (int, float)):
            raise ValidationError("Frequency must be a number")
        
        if not math.isfinite(frequency):
            raise ValidationError("Frequency must be finite")
        
        if frequency <= 0:
            raise ValidationError("Frequency must be positive")
        
        if frequency < self.LIMITS['frequency_min']:
            raise ValidationError(f"Frequency {frequency} Hz is below minimum {self.LIMITS['frequency_min']} Hz")
        
        if frequency > self.LIMITS['frequency_max']:
            raise ValidationError(f"Frequency {frequency} Hz exceeds maximum {self.LIMITS['frequency_max']} Hz")
        
        return float(frequency)
    
    def validate_pressure(self, pressure: float) -> float:
        """
        Validate pressure parameter.
        
        Args:
            pressure: Pressure in Pa
            
        Returns:
            Validated pressure
        """
        if not isinstance(pressure, (int, float)):
            raise ValidationError("Pressure must be a number")
        
        if not math.isfinite(pressure):
            raise ValidationError("Pressure must be finite")
        
        if pressure < self.LIMITS['pressure_min']:
            raise ValidationError(f"Pressure {pressure} Pa is below minimum {self.LIMITS['pressure_min']} Pa")
        
        if pressure > self.LIMITS['pressure_max']:
            raise ValidationError(f"Pressure {pressure} Pa exceeds maximum {self.LIMITS['pressure_max']} Pa")
        
        return float(pressure)
    
    def validate_position(self, position: Union[List, Tuple, np.ndarray]) -> np.ndarray:
        """
        Validate 3D position coordinates.
        
        Args:
            position: Position coordinates [x, y, z]
            
        Returns:
            Validated position as numpy array
        """
        if not isinstance(position, (list, tuple, np.ndarray)):
            raise ValidationError("Position must be a list, tuple, or array")
        
        position = np.array(position, dtype=float)
        
        if position.shape != (3,):
            raise ValidationError("Position must have exactly 3 coordinates [x, y, z]")
        
        if not np.all(np.isfinite(position)):
            raise ValidationError("All position coordinates must be finite")
        
        for i, coord in enumerate(['x', 'y', 'z']):
            if position[i] < self.LIMITS['position_min']:
                raise ValidationError(f"{coord} coordinate {position[i]} is below minimum {self.LIMITS['position_min']}")
            if position[i] > self.LIMITS['position_max']:
                raise ValidationError(f"{coord} coordinate {position[i]} exceeds maximum {self.LIMITS['position_max']}")
        
        return position
    
    def validate_phase_array(self, phases: Union[List, np.ndarray]) -> np.ndarray:
        """
        Validate array of phase values.
        
        Args:
            phases: Array of phase values in radians
            
        Returns:
            Validated phase array
        """
        if not isinstance(phases, (list, np.ndarray)):
            raise ValidationError("Phases must be a list or array")
        
        phases = np.array(phases, dtype=float)
        
        if phases.ndim != 1:
            raise ValidationError("Phase array must be 1-dimensional")
        
        if len(phases) < self.LIMITS['array_size_min']:
            raise ValidationError(f"Phase array too small: {len(phases)} < {self.LIMITS['array_size_min']}")
        
        if len(phases) > self.LIMITS['array_size_max']:
            raise ValidationError(f"Phase array too large: {len(phases)} > {self.LIMITS['array_size_max']}")
        
        if not np.all(np.isfinite(phases)):
            raise ValidationError("All phase values must be finite")
        
        # Normalize phases to [-π, π] range
        phases = np.mod(phases + np.pi, 2 * np.pi) - np.pi
        
        return phases
    
    def validate_amplitude_array(self, amplitudes: Union[List, np.ndarray]) -> np.ndarray:
        """
        Validate array of amplitude values.
        
        Args:
            amplitudes: Array of amplitude values [0, 1]
            
        Returns:
            Validated amplitude array
        """
        if not isinstance(amplitudes, (list, np.ndarray)):
            raise ValidationError("Amplitudes must be a list or array")
        
        amplitudes = np.array(amplitudes, dtype=float)
        
        if amplitudes.ndim != 1:
            raise ValidationError("Amplitude array must be 1-dimensional")
        
        if not np.all(np.isfinite(amplitudes)):
            raise ValidationError("All amplitude values must be finite")
        
        if np.any(amplitudes < self.LIMITS['amplitude_min']):
            raise ValidationError(f"Amplitude values must be >= {self.LIMITS['amplitude_min']}")
        
        if np.any(amplitudes > self.LIMITS['amplitude_max']):
            raise ValidationError(f"Amplitude values must be <= {self.LIMITS['amplitude_max']}")
        
        return amplitudes
    
    def validate_field_bounds(self, bounds: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Validate field computation bounds.
        
        Args:
            bounds: List of (min, max) tuples for each dimension
            
        Returns:
            Validated bounds
        """
        if not isinstance(bounds, list):
            raise ValidationError("Bounds must be a list")
        
        if len(bounds) != 3:
            raise ValidationError("Bounds must specify 3 dimensions")
        
        validated_bounds = []
        
        for i, bound in enumerate(bounds):
            if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                raise ValidationError(f"Bound {i} must be a tuple/list of (min, max)")
            
            min_val, max_val = float(bound[0]), float(bound[1])
            
            if not (math.isfinite(min_val) and math.isfinite(max_val)):
                raise ValidationError(f"Bound {i} values must be finite")
            
            if min_val >= max_val:
                raise ValidationError(f"Bound {i} minimum {min_val} must be less than maximum {max_val}")
            
            dimension_size = max_val - min_val
            if dimension_size < 0.001:  # 1mm minimum
                raise ValidationError(f"Bound {i} dimension {dimension_size} m is too small")
            
            if dimension_size > 2.0:  # 2m maximum
                raise ValidationError(f"Bound {i} dimension {dimension_size} m is too large")
            
            validated_bounds.append((min_val, max_val))
        
        return validated_bounds
    
    def validate_resolution(self, resolution: float) -> float:
        """
        Validate spatial resolution parameter.
        
        Args:
            resolution: Spatial resolution in meters
            
        Returns:
            Validated resolution
        """
        if not isinstance(resolution, (int, float)):
            raise ValidationError("Resolution must be a number")
        
        if not math.isfinite(resolution):
            raise ValidationError("Resolution must be finite")
        
        if resolution <= 0:
            raise ValidationError("Resolution must be positive")
        
        if resolution < 1e-5:  # 10 micrometers minimum
            raise ValidationError(f"Resolution {resolution} m is too fine (minimum 10 µm)")
        
        if resolution > 0.01:  # 1 cm maximum
            raise ValidationError(f"Resolution {resolution} m is too coarse (maximum 1 cm)")
        
        return float(resolution)
    
    def validate_focal_points(self, focal_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate focal point specifications.
        
        Args:
            focal_points: List of focal point dictionaries
            
        Returns:
            Validated focal points
        """
        if not isinstance(focal_points, list):
            raise ValidationError("Focal points must be a list")
        
        if len(focal_points) == 0:
            raise ValidationError("At least one focal point must be specified")
        
        if len(focal_points) > 10:  # Practical limit
            raise ValidationError("Too many focal points (maximum 10)")
        
        validated_points = []
        
        for i, point in enumerate(focal_points):
            if not isinstance(point, dict):
                raise ValidationError(f"Focal point {i} must be a dictionary")
            
            # Validate required fields
            if 'position' not in point:
                raise ValidationError(f"Focal point {i} missing 'position' field")
            
            validated_point = {}
            
            # Validate position
            validated_point['position'] = self.validate_position(point['position']).tolist()
            
            # Validate pressure (optional)
            if 'pressure' in point:
                validated_point['pressure'] = self.validate_pressure(point['pressure'])
            else:
                validated_point['pressure'] = 3000  # Default pressure
            
            # Validate width (optional)
            if 'width' in point:
                width = float(point['width'])
                if width <= 0 or width > 0.1:  # 10cm maximum width
                    raise ValidationError(f"Focal point {i} width {width} m is invalid")
                validated_point['width'] = width
            else:
                validated_point['width'] = 0.005  # Default 5mm width
            
            validated_points.append(validated_point)
        
        return validated_points


class APIInputValidator:
    """Validator for API request inputs."""
    
    def __init__(self):
        """Initialize API validator."""
        self.acoustic_validator = AcousticParameterValidator()
        
        # Common validation rules
        self.common_rules = {
            'optimization_request': [
                ValidationRule(
                    field='target_type',
                    validation_type=ValidationType.PATTERN,
                    parameters={'pattern': r'^[a-z_]+$'},
                    message='Target type must contain only lowercase letters and underscores'
                ),
                ValidationRule(
                    field='iterations',
                    validation_type=ValidationType.RANGE,
                    parameters={'min': 1, 'max': 10000},
                    message='Iterations must be between 1 and 10000'
                ),
                ValidationRule(
                    field='learning_rate',
                    validation_type=ValidationType.RANGE,
                    parameters={'min': 1e-6, 'max': 1.0},
                    message='Learning rate must be between 1e-6 and 1.0'
                )
            ],
            'hardware_command': [
                ValidationRule(
                    field='command',
                    validation_type=ValidationType.PATTERN,
                    parameters={'pattern': r'^[a-z_]+$'},
                    message='Command must contain only lowercase letters and underscores'
                )
            ]
        }
        
        # Blacklisted strings for security
        self.sql_injection_patterns = [
            r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bUPDATE\b|\bDROP\b)',
            r'(--|\/\*|\*\/)',
            r'(\bOR\b.*=.*|AND.*=.*)',
        ]
        
        self.script_injection_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
        ]
    
    def validate_request_data(
        self,
        data: Dict[str, Any],
        request_type: str
    ) -> Dict[str, Any]:
        """
        Validate API request data.
        
        Args:
            data: Request data dictionary
            request_type: Type of request for validation rules
            
        Returns:
            Validated and sanitized data
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError("Request data must be a dictionary")
        
        # Security checks
        self._check_security_threats(data)
        
        # Apply common validation rules
        if request_type in self.common_rules:
            for rule in self.common_rules[request_type]:
                self._apply_validation_rule(data, rule)
        
        # Request-specific validation
        if request_type == 'optimization_request':
            return self._validate_optimization_request(data)
        elif request_type == 'field_calculation':
            return self._validate_field_calculation_request(data)
        elif request_type == 'hardware_command':
            return self._validate_hardware_command(data)
        elif request_type == 'particle_request':
            return self._validate_particle_request(data)
        else:
            # Generic validation for unknown request types
            return self._sanitize_strings(data)
    
    def _validate_optimization_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimization request data."""
        validated_data = {}
        
        # Required fields
        required_fields = ['target_type']
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Target type
        target_type = str(data['target_type']).strip().lower()
        allowed_types = ['single_focus', 'multi_focus', 'twin_trap', 'line_trap', 'custom']
        if target_type not in allowed_types:
            raise ValidationError(f"Invalid target type. Must be one of: {allowed_types}")
        validated_data['target_type'] = target_type
        
        # Target position (optional)
        if 'target_position' in data and data['target_position'] is not None:
            validated_data['target_position'] = self.acoustic_validator.validate_position(
                data['target_position']
            ).tolist()
        
        # Target pressure
        if 'target_pressure' in data:
            validated_data['target_pressure'] = self.acoustic_validator.validate_pressure(
                float(data['target_pressure'])
            )
        else:
            validated_data['target_pressure'] = 3000  # Default
        
        # Focal points
        if 'focal_points' in data:
            validated_data['focal_points'] = self.acoustic_validator.validate_focal_points(
                data['focal_points']
            )
        else:
            validated_data['focal_points'] = []
        
        # Optimization parameters
        if 'iterations' in data:
            iterations = int(data['iterations'])
            if iterations < 1 or iterations > 10000:
                raise ValidationError("Iterations must be between 1 and 10000")
            validated_data['iterations'] = iterations
        else:
            validated_data['iterations'] = 1000
        
        if 'method' in data:
            method = str(data['method']).strip().lower()
            allowed_methods = ['adam', 'sgd', 'lbfgs', 'genetic', 'neural']
            if method not in allowed_methods:
                raise ValidationError(f"Invalid method. Must be one of: {allowed_methods}")
            validated_data['method'] = method
        else:
            validated_data['method'] = 'adam'
        
        if 'learning_rate' in data:
            lr = float(data['learning_rate'])
            if lr <= 0 or lr > 1.0:
                raise ValidationError("Learning rate must be between 0 and 1.0")
            validated_data['learning_rate'] = lr
        else:
            validated_data['learning_rate'] = 0.01
        
        if 'convergence_threshold' in data:
            threshold = float(data['convergence_threshold'])
            if threshold <= 0 or threshold > 1.0:
                raise ValidationError("Convergence threshold must be between 0 and 1.0")
            validated_data['convergence_threshold'] = threshold
        else:
            validated_data['convergence_threshold'] = 1e-6
        
        return validated_data
    
    def _validate_field_calculation_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate field calculation request."""
        validated_data = {}
        
        # Required fields
        required_fields = ['phases', 'amplitudes']
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate phases
        validated_data['phases'] = self.acoustic_validator.validate_phase_array(
            data['phases']
        ).tolist()
        
        # Validate amplitudes
        validated_data['amplitudes'] = self.acoustic_validator.validate_amplitude_array(
            data['amplitudes']
        ).tolist()
        
        # Check array lengths match
        if len(validated_data['phases']) != len(validated_data['amplitudes']):
            raise ValidationError("Phases and amplitudes arrays must have same length")
        
        # Field bounds
        if 'field_bounds' in data:
            # Convert to proper format
            bounds_dict = data['field_bounds']
            bounds_list = [
                (bounds_dict.get('x_min', -0.05), bounds_dict.get('x_max', 0.05)),
                (bounds_dict.get('y_min', -0.05), bounds_dict.get('y_max', 0.05)),
                (bounds_dict.get('z_min', 0.05), bounds_dict.get('z_max', 0.15))
            ]
            validated_bounds = self.acoustic_validator.validate_field_bounds(bounds_list)
            validated_data['field_bounds'] = {
                'x_min': validated_bounds[0][0],
                'x_max': validated_bounds[0][1],
                'y_min': validated_bounds[1][0],
                'y_max': validated_bounds[1][1],
                'z_min': validated_bounds[2][0],
                'z_max': validated_bounds[2][1]
            }
        
        # Resolution
        if 'resolution' in data:
            validated_data['resolution'] = self.acoustic_validator.validate_resolution(
                float(data['resolution'])
            )
        
        # Frequency
        if 'frequency' in data:
            validated_data['frequency'] = self.acoustic_validator.validate_frequency(
                float(data['frequency'])
            )
        else:
            validated_data['frequency'] = 40000  # Default
        
        # Medium
        if 'medium' in data:
            medium = str(data['medium']).strip().lower()
            allowed_media = ['air', 'water', 'tissue']
            if medium not in allowed_media:
                raise ValidationError(f"Invalid medium. Must be one of: {allowed_media}")
            validated_data['medium'] = medium
        else:
            validated_data['medium'] = 'air'
        
        return validated_data
    
    def _validate_hardware_command(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hardware command."""
        validated_data = {}
        
        # Required fields
        if 'command' not in data:
            raise ValidationError("Missing required field: command")
        
        # Command validation
        command = str(data['command']).strip().lower()
        allowed_commands = [
            'activate', 'deactivate', 'emergency_stop',
            'set_phases', 'set_amplitudes', 'calibrate',
            'get_status', 'reset'
        ]
        
        if command not in allowed_commands:
            raise ValidationError(f"Invalid command. Must be one of: {allowed_commands}")
        
        validated_data['command'] = command
        
        # Parameters (optional)
        if 'parameters' in data and data['parameters'] is not None:
            params = data['parameters']
            if not isinstance(params, dict):
                raise ValidationError("Parameters must be a dictionary")
            
            # Validate parameter values based on command
            if command in ['set_phases', 'set_amplitudes']:
                if command == 'set_phases' and 'phases' in params:
                    params['phases'] = self.acoustic_validator.validate_phase_array(
                        params['phases']
                    ).tolist()
                elif command == 'set_amplitudes' and 'amplitudes' in params:
                    params['amplitudes'] = self.acoustic_validator.validate_amplitude_array(
                        params['amplitudes']
                    ).tolist()
            
            validated_data['parameters'] = self._sanitize_strings(params)
        
        return validated_data
    
    def _validate_particle_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate particle manipulation request."""
        validated_data = {}
        
        # Required fields
        required_fields = ['position']
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
        
        # Position
        validated_data['position'] = self.acoustic_validator.validate_position(
            data['position']
        ).tolist()
        
        # Radius
        if 'radius' in data:
            radius = float(data['radius'])
            if radius <= 0 or radius > 0.01:  # 1cm maximum
                raise ValidationError("Particle radius must be between 0 and 0.01 m")
            validated_data['radius'] = radius
        else:
            validated_data['radius'] = 1e-3  # Default 1mm
        
        # Density
        if 'density' in data:
            density = float(data['density'])
            if density <= 0 or density > 10000:  # 10 g/cm³ maximum
                raise ValidationError("Particle density must be between 0 and 10000 kg/m³")
            validated_data['density'] = density
        else:
            validated_data['density'] = 25  # Default polystyrene
        
        return validated_data
    
    def _apply_validation_rule(self, data: Dict[str, Any], rule: ValidationRule) -> None:
        """Apply individual validation rule."""
        field_value = data.get(rule.field)
        
        # Check if field is required
        if rule.required and field_value is None:
            raise ValidationError(f"Required field '{rule.field}' is missing")
        
        if field_value is None:
            return  # Skip validation for optional missing fields
        
        # Apply validation based on type
        if rule.validation_type == ValidationType.RANGE:
            value = float(field_value)
            min_val = rule.parameters.get('min')
            max_val = rule.parameters.get('max')
            
            if min_val is not None and value < min_val:
                raise ValidationError(f"{rule.field}: {rule.message}")
            if max_val is not None and value > max_val:
                raise ValidationError(f"{rule.field}: {rule.message}")
        
        elif rule.validation_type == ValidationType.PATTERN:
            pattern = rule.parameters.get('pattern')
            if pattern and not re.match(pattern, str(field_value)):
                raise ValidationError(f"{rule.field}: {rule.message}")
        
        elif rule.validation_type == ValidationType.LENGTH:
            length = len(str(field_value))
            min_len = rule.parameters.get('min', 0)
            max_len = rule.parameters.get('max', float('inf'))
            
            if length < min_len or length > max_len:
                raise ValidationError(f"{rule.field}: {rule.message}")
    
    def _check_security_threats(self, data: Any) -> None:
        """Check for security threats in input data."""
        if isinstance(data, dict):
            for key, value in data.items():
                self._check_security_threats(value)
        elif isinstance(data, list):
            for item in data:
                self._check_security_threats(item)
        elif isinstance(data, str):
            # Check for SQL injection patterns
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    raise ValidationError("Potentially malicious input detected")
            
            # Check for script injection patterns
            for pattern in self.script_injection_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    raise ValidationError("Potentially malicious input detected")
    
    def _sanitize_strings(self, data: Any) -> Any:
        """Recursively sanitize string data."""
        if isinstance(data, dict):
            return {key: self._sanitize_strings(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_strings(item) for item in data]
        elif isinstance(data, str):
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[<>&"\'`]', '', data)
            return sanitized.strip()
        else:
            return data


def create_validation_schema(schema_type: str) -> Dict[str, Any]:
    """
    Create validation schema for common request types.
    
    Args:
        schema_type: Type of schema to create
        
    Returns:
        Validation schema dictionary
    """
    schemas = {
        'optimization_request': {
            'type': 'object',
            'required': ['target_type'],
            'properties': {
                'target_type': {
                    'type': 'string',
                    'enum': ['single_focus', 'multi_focus', 'twin_trap', 'line_trap', 'custom']
                },
                'target_position': {
                    'type': 'array',
                    'items': {'type': 'number'},
                    'minItems': 3,
                    'maxItems': 3
                },
                'target_pressure': {
                    'type': 'number',
                    'minimum': 0,
                    'maximum': 50000
                },
                'focal_points': {
                    'type': 'array',
                    'maxItems': 10,
                    'items': {
                        'type': 'object',
                        'required': ['position'],
                        'properties': {
                            'position': {
                                'type': 'array',
                                'items': {'type': 'number'},
                                'minItems': 3,
                                'maxItems': 3
                            },
                            'pressure': {
                                'type': 'number',
                                'minimum': 0,
                                'maximum': 50000
                            },
                            'width': {
                                'type': 'number',
                                'minimum': 0.001,
                                'maximum': 0.1
                            }
                        }
                    }
                },
                'iterations': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 10000
                },
                'method': {
                    'type': 'string',
                    'enum': ['adam', 'sgd', 'lbfgs', 'genetic', 'neural']
                },
                'learning_rate': {
                    'type': 'number',
                    'minimum': 1e-6,
                    'maximum': 1.0
                }
            }
        },
        'field_calculation': {
            'type': 'object',
            'required': ['phases', 'amplitudes'],
            'properties': {
                'phases': {
                    'type': 'array',
                    'minItems': 1,
                    'maxItems': 1000,
                    'items': {
                        'type': 'number',
                        'minimum': -np.pi,
                        'maximum': np.pi
                    }
                },
                'amplitudes': {
                    'type': 'array',
                    'minItems': 1,
                    'maxItems': 1000,
                    'items': {
                        'type': 'number',
                        'minimum': 0,
                        'maximum': 1
                    }
                },
                'frequency': {
                    'type': 'number',
                    'minimum': 1000,
                    'maximum': 10000000
                },
                'medium': {
                    'type': 'string',
                    'enum': ['air', 'water', 'tissue']
                }
            }
        },
        'hardware_command': {
            'type': 'object',
            'required': ['command'],
            'properties': {
                'command': {
                    'type': 'string',
                    'enum': ['activate', 'deactivate', 'emergency_stop', 
                            'set_phases', 'set_amplitudes', 'calibrate']
                },
                'parameters': {
                    'type': 'object'
                }
            }
        }
    }
    
    return schemas.get(schema_type, {})
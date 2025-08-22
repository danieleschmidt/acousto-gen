"""
Comprehensive validation framework for Acousto-Gen Generation 2.
Implements robust input validation, safety checks, and error handling.
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation process."""
    is_valid: bool
    issues: List[ValidationIssue]
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues of specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]


class SafetyValidator:
    """Validates safety constraints for acoustic operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize safety validator with configuration."""
        self.config = config or self._get_default_safety_config()
    
    def _get_default_safety_config(self) -> Dict[str, Any]:
        """Get default safety configuration."""
        return {
            "max_pressure_pa": 5000.0,  # 5 kPa max pressure
            "max_intensity_w_cm2": 15.0,  # 15 W/cm² max intensity
            "max_temperature_c": 45.0,  # 45°C max temperature
            "max_exposure_time_s": 300.0,  # 5 minutes max exposure
            "min_frequency_hz": 18000.0,  # 18 kHz minimum
            "max_frequency_hz": 100000.0,  # 100 kHz maximum
            "max_array_power_w": 50.0,  # 50W max total power
            "biomedical_pressure_limit_pa": 1000.0,  # Stricter limit for biomedical
            "safety_margin": 0.8  # 20% safety margin
        }
    
    def validate_pressure_field(self, field: np.ndarray, frequency: float) -> ValidationResult:
        """Validate acoustic pressure field for safety."""
        issues = []
        
        # Convert to amplitude field
        if np.iscomplexobj(field):
            amplitude_field = np.abs(field)
        else:
            amplitude_field = np.abs(field)
        
        max_pressure = np.max(amplitude_field)
        mean_pressure = np.mean(amplitude_field)
        
        # Check maximum pressure
        pressure_limit = self.config["max_pressure_pa"] * self.config["safety_margin"]
        if max_pressure > pressure_limit:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Maximum pressure {max_pressure:.1f} Pa exceeds safety limit {pressure_limit:.1f} Pa",
                field="max_pressure",
                value=max_pressure,
                suggestion=f"Reduce amplitudes or modify target to stay below {pressure_limit:.1f} Pa"
            ))
        
        # Check frequency range
        if frequency < self.config["min_frequency_hz"]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Frequency {frequency:.0f} Hz below minimum {self.config['min_frequency_hz']:.0f} Hz",
                field="frequency",
                value=frequency
            ))
        
        if frequency > self.config["max_frequency_hz"]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Frequency {frequency:.0f} Hz above maximum {self.config['max_frequency_hz']:.0f} Hz",
                field="frequency", 
                value=frequency
            ))
        
        # Calculate intensity (simplified)
        c = 343  # Speed of sound in air
        rho = 1.2  # Air density
        intensity_field = amplitude_field ** 2 / (2 * rho * c) * 1e-4  # Convert to W/cm²
        max_intensity = np.max(intensity_field)
        
        intensity_limit = self.config["max_intensity_w_cm2"] * self.config["safety_margin"]
        if max_intensity > intensity_limit:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Maximum intensity {max_intensity:.2f} W/cm² exceeds limit {intensity_limit:.2f} W/cm²",
                field="max_intensity",
                value=max_intensity
            ))
        
        # Check for excessive focal points (heating risk)
        pressure_threshold = self.config["max_pressure_pa"] * 0.5
        high_pressure_points = np.sum(amplitude_field > pressure_threshold)
        total_points = amplitude_field.size
        hot_spot_ratio = high_pressure_points / total_points
        
        if hot_spot_ratio > 0.01:  # More than 1% of field is high pressure
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"High concentration of focal points ({hot_spot_ratio*100:.1f}% of field) may cause heating",
                field="hot_spot_ratio",
                value=hot_spot_ratio,
                suggestion="Consider spreading focal points or reducing intensity"
            ))
        
        return ValidationResult(
            is_valid=len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0,
            issues=issues
        )
    
    def validate_phases_and_amplitudes(self, phases: np.ndarray, amplitudes: np.ndarray) -> ValidationResult:
        """Validate phase and amplitude arrays."""
        issues = []
        
        # Check array lengths match
        if len(phases) != len(amplitudes):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Phase array length {len(phases)} != amplitude array length {len(amplitudes)}",
                field="array_lengths"
            ))
        
        # Check phase range
        if np.any(phases < -np.pi) or np.any(phases > np.pi):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Phases must be in range [-π, π]",
                field="phases",
                suggestion="Normalize phases using np.angle() or manual wrapping"
            ))
        
        # Check amplitude range
        if np.any(amplitudes < 0) or np.any(amplitudes > 1):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Amplitudes must be in range [0, 1]",
                field="amplitudes",
                value=f"Range: [{np.min(amplitudes):.3f}, {np.max(amplitudes):.3f}]"
            ))
        
        # Check for NaN/infinity
        if np.any(~np.isfinite(phases)):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message="Phases contain NaN or infinity values",
                field="phases"
            ))
        
        if np.any(~np.isfinite(amplitudes)):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message="Amplitudes contain NaN or infinity values", 
                field="amplitudes"
            ))
        
        # Calculate total power
        total_power = np.sum(amplitudes ** 2) * 0.1  # Estimate in watts
        power_limit = self.config["max_array_power_w"]
        
        if total_power > power_limit:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Estimated power {total_power:.1f}W exceeds recommended {power_limit:.1f}W",
                field="total_power",
                value=total_power,
                suggestion=f"Reduce amplitudes by factor of {np.sqrt(power_limit/total_power):.2f}"
            ))
        
        return ValidationResult(
            is_valid=not any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in issues),
            issues=issues
        )


class InputValidator:
    """Validates input parameters and data structures."""
    
    @staticmethod
    def validate_target_specification(target: Dict[str, Any]) -> ValidationResult:
        """Validate target specification dictionary."""
        issues = []
        
        # Required fields
        required_fields = ["type", "focal_points"]
        for field in required_fields:
            if field not in target:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required field: {field}",
                    field=field
                ))
        
        if "type" in target:
            valid_types = ["single_focus", "multi_focus", "shaped", "custom"]
            if target["type"] not in valid_types:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid target type: {target['type']}. Must be one of {valid_types}",
                    field="type",
                    value=target["type"]
                ))
        
        # Validate focal points
        if "focal_points" in target:
            focal_points = target["focal_points"]
            if not isinstance(focal_points, list):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="focal_points must be a list",
                    field="focal_points"
                ))
            else:
                for i, point in enumerate(focal_points):
                    if not isinstance(point, dict):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Focal point {i} must be a dictionary",
                            field=f"focal_points[{i}]"
                        ))
                        continue
                    
                    # Check required point fields
                    if "position" not in point:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Focal point {i} missing position",
                            field=f"focal_points[{i}].position"
                        ))
                    else:
                        pos = point["position"]
                        if not isinstance(pos, (list, tuple, np.ndarray)) or len(pos) != 3:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                message=f"Focal point {i} position must be 3D coordinate [x, y, z]",
                                field=f"focal_points[{i}].position"
                            ))
                        else:
                            # Check position bounds (reasonable workspace)
                            x, y, z = pos
                            if not (-0.2 <= x <= 0.2 and -0.2 <= y <= 0.2 and 0.01 <= z <= 0.3):
                                issues.append(ValidationIssue(
                                    severity=ValidationSeverity.WARNING,
                                    message=f"Focal point {i} position {pos} may be outside typical workspace",
                                    field=f"focal_points[{i}].position",
                                    suggestion="Typical workspace: x,y ∈ [-20cm, 20cm], z ∈ [1cm, 30cm]"
                                ))
                    
                    if "pressure" in point:
                        pressure = point["pressure"]
                        if not isinstance(pressure, (int, float)) or pressure <= 0:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                message=f"Focal point {i} pressure must be positive number",
                                field=f"focal_points[{i}].pressure",
                                value=pressure
                            ))
                        elif pressure > 10000:  # 10 kPa
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"Focal point {i} pressure {pressure:.0f} Pa is very high",
                                field=f"focal_points[{i}].pressure",
                                value=pressure
                            ))
        
        return ValidationResult(
            is_valid=not any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in issues),
            issues=issues
        )
    
    @staticmethod
    def validate_optimization_parameters(params: Dict[str, Any]) -> ValidationResult:
        """Validate optimization parameters."""
        issues = []
        
        # Method validation
        if "method" in params:
            valid_methods = ["adam", "sgd", "lbfgs", "genetic", "neural"]
            if params["method"] not in valid_methods:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid optimization method: {params['method']}. Must be one of {valid_methods}",
                    field="method"
                ))
        
        # Iterations
        if "iterations" in params:
            iterations = params["iterations"]
            if not isinstance(iterations, int) or iterations <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="iterations must be positive integer",
                    field="iterations",
                    value=iterations
                ))
            elif iterations > 10000:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"High iteration count {iterations} may take very long time",
                    field="iterations",
                    value=iterations
                ))
        
        # Learning rate
        if "learning_rate" in params:
            lr = params["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="learning_rate must be positive number",
                    field="learning_rate",
                    value=lr
                ))
            elif lr > 1.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Learning rate {lr} is very high, may cause instability",
                    field="learning_rate",
                    value=lr,
                    suggestion="Try learning rate between 0.001 and 0.1"
                ))
        
        return ValidationResult(
            is_valid=not any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in issues),
            issues=issues
        )


class TensorValidator:
    """Validates PyTorch tensors and arrays."""
    
    @staticmethod
    def validate_tensor_properties(tensor: Union[torch.Tensor, np.ndarray], 
                                 expected_shape: Optional[Tuple] = None,
                                 expected_dtype: Optional[torch.dtype] = None,
                                 expected_device: Optional[torch.device] = None) -> ValidationResult:
        """Validate tensor properties."""
        issues = []
        
        # Convert numpy to tensor for validation
        if isinstance(tensor, np.ndarray):
            if expected_dtype and expected_dtype != torch.float32:
                # Handle dtype conversion
                pass
            tensor = torch.from_numpy(tensor)
        
        # Check for finite values
        if not torch.all(torch.isfinite(tensor)):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message="Tensor contains NaN or infinity values",
                field="tensor_values"
            ))
        
        # Check shape
        if expected_shape and tensor.shape != expected_shape:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Tensor shape {tensor.shape} != expected {expected_shape}",
                field="tensor_shape",
                value=str(tensor.shape)
            ))
        
        # Check dtype
        if expected_dtype and tensor.dtype != expected_dtype:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Tensor dtype {tensor.dtype} != expected {expected_dtype}",
                field="tensor_dtype",
                value=str(tensor.dtype),
                suggestion="Use .to() to convert dtype"
            ))
        
        # Check device
        if expected_device and tensor.device != expected_device:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Tensor device {tensor.device} != expected {expected_device}",
                field="tensor_device",
                value=str(tensor.device),
                suggestion="Use .to() to move tensor to correct device"
            ))
        
        # Memory check for large tensors
        tensor_size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
        if tensor_size_mb > 1000:  # 1GB
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Large tensor size: {tensor_size_mb:.1f} MB",
                field="tensor_memory",
                value=tensor_size_mb,
                suggestion="Consider using smaller resolution or batch processing"
            ))
        
        return ValidationResult(
            is_valid=not any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in issues),
            issues=issues
        )


class ComprehensiveValidator:
    """Main validation orchestrator for all Acousto-Gen operations."""
    
    def __init__(self, safety_config: Optional[Dict[str, Any]] = None):
        """Initialize comprehensive validator."""
        self.safety_validator = SafetyValidator(safety_config)
        self.input_validator = InputValidator()
        self.tensor_validator = TensorValidator()
    
    def validate_hologram_request(self, request: Dict[str, Any]) -> ValidationResult:
        """Validate complete hologram generation request."""
        all_issues = []
        
        # Validate target specification
        if "target" in request:
            target_result = self.input_validator.validate_target_specification(request["target"])
            all_issues.extend(target_result.issues)
        
        # Validate optimization parameters
        if "optimization" in request:
            opt_result = self.input_validator.validate_optimization_parameters(request["optimization"])
            all_issues.extend(opt_result.issues)
        
        # Validate array configuration
        if "array_config" in request:
            # Add array-specific validation here
            pass
        
        return ValidationResult(
            is_valid=not any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in all_issues),
            issues=all_issues
        )
    
    def validate_field_generation(self, phases: np.ndarray, amplitudes: np.ndarray, 
                                frequency: float, field: Optional[np.ndarray] = None) -> ValidationResult:
        """Validate complete field generation process."""
        all_issues = []
        
        # Validate inputs
        input_result = self.safety_validator.validate_phases_and_amplitudes(phases, amplitudes)
        all_issues.extend(input_result.issues)
        
        # Validate field if provided
        if field is not None:
            field_result = self.safety_validator.validate_pressure_field(field, frequency)
            all_issues.extend(field_result.issues)
        
        return ValidationResult(
            is_valid=not any(i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for i in all_issues),
            issues=all_issues
        )
    
    def create_validation_report(self, result: ValidationResult) -> str:
        """Create human-readable validation report."""
        if not result.issues:
            return "✅ Validation passed - no issues found"
        
        report_lines = []
        report_lines.append("🔍 Validation Report:")
        report_lines.append("=" * 50)
        
        # Group by severity
        for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, 
                        ValidationSeverity.WARNING, ValidationSeverity.INFO]:
            severity_issues = result.get_issues_by_severity(severity)
            if not severity_issues:
                continue
            
            emoji = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}[severity.value]
            report_lines.append(f"\n{emoji} {severity.value.upper()} ({len(severity_issues)} issues):")
            
            for issue in severity_issues:
                report_lines.append(f"  • {issue.message}")
                if issue.field:
                    report_lines.append(f"    Field: {issue.field}")
                if issue.value is not None:
                    report_lines.append(f"    Value: {issue.value}")
                if issue.suggestion:
                    report_lines.append(f"    💡 Suggestion: {issue.suggestion}")
        
        report_lines.append("\n" + "=" * 50)
        report_lines.append(f"Overall: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
        
        return "\n".join(report_lines)
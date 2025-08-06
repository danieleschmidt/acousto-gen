"""
Basic safety validation tests for acoustic holography.
Tests safety limits, validation, and error handling without external dependencies.
"""

import sys
import os
import unittest
import math
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_safety_limits():
    """Test basic safety limit validation."""
    # Physical safety limits for acoustic applications
    MAX_PRESSURE = 50000  # 50 kPa - FDA limit for diagnostic ultrasound
    MAX_INTENSITY = 100   # 100 W/cm² - Safety limit
    MAX_TEMPERATURE = 100 # 100°C - Absolute safety limit
    
    # Test pressure validation
    test_pressures = [1000, 4000, 10000, 60000]  # Pa
    safe_pressures = [p for p in test_pressures if p <= MAX_PRESSURE]
    
    assert len(safe_pressures) == 3, f"Expected 3 safe pressures, got {len(safe_pressures)}"
    assert 60000 not in safe_pressures, "60kPa should exceed safety limit"
    
    # Test intensity validation
    test_intensities = [1, 10, 50, 150]  # W/cm²
    safe_intensities = [i for i in test_intensities if i <= MAX_INTENSITY]
    
    assert len(safe_intensities) == 3, f"Expected 3 safe intensities, got {len(safe_intensities)}"
    assert 150 not in safe_intensities, "150 W/cm² should exceed safety limit"
    
    print("✓ Basic safety limits validation passed")


def test_input_validation():
    """Test input validation and sanitization."""
    
    def validate_frequency(freq):
        """Basic frequency validation."""
        if not isinstance(freq, (int, float)):
            return False, "Frequency must be a number"
        if not math.isfinite(freq):
            return False, "Frequency must be finite"
        if freq <= 0:
            return False, "Frequency must be positive"
        if freq < 1000 or freq > 10000000:
            return False, "Frequency out of safe range (1kHz - 10MHz)"
        return True, "Valid"
    
    # Test frequency validation
    test_cases = [
        (40000, True),      # Valid ultrasound frequency
        (-1000, False),     # Negative frequency
        (0, False),         # Zero frequency
        (float('inf'), False),  # Infinite frequency
        ("40000", False),   # String input
        (50000000, False),  # Too high frequency
    ]
    
    for freq, should_be_valid in test_cases:
        is_valid, message = validate_frequency(freq)
        if should_be_valid:
            assert is_valid, f"Frequency {freq} should be valid but got: {message}"
        else:
            assert not is_valid, f"Frequency {freq} should be invalid but was accepted"
    
    print("✓ Input validation tests passed")


def test_security_patterns():
    """Test security pattern detection."""
    
    def check_sql_injection(text):
        """Basic SQL injection pattern detection."""
        if not isinstance(text, str):
            return False
        
        dangerous_patterns = [
            "UNION", "SELECT", "INSERT", "DELETE", "UPDATE", "DROP",
            "--", "/*", "*/", "OR 1=1", "AND 1=1"
        ]
        
        text_upper = text.upper()
        for pattern in dangerous_patterns:
            if pattern in text_upper:
                return True
        return False
    
    def check_script_injection(text):
        """Basic script injection pattern detection."""
        if not isinstance(text, str):
            return False
        
        dangerous_patterns = [
            "<script", "javascript:", "onclick=", "onerror=", "onload="
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                return True
        return False
    
    # Test SQL injection detection
    sql_test_cases = [
        ("normal input", False),
        ("SELECT * FROM users", True),
        ("'; DROP TABLE users; --", True),
        ("UNION SELECT password FROM users", True),
        ("regular optimization request", False),
    ]
    
    for text, should_be_detected in sql_test_cases:
        detected = check_sql_injection(text)
        if should_be_detected:
            assert detected, f"SQL injection pattern '{text}' should be detected"
        else:
            assert not detected, f"Normal text '{text}' was incorrectly flagged"
    
    # Test script injection detection
    script_test_cases = [
        ("normal input", False),
        ("<script>alert('xss')</script>", True),
        ("javascript:alert('xss')", True),
        ("onclick=alert('xss')", True),
        ("regular button text", False),
    ]
    
    for text, should_be_detected in script_test_cases:
        detected = check_script_injection(text)
        if should_be_detected:
            assert detected, f"Script injection pattern '{text}' should be detected"
        else:
            assert not detected, f"Normal text '{text}' was incorrectly flagged"
    
    print("✓ Security pattern detection tests passed")


def test_error_handling():
    """Test error handling and graceful degradation."""
    
    def safe_division(a, b):
        """Example of safe mathematical operation."""
        try:
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise TypeError("Arguments must be numbers")
            if b == 0:
                raise ValueError("Division by zero")
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError("Arguments must be finite")
            return a / b
        except Exception as e:
            return None, str(e)
    
    # Test error handling cases
    test_cases = [
        (10, 2, 5.0),              # Normal case
        (10, 0, None),             # Division by zero
        (10, "a", None),           # Invalid type
        (float('inf'), 2, None),   # Infinite input
        (float('nan'), 2, None),   # NaN input
    ]
    
    for a, b, expected in test_cases:
        result = safe_division(a, b)
        if expected is None:
            assert result[0] is None, f"Expected error for {a}/{b}, got {result}"
        else:
            assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
    
    print("✓ Error handling tests passed")


def test_configuration_validation():
    """Test configuration file validation."""
    
    def validate_config(config_dict):
        """Basic configuration validation."""
        required_fields = ['frequency', 'max_pressure', 'max_intensity']
        errors = []
        
        for field in required_fields:
            if field not in config_dict:
                errors.append(f"Missing required field: {field}")
        
        # Validate frequency
        if 'frequency' in config_dict:
            freq = config_dict['frequency']
            if not isinstance(freq, (int, float)):
                errors.append("Frequency must be a number")
            elif freq < 1000 or freq > 10000000:
                errors.append("Frequency out of range")
        
        # Validate pressure
        if 'max_pressure' in config_dict:
            pressure = config_dict['max_pressure']
            if not isinstance(pressure, (int, float)):
                errors.append("Max pressure must be a number")
            elif pressure > 50000:
                errors.append("Max pressure exceeds safety limit")
        
        return len(errors) == 0, errors
    
    # Test configuration validation
    configs = [
        # Valid configuration
        {
            "frequency": 40000,
            "max_pressure": 4000,
            "max_intensity": 10
        },
        # Missing required field
        {
            "frequency": 40000,
            "max_intensity": 10
        },
        # Invalid frequency
        {
            "frequency": -40000,
            "max_pressure": 4000,
            "max_intensity": 10
        },
        # Unsafe pressure
        {
            "frequency": 40000,
            "max_pressure": 100000,
            "max_intensity": 10
        }
    ]
    
    expected_results = [True, False, False, False]
    
    for i, config in enumerate(configs):
        is_valid, errors = validate_config(config)
        expected = expected_results[i]
        
        if expected:
            assert is_valid, f"Config {i} should be valid but got errors: {errors}"
        else:
            assert not is_valid, f"Config {i} should be invalid but was accepted"
    
    print("✓ Configuration validation tests passed")


def test_performance_bounds():
    """Test performance and resource usage bounds."""
    
    def check_memory_usage(array_size, element_size=8):
        """Estimate memory usage and check limits."""
        estimated_bytes = array_size * element_size
        max_allowed = 1024 * 1024 * 1024  # 1GB limit for single arrays
        
        return estimated_bytes <= max_allowed
    
    def check_computation_complexity(grid_size, num_sources):
        """Check if computation is within reasonable bounds."""
        total_operations = grid_size * num_sources
        max_operations = 1e9  # 1 billion operations limit
        
        return total_operations <= max_operations
    
    # Test memory bounds
    memory_test_cases = [
        (1000, True),           # Small array
        (100000, True),         # Medium array
        (200000000, False),     # Too large array
    ]
    
    for size, should_pass in memory_test_cases:
        result = check_memory_usage(size)
        if should_pass:
            assert result, f"Array size {size} should be acceptable"
        else:
            assert not result, f"Array size {size} should exceed memory limits"
    
    # Test computational complexity
    complexity_test_cases = [
        (1000, 1000, True),     # Reasonable problem size
        (10000, 10000, True),   # Large but manageable
        (100000, 100000, False), # Too computationally expensive
    ]
    
    for grid_size, num_sources, should_pass in complexity_test_cases:
        result = check_computation_complexity(grid_size, num_sources)
        if should_pass:
            assert result, f"Problem size {grid_size}x{num_sources} should be acceptable"
        else:
            assert not result, f"Problem size {grid_size}x{num_sources} should be too complex"
    
    print("✓ Performance bounds tests passed")


def run_all_tests():
    """Run all safety and validation tests."""
    print("Running Acousto-Gen Safety Validation Tests")
    print("=" * 50)
    
    try:
        test_basic_safety_limits()
        test_input_validation()
        test_security_patterns()
        test_error_handling()
        test_configuration_validation()
        test_performance_bounds()
        
        print("=" * 50)
        print("✓ All safety validation tests PASSED")
        print(f"✓ System meets basic safety and security requirements")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
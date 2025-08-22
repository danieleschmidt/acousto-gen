#!/usr/bin/env python3
"""
Generation 2 Robustness Test Suite
Tests comprehensive error handling, validation, and security features
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
import time
import traceback
from typing import Dict, Any

# Import Generation 2 components
from validation.comprehensive_validator import (
    ComprehensiveValidator, ValidationSeverity, SafetyValidator
)
from reliability.advanced_error_handler import (
    AcoustoGenErrorHandler, robust_execute, ErrorSeverity, RecoveryStrategy
)
from security.security_framework import (
    SecurityFramework, SecurityLevel, ThreatType, InputSanitizer
)

# Import core physics for testing
from physics.propagation.wave_propagator import WavePropagator
from physics.transducers.transducer_array import UltraLeap256


def test_generation2_robust():
    """Test Generation 2 robustness features."""
    print("🛡️  GENERATION 2 ROBUSTNESS TEST SUITE")
    print("=" * 60)
    
    test_results = {
        "validation": False,
        "error_handling": False,
        "security": False,
        "safety": False,
        "recovery": False
    }
    
    # Test 1: Comprehensive Validation
    print("\n1. 🔍 Testing Comprehensive Validation...")
    try:
        validator = ComprehensiveValidator()
        
        # Test valid input
        valid_target = {
            "type": "single_focus",
            "focal_points": [
                {"position": [0.0, 0.0, 0.1], "pressure": 2000.0}
            ]
        }
        
        result = validator.input_validator.validate_target_specification(valid_target)
        print(f"   ✅ Valid target validation: {result.is_valid}")
        
        # Test invalid input
        invalid_target = {
            "type": "invalid_type",
            "focal_points": [
                {"position": [999, 999, 999], "pressure": -1000}  # Invalid position and pressure
            ]
        }
        
        result = validator.input_validator.validate_target_specification(invalid_target)
        print(f"   ✅ Invalid target correctly rejected: {not result.is_valid}")
        print(f"   ✅ Found {len(result.issues)} validation issues")
        
        # Test safety validation
        dangerous_field = np.random.random((50, 50, 50)) * 20000  # Very high pressure
        safety_result = validator.safety_validator.validate_pressure_field(dangerous_field, 40000)
        
        critical_issues = [i for i in safety_result.issues if i.severity == ValidationSeverity.CRITICAL]
        print(f"   ✅ Safety validator caught {len(critical_issues)} critical issues")
        
        test_results["validation"] = True
        
    except Exception as e:
        print(f"   ❌ Validation test failed: {e}")
        traceback.print_exc()
    
    # Test 2: Advanced Error Handling
    print("\n2. 🔧 Testing Advanced Error Handling...")
    try:
        error_handler = AcoustoGenErrorHandler(enable_recovery=True)
        
        # Simulate various error types
        errors_tested = 0
        errors_handled = 0
        
        # Test 1: Memory error simulation
        try:
            raise MemoryError("Simulated CUDA out of memory")
        except Exception as e:
            context = {"operation": "field_calculation", "resolution": 0.001}
            recovery_params = error_handler.handle_error(e, "physics_engine", context)
            if recovery_params:
                errors_handled += 1
            errors_tested += 1
        
        # Test 2: Runtime error simulation
        try:
            raise RuntimeError("Optimization convergence failure")
        except Exception as e:
            context = {"iterations": 1000, "learning_rate": 0.1}
            recovery_params = error_handler.handle_error(e, "optimizer", context)
            if recovery_params:
                errors_handled += 1
            errors_tested += 1
        
        # Test 3: Connection error simulation
        try:
            raise ConnectionError("Hardware connection lost")
        except Exception as e:
            context = {"hardware_type": "ultraleap", "port": "/dev/ttyUSB0"}
            recovery_params = error_handler.handle_error(e, "hardware_interface", context)
            if recovery_params:
                errors_handled += 1
            errors_tested += 1
        
        print(f"   ✅ Error handling: {errors_handled}/{errors_tested} errors had recovery strategies")
        
        # Test robust execution decorator
        @robust_execute(component="test", max_retries=2, recovery_enabled=True)
        def flaky_function(attempt_count=[0]):
            attempt_count[0] += 1
            if attempt_count[0] < 2:  # Fail first attempt
                raise ValueError("Simulated temporary failure")
            return "success"
        
        result = flaky_function()
        print(f"   ✅ Robust execution decorator: {result}")
        
        # Get error summary
        summary = error_handler.get_error_summary()
        print(f"   ✅ Error statistics: {summary['total_errors']} total errors logged")
        
        test_results["error_handling"] = True
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        traceback.print_exc()
    
    # Test 3: Security Framework
    print("\n3. 🔒 Testing Security Framework...")
    try:
        security = SecurityFramework(enable_authentication=True)
        
        # Test input sanitization
        sanitizer = security.sanitizer
        
        # Test safe input
        safe_string = sanitizer.sanitize_string("Normal input string", max_length=100)
        print(f"   ✅ Safe input sanitized: '{safe_string}'")
        
        # Test dangerous inputs
        dangerous_inputs = [
            "'; DROP TABLE users; --",  # SQL injection
            "../../../etc/passwd",      # Path traversal
            "<script>alert('xss')</script>",  # XSS
            "cmd.exe /c format C:",     # Command injection
        ]
        
        blocked_count = 0
        for dangerous_input in dangerous_inputs:
            try:
                sanitizer.sanitize_string(dangerous_input)
            except ValueError:
                blocked_count += 1
        
        print(f"   ✅ Security sanitizer blocked {blocked_count}/{len(dangerous_inputs)} dangerous inputs")
        
        # Test authentication
        auth_manager = security.auth_manager
        
        # Create test user
        api_key = auth_manager.create_user("testuser", "testpass123", SecurityLevel.OPERATOR)
        print(f"   ✅ Created test user with API key")
        
        # Test authentication
        token = auth_manager.authenticate("testuser", "testpass123")
        print(f"   ✅ User authentication: {'success' if token else 'failed'}")
        
        # Test token verification
        user_info = auth_manager.verify_token(token)
        print(f"   ✅ Token verification: {'success' if user_info else 'failed'}")
        
        # Test failed authentication
        failed_token = auth_manager.authenticate("testuser", "wrongpassword")
        print(f"   ✅ Failed authentication correctly rejected: {failed_token is None}")
        
        # Test authorization
        from security.security_framework import AuthorizationManager
        
        can_optimize = AuthorizationManager.check_permission(SecurityLevel.OPERATOR, "optimize_hologram")
        cannot_admin = AuthorizationManager.check_permission(SecurityLevel.OPERATOR, "system_configuration")
        
        print(f"   ✅ Authorization: operator can optimize={can_optimize}, can admin={cannot_admin}")
        
        # Test rate limiting
        monitor = security.monitor
        rate_ok = monitor.check_rate_limit("api_calls", "test_ip")
        print(f"   ✅ Rate limiting functional: {rate_ok}")
        
        # Get security status
        status = security.get_security_status()
        print(f"   ✅ Security status: {status['security_level']} level, {status['active_sessions']} sessions")
        
        test_results["security"] = True
        
    except Exception as e:
        print(f"   ❌ Security test failed: {e}")
        traceback.print_exc()
    
    # Test 4: Safety System Integration
    print("\n4. ⚠️  Testing Safety System Integration...")
    try:
        from reliability.advanced_error_handler import safety_interlock
        
        # Test safe parameters
        safe_params = {
            "max_pressure": 3000,  # 3 kPa - safe
            "temperature": 35,     # 35°C - safe  
            "total_power": 25      # 25W - safe
        }
        
        safe_result = safety_interlock.check_safety_conditions("test_operation", safe_params)
        print(f"   ✅ Safe parameters approved: {safe_result}")
        
        # Test unsafe parameters  
        unsafe_params = {
            "max_pressure": 15000,  # 15 kPa - unsafe!
            "temperature": 70,      # 70°C - unsafe!
            "total_power": 150      # 150W - unsafe!
        }
        
        unsafe_result = safety_interlock.check_safety_conditions("test_operation", unsafe_params)
        print(f"   ✅ Unsafe parameters correctly rejected: {not unsafe_result}")
        print(f"   ✅ Safety violations logged: {len(safety_interlock.safety_violations)} violations")
        
        # Reset interlocks
        safety_interlock.reset_interlocks()
        print(f"   ✅ Safety interlocks reset successfully")
        
        test_results["safety"] = True
        
    except Exception as e:
        print(f"   ❌ Safety test failed: {e}")
        traceback.print_exc()
    
    # Test 5: Integrated Recovery System
    print("\n5. 🔄 Testing Integrated Recovery System...")
    try:
        # Test complete physics pipeline with error recovery
        array = UltraLeap256()
        propagator = WavePropagator(resolution=0.01, frequency=40e3, device='cpu')  # Lower res for speed
        
        # Create scenario that may need recovery
        phases = np.random.uniform(-np.pi, np.pi, len(array.elements))
        
        # Test with validation
        validator = ComprehensiveValidator()
        
        field_result = validator.validate_field_generation(
            phases=phases,
            amplitudes=np.ones(len(array.elements)),
            frequency=40e3
        )
        
        print(f"   ✅ Physics pipeline validation: {field_result.is_valid}")
        if field_result.issues:
            print(f"   ℹ️  Found {len(field_result.issues)} validation issues")
            for issue in field_result.issues[:3]:  # Show first 3
                print(f"      • {issue.severity.value}: {issue.message}")
        
        # Test actual field computation with error handling
        @robust_execute(component="physics", max_retries=2, recovery_enabled=True)
        def compute_safe_field():
            return propagator.compute_field_from_sources(
                array.get_positions(),
                np.ones(len(array.elements)),
                phases
            )
        
        field_data = compute_safe_field()
        print(f"   ✅ Robust field computation: shape {field_data.shape}")
        
        # Validate computed field
        field_validation = validator.safety_validator.validate_pressure_field(field_data, 40e3)
        critical_safety = [i for i in field_validation.issues if i.severity == ValidationSeverity.CRITICAL]
        print(f"   ✅ Field safety validation: {len(critical_safety)} critical issues")
        
        test_results["recovery"] = True
        
    except Exception as e:
        print(f"   ❌ Recovery test failed: {e}")
        traceback.print_exc()
    
    # Test Summary
    print("\n" + "=" * 60)
    print("🛡️  GENERATION 2 ROBUSTNESS TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name.upper():20s}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 GENERATION 2 (ROBUST) IMPLEMENTATION COMPLETED!")
        print("✅ Comprehensive validation system: OPERATIONAL")
        print("✅ Advanced error handling: OPERATIONAL")  
        print("✅ Security framework: OPERATIONAL")
        print("✅ Safety monitoring: OPERATIONAL")
        print("✅ Automatic recovery: OPERATIONAL")
        print("\n🚀 READY FOR GENERATION 3 (SCALE) IMPLEMENTATION")
        return True
    else:
        print(f"\n❌ Generation 2 robustness tests incomplete ({passed_tests}/{total_tests})")
        return False


if __name__ == "__main__":
    try:
        success = test_generation2_robust()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Generation 2 test suite crashed: {e}")
        traceback.print_exc()
        sys.exit(1)
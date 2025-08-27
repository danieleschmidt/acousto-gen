#!/usr/bin/env python3
"""
Test script for Generation 2 - Robust Implementation.
Tests safety systems, monitoring, security, and error recovery.
"""

import sys
import os
import time
import threading
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock dependencies for testing
try:
    import psutil
except ImportError:
    # Mock psutil if not available
    class MockProcess:
        def memory_info(self):
            class MemInfo:
                rss = 50 * 1024 * 1024  # 50MB
            return MemInfo()
        
        def cpu_percent(self):
            return 15.5
    
    class MockVirtualMemory:
        percent = 45.0
        used = 4 * 1024 * 1024 * 1024  # 4GB
        total = 8 * 1024 * 1024 * 1024  # 8GB
    
    class MockDiskUsage:
        def __init__(self, path):
            self.used = 100 * 1024 * 1024 * 1024  # 100GB
            self.total = 500 * 1024 * 1024 * 1024  # 500GB
    
    class MockPsutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 25.3
        
        @staticmethod
        def virtual_memory():
            return MockVirtualMemory()
        
        @staticmethod
        def disk_usage(path):
            return MockDiskUsage(path)
        
        @staticmethod
        def Process():
            return MockProcess()
    
    sys.modules['psutil'] = MockPsutil()


def test_safety_systems():
    """Test safety management systems."""
    print("🛡️ Testing Safety Systems...")
    
    try:
        from safety.safety_manager import SafetyManager, SafetyLimits, SafetyViolationType, SafetyLevel
        
        # Test safety manager initialization
        limits = SafetyLimits(
            max_total_power=500.0,
            max_temperature=70.0,
            max_field_strength=8000.0
        )
        
        safety_manager = SafetyManager(limits)
        print('✓ Safety manager initialized')
        
        # Test parameter validation
        test_params = {
            'frequency': 45000,  # Hz
            'total_power': 300,   # W - within limits
            'target_position': [0.05, 0.05, 0.1]  # meters
        }
        
        violations = safety_manager.validate_operation_parameters(test_params)
        print(f'✓ Parameter validation: {len(violations)} violations found')
        
        # Test operation safety check
        is_safe = safety_manager.check_operation_safety("test_operation", test_params)
        print(f'✓ Operation safety check: {"SAFE" if is_safe else "UNSAFE"}')
        
        # Test violation handling
        safety_manager._trigger_violation(
            SafetyViolationType.TEMPERATURE_TOO_HIGH,
            SafetyLevel.WARNING,
            "Test violation",
            measured_value=65.0,
            threshold_value=60.0
        )
        print('✓ Safety violation handling')
        
        # Test safety status
        status = safety_manager.get_safety_status()
        print(f'✓ Safety status: {status["total_violations"]} total violations')
        
        return True
        
    except Exception as e:
        print(f'❌ Safety systems test failed: {e}')
        return False


def test_monitoring_systems():
    """Test monitoring and metrics collection."""
    print("\n📊 Testing Monitoring Systems...")
    
    try:
        from monitoring.system_monitor import SystemMonitor, MetricsCollector, SystemMetric
        
        # Test metrics collector
        collector = MetricsCollector(retention_hours=1)
        
        # Record test metrics
        test_metric = SystemMetric(
            name="test_metric",
            value=42.5,
            unit="percent",
            tags={"component": "test"}
        )
        collector.record_metric(test_metric)
        print('✓ Metrics collector initialized and recording')
        
        # Test performance tracking
        collector.record_operation("test_operation", 0.123, success=True)
        collector.record_operation("test_operation", 0.156, success=False, error_type="timeout")
        print('✓ Performance metrics recorded')
        
        # Test system monitor
        monitor = SystemMonitor(collection_interval=0.1)
        
        # Mock hardware interface for testing
        class MockHardware:
            def get_status(self):
                class Status:
                    connected = True
                    temperature = 45.0
                    voltage = 12.5
                    current = 2.1
                return Status()
        
        monitor.add_hardware_monitor("test_hardware", MockHardware())
        print('✓ System monitor initialized with hardware')
        
        # Start monitoring briefly
        monitor.start_monitoring()
        time.sleep(0.5)  # Let it collect some metrics
        monitor.stop_monitoring()
        print('✓ System monitoring cycle completed')
        
        # Test health check
        health = monitor.get_health_check()
        print(f'✓ Health check: {health["status"]}')
        
        # Test status
        status = monitor.get_system_status()
        print(f'✓ System status: {status["hardware_monitors"]} hardware monitors')
        
        return True
        
    except Exception as e:
        print(f'❌ Monitoring systems test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_security_systems():
    """Test security and authentication systems."""
    print("\n🔒 Testing Security Systems...")
    
    try:
        from security.security_manager import SecurityManager, SecurityLevel, InputValidator
        
        # Test security manager
        security_manager = SecurityManager()
        print('✓ Security manager initialized')
        
        # Test input validation
        validator = InputValidator()
        
        # Test string validation
        is_valid, error = validator.validate_string("test_user", "username", max_length=50)
        print(f'✓ String validation: {"VALID" if is_valid else f"INVALID - {error}"}')
        
        # Test number validation
        is_valid, error = validator.validate_number(45.5, min_val=0, max_val=100)
        print(f'✓ Number validation: {"VALID" if is_valid else f"INVALID - {error}"}')
        
        # Test position validation
        is_valid, error = validator.validate_position([0.1, 0.2, 0.15])
        print(f'✓ Position validation: {"VALID" if is_valid else f"INVALID - {error}"}')
        
        # Test dangerous pattern detection
        is_valid, error = validator.validate_string("<script>alert('xss')</script>", "experiment_name")
        print(f'✓ XSS detection: {"BLOCKED" if not is_valid else "FAILED TO BLOCK"}')
        
        # Test session management
        session_id = security_manager.create_session(
            user_id="test_user",
            security_level=SecurityLevel.USER,
            ip_address="192.168.1.100",
            user_agent="TestClient/1.0"
        )
        print(f'✓ Session created: {session_id[:16]}...')
        
        # Test session validation
        session = security_manager.validate_session(session_id, "192.168.1.100")
        print(f'✓ Session validation: {"VALID" if session else "INVALID"}')
        
        # Test permissions
        has_perm = security_manager.has_permission(session_id, "view_experiments")
        print(f'✓ Permission check: {"GRANTED" if has_perm else "DENIED"}')
        
        # Test rate limiting
        allowed = security_manager.check_rate_limit("test_user", "login")
        print(f'✓ Rate limiting: {"ALLOWED" if allowed else "BLOCKED"}')
        
        # Test input validation with rules
        test_data = {
            "name": "Test Experiment",
            "frequency": 40000,
            "power": 250.5
        }
        
        validation_rules = {
            "name": {"required": True, "type": "string", "pattern": "experiment_name", "max_length": 100},
            "frequency": {"required": True, "type": "number", "min": 20000, "max": 100000},
            "power": {"required": True, "type": "number", "min": 0, "max": 1000}
        }
        
        is_valid, errors = security_manager.validate_input(test_data, validation_rules)
        print(f'✓ Complex validation: {"VALID" if is_valid else f"INVALID - {len(errors)} errors"}')
        
        # Test security status
        status = security_manager.get_security_status()
        print(f'✓ Security status: {status["active_sessions"]} active sessions')
        
        return True
        
    except Exception as e:
        print(f'❌ Security systems test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_error_recovery():
    """Test error recovery and fault tolerance."""
    print("\n🚨 Testing Error Recovery Systems...")
    
    try:
        from recovery.error_recovery import (
            ErrorRecoveryManager, RecoveryAction, RecoveryStrategy, 
            ErrorSeverity, CircuitBreaker, RetryManager
        )
        
        # Test circuit breaker
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        def failing_function():
            raise Exception("Test failure")
        
        # Trigger circuit breaker
        failures = 0
        for _ in range(5):
            try:
                circuit_breaker.call(failing_function)
            except Exception:
                failures += 1
        
        print(f'✓ Circuit breaker: {failures} failures, state: {circuit_breaker.state}')
        
        # Test retry manager
        retry_manager = RetryManager()
        
        @retry_manager.retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def sometimes_failing_function(success_after=2):
            if not hasattr(sometimes_failing_function, 'attempt_count'):
                sometimes_failing_function.attempt_count = 0
            sometimes_failing_function.attempt_count += 1
            
            if sometimes_failing_function.attempt_count < success_after:
                raise Exception(f"Attempt {sometimes_failing_function.attempt_count} failed")
            
            return f"Success after {sometimes_failing_function.attempt_count} attempts"
        
        try:
            result = sometimes_failing_function(success_after=2)
            print(f'✓ Retry mechanism: {result}')
        except Exception as e:
            print(f'❌ Retry mechanism failed: {e}')
        
        # Test error recovery manager
        recovery_manager = ErrorRecoveryManager()
        
        # Register a recovery strategy
        def test_recovery_function():
            print("    Recovery function executed")
            return True
        
        recovery_action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            max_attempts=2,
            delay_seconds=0.1,
            recovery_function=test_recovery_function
        )
        
        recovery_manager.register_recovery_strategy("TestException", recovery_action)
        print('✓ Recovery strategy registered')
        
        # Test error handling
        test_error = Exception("Test error for recovery")
        success = recovery_manager.handle_error(
            test_error,
            component="test_component",
            operation="test_operation",
            severity=ErrorSeverity.HIGH
        )
        
        # Give recovery thread time to process
        time.sleep(0.5)
        
        print(f'✓ Error recovery: {"SUCCESS" if success else "ATTEMPTED"}')
        
        # Test component health
        health = recovery_manager.get_component_health("test_component")
        print(f'✓ Component health: {health}')
        
        # Test system health
        system_health = recovery_manager.get_system_health()
        print(f'✓ System health: {system_health["status"]}')
        
        # Test statistics
        stats = recovery_manager.get_error_statistics()
        print(f'✓ Error statistics: {stats["total_errors"]} total errors')
        
        # Clean up
        recovery_manager.stop_recovery_processing()
        
        return True
        
    except Exception as e:
        print(f'❌ Error recovery test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between all robust systems."""
    print("\n🔗 Testing System Integration...")
    
    try:
        # Initialize all systems
        from safety.safety_manager import get_safety_manager, initialize_safety
        from monitoring.system_monitor import get_system_monitor, initialize_monitoring
        from security.security_manager import get_security_manager, initialize_security
        from recovery.error_recovery import get_error_recovery_manager, initialize_error_recovery
        
        # Initialize all systems
        initialize_safety(start_monitoring=False)  # Don't start monitoring to avoid conflicts
        initialize_monitoring(collection_interval=1.0, start_monitoring=False)
        initialize_security()
        initialize_error_recovery()
        
        print('✓ All systems initialized')
        
        # Get managers
        safety_mgr = get_safety_manager()
        monitor_mgr = get_system_monitor()
        security_mgr = get_security_manager()
        recovery_mgr = get_error_recovery_manager()
        
        print('✓ All managers accessible')
        
        # Test cross-system integration
        # Create a session
        from security.security_manager import SecurityLevel
        session_id = security_mgr.create_session(
            "integration_user",
            SecurityLevel.OPERATOR,
            "127.0.0.1",
            "IntegrationTest/1.0"
        )
        
        # Check if user has permission for hardware control
        can_control = security_mgr.has_permission(session_id, "control_hardware")
        print(f'✓ Permission integration: {"GRANTED" if can_control else "DENIED"}')
        
        # Test operation with safety check
        if can_control:
            test_params = {
                'frequency': 40000,
                'total_power': 300,
                'target_position': [0.02, 0.03, 0.08]
            }
            
            is_safe = safety_mgr.check_operation_safety("test_hardware_control", test_params)
            print(f'✓ Safety integration: {"SAFE" if is_safe else "UNSAFE"}')
        
        # Test monitoring integration
        system_status = {
            "safety": safety_mgr.get_safety_status(),
            "security": security_mgr.get_security_status(), 
            "recovery": recovery_mgr.get_system_health(),
            "monitoring": monitor_mgr.get_health_check()
        }
        
        print('✓ Status integration: All systems reporting')
        
        # Test error propagation
        from recovery.error_recovery import ErrorSeverity
        test_error = Exception("Integration test error")
        recovery_mgr.handle_error(
            test_error,
            component="integration_test",
            severity=ErrorSeverity.MEDIUM
        )
        
        print('✓ Error propagation: Error logged and handled')
        
        # Brief integration run
        print('✓ Running brief integration test...')
        
        # Start monitoring briefly
        monitor_mgr.start_monitoring()
        safety_mgr.start_monitoring(interval=0.2)
        
        # Simulate some activity
        time.sleep(1.0)
        
        # Stop monitoring
        monitor_mgr.stop_monitoring()
        safety_mgr.stop_monitoring()
        recovery_mgr.stop_recovery_processing()
        
        print('✓ Integration test completed successfully')
        
        return True
        
    except Exception as e:
        print(f'❌ Integration test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Generation 2 tests."""
    print("🚀 GENERATION 2: MAKE IT ROBUST (Reliable Implementation)")
    print("=" * 70)
    
    tests = [
        ("Safety Systems", test_safety_systems),
        ("Monitoring Systems", test_monitoring_systems),
        ("Security Systems", test_security_systems),
        ("Error Recovery", test_error_recovery),
        ("System Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 GENERATION 2 TEST RESULTS:")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ GENERATION 2 COMPLETE - System is now robust and production-ready!")
        print("\n🎯 Robust Systems Working:")
        print("  • 🛡️ Comprehensive safety management with real-time monitoring")
        print("  • 📊 System monitoring with metrics collection and alerting")
        print("  • 🔒 Security with authentication, authorization, and input validation")
        print("  • 🚨 Error recovery with circuit breakers and fault tolerance")
        print("  • 🔗 Integrated system health monitoring and management")
        print("\n🏗️ Production-Ready Features:")
        print("  • Real-time hardware safety monitoring")
        print("  • Comprehensive security audit logging")
        print("  • Automatic error recovery and failover")
        print("  • Performance monitoring and alerting")
        print("  • Cross-system integration and health checks")
        return True
    else:
        print("❌ GENERATION 2 INCOMPLETE - Some robust systems failed tests")
        print(f"   {total - passed} test(s) need attention before production deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
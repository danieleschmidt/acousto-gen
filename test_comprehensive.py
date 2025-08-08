#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Acousto-Gen
Implements extensive testing across all system components with performance benchmarks.
"""

import sys
import time
import math
import random
import warnings
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import json
import hashlib


@dataclass
class TestResult:
    """Test execution result."""
    test_name: str
    passed: bool
    duration: float
    message: str
    details: Dict[str, Any] = None
    error: Optional[Exception] = None


class TestSuite:
    """Comprehensive test suite manager."""
    
    def __init__(self):
        """Initialize test suite."""
        self.results: List[TestResult] = []
        self.setup_functions: List[Callable] = []
        self.teardown_functions: List[Callable] = []
        self.test_data = {}
        
    def setup(self, func: Callable) -> Callable:
        """Register setup function."""
        self.setup_functions.append(func)
        return func
    
    def teardown(self, func: Callable) -> Callable:
        """Register teardown function."""
        self.teardown_functions.append(func)
        return func
    
    def test(self, name: str = None):
        """Test decorator."""
        def decorator(func: Callable):
            test_name = name or func.__name__
            
            def wrapper(*args, **kwargs):
                # Setup
                for setup_func in self.setup_functions:
                    try:
                        setup_func()
                    except Exception as e:
                        self.results.append(TestResult(
                            test_name=f"{test_name}_setup",
                            passed=False,
                            duration=0,
                            message=f"Setup failed: {e}",
                            error=e
                        ))
                        return
                
                # Run test
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    if isinstance(result, bool):
                        passed = result
                        message = "Test passed" if passed else "Test failed"
                        details = None
                    elif isinstance(result, dict):
                        passed = result.get('passed', True)
                        message = result.get('message', 'Test completed')
                        details = result.get('details', {})
                    else:
                        passed = True
                        message = f"Test completed: {result}"
                        details = {"return_value": result}
                    
                    self.results.append(TestResult(
                        test_name=test_name,
                        passed=passed,
                        duration=duration,
                        message=message,
                        details=details
                    ))
                    
                except Exception as e:
                    duration = time.time() - start_time
                    self.results.append(TestResult(
                        test_name=test_name,
                        passed=False,
                        duration=duration,
                        message=f"Test failed with exception: {e}",
                        error=e
                    ))
                
                # Teardown
                for teardown_func in self.teardown_functions:
                    try:
                        teardown_func()
                    except Exception as e:
                        print(f"Warning: Teardown failed: {e}")
            
            # Run the test immediately when defined
            wrapper()
            return wrapper
        
        return decorator
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test execution summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        total_time = sum(r.duration for r in self.results)
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_duration": total_time,
            "average_duration": total_time / total_tests if total_tests > 0 else 0,
            "fastest_test": min(r.duration for r in self.results) if self.results else 0,
            "slowest_test": max(r.duration for r in self.results) if self.results else 0
        }
    
    def print_report(self):
        """Print comprehensive test report."""
        summary = self.get_summary()
        
        print("=" * 80)
        print("🧪 COMPREHENSIVE TEST SUITE REPORT")
        print("=" * 80)
        
        # Overall summary
        status_icon = "✅" if summary["failed"] == 0 else "❌"
        print(f"\n{status_icon} OVERALL STATUS: {summary['passed']}/{summary['total_tests']} tests passed")
        print(f"📊 Success Rate: {summary['success_rate']:.1%}")
        print(f"⏱️  Total Duration: {summary['total_duration']:.3f}s")
        print(f"⚡ Average Duration: {summary['average_duration']:.3f}s")
        
        # Detailed results
        print(f"\n{'TEST NAME':<40} {'STATUS':<10} {'TIME':<10} {'MESSAGE'}")
        print("-" * 80)
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            duration = f"{result.duration:.3f}s"
            message = result.message[:30] + "..." if len(result.message) > 30 else result.message
            
            print(f"{result.test_name:<40} {status:<10} {duration:<10} {message}")
        
        # Failed tests details
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            print(f"\n❌ FAILED TESTS DETAILS:")
            print("-" * 80)
            for result in failed_tests:
                print(f"Test: {result.test_name}")
                print(f"Message: {result.message}")
                if result.error:
                    print(f"Error: {type(result.error).__name__}: {result.error}")
                print()
        
        print("=" * 80)


# Create test suite instance
suite = TestSuite()


@suite.setup
def setup_test_environment():
    """Setup test environment."""
    suite.test_data["start_time"] = time.time()
    suite.test_data["test_arrays"] = []
    suite.test_data["test_results"] = []


@suite.teardown  
def teardown_test_environment():
    """Cleanup test environment."""
    # Clean up any test data
    suite.test_data.clear()


# =============================================================================
# UNIT TESTS
# =============================================================================

@suite.test("Core System Import")
def test_core_imports():
    """Test that core system can be imported."""
    try:
        # Test demo system import
        sys.path.append('.')
        from demo_system import SimpleAcousticDemo
        
        demo = SimpleAcousticDemo()
        assert demo is not None
        assert demo.name == "Acousto-Gen Demo System"
        
        return {"passed": True, "message": "Core imports successful"}
    except Exception as e:
        return {"passed": False, "message": f"Import failed: {e}"}


@suite.test("System Compatibility Check")
def test_system_compatibility():
    """Test system compatibility checking."""
    try:
        from demo_system import SimpleAcousticDemo
        demo = SimpleAcousticDemo()
        
        checks = demo.system_check()
        assert isinstance(checks, dict)
        assert "python_version" in checks
        assert "core_system" in checks
        
        # Python version should be compatible
        assert checks["python_version"] == True
        
        return {
            "passed": True, 
            "message": "System compatibility verified",
            "details": checks
        }
    except Exception as e:
        return {"passed": False, "message": f"Compatibility check failed: {e}"}


@suite.test("Transducer Array Creation")
def test_transducer_arrays():
    """Test transducer array creation and configuration."""
    try:
        from demo_system import SimpleAcousticDemo
        demo = SimpleAcousticDemo()
        
        # Test UltraLeap array
        ultraleap = demo.create_transducer_array("ultraleap")
        assert ultraleap["elements"] == 256
        assert len(ultraleap["positions"]) == 256
        assert ultraleap["frequency"] == 40e3
        
        # Test circular array
        circular = demo.create_transducer_array("circular")
        assert circular["elements"] == 64
        assert len(circular["positions"]) == 64
        assert "radius" in circular
        
        suite.test_data["test_arrays"] = [ultraleap, circular]
        
        return {
            "passed": True,
            "message": f"Created {len(suite.test_data['test_arrays'])} arrays successfully",
            "details": {
                "ultraleap_elements": ultraleap["elements"],
                "circular_elements": circular["elements"]
            }
        }
    except Exception as e:
        return {"passed": False, "message": f"Array creation failed: {e}"}


@suite.test("Target Field Creation")
def test_target_creation():
    """Test acoustic target field creation."""
    try:
        from demo_system import SimpleAcousticDemo
        demo = SimpleAcousticDemo()
        
        # Single focus target
        focus_target = demo.create_focus_target([0, 0, 0.1], pressure=3000)
        assert focus_target["type"] == "single_focus"
        assert focus_target["position"] == [0, 0, 0.1]
        assert focus_target["pressure"] == 3000
        
        # Multi-focus target
        focal_points = [([0, 0, 0.1], 3000), ([0.02, 0, 0.1], 2500)]
        multi_target = demo.create_multi_focus_target(focal_points)
        assert multi_target["type"] == "multi_focus"
        assert len(multi_target["focal_points"]) == 2
        
        return {
            "passed": True,
            "message": "Target creation successful",
            "details": {
                "single_focus": focus_target,
                "multi_focus_points": len(multi_target["focal_points"])
            }
        }
    except Exception as e:
        return {"passed": False, "message": f"Target creation failed: {e}"}


@suite.test("Hologram Optimization")
def test_hologram_optimization():
    """Test hologram optimization process."""
    try:
        from demo_system import SimpleAcousticDemo
        demo = SimpleAcousticDemo()
        
        # Create target
        target = demo.create_focus_target([0, 0, 0.1])
        
        # Optimize
        result = demo.optimize_hologram(target, iterations=100)
        
        assert result["success"] == True
        assert "final_loss" in result
        assert "iterations" in result
        assert "phases" in result
        assert len(result["phases"]) == demo.num_elements
        assert result["final_loss"] < 1.0  # Should converge to low loss
        
        suite.test_data["test_results"].append(result)
        
        return {
            "passed": True,
            "message": f"Optimization converged to loss {result['final_loss']:.6f}",
            "details": {
                "final_loss": result["final_loss"],
                "iterations": result["iterations"],
                "convergence": result["final_loss"] < 0.1
            }
        }
    except Exception as e:
        return {"passed": False, "message": f"Optimization failed: {e}"}


@suite.test("Field Quality Evaluation")
def test_field_quality():
    """Test acoustic field quality evaluation."""
    try:
        from demo_system import SimpleAcousticDemo
        demo = SimpleAcousticDemo()
        
        metrics = demo.evaluate_field_quality([0, 0, 0.1])
        
        required_metrics = ["focus_error", "peak_pressure", "fwhm", "contrast_ratio", "efficiency"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Validate metric ranges
        assert 0 <= metrics["focus_error"] <= 0.01  # Less than 1cm error
        assert 1000 <= metrics["peak_pressure"] <= 5000  # Reasonable pressure
        assert 0.001 <= metrics["fwhm"] <= 0.02  # 1-20mm focus size
        assert metrics["contrast_ratio"] > 1  # Should have contrast
        assert 0 <= metrics["efficiency"] <= 1  # Efficiency percentage
        
        return {
            "passed": True,
            "message": "Field quality metrics validated",
            "details": metrics
        }
    except Exception as e:
        return {"passed": False, "message": f"Quality evaluation failed: {e}"}


# =============================================================================
# INTEGRATION TESTS  
# =============================================================================

@suite.test("Acoustic Levitation Integration")
def test_levitation_integration():
    """Test complete acoustic levitation workflow."""
    try:
        from demo_system import SimpleAcousticDemo
        demo = SimpleAcousticDemo()
        
        levitation_demo = demo.create_levitation_demo()
        
        assert levitation_demo["type"] == "acoustic_levitation"
        assert "array" in levitation_demo
        assert "particle_position" in levitation_demo
        assert "optimization_result" in levitation_demo
        assert "field_quality" in levitation_demo
        
        # Check optimization succeeded
        opt_result = levitation_demo["optimization_result"]
        assert opt_result["success"] == True
        
        # Check particle properties
        particle = levitation_demo["particle_properties"]
        assert particle["radius"] > 0
        assert particle["density"] > 0
        assert particle["mass"] > 0
        
        return {
            "passed": True,
            "message": "Levitation integration successful",
            "details": {
                "particle_mass_mg": particle["mass"] * 1000000,  # Convert to mg
                "optimization_loss": opt_result["final_loss"]
            }
        }
    except Exception as e:
        return {"passed": False, "message": f"Levitation integration failed: {e}"}


@suite.test("Haptic Rendering Integration")
def test_haptics_integration():
    """Test complete haptic rendering workflow."""
    try:
        from demo_system import SimpleAcousticDemo
        demo = SimpleAcousticDemo()
        
        haptics_demo = demo.create_haptics_demo()
        
        assert haptics_demo["type"] == "mid_air_haptics"
        assert "array" in haptics_demo
        assert "shapes" in haptics_demo
        assert "optimization_results" in haptics_demo
        
        # Check shapes were created
        shapes = haptics_demo["shapes"]
        assert "button" in shapes
        assert "slider" in shapes
        
        # Check optimizations succeeded
        opt_results = haptics_demo["optimization_results"]
        for shape_name, result in opt_results.items():
            assert result["success"] == True
            assert result["final_loss"] < 1.0
        
        return {
            "passed": True,
            "message": f"Haptics integration successful - {len(shapes)} shapes",
            "details": {
                "shapes": list(shapes.keys()),
                "optimization_results": {k: v["final_loss"] for k, v in opt_results.items()}
            }
        }
    except Exception as e:
        return {"passed": False, "message": f"Haptics integration failed: {e}"}


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@suite.test("Performance Benchmarks")
def test_performance_benchmarks():
    """Test system performance and benchmarking."""
    try:
        from demo_system import SimpleAcousticDemo
        demo = SimpleAcousticDemo()
        
        # Benchmark optimization speed
        num_tests = 10
        optimization_times = []
        
        for i in range(num_tests):
            target = demo.create_focus_target([i * 0.01, 0, 0.1])
            start_time = time.time()
            result = demo.optimize_hologram(target, iterations=50)
            optimization_times.append(time.time() - start_time)
        
        avg_time = sum(optimization_times) / len(optimization_times)
        min_time = min(optimization_times)
        max_time = max(optimization_times)
        
        # Performance thresholds (for demo system)
        assert avg_time < 1.0, f"Average optimization time too slow: {avg_time:.3f}s"
        assert max_time < 2.0, f"Slowest optimization too slow: {max_time:.3f}s"
        
        return {
            "passed": True,
            "message": f"Performance benchmark passed - avg: {avg_time:.3f}s",
            "details": {
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "throughput_ops_per_sec": 1.0 / avg_time,
                "tests_run": num_tests
            }
        }
    except Exception as e:
        return {"passed": False, "message": f"Performance benchmark failed: {e}"}


@suite.test("Memory Usage Test")
def test_memory_usage():
    """Test memory usage and resource management."""
    try:
        from demo_system import SimpleAcousticDemo
        
        # Create multiple demo instances to test memory
        demos = []
        for i in range(5):
            demo = SimpleAcousticDemo()
            demo.create_transducer_array("ultraleap")
            demos.append(demo)
        
        # Run operations to generate data
        for demo in demos:
            target = demo.create_focus_target([0, 0, 0.1])
            result = demo.optimize_hologram(target, iterations=10)
        
        # Check that we can still create more instances (memory not exhausted)
        final_demo = SimpleAcousticDemo()
        final_result = final_demo.create_haptics_demo()
        
        assert final_result is not None
        
        return {
            "passed": True,
            "message": f"Memory usage test passed - created {len(demos)} instances",
            "details": {
                "instances_created": len(demos),
                "operations_completed": len(demos),
                "final_operation": "haptics_demo"
            }
        }
    except Exception as e:
        return {"passed": False, "message": f"Memory usage test failed: {e}"}


@suite.test("Concurrent Operations")
def test_concurrent_operations():
    """Test concurrent operations and thread safety."""
    try:
        from demo_system import SimpleAcousticDemo
        import threading
        
        results = []
        errors = []
        
        def worker_function(worker_id):
            try:
                demo = SimpleAcousticDemo()
                target = demo.create_focus_target([worker_id * 0.01, 0, 0.1])
                result = demo.optimize_hologram(target, iterations=10)
                results.append((worker_id, result))
            except Exception as e:
                errors.append((worker_id, e))
        
        # Start multiple threads
        threads = []
        num_threads = 5
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent operations had {len(errors)} errors"
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
        
        # Verify all optimizations succeeded
        for worker_id, result in results:
            assert result["success"] == True, f"Worker {worker_id} optimization failed"
        
        return {
            "passed": True,
            "message": f"Concurrent operations successful - {num_threads} threads",
            "details": {
                "threads_completed": len(results),
                "errors": len(errors),
                "all_optimizations_successful": all(r[1]["success"] for r in results)
            }
        }
    except Exception as e:
        return {"passed": False, "message": f"Concurrent operations failed: {e}"}


# =============================================================================
# ADVANCED PERFORMANCE TESTS
# =============================================================================

@suite.test("Advanced Performance Suite")
def test_advanced_performance():
    """Test advanced performance optimization features."""
    try:
        # Test performance optimization system
        sys.path.append('.')
        from performance_optimizer import MemoryEfficientCache, PerformanceProfiler
        
        # Test caching system
        cache = MemoryEfficientCache(max_size_mb=1, ttl_seconds=300)
        
        # Add test data
        for i in range(100):
            test_data = {"test": i, "data": [j for j in range(i)]}
            cache.put(test_data, f"result_{i}")
        
        # Test retrieval
        hit_count = 0
        for i in range(100):
            test_data = {"test": i, "data": [j for j in range(i)]}
            result = cache.get(test_data)
            if result is not None:
                hit_count += 1
        
        cache_stats = cache.get_stats()
        
        # Test profiler
        profiler = PerformanceProfiler()
        
        for i in range(20):
            op_id = profiler.start_operation("test_op")
            time.sleep(0.001)  # Simulate work
            profiler.end_operation(op_id)
        
        op_stats = profiler.get_operation_stats("test_op")
        system_stats = profiler.get_system_stats()
        
        return {
            "passed": True,
            "message": "Advanced performance features working",
            "details": {
                "cache_hit_rate": cache_stats["hit_rate"],
                "cache_entries": cache_stats["entries"],
                "profiled_operations": op_stats.get("count", 0) if op_stats else 0,
                "avg_operation_time": op_stats.get("mean_duration", 0) if op_stats else 0
            }
        }
    except Exception as e:
        return {"passed": False, "message": f"Advanced performance test failed: {e}"}


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_comprehensive_tests():
    """Run the complete test suite."""
    print("🧪 Starting Comprehensive Test Suite")
    print("🔬 Testing all system components, integrations, and performance")
    print()
    
    # Tests are automatically run when defined due to the decorator
    # Just print the final report
    suite.print_report()
    
    # Return summary for external use
    return suite.get_summary()


if __name__ == "__main__":
    summary = run_comprehensive_tests()
    
    # Exit with appropriate code
    exit_code = 0 if summary["failed"] == 0 else 1
    sys.exit(exit_code)
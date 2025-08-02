# Testing Guide

This document describes the comprehensive testing strategy for Acousto-Gen.

## Test Organization

### Directory Structure

```
tests/
├── unit/                   # Unit tests for individual components
├── integration/            # Integration tests for component interactions
├── e2e/                   # End-to-end workflow tests
├── fixtures/              # Test data and configuration files
├── data/                  # Test datasets and reference data
├── references/            # Reference results for validation
├── conftest.py           # Pytest configuration and shared fixtures
└── test_*.py             # Legacy test files (being migrated)
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation:

- **Physics modules**: Wave propagation, transducer modeling, field calculations
- **Optimization algorithms**: Gradient descent, genetic algorithms, neural networks
- **Safety systems**: Pressure limits, thermal monitoring, emergency stops
- **Hardware interfaces**: Device drivers, calibration routines
- **Data structures**: Field representations, array configurations

**Run unit tests:**
```bash
pytest tests/unit/ -v
```

### Integration Tests (`tests/integration/`)

Test component interactions and data flow:

- **Full pipeline**: From target specification to hardware control
- **Multi-component workflows**: Physics + optimization + safety
- **Hardware integration**: Real device communication
- **Performance integration**: GPU acceleration, memory management

**Run integration tests:**
```bash
pytest tests/integration/ -v --integration
```

### End-to-End Tests (`tests/e2e/`)

Test complete user workflows:

- **Researcher workflows**: Academic research scenarios
- **Engineering workflows**: Hardware development and validation
- **Medical workflows**: Clinical application testing
- **Educational workflows**: Teaching and demonstration use cases

**Run E2E tests:**
```bash
pytest tests/e2e/ -v --runslow
```

## Test Execution

### Quick Test Suite

For rapid development feedback:

```bash
# Run fast unit tests only
pytest tests/unit/ -v -m "not slow"

# Run with coverage
pytest tests/unit/ --cov=acousto_gen --cov-report=html
```

### Comprehensive Test Suite

For thorough validation:

```bash
# Run all tests
pytest -v

# Run with all optional features
pytest -v --runslow --hardware --gpu --integration
```

### Specific Test Categories

```bash
# Performance tests
pytest -m performance -v

# Hardware tests (requires hardware)
pytest -m hardware -v --hardware

# GPU tests (requires CUDA)
pytest -m gpu -v --gpu

# Slow tests
pytest -m slow -v --runslow
```

## Test Fixtures and Data

### Mock Objects

- **`mock_transducer`**: Standard 256-element array simulation
- **`large_mock_transducer`**: 1024-element array for performance testing
- **`medical_mock_transducer`**: Medical-grade array with safety features
- **`mock_hardware_interface`**: Hardware communication simulation

### Test Data

- **`sample_pressure_field`**: Realistic 3D pressure field with focus
- **`complex_pressure_field`**: Complex field with phase information
- **`test_configuration`**: JSON configuration for test parameters
- **`performance_benchmark_data`**: Large datasets for performance testing

### Application Configurations

- **`levitation_test_config`**: Particle properties and trap parameters
- **`haptic_test_config`**: Perception thresholds and modulation settings
- **`medical_test_config`**: Regulatory limits and tissue properties

## Writing Tests

### Unit Test Template

```python
import pytest
import numpy as np
from acousto_gen.module import ComponentToTest

class TestComponent:
    """Test ComponentToTest functionality."""
    
    def test_basic_functionality(self):
        """Test basic component behavior."""
        component = ComponentToTest(param1=value1)
        result = component.method(input_data)
        
        assert isinstance(result, expected_type)
        assert result.shape == expected_shape
        assert np.allclose(result, expected_values, rtol=1e-5)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        component = ComponentToTest()
        
        with pytest.raises(ValueError):
            component.method(invalid_input)
    
    @pytest.mark.parametrize("input_val,expected", [
        (1.0, 2.0),
        (2.0, 4.0),
        (3.0, 6.0)
    ])
    def test_parametrized(self, input_val, expected):
        """Test with multiple parameter sets."""
        component = ComponentToTest()
        result = component.method(input_val)
        assert result == expected
```

### Integration Test Template

```python
import pytest
from acousto_gen import AcousticHologram

class TestIntegration:
    """Test component integration."""
    
    def test_complete_workflow(self, mock_transducer, test_configuration):
        """Test complete workflow integration."""
        # Setup
        hologram = AcousticHologram(
            transducer=mock_transducer,
            **test_configuration["acoustic_parameters"]
        )
        
        # Execute workflow
        target = hologram.create_focus_point([0, 0, 0.1], 3000)
        phases = hologram.optimize(target, iterations=100)
        
        # Validate integration
        assert len(phases) == mock_transducer.elements
        mock_transducer.set_phases.assert_not_called()  # Not called yet
        
        # Apply to hardware
        mock_transducer.set_phases(phases)
        mock_transducer.set_phases.assert_called_once()
```

### Performance Test Template

```python
import pytest
import time
import numpy as np

@pytest.mark.performance
class TestPerformance:
    """Test performance characteristics."""
    
    def test_optimization_speed(self, large_mock_transducer):
        """Test optimization performance."""
        hologram = AcousticHologram(transducer=large_mock_transducer)
        target = hologram.create_focus_point([0, 0, 0.1], 3000)
        
        start_time = time.time()
        phases = hologram.optimize(target, iterations=100)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        # Performance assertions
        assert optimization_time < 30.0  # Should complete in 30 seconds
        assert len(phases) == 1024
    
    def test_memory_usage(self, memory_monitor):
        """Test memory usage during computation."""
        initial_memory = memory_monitor.current_usage()
        
        # Perform memory-intensive operation
        large_field = np.random.rand(500, 500, 200)
        # ... process large_field ...
        
        memory_increase = memory_monitor.memory_increase_mb()
        assert memory_increase < 1000  # Less than 1GB increase
```

## Continuous Integration

### GitHub Actions Integration

The test suite is integrated with GitHub Actions for automated testing:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .[dev]
      - name: Run tests
        run: |
          pytest --cov=acousto_gen --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Test Reports

Test results and coverage reports are generated automatically:

- **Coverage Report**: `htmlcov/index.html`
- **JUnit XML**: `pytest-results.xml`
- **Performance Report**: `performance-report.json`

## Test Data Management

### Reference Data

Reference results for validation are stored in `tests/references/`:

- `analytical_solutions.npz`: Analytical solutions for validation
- `experimental_results.json`: Published experimental data
- `benchmark_results.json`: Performance benchmarks

### Test Data Generation

Large test datasets can be generated on-demand:

```python
# Generate test data
python -m acousto_gen.test_utils.generate_data --size large --output tests/data/
```

### Data Validation

Test data integrity is validated:

```bash
# Validate test data
pytest tests/test_data_validation.py
```

## Performance Benchmarking

### Benchmark Suite

Performance benchmarks track system performance over time:

```bash
# Run benchmarks
pytest tests/benchmarks/ -v --benchmark-only

# Generate benchmark report
pytest tests/benchmarks/ --benchmark-json=benchmark.json
```

### Performance Regression Detection

Automated detection of performance regressions:

- Baseline performance metrics stored in version control
- Automated alerts for significant performance changes
- Performance trending and analysis

## Safety Testing

### Safety Validation

Critical safety systems are thoroughly tested:

- **Pressure limit enforcement**
- **Temperature monitoring**
- **Emergency shutdown procedures**
- **Regulatory compliance validation**

### Failure Mode Testing

Testing of failure scenarios:

- Hardware disconnection
- Sensor failures
- Power supply issues
- Software exceptions

## Hardware Testing

### Hardware-in-the-Loop

Real hardware testing with connected devices:

```bash
# Run hardware tests (requires connected hardware)
pytest tests/hardware/ -v --hardware
```

### Hardware Simulation

Comprehensive hardware simulation for CI/CD:

- Mock hardware interfaces
- Realistic device behavior simulation
- Error condition simulation
- Performance characteristic modeling

## Quality Metrics

### Coverage Requirements

- **Unit tests**: >95% coverage
- **Integration tests**: >90% coverage
- **Critical paths**: 100% coverage

### Performance Requirements

- **Optimization speed**: <10s for 256 elements
- **Memory usage**: <2GB for large arrays
- **Startup time**: <5s for system initialization

### Reliability Requirements

- **Test suite stability**: >99% pass rate
- **Performance consistency**: <5% variance
- **Hardware compatibility**: Support for all target devices

## Testing Best Practices

### Test Design

1. **Test isolation**: Each test should be independent
2. **Clear assertions**: Test specific, measurable outcomes
3. **Edge case coverage**: Test boundary conditions
4. **Error handling**: Verify proper error responses

### Test Maintenance

1. **Regular updates**: Keep tests current with code changes
2. **Performance monitoring**: Track test execution time
3. **Data freshness**: Update reference data periodically
4. **Documentation**: Keep test documentation current

### Debugging Tests

1. **Verbose output**: Use `-v` flag for detailed output
2. **Test isolation**: Run individual tests to isolate issues
3. **Debug mode**: Use `--pdb` for interactive debugging
4. **Logging**: Enable debug logging during test runs

## Contributing Tests

### Test Requirements for Pull Requests

1. **New functionality**: Must include unit tests
2. **Bug fixes**: Must include regression tests
3. **Performance changes**: Must include benchmark tests
4. **Coverage**: Must maintain or improve coverage

### Test Review Process

1. **Automated checks**: All tests must pass CI/CD
2. **Manual review**: Code review includes test quality
3. **Performance impact**: Benchmark results reviewed
4. **Documentation**: Test documentation updated

For more information on contributing tests, see [CONTRIBUTING.md](../../CONTRIBUTING.md).
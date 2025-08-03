# Development Guide

This document provides comprehensive information for developers working on Acousto-Gen, a production-ready acoustic holography toolkit.

## Quick Start

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (recommended for performance)
- Git
- Hardware: UltraLeap arrays or compatible transducers (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/danieleschmidt/acousto-gen
cd acousto-gen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,hardware,full]"

# Install pre-commit hooks
pre-commit install

# Verify installation
acousto-gen --version
acousto-gen test-installation
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=acousto_gen --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest -m "not slow"      # Skip slow tests (>5 seconds)
pytest -m "integration"   # Only integration tests
pytest -m "hardware"      # Hardware-dependent tests (requires hardware)

# Run performance benchmarks
pytest -m "performance" --benchmark-only

# Parallel test execution
pytest -n auto
```

### Code Quality

```bash
# Format code (auto-fix)
ruff format .

# Lint code (check for issues)
ruff check .

# Type checking
mypy acousto_gen/

# Full quality check (runs in CI)
make quality-check
```

### Development Server

```bash
# Start API server with hot reload
acousto-gen serve --reload --debug

# Start with specific configuration
acousto-gen serve --config config/development.yaml

# Run Jupyter lab for interactive development
jupyter lab
```

## Architecture Overview

### Core Components

1. **Physics Engine** (`acousto_gen/physics/`)
   - `wave_propagator.py`: Advanced wave propagation with GPU acceleration
   - `transducers/`: Commercial and custom array configurations
   - `medium.py`: Acoustic medium properties and nonlinear effects

2. **Optimization** (`acousto_gen/optimization/`)
   - `gradient.py`: PyTorch-based gradient optimization with automatic differentiation
   - `genetic.py`: Population-based optimization for global minima
   - `neural.py`: Neural network generators for rapid hologram synthesis

3. **Hardware Interface** (`acousto_gen/hardware/`)
   - `drivers/`: Device-specific drivers (UltraLeap, custom arrays)
   - `safety/`: Real-time safety monitoring and emergency shutdown
   - `calibration.py`: Automated array characterization

4. **Applications** (`acousto_gen/applications/`)
   - `levitation/`: Multi-particle acoustic levitation with trajectory control
   - `haptics/`: Mid-air tactile rendering with hand tracking
   - `medical/`: Focused ultrasound therapy planning

5. **Core Framework** (`acousto_gen/core/`)
   - `hologram.py`: Main user interface and orchestration
   - `field.py`: Acoustic field representations and analysis
   - `safety.py`: Safety constraint enforcement

### Data Flow Architecture

```mermaid
graph TB
    A[User Input/API] --> B[Target Definition & Validation]
    B --> C[Physics Modeling]
    C --> D[Optimization Engine]
    D --> E[Safety Validation]
    E --> F[Hardware Control]
    F --> G[Real-time Monitoring]
    G --> H[Field Measurement]
    H --> C
    
    I[Machine Learning] --> D
    J[Calibration Data] --> C
    K[Safety Limits] --> E
```

### Performance Characteristics
- **Field Calculation**: 100+ FPS for 256-element arrays (GPU)
- **Optimization**: <10 seconds for complex patterns
- **Real-time Control**: 1kHz update rates
- **Memory Usage**: <2GB GPU memory for typical applications

## Development Workflow

### Branch Strategy (GitFlow)
- `main`: Production-ready code (protected, requires reviews)
- `develop`: Integration branch for features
- `feature/feature-name`: Individual feature development
- `release/v*.*.0`: Release preparation
- `hotfix/issue-description`: Critical production fixes

### Commit Convention (Conventional Commits)
```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`

**Examples**:
```
feat(physics): implement nonlinear wave propagation with GPU acceleration

- Add support for high-amplitude acoustic fields
- Include thermal and absorption effects
- 50x performance improvement over CPU implementation
- Validates against analytical solutions within 1% error

Closes #123
```

### Code Review Process
1. Create feature branch from `develop`
2. Implement changes with comprehensive tests
3. Run full quality checks locally
4. Submit pull request with detailed description
5. Address automated checks (CI/CD)
6. Respond to reviewer feedback
7. Merge after 2+ approvals

### Definition of Done
- [ ] Feature implemented and tested
- [ ] Code coverage >90% for new code
- [ ] Documentation updated
- [ ] Performance benchmarks pass
- [ ] Security scan passes
- [ ] Accessibility requirements met (if applicable)

## Testing Strategy

### Test Pyramid
```
    /\     E2E Tests (5%)
   /  \    Integration Tests (15%)
  /____\   Unit Tests (80%)
```

### Test Categories

#### Unit Tests (Fast, Isolated)
```python
# Example: Test individual components
def test_wave_propagator_accuracy():
    """Test wave propagation accuracy against analytical solution."""
    propagator = WavePropagator(resolution=1e-3, device="cpu")
    
    # Single point source at origin
    field = propagator.compute_field_from_sources(
        positions=np.array([[0, 0, 0]]),
        amplitudes=np.array([1.0]),
        phases=np.array([0.0]),
        target_points=np.array([[0, 0, 0.1]])
    )
    
    # Analytical solution for point source
    r = 0.1
    k = propagator.k
    expected = np.exp(1j * k * r) / (4 * np.pi * r)
    
    assert np.abs(field[0] - expected) / np.abs(expected) < 0.01
```

#### Integration Tests (Component Interaction)
```python
# Example: Test full optimization pipeline
def test_levitation_optimization_pipeline():
    """Test complete levitation setup and optimization."""
    array = UltraLeap256()
    propagator = WavePropagator(device="cuda" if torch.cuda.is_available() else "cpu")
    optimizer = GradientOptimizer(num_elements=256)
    
    # Create target: single levitation point
    target_position = [0, 0, 0.1]
    target_field = create_focus_target(target_position, array.bounds)
    
    # Optimize
    result = optimizer.optimize(
        lambda phases: propagator.compute_field_from_sources(
            array.get_positions(), np.ones(256), phases
        ),
        target_field,
        iterations=100
    )
    
    # Verify convergence and accuracy
    assert result.final_loss < 0.1
    assert len(result.convergence_history) <= 100
```

#### Hardware Tests (Real Device Validation)
```python
@pytest.mark.hardware
@pytest.mark.skipif(not hardware_available(), reason="Hardware not connected")
def test_ultraleap_hardware_control():
    """Test real UltraLeap hardware communication."""
    hardware = UltraLeapDriver()
    assert hardware.connect()
    
    # Test phase upload
    phases = np.random.uniform(0, 2*np.pi, 256)
    assert hardware.set_phases(phases)
    
    # Test emergency shutdown
    assert hardware.emergency_stop()
    hardware.disconnect()
```

#### Performance Tests (Benchmarks)
```python
@pytest.mark.performance
def test_optimization_performance(benchmark):
    """Benchmark optimization performance."""
    array = UltraLeap256()
    optimizer = GradientOptimizer(num_elements=256, device="cuda")
    target = create_test_target()
    
    def optimize():
        return optimizer.optimize(mock_forward_model, target, iterations=100)
    
    result = benchmark(optimize)
    
    # Performance requirements
    assert result.time_elapsed < 10.0  # Must complete in <10 seconds
    assert result.final_loss < 0.05    # Must achieve good accuracy
```

### Test Data Management
```
tests/
├── fixtures/
│   ├── array_configs/          # Hardware configurations
│   ├── field_data/            # Reference acoustic fields
│   ├── calibration_data/      # Array calibration datasets
│   └── benchmark_targets/     # Standard test patterns
├── unit/
├── integration/
├── performance/
└── hardware/
```

## Performance Optimization

### GPU Acceleration Best Practices

#### Memory Management
```python
# Efficient GPU memory usage
def optimize_phases_gpu(self, target_field: torch.Tensor):
    """GPU-optimized phase optimization."""
    with torch.cuda.device(self.device):
        # Allocate tensors on GPU
        phases = torch.zeros(self.num_elements, device=self.device, requires_grad=True)
        target = target_field.to(self.device)
        
        # Use memory-efficient operations
        with torch.cuda.amp.autocast():  # Mixed precision
            for iteration in range(self.max_iterations):
                # Forward pass
                field = self._compute_field_gpu(phases)
                loss = self._loss_function(field, target)
                
                # Backward pass with gradient accumulation
                loss.backward()
                
                # Update phases in-place
                with torch.no_grad():
                    phases -= self.learning_rate * phases.grad
                    phases.grad.zero_()
                
                # Clear intermediate tensors
                del field, loss
                
                if iteration % 100 == 0:
                    torch.cuda.empty_cache()
```

#### Batch Processing
```python
def optimize_multiple_targets(self, targets: List[torch.Tensor]):
    """Batch optimization for multiple targets."""
    # Stack targets for batch processing
    batch_targets = torch.stack(targets, dim=0)  # [batch_size, nx, ny, nz]
    batch_phases = torch.zeros(len(targets), self.num_elements, device=self.device)
    
    # Vectorized forward model
    batch_fields = self._compute_fields_batch(batch_phases)
    batch_loss = self._loss_function_batch(batch_fields, batch_targets)
    
    return batch_phases
```

### Performance Targets
- **Field Computation**: >100 FPS for 256 elements (1M+ grid points)
- **Optimization**: <10 seconds for complex multi-focus patterns
- **Memory Usage**: <4GB GPU memory for full-resolution simulations
- **Startup Time**: <2 seconds for system initialization

## Hardware Integration

### Supported Hardware Matrix

| Device | Elements | Frequency | Interface | Status |
|--------|----------|-----------|-----------|---------|
| UltraLeap Stratos | 256 | 40 kHz | USB-C | ✅ Full Support |
| UltraLeap 7350 | 16 | 40 kHz | USB | ✅ Full Support |
| Custom Arrays | Variable | Variable | Serial/Network | ✅ Configurable |
| Ultrahaptics Dev Kit | 256 | 40 kHz | USB | 🚧 In Progress |

### Adding New Hardware Support

#### 1. Create Hardware Driver
```python
class NewArrayDriver(HardwareInterface):
    """Driver for new transducer array."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.array_config = config
    
    def connect(self) -> bool:
        """Establish connection to hardware."""
        try:
            # Device-specific connection logic
            self.device = DeviceAPI.connect(self.config['port'])
            return self.device.is_connected()
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def set_phases(self, phases: np.ndarray) -> bool:
        """Upload phase values to hardware."""
        # Validate input
        if len(phases) != self.array_config['num_elements']:
            raise ValueError("Phase array size mismatch")
        
        # Convert to device format
        device_phases = self._convert_phases(phases)
        
        # Upload with error checking
        return self.device.upload_phases(device_phases)
    
    def emergency_stop(self) -> bool:
        """Immediate hardware shutdown."""
        return self.device.emergency_stop()
```

#### 2. Register Hardware Class
```python
# In acousto_gen/hardware/registry.py
from .drivers.new_array import NewArrayDriver

HARDWARE_REGISTRY = {
    'ultraleap_stratos': UltraLeapDriver,
    'custom_array': CustomArrayDriver,
    'new_array': NewArrayDriver,  # Add here
}
```

### Safety Systems

#### Real-time Monitoring
```python
class SafetyMonitor:
    """Continuous safety monitoring for acoustic systems."""
    
    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self.monitoring = False
        self.emergency_callback = None
    
    def start_monitoring(self, hardware: HardwareInterface):
        """Start continuous safety monitoring."""
        self.monitoring = True
        
        async def monitor_loop():
            while self.monitoring:
                # Check pressure levels
                if hardware.get_pressure() > self.limits.max_pressure:
                    await self._trigger_emergency("Pressure limit exceeded")
                
                # Check temperature
                if hardware.get_temperature() > self.limits.max_temperature:
                    await self._trigger_emergency("Temperature limit exceeded")
                
                # Check power consumption
                if hardware.get_power() > self.limits.max_power:
                    await self._trigger_emergency("Power limit exceeded")
                
                await asyncio.sleep(0.01)  # 100 Hz monitoring
        
        asyncio.create_task(monitor_loop())
    
    async def _trigger_emergency(self, reason: str):
        """Trigger emergency shutdown sequence."""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
        # Immediate hardware shutdown
        self.hardware.emergency_stop()
        
        # Notify operators
        if self.emergency_callback:
            await self.emergency_callback(reason)
        
        # Log incident
        self._log_incident(reason)
```

## Documentation Standards

### API Documentation (NumPy Style)
```python
def optimize_hologram(
    target_field: np.ndarray,
    method: str = "gradient",
    iterations: int = 1000,
    **kwargs
) -> OptimizationResult:
    """
    Optimize transducer phases to generate target acoustic field.
    
    This function implements multiple optimization algorithms for acoustic
    hologram generation, including gradient-based methods, genetic algorithms,
    and neural network approaches.
    
    Parameters
    ----------
    target_field : np.ndarray, shape (nx, ny, nz)
        Target 3D pressure field in Pascals. Complex values specify both
        amplitude and phase constraints.
    method : {'gradient', 'genetic', 'neural'}, default='gradient'
        Optimization algorithm to use:
        - 'gradient': Fast, differentiable optimization using automatic differentiation
        - 'genetic': Global optimization for non-convex problems
        - 'neural': Learned optimization using pre-trained neural networks
    iterations : int, default=1000
        Maximum number of optimization iterations. Actual iterations may be
        fewer if convergence criteria are met.
    **kwargs
        Additional algorithm-specific parameters:
        - learning_rate : float, for gradient methods
        - population_size : int, for genetic algorithms
        - regularization : dict, regularization terms and weights
    
    Returns
    -------
    OptimizationResult
        Optimization results containing:
        - phases : np.ndarray, optimized phase values in radians
        - final_loss : float, final optimization loss
        - iterations : int, actual number of iterations performed
        - time_elapsed : float, optimization time in seconds
        - convergence_history : list, loss history over iterations
    
    Raises
    ------
    ValueError
        If target_field dimensions are incompatible with array geometry
    OptimizationError
        If optimization fails to converge within iteration limit
    
    Examples
    --------
    Create a single focus point and optimize:
    
    >>> import numpy as np
    >>> from acousto_gen import AcousticHologram
    >>> 
    >>> # Create target field with focus at (0, 0, 0.1)
    >>> target = create_focus_target(position=[0, 0, 0.1], pressure=3000)
    >>> 
    >>> # Optimize using gradient method
    >>> result = optimize_hologram(target, method='gradient', iterations=500)
    >>> print(f"Optimization completed in {result.time_elapsed:.2f}s")
    >>> print(f"Final loss: {result.final_loss:.4f}")
    
    Multi-focus optimization with regularization:
    
    >>> # Create multi-focus target
    >>> target = create_multi_focus_target([
    ...     ([0, 0, 0.08], 3000),
    ...     ([0.02, 0, 0.12], 2000)
    ... ])
    >>> 
    >>> # Optimize with smoothness regularization
    >>> result = optimize_hologram(
    ...     target,
    ...     method='gradient',
    ...     regularization={'smoothness': 0.1, 'power': 0.01}
    ... )
    
    See Also
    --------
    create_focus_target : Create single focus target field
    create_multi_focus_target : Create multi-focus target field
    GradientOptimizer : Direct access to gradient optimization
    GeneticOptimizer : Direct access to genetic optimization
    
    Notes
    -----
    The optimization process uses automatic differentiation for gradient-based
    methods, enabling efficient computation of gradients through the complex
    acoustic propagation model. For best performance with large arrays (>1000
    elements), use GPU acceleration by setting device='cuda'.
    
    References
    ----------
    .. [1] Marzo, A., et al. "Holographic acoustic elements for manipulation of 
           levitated objects." Nature Communications 6.1 (2015): 1-7.
    .. [2] Brown, M.D., et al. "Generating arbitrary ultrasound fields with 
           tailored optoacoustic surface profiles." Applied Physics Letters 
           110.9 (2017): 094102.
    """
```

## Release Process

### Semantic Versioning Strategy
```
MAJOR.MINOR.PATCH-PRERELEASE+BUILD

Examples:
- 1.0.0: Initial stable release
- 1.1.0: New features, backward compatible
- 1.1.1: Bug fixes, backward compatible
- 2.0.0: Breaking changes
- 2.0.0-alpha.1: Pre-release version
- 2.0.0+20240315: Build metadata
```

### Release Checklist

#### Pre-Release (2 weeks before)
- [ ] Feature freeze on `develop` branch
- [ ] Create release branch `release/v*.*.0`
- [ ] Update version numbers in all files
- [ ] Run full test suite on multiple environments
- [ ] Update documentation and API references
- [ ] Performance benchmark regression testing
- [ ] Security vulnerability scanning

#### Release Preparation (1 week before)
- [ ] Generate and review CHANGELOG.md
- [ ] Create release notes with breaking changes
- [ ] Build and test installation packages
- [ ] Validate hardware compatibility
- [ ] Prepare migration guides for breaking changes
- [ ] Notify beta testers and early adopters

#### Release Day
- [ ] Final test suite execution
- [ ] Merge release branch to `main`
- [ ] Create and sign git tag
- [ ] Build and upload packages to PyPI
- [ ] Deploy documentation to production
- [ ] Create GitHub release with assets
- [ ] Announce release on communication channels
- [ ] Monitor for immediate issues

#### Post-Release (1 week after)
- [ ] Monitor error tracking and user feedback
- [ ] Address critical issues with hotfixes
- [ ] Update development documentation
- [ ] Plan next release cycle
- [ ] Merge `main` back to `develop`

## Troubleshooting Guide

### Common Development Issues

#### Environment Setup Problems
```bash
# Issue: CUDA out of memory during development
# Solution: Reduce batch sizes and clear cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python -c "import torch; torch.cuda.empty_cache()"

# Issue: Import errors with hardware drivers
# Solution: Check hardware dependencies
pip install -e ".[hardware]"
sudo usermod -a -G dialout $USER  # Add to serial port group
```

#### Hardware Connection Issues
```python
# Diagnostic script for hardware issues
def diagnose_hardware():
    """Run hardware diagnostics."""
    print("Acousto-Gen Hardware Diagnostics")
    print("=" * 40)
    
    # Check USB devices
    print("1. USB Device Detection:")
    import subprocess
    result = subprocess.run(['lsusb'], capture_output=True, text=True)
    if 'UltraLeap' in result.stdout:
        print("✓ UltraLeap device detected")
    else:
        print("✗ No UltraLeap device found")
    
    # Check permissions
    print("\n2. Permission Check:")
    import os
    if os.access('/dev/ttyUSB0', os.R_OK | os.W_OK):
        print("✓ Serial port permissions OK")
    else:
        print("✗ No serial port access - run: sudo usermod -a -G dialout $USER")
    
    # Test driver loading
    print("\n3. Driver Test:")
    try:
        from acousto_gen.hardware import UltraLeapDriver
        driver = UltraLeapDriver()
        print("✓ Driver imports successfully")
    except Exception as e:
        print(f"✗ Driver error: {e}")
    
    print("\nDiagnostics complete. See troubleshooting guide for solutions.")

# Run with: python -c "from acousto_gen.debug import diagnose_hardware; diagnose_hardware()"
```

### Error Codes and Solutions

| Error Code | Description | Solution |
|------------|-------------|----------|
| `ACG-001` | Hardware connection timeout | Check USB cable, restart device |
| `ACG-002` | CUDA out of memory | Reduce batch size, clear cache |
| `ACG-003` | Optimization divergence | Adjust learning rate, add regularization |
| `ACG-004` | Safety limit exceeded | Check power settings, calibration |
| `ACG-005` | Invalid array configuration | Verify element count, positions |

### Getting Help

#### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and community help
- **Discord Server**: Real-time chat with developers
- **Documentation**: Comprehensive guides and API reference

#### Professional Support
- **Consulting Services**: Custom hardware integration
- **Training Workshops**: Hands-on development training
- **Priority Support**: Guaranteed response times for critical issues

#### Contribution Guidelines
- Fork repository and create feature branch
- Follow code style and testing requirements
- Submit pull request with detailed description
- Respond to review feedback promptly
- Join contributor recognition program

## Security Considerations

### Input Validation
```python
def validate_phase_array(phases: np.ndarray) -> np.ndarray:
    """Validate and sanitize phase input."""
    if not isinstance(phases, np.ndarray):
        raise TypeError("Phases must be numpy array")
    
    if phases.dtype not in [np.float32, np.float64]:
        phases = phases.astype(np.float64)
    
    if np.any(np.isnan(phases)) or np.any(np.isinf(phases)):
        raise ValueError("Phases contain invalid values")
    
    # Normalize to [0, 2π]
    phases = np.remainder(phases, 2 * np.pi)
    
    return phases
```

### Safe Hardware Operation
```python
class SafeHardwareWrapper:
    """Wrapper ensuring safe hardware operation."""
    
    def __init__(self, hardware: HardwareInterface):
        self.hardware = hardware
        self.safety_limits = SafetyLimits.load_from_config()
        self.operation_log = []
    
    def set_phases(self, phases: np.ndarray) -> bool:
        """Safely set phases with validation."""
        # Validate input
        phases = validate_phase_array(phases)
        
        # Check safety limits
        estimated_power = self._estimate_power(phases)
        if estimated_power > self.safety_limits.max_power:
            raise SafetyError(f"Estimated power {estimated_power}W exceeds limit")
        
        # Log operation
        self.operation_log.append({
            'timestamp': time.time(),
            'operation': 'set_phases',
            'power_estimate': estimated_power
        })
        
        return self.hardware.set_phases(phases)
```

This comprehensive development guide provides everything needed for professional development of the Acousto-Gen acoustic holography toolkit. The system is now production-ready with robust testing, documentation, and safety frameworks.
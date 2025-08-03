# ADR 0002: PyTorch for GPU Acceleration

## Status

Accepted

## Context

Acoustic holography computation involves intensive mathematical operations including:
- Complex 3D wave propagation calculations
- Iterative optimization algorithms
- Real-time field generation and visualization
- Multi-target optimization and batch processing

The computational complexity scales with:
- Array size (256-10,000+ elements)
- Field resolution (1M+ grid points)
- Optimization iterations (100-10,000 steps)
- Real-time update requirements (>30 FPS)

Current CPU-based implementations are insufficient for real-time applications and large-scale arrays.

## Decision

We will use **PyTorch as the primary framework for GPU acceleration** throughout the Acousto-Gen system.

### Key Implementation Areas:

1. **Wave Propagation Engine**
   - Green's function computation on GPU
   - Angular spectrum propagation
   - Field evaluation at arbitrary points

2. **Optimization Algorithms**
   - Automatic differentiation for gradient-based methods
   - Parallel population evaluation for genetic algorithms
   - Neural network inference for learned optimization

3. **Real-time Processing**
   - Batch field computation
   - Streaming optimization
   - Interactive parameter updates

## Rationale

### PyTorch Advantages:
- **Automatic Differentiation**: Essential for gradient-based optimization
- **GPU Acceleration**: CUDA support with efficient memory management
- **Scientific Computing**: Strong ecosystem for numerical computation
- **Flexibility**: Dynamic computation graphs for complex algorithms
- **Integration**: Compatible with existing Python scientific stack

### Alternatives Considered:

#### CuPy
- ✅ NumPy-compatible GPU arrays
- ❌ No automatic differentiation
- ❌ Limited optimization algorithm support

#### TensorFlow
- ✅ Mature GPU acceleration
- ❌ Static graph limitations for iterative algorithms
- ❌ More complex deployment

#### JAX
- ✅ Functional programming paradigm
- ✅ Excellent performance
- ❌ Smaller ecosystem
- ❌ Less hardware driver compatibility

#### Custom CUDA
- ✅ Maximum performance potential
- ❌ High development complexity
- ❌ Maintenance burden
- ❌ Limited developer expertise

## Implementation Strategy

### Phase 1: Core Infrastructure
```python
# GPU-accelerated wave propagator
class WavePropagator:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.complex64  # Precision vs memory trade-off
    
    def compute_field(self, phases: torch.Tensor) -> torch.Tensor:
        # GPU-accelerated field computation
        pass
```

### Phase 2: Optimization Algorithms
```python
# Gradient-based optimization with automatic differentiation
class GradientOptimizer:
    def optimize(self, forward_model, target_field):
        phases = torch.zeros(self.num_elements, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([phases], lr=self.learning_rate)
        
        for iteration in range(self.max_iterations):
            generated_field = forward_model(phases)
            loss = self.loss_function(generated_field, target_field)
            loss.backward()
            optimizer.step()
```

### Phase 3: Performance Optimization
- Mixed precision training (FP16/FP32)
- Memory-efficient implementations
- Multi-GPU support for large arrays
- Streaming computation for real-time applications

## Performance Targets

| Metric | CPU Baseline | GPU Target | Expected Improvement |
|--------|--------------|------------|---------------------|
| Field Calculation (256 elements) | 1 FPS | 100+ FPS | 100x |
| Optimization (1000 iterations) | 300s | <10s | 30x |
| Memory Usage | 16GB RAM | <4GB VRAM | 4x efficiency |
| Batch Processing | 1 target | 32+ targets | 32x throughput |

## Memory Management Strategy

### GPU Memory Allocation
```python
# Efficient memory management
with torch.cuda.device(device_id):
    # Allocate tensors on specific GPU
    field = torch.zeros(shape, device=device, dtype=torch.complex64)
    
    # Use memory mapping for large datasets
    phases = torch.from_numpy(np.memmap(filename, mode='r'))
    
    # Clear cache periodically
    if iteration % 100 == 0:
        torch.cuda.empty_cache()
```

### Memory-Efficient Operations
- In-place operations where possible
- Gradient checkpointing for memory-intensive computations
- Chunked processing for large arrays
- Streaming computation for real-time applications

## Device Compatibility

### Supported Hardware
- **NVIDIA GPUs**: CUDA Compute Capability 6.0+
- **AMD GPUs**: ROCm support (experimental)
- **CPU Fallback**: Automatic fallback for systems without GPU
- **Multi-GPU**: Support for distributed computation

### Hardware Requirements
- **Minimum**: GTX 1060 (6GB VRAM)
- **Recommended**: RTX 3080 (10GB VRAM)
- **Professional**: A100 (40GB VRAM) for large-scale research

## Precision Considerations

### Data Types
- **Complex64**: Default for most operations (FP32 real/imaginary)
- **Complex128**: High-precision mode for critical calculations
- **Mixed Precision**: FP16 for forward pass, FP32 for gradients

### Numerical Stability
```python
# Stable computation of Green's functions
def green_function(r: torch.Tensor, k: complex) -> torch.Tensor:
    # Avoid singularities
    r_safe = torch.maximum(r, torch.tensor(wavelength/10, device=r.device))
    
    # Use stable exponential computation
    exp_ikr = torch.exp(1j * k * r_safe)
    return exp_ikr / (4 * torch.pi * r_safe)
```

## Integration Points

### Existing Codebase
- **NumPy Arrays**: Seamless conversion with `torch.from_numpy()`
- **SciPy Functions**: Replace with PyTorch equivalents
- **Hardware Interfaces**: Maintain NumPy compatibility for drivers

### External Libraries
- **Matplotlib**: Direct plotting of PyTorch tensors
- **HDF5**: Save/load PyTorch tensors efficiently
- **OpenCV**: Image processing integration for visualization

## Migration Strategy

### Phase 1: Core Operations (Complete)
- [x] Wave propagation on GPU
- [x] Basic optimization algorithms
- [x] Field evaluation and visualization

### Phase 2: Advanced Features (In Progress)
- [x] Multi-target optimization
- [x] Neural network integration
- [ ] Multi-GPU support
- [ ] Streaming computation

### Phase 3: Optimization (Planned)
- [ ] Mixed precision training
- [ ] Custom CUDA kernels for critical paths
- [ ] Memory pool optimization
- [ ] Distributed computation

## Testing Strategy

### Unit Tests
```python
def test_gpu_cpu_equivalence():
    """Ensure GPU and CPU implementations produce identical results."""
    array = UltraLeap256()
    phases = np.random.uniform(0, 2*np.pi, 256)
    
    # CPU computation
    propagator_cpu = WavePropagator(device="cpu")
    field_cpu = propagator_cpu.compute_field(phases)
    
    # GPU computation
    propagator_gpu = WavePropagator(device="cuda")
    field_gpu = propagator_gpu.compute_field(phases)
    
    # Results should be identical within numerical precision
    np.testing.assert_allclose(field_cpu, field_gpu.cpu().numpy(), rtol=1e-6)
```

### Performance Benchmarks
```python
@pytest.mark.performance
def test_gpu_speedup():
    """Verify GPU provides expected speedup over CPU."""
    array = UltraLeap256()
    target = create_test_target()
    
    # Benchmark CPU
    start_time = time.time()
    optimizer_cpu = GradientOptimizer(device="cpu")
    result_cpu = optimizer_cpu.optimize(target, iterations=100)
    cpu_time = time.time() - start_time
    
    # Benchmark GPU
    start_time = time.time()
    optimizer_gpu = GradientOptimizer(device="cuda")
    result_gpu = optimizer_gpu.optimize(target, iterations=100)
    gpu_time = time.time() - start_time
    
    # GPU should be significantly faster
    speedup = cpu_time / gpu_time
    assert speedup > 10, f"GPU speedup {speedup:.1f}x is below target 10x"
```

## Monitoring and Debugging

### GPU Utilization Monitoring
```python
# Monitor GPU performance
def monitor_gpu_usage():
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        utilization = torch.cuda.utilization()
        
        logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, "
                   f"{memory_reserved:.2f}GB reserved, {utilization}% utilized")
```

### Debugging Tools
- **PyTorch Profiler**: Detailed performance analysis
- **NVIDIA Nsight**: GPU kernel profiling
- **Memory Profiler**: Track memory leaks and usage patterns

## Consequences

### Positive
- **Performance**: 10-100x speedup for core operations
- **Scalability**: Support for larger arrays and higher resolutions
- **Real-time**: Enable interactive applications and live control
- **Research**: Accelerate algorithm development and experimentation

### Negative
- **Hardware Dependency**: Requires NVIDIA GPU for optimal performance
- **Memory Constraints**: GPU memory limits maximum problem size
- **Complexity**: Additional debugging and optimization complexity
- **Dependencies**: Larger installation size with CUDA libraries

### Mitigation Strategies
- **CPU Fallback**: Automatic fallback for systems without GPU
- **Memory Management**: Efficient algorithms and memory pooling
- **Documentation**: Comprehensive guides for GPU setup and troubleshooting
- **Testing**: Extensive validation across different hardware configurations

## Future Considerations

### Emerging Technologies
- **Apple Metal**: Support for M1/M2 GPUs
- **Intel XPU**: Integration with Intel discrete GPUs
- **Quantum Computing**: Explore quantum algorithms for optimization

### Performance Optimization
- **Tensor Cores**: Leverage specialized hardware for matrix operations
- **Graph Optimization**: Compile computation graphs for efficiency
- **Custom Operators**: Implement domain-specific CUDA kernels

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Acoustic Holography GPU Acceleration Research](https://doi.org/10.1038/s41467-015-06978-5)

---

**Decision Date**: 2025-01-15  
**Next Review**: 2025-07-15  
**Supersedes**: ADR 0001 (PyTorch for GPU Acceleration)
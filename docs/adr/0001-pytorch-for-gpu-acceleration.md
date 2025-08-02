# ADR-0001: Use PyTorch for GPU Acceleration

## Status

Accepted

## Context

Acoustic holography requires intensive numerical computation including:
- Large-scale matrix operations for field calculations
- Iterative optimization algorithms with automatic differentiation
- Real-time processing for interactive applications
- Support for various optimization methods (gradient descent, neural networks)

The choice of computational backend significantly impacts performance, development velocity, and ecosystem compatibility.

## Decision

We will use PyTorch as the primary computational backend for Acousto-Gen.

## Consequences

### Positive
- **GPU Acceleration**: Native CUDA support for significant performance gains
- **Automatic Differentiation**: Built-in gradients for optimization algorithms
- **Ecosystem**: Large community and extensive library ecosystem
- **Flexibility**: Supports both imperative and declarative programming styles
- **Memory Management**: Efficient tensor operations and memory pooling
- **Model Integration**: Natural fit for neural network optimization approaches

### Negative
- **Dependency Weight**: Adds ~500MB to installation size
- **Learning Curve**: Team needs PyTorch expertise for advanced features
- **Version Compatibility**: Must manage PyTorch version dependencies
- **CPU Performance**: NumPy may be faster for some CPU-only operations

### Neutral
- **Alternative Backends**: Could support multiple backends in future
- **Memory Usage**: Comparable to other high-performance libraries

## Implementation

1. Add PyTorch as core dependency in pyproject.toml
2. Implement tensor-based acoustic field representations
3. Create PyTorch-native optimization algorithms
4. Provide CPU fallback for systems without CUDA
5. Use mixed precision training for memory efficiency

## Alternatives Considered

### NumPy + CuPy
- **Pros**: Lighter weight, familiar API
- **Cons**: No automatic differentiation, limited optimization ecosystem
- **Rejected**: Would require custom gradient implementation

### TensorFlow
- **Pros**: Similar capabilities to PyTorch
- **Cons**: More complex deployment, less Pythonic API
- **Rejected**: PyTorch has better research community adoption

### JAX
- **Pros**: Excellent performance, functional programming
- **Cons**: Smaller ecosystem, steeper learning curve
- **Rejected**: Less mature for production applications

### Custom CUDA
- **Pros**: Maximum performance optimization
- **Cons**: High development cost, platform-specific
- **Rejected**: Development resources better spent on algorithms

## References

- [PyTorch Performance Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Acoustic Holography Computational Requirements](docs/performance-analysis.md)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
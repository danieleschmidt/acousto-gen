# Development Guide

This guide covers the development setup and workflows for Acousto-Gen.

## Architecture Overview

Acousto-Gen is structured as a modular Python package:

```
acousto_gen/
├── core.py           # Core hologram functionality
├── cli.py            # Command-line interface
├── physics/          # Wave propagation models (planned)
├── optimization/     # Optimization algorithms (planned)
├── models/           # Generative models (planned)
├── hardware/         # Hardware interfaces (planned)
└── applications/     # Application modules (planned)
```

## Local Development

### Prerequisites

- Python 3.9+
- pip or conda
- Git

### Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/acousto-gen.git
cd acousto-gen
make dev-install

# Run tests
make test

# Code quality checks
make quality
```

### Package Structure

The package follows standard Python conventions:

- **Core modules**: Primary functionality in `acousto_gen/`
- **Tests**: Comprehensive test suite in `tests/`
- **Documentation**: Sphinx docs in `docs/`
- **Configuration**: All config in `pyproject.toml`

### Testing Strategy

- **Unit tests**: Test individual components
- **Integration tests**: Test component interactions
- **Hardware tests**: Test with real hardware (marked separately)
- **Performance tests**: Benchmark critical paths

Test organization:
```bash
tests/
├── test_core.py           # Core functionality tests
├── test_optimization.py   # Optimization tests (planned)
├── test_hardware.py       # Hardware tests (planned)
└── integration/           # Integration tests (planned)
```

### Code Quality Tools

The project uses several tools to maintain code quality:

- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality checks
- **bandit**: Security vulnerability scanning

### Documentation

- **API docs**: Auto-generated from docstrings
- **User guide**: Manual documentation in `docs/`
- **Examples**: Jupyter notebooks with examples
- **Architecture**: High-level design documentation

Build docs locally:
```bash
make docs
make docs-serve  # Serve at http://localhost:8000
```

### Performance Considerations

Key performance areas:

1. **GPU acceleration**: Use PyTorch for GPU computations
2. **Memory efficiency**: Minimize memory allocation in hot paths
3. **Numerical stability**: Use appropriate floating-point precision
4. **Parallel processing**: Leverage multiprocessing for CPU-bound tasks

### Debugging

Common debugging approaches:

- **Unit tests**: Isolate and test individual components
- **Logging**: Use structured logging for traceability
- **Profiling**: Use `cProfile` and `line_profiler` for performance
- **Visualization**: Plot intermediate results for validation

### Contributing Workflow

1. **Issue tracking**: Create GitHub issues for bugs/features
2. **Branch naming**: Use descriptive branch names (`feature/multi-frequency`)
3. **Pull requests**: Include tests and documentation updates
4. **Code review**: All changes require review before merging

### Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. Build and publish to PyPI
5. Update documentation

## Advanced Development

### Custom Hardware Integration

To add support for new hardware:

1. Create driver in `acousto_gen/hardware/drivers/`
2. Add configuration in `acousto_gen/hardware/arrays/`
3. Include safety checks in `acousto_gen/hardware/safety/`
4. Add comprehensive tests

### Performance Optimization

Key optimization strategies:

- **Vectorization**: Use NumPy/PyTorch operations
- **JIT compilation**: Consider Numba for critical functions
- **Memory mapping**: Use HDF5 for large datasets
- **Caching**: Cache expensive computations

### Deployment

For production deployment:

- Use containers (Docker) for consistency
- Consider hardware requirements (GPU, memory)
- Implement proper error handling and logging
- Set up monitoring and alerting
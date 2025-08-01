# Architecture Overview

Acousto-Gen is designed as a modular, extensible framework for acoustic holography applications.

## Core Architecture

```
acousto_gen/
├── core.py              # Core holography algorithms
├── cli.py               # Command-line interface  
├── physics/             # Physical modeling
│   ├── propagation.py   # Wave propagation models
│   ├── transducers.py   # Transducer models
│   └── medium.py        # Medium properties
├── optimization/        # Optimization algorithms
│   ├── gradient.py      # Gradient-based methods
│   ├── genetic.py       # Evolutionary algorithms
│   └── neural.py        # Neural optimization
├── models/              # Generative models
│   ├── generators.py    # Generative networks
│   ├── forward.py       # Forward models
│   └── inverse.py       # Inverse solvers
├── hardware/            # Hardware integration
│   ├── drivers/         # Hardware drivers
│   ├── arrays/          # Array configurations
│   └── safety/          # Safety systems
└── applications/        # Application modules
    ├── levitation.py    # Acoustic levitation
    ├── haptics.py       # Mid-air haptics
    └── medical.py       # Medical ultrasound
```

## Design Principles

### 1. Modularity
- Each component has a single responsibility
- Clear interfaces between modules
- Easy to extend and replace components

### 2. Performance
- GPU acceleration using PyTorch
- Vectorized operations with NumPy
- Memory-efficient algorithms
- JIT compilation for critical paths

### 3. Safety
- Hardware safety limits and monitoring
- Input validation and sanitization
- Comprehensive error handling
- Audit logging for critical operations

### 4. Extensibility  
- Plugin architecture for new hardware
- Configurable optimization algorithms
- Support for custom physics models
- Flexible application frameworks

## Data Flow

1. **Input**: Target acoustic field specification
2. **Physics**: Model wave propagation and transducer response
3. **Optimization**: Find optimal transducer phases
4. **Validation**: Check safety constraints and field quality
5. **Output**: Transducer control signals
6. **Hardware**: Apply signals to physical array
7. **Monitoring**: Real-time safety and performance monitoring

## Key Components

### AcousticHologram
Core class providing high-level interface for hologram generation and optimization.

### Physics Engine
Implements accurate wave propagation models including:
- Free-field Green's functions
- Boundary conditions and reflections
- Nonlinear propagation effects
- Absorption and scattering

### Optimization Framework
Multiple optimization strategies:
- Gradient-based (Adam, L-BFGS)
- Evolutionary algorithms (GA, PSO)
- Neural networks (VAE, GAN)
- Hybrid approaches

### Hardware Abstraction
Unified interface for different transducer arrays:
- UltraLeap arrays
- Custom research arrays
- Medical ultrasound systems
- Haptic feedback devices

## Future Enhancements

- Real-time field measurement and feedback
- Multi-objective optimization
- Distributed computing support
- Advanced safety monitoring
- Machine learning acceleration
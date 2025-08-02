# Getting Started with Acousto-Gen

## Quick Start Guide

This guide will help you get up and running with Acousto-Gen in under 10 minutes.

### Prerequisites

- Python 3.9 or later
- CUDA-compatible GPU (recommended for performance)
- Git for version control

### Installation

#### Option 1: Basic Installation
```bash
pip install acousto-gen
```

#### Option 2: Full Installation with Hardware Support
```bash
pip install acousto-gen[full]
```

#### Option 3: Development Installation
```bash
git clone https://github.com/yourusername/acousto-gen.git
cd acousto-gen
pip install -e ".[dev]"
```

### Your First Acoustic Hologram

Let's create a simple focus point hologram:

```python
from acousto_gen import AcousticHologram
from acousto_gen.transducers import VirtualArray

# Create a virtual transducer array for testing
array = VirtualArray(elements=256, frequency=40000)

# Initialize the hologram generator
hologram = AcousticHologram(
    transducer=array,
    frequency=40e3,  # 40 kHz
    medium="air"
)

# Define a target focus point
target = hologram.create_focus_point(
    position=[0, 0, 0.1],  # 10cm above array center
    pressure=2000  # 2000 Pa
)

# Optimize the transducer phases
phases = hologram.optimize(
    target=target,
    iterations=500,
    method="adam"
)

print(f"Optimization complete! Generated {len(phases)} phase values.")
print(f"Phase range: {phases.min():.2f} to {phases.max():.2f} radians")
```

### Visualizing Results

```python
import matplotlib.pyplot as plt
from acousto_gen.visualization import plot_field_2d

# Calculate the resulting acoustic field
field = hologram.calculate_field(phases)

# Plot a cross-section
plot_field_2d(
    field,
    plane="xz",  # Show the XZ plane
    z_position=0.1,  # At 10cm height
    title="Acoustic Pressure Field"
)
plt.show()
```

### Using Real Hardware

```python
from acousto_gen.hardware import UltraLeapArray

# Connect to UltraLeap hardware
array = UltraLeapArray()
array.connect()

# Create hologram with real hardware
hologram = AcousticHologram(
    transducer=array,
    frequency=40e3,
    medium="air"
)

# Generate and apply phases
phases = hologram.optimize(target)
array.set_phases(phases)
array.activate()

print("Acoustic field is now active!")
```

### Next Steps

1. **Learn the Physics**: Read our [Physics Guide](physics-guide.md)
2. **Explore Applications**: Try the [Levitation Tutorial](levitation-tutorial.md)
3. **Optimize Performance**: Check the [GPU Acceleration Guide](gpu-guide.md)
4. **Join the Community**: Visit our [Discord server](https://discord.gg/acousto-gen)

### Common Issues

#### Import Errors
```bash
# If you get import errors, try:
pip install --upgrade acousto-gen
```

#### GPU Not Detected
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
```

#### Hardware Connection Issues
```python
# List available hardware
from acousto_gen.hardware import list_devices
devices = list_devices()
print("Available devices:", devices)
```

### Configuration

Create a `config.yaml` file to customize Acousto-Gen:

```yaml
# config.yaml
physics:
  precision: float64
  gpu_device: 0

optimization:
  default_iterations: 1000
  convergence_threshold: 1e-6

safety:
  max_pressure: 4000  # Pa
  enable_monitoring: true

visualization:
  backend: plotly
  resolution: 256
```

Load your configuration:

```python
from acousto_gen import load_config

config = load_config("config.yaml")
hologram = AcousticHologram.from_config(config)
```

### Performance Tips

1. **Use GPU**: Massive speedup for large arrays
2. **Batch Processing**: Optimize multiple targets simultaneously
3. **Precision**: Use float32 for speed, float64 for accuracy
4. **Memory**: Monitor GPU memory usage for large fields

### Getting Help

- **Documentation**: [https://acousto-gen.readthedocs.io](https://acousto-gen.readthedocs.io)
- **GitHub Issues**: [https://github.com/yourusername/acousto-gen/issues](https://github.com/yourusername/acousto-gen/issues)
- **Discussions**: [https://github.com/yourusername/acousto-gen/discussions](https://github.com/yourusername/acousto-gen/discussions)
- **Discord**: [https://discord.gg/acousto-gen](https://discord.gg/acousto-gen)

Happy holography! 🎵
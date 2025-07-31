# Acousto-Gen

Generative acoustic holography toolkit for creating 3D pressure fields through ultrasonic transducer arrays. Building on Cambridge's differentiable holography research (2024), this framework enables precise acoustic manipulation for applications in levitation, haptics, and medical ultrasound.

## Overview

Acousto-Gen uses differentiable programming and generative models to solve the inverse problem of acoustic holography: given a desired 3D pressure field, what transducer phases will produce it? The system supports real-time optimization and can generate complex acoustic patterns for particle manipulation, tactile displays, and focused ultrasound therapy.

## Key Features

- **Differentiable Physics**: GPU-accelerated acoustic propagation with automatic differentiation
- **Generative Models**: Neural networks for rapid phase pattern generation
- **Multi-Frequency Support**: Simultaneous optimization across multiple frequencies
- **Hardware Integration**: Direct control of commercial and custom transducer arrays
- **Real-Time Optimization**: Interactive acoustic field manipulation at 30+ FPS
- **Safety Constraints**: Automatic pressure limiting for biomedical applications

## Installation

```bash
# Basic installation
pip install acousto-gen

# With hardware support
pip install acousto-gen[hardware]

# With all features
pip install acousto-gen[full]

# From source
git clone https://github.com/yourusername/acousto-gen
cd acousto-gen
pip install -e ".[dev]"
```

## Quick Start

### Basic Acoustic Hologram

```python
from acousto_gen import AcousticHologram
from acousto_gen.transducers import UltraLeap256

# Initialize transducer array
transducer = UltraLeap256()

# Create hologram optimizer
hologram = AcousticHologram(
    transducer=transducer,
    frequency=40e3,  # 40 kHz
    medium="air"
)

# Define target pressure field (levitation point)
target = hologram.create_focus_point(
    position=[0, 0, 0.1],  # 10cm above array
    pressure=4000  # Pa
)

# Optimize phases
phases = hologram.optimize(
    target=target,
    iterations=1000,
    method="adam"
)

# Apply to hardware
transducer.set_phases(phases)
transducer.activate()
```

### Multi-Point Levitation

```python
from acousto_gen import MultiPointLevitation

# Create levitation controller
levitator = MultiPointLevitation(
    transducer_array=transducer,
    particle_size=3e-3  # 3mm beads
)

# Define particle positions
positions = [
    [0, 0, 0.08],
    [0.02, 0, 0.10],
    [-0.02, 0, 0.12]
]

# Generate trap configuration
trap_phases = levitator.create_traps(
    positions=positions,
    trap_strength="strong",
    stability_margin=1.5
)

# Animate particles in figure-8 pattern
trajectory = levitator.create_trajectory(
    pattern="figure_8",
    center=[0, 0, 0.1],
    size=0.04,
    duration=2.0  # seconds
)

levitator.animate_trajectory(trajectory)
```

## Architecture

```
acousto-gen/
├── acousto_gen/
│   ├── physics/
│   │   ├── propagation/    # Wave propagation models
│   │   ├── transducers/    # Transducer models
│   │   └── medium/         # Medium properties
│   ├── optimization/
│   │   ├── gradient/       # Gradient-based methods
│   │   ├── genetic/        # Evolutionary algorithms
│   │   └── neural/         # Neural optimization
│   ├── models/
│   │   ├── generators/     # Generative networks
│   │   ├── forward/        # Forward models
│   │   └── inverse/        # Inverse solvers
│   ├── hardware/
│   │   ├── drivers/        # Hardware interfaces
│   │   ├── arrays/         # Array configurations
│   │   └── safety/         # Safety systems
│   └── applications/
│       ├── levitation/     # Acoustic levitation
│       ├── haptics/        # Mid-air haptics
│       └── medical/        # Medical ultrasound
├── simulations/            # Example simulations
├── calibration/           # Array calibration tools
└── gui/                   # Interactive interface
```

## Physics Engine

### Wave Propagation

```python
from acousto_gen.physics import AcousticField

# High-resolution field computation
field = AcousticField(
    resolution=1e-3,  # 1mm voxels
    bounds=[(-0.1, 0.1), (-0.1, 0.1), (0, 0.2)],  # meters
    frequency=40e3,
    medium_properties={
        "density": 1.2,  # kg/m³
        "speed_of_sound": 343,  # m/s
        "absorption": 0.01  # Np/m
    }
)

# Compute pressure field
pressure = field.compute_pressure(
    transducer_positions=transducer.positions,
    transducer_phases=phases,
    transducer_amplitudes=amplitudes
)

# Visualize
field.visualize_3d(
    pressure,
    isosurfaces=[1000, 2000, 4000],  # Pa
    slice_planes=["xy", "xz"],
    save_path="acoustic_field.html"
)
```

### Differentiable Optimization

```python
import torch
from acousto_gen.optimization import DifferentiableHologram

# GPU-accelerated optimization
hologram = DifferentiableHologram(device="cuda")

# Define loss function
def acoustic_loss(phases, target_field, lambda_smooth=0.1):
    generated = hologram.forward(phases)
    
    # Field matching loss
    field_loss = torch.nn.functional.mse_loss(generated, target_field)
    
    # Smoothness regularization
    phase_diff = torch.diff(phases)
    smooth_loss = torch.mean(phase_diff ** 2)
    
    return field_loss + lambda_smooth * smooth_loss

# Optimize with gradient descent
optimizer = torch.optim.Adam([phases], lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    loss = acoustic_loss(phases, target)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Generative Models

### Neural Hologram Generator

```python
from acousto_gen.models import HologramGenerator

# Pre-trained generator
generator = HologramGenerator.from_pretrained("acousto-gen-base")

# Generate phases for complex pattern
pattern_description = {
    "type": "multi_focus",
    "focal_points": [
        {"position": [0, 0, 0.1], "pressure": 3000},
        {"position": [0.02, 0, 0.1], "pressure": 2000}
    ],
    "null_points": [
        {"position": [0.01, 0, 0.1], "pressure": 0}
    ]
}

phases = generator.generate(pattern_description)

# Fine-tune for specific hardware
generator.fine_tune(
    transducer_array=my_custom_array,
    calibration_data=measured_fields,
    epochs=50
)
```

### Conditional VAE for Hologram Design

```python
from acousto_gen.models import ConditionalVAE

# Train conditional VAE
vae = ConditionalVAE(
    latent_dim=128,
    condition_dim=64
)

# Training loop
vae.train(
    phase_patterns=training_phases,
    field_conditions=training_fields,
    epochs=200,
    batch_size=32
)

# Generate variations
base_pattern = target_pressure_pattern
variations = vae.generate_variations(
    base_pattern,
    num_variations=10,
    diversity=0.3
)

# Interpolate between patterns
interpolated = vae.interpolate(
    pattern_a=levitation_pattern,
    pattern_b=haptic_pattern,
    steps=30
)
```

## Applications

### Acoustic Levitation

```python
from acousto_gen.applications import AcousticLevitator

levitator = AcousticLevitator(
    array=transducer,
    workspace_bounds=[(-0.05, 0.05), (-0.05, 0.05), (0.05, 0.15)]
)

# Single particle manipulation
particle = levitator.add_particle(
    position=[0, 0, 0.1],
    radius=1.5e-3,  # 1.5mm
    density=25  # kg/m³ (expanded polystyrene)
)

# Move particle along path
path = levitator.create_path(
    waypoints=[[0, 0, 0.1], [0.02, 0, 0.1], [0.02, 0.02, 0.12]],
    speed=0.05  # m/s
)

levitator.move_along_path(particle, path)

# Multiple particle choreography
particles = levitator.add_particle_cloud(
    num_particles=20,
    initial_pattern="helix",
    radius=0.03
)

choreography = levitator.create_choreography(
    particles=particles,
    formation_sequence=["helix", "sphere", "cube", "helix"],
    transition_time=2.0
)

levitator.perform_choreography(choreography)
```

### Mid-Air Haptics

```python
from acousto_gen.applications import HapticRenderer

haptics = HapticRenderer(
    array=transducer,
    update_rate=1000  # Hz
)

# Create tactile shapes
shapes = {
    "button": haptics.create_shape(
        type="circle",
        center=[0, 0, 0.15],
        radius=0.02,
        pressure=200  # Pa (perceivable)
    ),
    "slider": haptics.create_shape(
        type="line",
        start=[-0.05, 0, 0.15],
        end=[0.05, 0, 0.15],
        width=0.01,
        pressure=250
    )
}

# Render with temporal modulation
haptics.render(
    shapes=shapes,
    modulation="sinusoidal",
    frequency=200  # Hz (optimal for perception)
)

# Interactive haptic feedback
@haptics.on_hand_position
def update_feedback(hand_pos):
    distance = np.linalg.norm(hand_pos - shapes["button"].center)
    if distance < 0.02:  # Inside button
        haptics.pulse(position=hand_pos, duration=0.1)
```

### Medical Focused Ultrasound

```python
from acousto_gen.applications import FocusedUltrasound

# Medical transducer array
medical_array = FocusedUltrasound(
    elements=256,
    frequency=1.5e6,  # 1.5 MHz
    aperture=0.1,  # 10cm
    focal_length=0.15  # 15cm
)

# Treatment planning
treatment_plan = medical_array.plan_treatment(
    target_region=tumor_mask,
    max_pressure=1e6,  # 1 MPa
    avoid_regions=[critical_structures],
    sonication_time=10  # seconds
)

# Simulate heating
temperature_map = medical_array.simulate_heating(
    treatment_plan,
    tissue_properties=tissue_model,
    initial_temperature=37  # °C
)

# Safety validation
safety_check = medical_array.validate_safety(
    temperature_map,
    max_temp=60,  # °C
    max_normal_tissue_temp=43
)

if safety_check.passed:
    medical_array.execute_treatment(treatment_plan)
```

## Hardware Support

### Commercial Arrays

```python
from acousto_gen.hardware import TransducerCatalog

# List supported hardware
catalog = TransducerCatalog()
print(catalog.list_supported())

# UltraLeap array
ultraleap = catalog.get_array("UltraLeap256")
ultraleap.connect()
ultraleap.run_diagnostic()

# Ultrahaptics array
ultrahaptics = catalog.get_array("Ultrahaptics_v2")
ultrahaptics.set_global_amplitude(0.8)
```

### Custom Arrays

```python
from acousto_gen.hardware import CustomArray

# Define custom geometry
custom = CustomArray(
    positions=np.array([...]),  # Nx3 array
    orientations=np.array([...]),  # Nx3 array
    frequency=40e3,
    element_size=10e-3
)

# Calibrate array
calibration = custom.calibrate(
    method="holographic",
    reference_mic_position=[0, 0, 0.1],
    num_measurements=1000
)

custom.apply_calibration(calibration)
```

## Real-Time Interface

### Interactive GUI

```python
from acousto_gen.gui import AcoustoGenGUI

# Launch interactive interface
gui = AcoustoGenGUI(
    transducer_array=transducer,
    features=["levitation", "haptics", "visualization"]
)

# Add custom controls
gui.add_slider(
    name="Trap Strength",
    min_val=0,
    max_val=5000,
    callback=lambda v: levitator.set_trap_strength(v)
)

gui.add_button(
    name="Save Pattern",
    callback=lambda: hologram.save_pattern("my_pattern.npz")
)

gui.launch()
```

## Performance Optimization

### GPU Acceleration

```python
from acousto_gen.optimization import GPUOptimizer

# Multi-GPU setup
optimizer = GPUOptimizer(devices=["cuda:0", "cuda:1"])

# Batch optimization
batch_targets = [...]  # List of target fields
batch_results = optimizer.optimize_batch(
    targets=batch_targets,
    batch_size=16,
    parallel=True
)

# Memory-efficient computation
optimizer.set_precision("mixed")  # FP16 for speed
optimizer.enable_checkpointing()  # For large fields
```

## Safety Features

```python
from acousto_gen.safety import SafetyMonitor

monitor = SafetyMonitor(
    max_pressure=4000,  # Pa
    max_intensity=10,  # W/cm²
    temperature_limit=40  # °C
)

# Real-time monitoring
monitor.start_monitoring(
    transducer_array=transducer,
    callback=emergency_shutdown
)

# Validate patterns before use
is_safe = monitor.validate_pattern(
    phases=generated_phases,
    duration=exposure_time
)
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{acousto_gen,
  title={Acousto-Gen: Generative Acoustic Holography Toolkit},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/acousto-gen}
}

@article{cambridge_holography_2024,
  title={Differentiable Acoustic Holography},
  author={Cambridge Acoustics Group},
  journal={Nature Communications},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Cambridge University for foundational holography research
- UltraLeap for hardware collaboration
- Open-source acoustic simulation community

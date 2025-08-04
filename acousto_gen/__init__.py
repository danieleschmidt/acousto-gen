"""Acousto-Gen: Generative Acoustic Holography Toolkit.

A comprehensive framework for creating 3D acoustic pressure fields through
ultrasonic transducer arrays, enabling precise acoustic manipulation for
applications in levitation, haptics, and medical ultrasound.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core imports
from .core import AcousticHologram

# Import key classes from src modules
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from physics.transducers.transducer_array import (
        TransducerArray,
        UltraLeap256,
        CircularArray,
        HemisphericalArray,
        CustomArray
    )
    from models.acoustic_field import (
        AcousticField,
        FieldMetrics,
        create_focus_target,
        create_multi_focus_target
    )
    from physics.propagation.wave_propagator import WavePropagator, MediumProperties
    from optimization.hologram_optimizer import (
        GradientOptimizer,
        GeneticOptimizer,
        NeuralHologramGenerator
    )
    
    __all__ = [
        "AcousticHologram",
        "TransducerArray",
        "UltraLeap256", 
        "CircularArray",
        "HemisphericalArray",
        "CustomArray",
        "AcousticField",
        "FieldMetrics",
        "WavePropagator",
        "MediumProperties",
        "GradientOptimizer",
        "GeneticOptimizer",
        "NeuralHologramGenerator",
        "__version__"
    ]
    
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    __all__ = ["AcousticHologram", "__version__"]
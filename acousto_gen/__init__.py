"""Acousto-Gen: Generative Acoustic Holography Toolkit.

A comprehensive framework for creating 3D acoustic pressure fields through
ultrasonic transducer arrays, enabling precise acoustic manipulation for
applications in levitation, haptics, and medical ultrasound.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"
__license__ = "MIT"

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
    from applications.levitation.acoustic_levitator import AcousticLevitator, Particle
    from applications.haptics.haptic_renderer import HapticRenderer
    
    # Complete import success
    IMPORT_SUCCESS = True
    
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
        "AcousticLevitator",
        "Particle",
        "HapticRenderer",
        "__version__"
    ]
    
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    IMPORT_SUCCESS = False
    __all__ = ["AcousticHologram", "__version__"]

# Quick system check
def system_check():
    """Perform basic system compatibility check."""
    checks = {
        "python_version": sys.version_info >= (3, 9),
        "numpy_available": False,
        "torch_available": False,
        "scipy_available": False,
    }
    
    try:
        import numpy
        checks["numpy_available"] = True
    except ImportError:
        pass
    
    try:
        import torch
        checks["torch_available"] = True
    except ImportError:
        pass
    
    try:
        import scipy
        checks["scipy_available"] = True
    except ImportError:
        pass
    
    return checks
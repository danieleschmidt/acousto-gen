"""
acousto-gen: Acoustic holography toolkit for simulating and optimizing
3D pressure fields from ultrasonic transducer arrays.
"""

from .transducer_array import TransducerArray
from .simulator import AcousticSimulator
from .optimizer import HologramOptimizer

__version__ = "0.1.0"
__all__ = ["TransducerArray", "AcousticSimulator", "HologramOptimizer"]

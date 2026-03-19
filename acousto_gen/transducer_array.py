"""
TransducerArray: models a flat phased array of ultrasonic transducers.

Each element has a 3D position, operating frequency, phase offset, and amplitude.
The array exposes positions/phases/amplitudes as numpy arrays for use by
AcousticSimulator and HologramOptimizer.
"""

import numpy as np
from typing import Optional


class TransducerArray:
    """
    Flat phased array of ultrasonic transducer elements.

    Parameters
    ----------
    positions : (N, 3) ndarray
        3D coordinates of each element in metres [x, y, z].
    frequency : float
        Common operating frequency in Hz (default 40 kHz).
    phases : (N,) ndarray, optional
        Initial phase offsets in radians (default all zeros).
    amplitudes : (N,) ndarray, optional
        Element amplitudes, typically in [0, 1] (default all ones).
    """

    def __init__(
        self,
        positions: np.ndarray,
        frequency: float = 40_000.0,
        phases: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None,
    ) -> None:
        self.positions = np.asarray(positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must be shape (N, 3)")

        self.N = self.positions.shape[0]
        self.frequency = float(frequency)

        self.phases = (
            np.asarray(phases, dtype=float) if phases is not None else np.zeros(self.N)
        )
        self.amplitudes = (
            np.asarray(amplitudes, dtype=float)
            if amplitudes is not None
            else np.ones(self.N)
        )

        if self.phases.shape != (self.N,):
            raise ValueError(f"phases must have length {self.N}")
        if self.amplitudes.shape != (self.N,):
            raise ValueError(f"amplitudes must have length {self.N}")

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def linear(
        cls,
        n: int,
        pitch: float = 10e-3,
        frequency: float = 40_000.0,
        axis: str = "x",
    ) -> "TransducerArray":
        """
        Create a 1-D linear array of *n* elements spaced by *pitch* metres.

        The array is centred at the origin and lies in the z=0 plane.

        Parameters
        ----------
        n : int
            Number of elements.
        pitch : float
            Centre-to-centre spacing in metres (default 10 mm).
        frequency : float
            Operating frequency in Hz.
        axis : {'x', 'y'}
            Axis along which to arrange elements.
        """
        coords = np.linspace(-(n - 1) / 2 * pitch, (n - 1) / 2 * pitch, n)
        positions = np.zeros((n, 3))
        if axis == "x":
            positions[:, 0] = coords
        else:
            positions[:, 1] = coords
        return cls(positions, frequency=frequency)

    @classmethod
    def grid(
        cls,
        nx: int,
        ny: int,
        pitch: float = 10e-3,
        frequency: float = 40_000.0,
    ) -> "TransducerArray":
        """
        Create a 2-D rectangular grid array of *nx* × *ny* elements.

        Centred at the origin, lying in the z=0 plane.
        """
        xs = np.linspace(-(nx - 1) / 2 * pitch, (nx - 1) / 2 * pitch, nx)
        ys = np.linspace(-(ny - 1) / 2 * pitch, (ny - 1) / 2 * pitch, ny)
        xx, yy = np.meshgrid(xs, ys)
        positions = np.column_stack(
            [xx.ravel(), yy.ravel(), np.zeros(nx * ny)]
        )
        return cls(positions, frequency=frequency)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wavelength(self) -> float:
        """Wavelength in air at sea level (m)."""
        return 343.0 / self.frequency

    @property
    def wavenumber(self) -> float:
        """Wave number k = 2π/λ (rad/m)."""
        return 2 * np.pi * self.frequency / 343.0

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def set_phases(self, phases: np.ndarray) -> None:
        phases = np.asarray(phases, dtype=float)
        if phases.shape != (self.N,):
            raise ValueError(f"phases must have length {self.N}")
        self.phases = phases

    def set_amplitudes(self, amplitudes: np.ndarray) -> None:
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.shape != (self.N,):
            raise ValueError(f"amplitudes must have length {self.N}")
        self.amplitudes = amplitudes

    def reset(self) -> None:
        """Reset phases to zero and amplitudes to one."""
        self.phases = np.zeros(self.N)
        self.amplitudes = np.ones(self.N)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TransducerArray(N={self.N}, frequency={self.frequency/1e3:.1f} kHz, "
            f"λ={self.wavelength*1e3:.1f} mm)"
        )

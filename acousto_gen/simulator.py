"""
AcousticSimulator: computes complex pressure fields in 3D space.

Physics
-------
The pressure contribution from a single point-source transducer element at
position **r_i** to a field point **r** is modelled by the monopole Green's
function (Rayleigh-Sommerfeld approximation):

    p_i(r) = A_i * exp(j * phi_i) * exp(j * k * |r - r_i|) / |r - r_i|

where:
    A_i   — amplitude of element i
    phi_i — phase offset of element i
    k     — wave number 2π/λ

The total pressure is the superposition over all N elements:

    p(r) = Σ_i p_i(r)

This is exact for omnidirectional point sources and a good first approximation
for small (sub-wavelength) transducer elements.
"""

import numpy as np
from .transducer_array import TransducerArray


class AcousticSimulator:
    """
    Computes the complex acoustic pressure field produced by a TransducerArray.

    Parameters
    ----------
    array : TransducerArray
        The transducer array to simulate.
    speed_of_sound : float
        Speed of sound in the medium (m/s). Default 343 m/s (air, ~20 °C).
    """

    def __init__(self, array: TransducerArray, speed_of_sound: float = 343.0) -> None:
        self.array = array
        self.c = float(speed_of_sound)

    # ------------------------------------------------------------------
    # Core field computation
    # ------------------------------------------------------------------

    def pressure_at_points(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the complex pressure at an arbitrary set of field points.

        Parameters
        ----------
        points : (M, 3) ndarray
            Cartesian coordinates of evaluation points [x, y, z] in metres.

        Returns
        -------
        pressure : (M,) complex ndarray
            Complex pressure amplitude at each point (SI units, up to
            a common prefactor that cancels in phase-only optimisation).
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]  # single point → (1, 3)

        k = 2 * np.pi * self.array.frequency / self.c  # wave number

        # positions: (N, 3)  points: (M, 3)
        # displacement vectors: (M, N, 3)
        diff = points[:, None, :] - self.array.positions[None, :, :]
        # distances: (M, N)
        r = np.linalg.norm(diff, axis=-1)
        # Avoid divide-by-zero for points coinciding with a transducer
        r = np.where(r < 1e-12, 1e-12, r)

        # Complex source weights: (N,)
        weights = self.array.amplitudes * np.exp(1j * self.array.phases)  # (N,)

        # Green's function: exp(j*k*r) / r  →  (M, N)
        G = np.exp(1j * k * r) / r

        # Superposition: (M,)
        pressure = G @ weights  # matrix-vector product
        return pressure

    def pressure_on_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: float,
    ) -> np.ndarray:
        """
        Compute the complex pressure on a regular 2-D XY grid at height *z*.

        Parameters
        ----------
        x : 1-D ndarray
            X coordinates in metres.
        y : 1-D ndarray
            Y coordinates in metres.
        z : float
            Fixed z height in metres.

        Returns
        -------
        pressure : (len(y), len(x)) complex ndarray
        """
        xx, yy = np.meshgrid(x, y)
        zz = np.full_like(xx, z)
        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        p = self.pressure_at_points(pts)
        return p.reshape(len(y), len(x))

    def pressure_along_axis(
        self,
        z_values: np.ndarray,
        x: float = 0.0,
        y: float = 0.0,
    ) -> np.ndarray:
        """
        Compute complex pressure along a line parallel to the z axis.

        Parameters
        ----------
        z_values : 1-D ndarray
            Z coordinates (metres).
        x, y : float
            Lateral position (metres).

        Returns
        -------
        pressure : (len(z_values),) complex ndarray
        """
        pts = np.column_stack(
            [
                np.full_like(z_values, x),
                np.full_like(z_values, y),
                z_values,
            ]
        )
        return self.pressure_at_points(pts)

    # ------------------------------------------------------------------
    # Convenience: SPL
    # ------------------------------------------------------------------

    @staticmethod
    def spl(pressure: np.ndarray, p_ref: float = 20e-6) -> np.ndarray:
        """
        Convert complex pressure amplitude to Sound Pressure Level (dB SPL).

        Parameters
        ----------
        pressure : ndarray (complex)
            Complex pressure field.
        p_ref : float
            Reference pressure (default 20 µPa for air).
        """
        return 20 * np.log10(np.abs(pressure) / p_ref + 1e-30)

    def __repr__(self) -> str:
        return f"AcousticSimulator(array={self.array!r}, c={self.c} m/s)"

"""
HologramOptimizer: find transducer phases that reproduce a target pressure pattern.

Two algorithms are provided:

1. **Gerchberg-Saxton (GS)**  — iterative phase retrieval.
   Works well when you want unit-amplitude transducers and a prescribed
   magnitude pattern at a set of focal points.

2. **Gradient-descent (GD)** — minimise MSE between |p(target)| and |p(achieved)|
   via automatic differentiation using pure-NumPy finite differences.
   Slower than GS but handles arbitrary amplitude constraints.
"""

import numpy as np
from typing import Optional, Tuple
from .transducer_array import TransducerArray
from .simulator import AcousticSimulator


class HologramOptimizer:
    """
    Optimise transducer phases to match a target pressure magnitude pattern.

    Parameters
    ----------
    array : TransducerArray
        The transducer array whose phases will be optimised in-place.
    simulator : AcousticSimulator
        Simulator bound to the same array.
    """

    def __init__(self, array: TransducerArray, simulator: AcousticSimulator) -> None:
        self.array = array
        self.sim = simulator

    # ------------------------------------------------------------------
    # Gerchberg-Saxton
    # ------------------------------------------------------------------

    def gerchberg_saxton(
        self,
        target_points: np.ndarray,
        target_magnitudes: Optional[np.ndarray] = None,
        n_iter: int = 100,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, list]:
        """
        Gerchberg-Saxton phase-retrieval algorithm.

        The algorithm alternates between the transducer plane and the target
        focal points, enforcing:
          - Unit amplitude (|A_i| = 1) at each transducer
          - Prescribed magnitude at each target point

        Parameters
        ----------
        target_points : (M, 3) ndarray
            Locations where pressure should be focused (metres).
        target_magnitudes : (M,) ndarray, optional
            Desired pressure magnitude at each target point.
            Defaults to equal-weight focusing (all ones).
        n_iter : int
            Number of GS iterations (default 100).
        verbose : bool
            Print error every 10 iterations.

        Returns
        -------
        phases : (N,) ndarray
            Optimised phase offsets in radians.
        errors : list of float
            Normalised error after each iteration.
        """
        target_points = np.asarray(target_points, dtype=float)
        if target_points.ndim == 1:
            target_points = target_points[None, :]
        M = target_points.shape[0]

        if target_magnitudes is None:
            target_magnitudes = np.ones(M)
        else:
            target_magnitudes = np.asarray(target_magnitudes, dtype=float)

        # Pre-compute propagation matrix H: (M, N)
        # H[m, n] = Green's function from transducer n to point m
        k = 2 * np.pi * self.array.frequency / self.sim.c
        diff = target_points[:, None, :] - self.array.positions[None, :, :]  # (M,N,3)
        r = np.linalg.norm(diff, axis=-1)  # (M, N)
        r = np.where(r < 1e-12, 1e-12, r)
        H = np.exp(1j * k * r) / r  # (M, N)
        H_conj_T = H.conj().T  # (N, M) — back-propagation operator

        # Initialise random phases on transducers
        phases = np.random.uniform(0, 2 * np.pi, self.array.N)
        self.array.set_phases(phases)

        errors = []
        for it in range(n_iter):
            # --- Forward pass: transducers → focal points ---
            weights = np.exp(1j * phases)  # unit amplitude, (N,)
            focal_field = H @ weights  # (M,)

            # --- Apply target magnitude constraint ---
            focal_field_constrained = target_magnitudes * np.exp(
                1j * np.angle(focal_field)
            )

            # --- Backward pass: focal points → transducers ---
            transducer_field = H_conj_T @ focal_field_constrained  # (N,)

            # --- Apply unit-amplitude constraint on transducers ---
            phases = np.angle(transducer_field)

            # --- Track error ---
            achieved = np.abs(H @ np.exp(1j * phases))
            err = np.linalg.norm(achieved - target_magnitudes) / (
                np.linalg.norm(target_magnitudes) + 1e-30
            )
            errors.append(float(err))

            if verbose and (it + 1) % 10 == 0:
                print(f"  GS iter {it+1:4d}/{n_iter}  error={err:.4f}")

        self.array.set_phases(phases)
        return phases.copy(), errors

    # ------------------------------------------------------------------
    # Gradient descent (phase-only, amplitude fixed at 1)
    # ------------------------------------------------------------------

    def gradient_descent(
        self,
        target_points: np.ndarray,
        target_magnitudes: Optional[np.ndarray] = None,
        n_iter: int = 200,
        lr: float = 0.05,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, list]:
        """
        Phase-only gradient descent using finite differences.

        Minimises:  L = Σ_m (|p_m| - T_m)²

        Parameters
        ----------
        target_points : (M, 3) ndarray
        target_magnitudes : (M,) ndarray, optional
        n_iter : int
            Number of gradient steps.
        lr : float
            Learning rate (radians per step).
        verbose : bool

        Returns
        -------
        phases : (N,) ndarray
        losses : list of float
        """
        target_points = np.asarray(target_points, dtype=float)
        if target_points.ndim == 1:
            target_points = target_points[None, :]
        M = target_points.shape[0]

        if target_magnitudes is None:
            target_magnitudes = np.ones(M)
        else:
            target_magnitudes = np.asarray(target_magnitudes, dtype=float)

        k = 2 * np.pi * self.array.frequency / self.sim.c
        diff = target_points[:, None, :] - self.array.positions[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        r = np.where(r < 1e-12, 1e-12, r)
        H = np.exp(1j * k * r) / r  # (M, N)

        phases = np.random.uniform(0, 2 * np.pi, self.array.N)
        losses = []

        def loss_fn(ph):
            achieved = np.abs(H @ np.exp(1j * ph))
            return np.sum((achieved - target_magnitudes) ** 2)

        eps = 1e-4  # finite-difference step
        for it in range(n_iter):
            L = loss_fn(phases)
            losses.append(float(L))

            # Analytic gradient via complex chain rule:
            #   dL/dφ_n = 2 * Re[ Σ_m (|p_m| - T_m) * p_m^* / |p_m| * H[m,n] * j*exp(jφ_n) ]
            p = H @ np.exp(1j * phases)  # (M,)
            p_mag = np.abs(p)
            residual = (p_mag - target_magnitudes) / (p_mag + 1e-30)  # (M,)
            # dL/dφ_n = 2 * Re[ Σ_m residual_m * p_m^* * H[m,n] * j * exp(jφ_n) ]
            grad_complex = (
                2 * (H.conj().T @ (residual * p.conj())) * 1j * np.exp(1j * phases)
            )
            grad = np.real(grad_complex)

            phases -= lr * grad

            if verbose and (it + 1) % 20 == 0:
                print(f"  GD iter {it+1:4d}/{n_iter}  loss={L:.4e}")

        self.array.set_phases(phases)
        return phases.copy(), losses

    # ------------------------------------------------------------------
    # Levitation trap: two-sided focusing (acoustic tweezers)
    # ------------------------------------------------------------------

    def levitation_trap(
        self,
        trap_centre: np.ndarray,
        n_iter: int = 150,
        method: str = "gs",
        verbose: bool = False,
    ) -> Tuple[np.ndarray, list]:
        """
        Set up a single-sided levitation trap.

        Creates a focused spot at *trap_centre* surrounded by higher-pressure
        regions above and below, forming a local minimum in which a particle
        can be trapped.

        Parameters
        ----------
        trap_centre : (3,) array_like
            Target levitation position in metres.
        n_iter : int
        method : {'gs', 'gd'}
        verbose : bool

        Returns
        -------
        phases : (N,) ndarray
        errors : list
        """
        c = np.asarray(trap_centre, dtype=float)
        dz = self.array.wavelength * 0.5
        # Four points: the trap minimum and a ring around it
        ring_r = self.array.wavelength * 0.3
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        ring = np.column_stack(
            [c[0] + ring_r * np.cos(angles),
             c[1] + ring_r * np.sin(angles),
             np.full(4, c[2])]
        )
        above = (c + [0, 0, dz])[None, :]
        below = (c - [0, 0, dz])[None, :]
        target_pts = np.vstack([above, below, ring])
        # High amplitude at surrounding points, low at centre
        mags = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 0.5])

        if method == "gs":
            return self.gerchberg_saxton(
                target_pts, mags, n_iter=n_iter, verbose=verbose
            )
        else:
            return self.gradient_descent(
                target_pts, mags, n_iter=n_iter, verbose=verbose
            )

    def __repr__(self) -> str:
        return f"HologramOptimizer(N={self.array.N})"

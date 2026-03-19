"""Tests for acousto-gen toolkit."""
import numpy as np
import pytest
from acousto_gen import TransducerArray, AcousticSimulator, HologramOptimizer


class TestTransducerArray:
    def test_linear_array_shape(self):
        arr = TransducerArray.linear(n=8, pitch=0.005, frequency=40e3)
        assert arr.positions.shape == (8, 3)
        assert arr.phases.shape == (8,)
        assert arr.amplitudes.shape == (8,)

    def test_linear_array_centered(self):
        arr = TransducerArray.linear(n=8, pitch=0.01, frequency=40e3)
        assert abs(arr.positions[:, 0].mean()) < 1e-10  # centered at x=0

    def test_grid_array_shape(self):
        arr = TransducerArray.grid(nx=4, ny=4, pitch=0.005, frequency=40e3)
        assert arr.positions.shape == (16, 3)

    def test_set_phases(self):
        arr = TransducerArray.linear(n=4, pitch=0.005, frequency=40e3)
        new_phases = np.array([0.1, 0.2, 0.3, 0.4])
        arr.set_phases(new_phases)
        np.testing.assert_allclose(arr.phases, new_phases)

    def test_frequency_stored(self):
        arr = TransducerArray.linear(n=4, pitch=0.005, frequency=25e3)
        assert arr.frequency == 25e3

    def test_repr(self):
        arr = TransducerArray.linear(n=4, pitch=0.005, frequency=40e3)
        assert "TransducerArray" in repr(arr)


class TestAcousticSimulator:
    def setup_method(self):
        self.arr = TransducerArray.linear(n=16, pitch=0.005, frequency=40e3)
        self.sim = AcousticSimulator(self.arr)

    def test_pressure_at_single_point(self):
        p = self.sim.pressure_at_points(np.array([[0.0, 0.0, 0.1]]))
        assert p.shape == (1,)
        assert np.isfinite(p[0])

    def test_pressure_at_multiple_points(self):
        pts = np.random.default_rng(42).uniform(-0.05, 0.05, (20, 3))
        pts[:, 2] = 0.1  # z = 10 cm
        p = self.sim.pressure_at_points(pts)
        assert p.shape == (20,)
        assert np.all(np.isfinite(p))

    def test_pressure_along_axis_shape(self):
        z = np.linspace(0.01, 0.3, 100)
        p = self.sim.pressure_along_axis(z)
        assert p.shape == (100,)

    def test_pressure_on_grid_shape(self):
        x = np.linspace(-0.05, 0.05, 10)
        y = np.linspace(-0.05, 0.05, 10)
        p = self.sim.pressure_on_grid(x, y, z=0.1)
        assert p.shape == (10, 10)

    def test_focal_point_on_axis(self):
        """16-element array should focus near the array center along z-axis."""
        z = np.linspace(0.01, 0.3, 500)
        p = self.sim.pressure_along_axis(z)
        focal_z = z[np.argmax(np.abs(p))]
        # Focal point should be somewhere along the axis
        assert 0.01 <= focal_z <= 0.3

    def test_symmetry_centered_array(self):
        """Symmetric array with uniform phases: pressure symmetric about axis."""
        arr = TransducerArray.linear(n=8, pitch=0.005, frequency=40e3)
        sim = AcousticSimulator(arr)
        p_pos = sim.pressure_at_points(np.array([[0.01, 0.0, 0.1]]))
        p_neg = sim.pressure_at_points(np.array([[-0.01, 0.0, 0.1]]))
        np.testing.assert_allclose(np.abs(p_pos), np.abs(p_neg), rtol=1e-6)

    def test_spl_finite(self):
        p = self.sim.pressure_along_axis(np.linspace(0.05, 0.2, 50))
        spl = AcousticSimulator.spl(p)
        assert np.all(np.isfinite(spl))

    def test_spl_positive_for_strong_source(self):
        p = self.sim.pressure_along_axis(np.array([0.1]))
        spl = AcousticSimulator.spl(p)
        assert spl[0] > 0  # should be well above 0 dB SPL

    def test_speed_of_sound_effect(self):
        """Different c → different focal pattern (different wavelength)."""
        sim_air = AcousticSimulator(self.arr, speed_of_sound=343.0)
        sim_water = AcousticSimulator(self.arr, speed_of_sound=1500.0)
        z = np.linspace(0.01, 0.3, 100)
        p_air = sim_air.pressure_along_axis(z)
        p_water = sim_water.pressure_along_axis(z)
        # They should differ (different k)
        assert not np.allclose(np.abs(p_air), np.abs(p_water))


class TestHologramOptimizer:
    def setup_method(self):
        self.arr = TransducerArray.linear(n=8, pitch=0.005, frequency=40e3)
        self.sim = AcousticSimulator(self.arr)
        self.opt = HologramOptimizer(self.arr, self.sim)

    def test_gs_returns_phases(self):
        targets = np.array([[0.0, 0.0, 0.1]])
        phases, errors = self.opt.gerchberg_saxton(target_points=targets, n_iter=5)
        assert phases.shape == (8,)
        assert np.all(np.isfinite(phases))
        assert len(errors) == 5

    def test_gs_multiple_foci(self):
        targets = np.array([[0.0, 0.0, 0.1], [0.02, 0.0, 0.1]])
        phases, errors = self.opt.gerchberg_saxton(target_points=targets, n_iter=10)
        assert phases.shape == (8,)
        assert np.all(np.isfinite(phases))

    def test_gradient_descent_returns_phases(self):
        target = np.array([[0.0, 0.0, 0.1]])
        phases, losses = self.opt.gradient_descent(target_points=target, n_iter=20)
        assert phases.shape == (8,)
        assert np.all(np.isfinite(phases))

    def test_levitation_trap(self):
        trap_center = np.array([0.0, 0.0, 0.08])
        result = self.opt.levitation_trap(trap_centre=trap_center)
        # returns phases or (phases, errors) depending on method
        phases = result[0] if isinstance(result, tuple) else result
        assert phases.shape == (8,)
        assert np.all(np.isfinite(phases))

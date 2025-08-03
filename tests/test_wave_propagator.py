"""
Tests for wave propagation module.
Validates physics calculations and field generation.
"""

import pytest
import numpy as np
import torch
from src.physics.propagation.wave_propagator import WavePropagator, MediumProperties


class TestMediumProperties:
    """Test medium properties calculations."""
    
    def test_air_properties(self):
        """Test standard air properties."""
        air = MediumProperties(
            density=1.2,
            speed_of_sound=343,
            absorption=0.01
        )
        
        # Test impedance
        impedance = air.get_impedance()
        assert np.isclose(impedance, 1.2 * 343, rtol=1e-5)
        
        # Test wavelength at 40 kHz
        wavelength = air.get_wavelength(40e3)
        assert np.isclose(wavelength, 343 / 40e3, rtol=1e-5)
        
        # Test wavenumber
        k = air.get_wavenumber(40e3)
        assert np.isclose(k.real, 2 * np.pi * 40e3 / 343, rtol=1e-5)
        assert np.isclose(k.imag, -0.01, rtol=1e-5)
    
    def test_water_properties(self):
        """Test water properties."""
        water = MediumProperties(
            density=1000,
            speed_of_sound=1480,
            absorption=0.002
        )
        
        impedance = water.get_impedance()
        assert impedance == 1000 * 1480


class TestWavePropagator:
    """Test wave propagation calculations."""
    
    @pytest.fixture
    def propagator(self):
        """Create test propagator."""
        return WavePropagator(
            resolution=5e-3,
            bounds=[(-0.05, 0.05), (-0.05, 0.05), (0, 0.1)],
            frequency=40e3,
            device="cpu"
        )
    
    def test_initialization(self, propagator):
        """Test propagator initialization."""
        assert propagator.frequency == 40e3
        assert propagator.resolution == 5e-3
        assert propagator.nx > 0
        assert propagator.ny > 0
        assert propagator.nz > 0
    
    def test_single_source_field(self, propagator):
        """Test field from single point source."""
        # Single source at origin
        source_positions = np.array([[0, 0, 0]])
        source_amplitudes = np.array([1.0])
        source_phases = np.array([0.0])
        
        # Compute field at test point
        test_points = np.array([[0, 0, 0.05]])
        field = propagator.compute_field_from_sources(
            source_positions,
            source_amplitudes,
            source_phases,
            target_points=test_points
        )
        
        # Check field is non-zero
        assert np.abs(field[0]) > 0
        
        # Check decay with distance
        test_points_far = np.array([[0, 0, 0.1]])
        field_far = propagator.compute_field_from_sources(
            source_positions,
            source_amplitudes,
            source_phases,
            target_points=test_points_far
        )
        
        assert np.abs(field_far[0]) < np.abs(field[0])
    
    def test_two_source_interference(self, propagator):
        """Test interference between two sources."""
        # Two sources with opposite phases
        source_positions = np.array([[-0.01, 0, 0], [0.01, 0, 0]])
        source_amplitudes = np.array([1.0, 1.0])
        
        # In phase - constructive interference
        source_phases = np.array([0.0, 0.0])
        test_point = np.array([[0, 0, 0.05]])
        
        field_constructive = propagator.compute_field_from_sources(
            source_positions,
            source_amplitudes,
            source_phases,
            target_points=test_point
        )
        
        # Out of phase - destructive interference
        source_phases = np.array([0.0, np.pi])
        field_destructive = propagator.compute_field_from_sources(
            source_positions,
            source_amplitudes,
            source_phases,
            target_points=test_point
        )
        
        # Constructive should be stronger than destructive
        assert np.abs(field_constructive[0]) > np.abs(field_destructive[0])
    
    def test_angular_spectrum_propagation(self, propagator):
        """Test angular spectrum propagation method."""
        # Create initial field
        nx, ny = 50, 50
        initial_field = np.zeros((nx, ny), dtype=complex)
        initial_field[nx//2, ny//2] = 1.0  # Point source
        
        # Propagate
        z_distance = 0.01  # 1cm
        propagated = propagator.angular_spectrum_propagation(
            initial_field,
            z_distance
        )
        
        # Check field has spread
        assert np.sum(np.abs(propagated) > 0.1) > 1
        
        # Check energy conservation (approximately)
        initial_energy = np.sum(np.abs(initial_field)**2)
        propagated_energy = np.sum(np.abs(propagated)**2)
        assert np.isclose(initial_energy, propagated_energy, rtol=0.1)
    
    def test_pressure_gradient(self, propagator):
        """Test pressure gradient calculation."""
        # Create test field with known gradient
        nx, ny, nz = 10, 10, 10
        x = np.linspace(-0.01, 0.01, nx)
        y = np.linspace(-0.01, 0.01, ny)
        z = np.linspace(0, 0.02, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Linear field for testing
        field = X + 2*Y + 3*Z + 0j
        
        grad_x, grad_y, grad_z = propagator.compute_pressure_gradient(field)
        
        # Check gradients are approximately constant
        assert np.allclose(grad_x[1:-1, 1:-1, 1:-1], 1.0, atol=0.2)
        assert np.allclose(grad_y[1:-1, 1:-1, 1:-1], 2.0, atol=0.2)
        assert np.allclose(grad_z[1:-1, 1:-1, 1:-1], 3.0, atol=0.2)
    
    def test_radiation_force(self, propagator):
        """Test acoustic radiation force calculation."""
        # Create focused field
        source_positions = np.array([[0, 0, 0]])
        source_amplitudes = np.array([1000.0])
        source_phases = np.array([0.0])
        
        field = propagator.compute_field_from_sources(
            source_positions,
            source_amplitudes,
            source_phases
        )
        
        # Calculate forces
        Fx, Fy, Fz = propagator.compute_acoustic_radiation_force(field)
        
        # Forces should be non-zero
        assert np.max(np.abs(Fx)) > 0
        assert np.max(np.abs(Fy)) > 0
        assert np.max(np.abs(Fz)) > 0
    
    def test_intensity_calculation(self, propagator):
        """Test acoustic intensity calculation."""
        # Create test field
        source_positions = np.array([[0, 0, 0]])
        source_amplitudes = np.array([100.0])
        source_phases = np.array([0.0])
        
        field = propagator.compute_field_from_sources(
            source_positions,
            source_amplitudes,
            source_phases
        )
        
        intensity = propagator.compute_acoustic_intensity(field)
        
        # Intensity should be positive
        assert np.all(intensity >= 0)
        
        # Intensity should decay with distance
        center_idx = (propagator.nx // 2, propagator.ny // 2, 0)
        edge_idx = (0, 0, propagator.nz - 1)
        
        assert intensity[center_idx] > intensity[edge_idx]
    
    def test_focus_detection(self, propagator):
        """Test focal point detection."""
        # Create focused field
        focal_position = np.array([0, 0, 0.05])
        source_positions = propagator.array.get_positions() if hasattr(propagator, 'array') else np.array([[0, 0, 0]])
        
        # Simple focusing - all sources in phase toward focal point
        source_phases = []
        for pos in source_positions:
            distance = np.linalg.norm(focal_position - pos)
            k = 2 * np.pi * propagator.frequency / propagator.medium.speed_of_sound
            phase = -k * distance  # Compensate for propagation
            source_phases.append(phase % (2 * np.pi))
        
        source_phases = np.array(source_phases)
        source_amplitudes = np.ones(len(source_positions))
        
        field = propagator.compute_field_from_sources(
            source_positions,
            source_amplitudes,
            source_phases
        )
        
        # Find focus points
        focal_points = propagator.find_focus_points(field, threshold=0.5)
        
        # Should find at least one focus
        assert len(focal_points) > 0
        
        # Focus should be near target
        if focal_points:
            found_focus = focal_points[0]
            error = np.linalg.norm(
                np.array(found_focus[:3]) - focal_position
            )
            # Allow 2cm error due to discretization
            assert error < 0.02


class TestFieldOperations:
    """Test field manipulation operations."""
    
    def test_field_superposition(self):
        """Test linear superposition of fields."""
        propagator = WavePropagator(
            resolution=5e-3,
            frequency=40e3
        )
        
        # Two individual fields
        source1 = np.array([[0, 0, 0]])
        field1 = propagator.compute_field_from_sources(
            source1,
            np.array([1.0]),
            np.array([0.0])
        )
        
        source2 = np.array([[0.01, 0, 0]])
        field2 = propagator.compute_field_from_sources(
            source2,
            np.array([1.0]),
            np.array([0.0])
        )
        
        # Combined field
        sources_combined = np.vstack([source1, source2])
        field_combined = propagator.compute_field_from_sources(
            sources_combined,
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0])
        )
        
        # Should approximately equal sum (within numerical error)
        difference = np.abs(field_combined - (field1 + field2))
        relative_error = np.mean(difference) / np.mean(np.abs(field_combined))
        assert relative_error < 0.01
    
    @pytest.mark.parametrize("frequency", [20e3, 40e3, 100e3])
    def test_frequency_dependence(self, frequency):
        """Test frequency-dependent behavior."""
        propagator = WavePropagator(
            resolution=5e-3,
            frequency=frequency
        )
        
        wavelength = propagator.medium.get_wavelength(frequency)
        
        # Wavelength should decrease with frequency
        assert wavelength == pytest.approx(343 / frequency, rel=1e-3)
        
        # Wavenumber should increase with frequency
        k = propagator.medium.get_wavenumber(frequency)
        assert k.real == pytest.approx(2 * np.pi * frequency / 343, rel=1e-3)
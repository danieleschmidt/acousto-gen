"""Tests for core acoustic holography functionality."""

import numpy as np
import pytest

from acousto_gen.core import AcousticHologram


class TestAcousticHologram:
    """Test cases for AcousticHologram class."""
    
    def test_initialization(self) -> None:
        """Test hologram initialization."""
        from src.physics.transducers.transducer_array import UltraLeap256
        mock_transducer = UltraLeap256()
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000,
            medium="air"
        )
        
        assert hologram.transducer == mock_transducer
        assert hologram.frequency == 40000
        assert hasattr(hologram.medium, 'speed_of_sound')
    
    def test_create_focus_point(self) -> None:
        """Test focus point creation."""
        from src.physics.transducers.transducer_array import UltraLeap256
        mock_transducer = UltraLeap256()
        hologram = AcousticHologram(
            transducer=mock_transducer, 
            frequency=40000
        )
        
        target_field = hologram.create_focus_point(
            position=(0, 0, 0.1),
            pressure=4000
        )
        
        # The method now returns an AcousticField object
        assert hasattr(target_field, 'data')
        assert hasattr(target_field, 'bounds')
        assert target_field.data.shape == (200, 200, 200)
    
    def test_optimize(self) -> None:
        """Test phase optimization."""
        from src.physics.transducers.transducer_array import UltraLeap256
        mock_transducer = UltraLeap256()
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        # Create a proper target field
        target_field = hologram.create_focus_point(
            position=(0, 0, 0.1),
            pressure=4000
        )
        
        result = hologram.optimize(target_field, iterations=10)
        
        assert isinstance(result, dict)
        assert 'phases' in result
        assert 'final_loss' in result
        phases = result['phases']
        assert isinstance(phases, np.ndarray)
        assert len(phases) == 256
        assert np.all(phases >= 0)
        assert np.all(phases <= 2 * np.pi)
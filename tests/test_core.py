"""Tests for core acoustic holography functionality."""

import numpy as np
import pytest

from acousto_gen.core import AcousticHologram


class TestAcousticHologram:
    """Test cases for AcousticHologram class."""
    
    def test_initialization(self) -> None:
        """Test hologram initialization."""
        mock_transducer = "mock_transducer"
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000,
            medium="air"
        )
        
        assert hologram.transducer == mock_transducer
        assert hologram.frequency == 40000
        assert hologram.medium == "air"
    
    def test_create_focus_point(self) -> None:
        """Test focus point creation."""
        hologram = AcousticHologram(
            transducer="mock", 
            frequency=40000
        )
        
        target_field = hologram.create_focus_point(
            position=(0, 0, 0.1),
            pressure=4000
        )
        
        assert isinstance(target_field, np.ndarray)
        assert target_field.shape == (100, 100, 100)
    
    def test_optimize(self) -> None:
        """Test phase optimization."""
        hologram = AcousticHologram(
            transducer="mock",
            frequency=40000
        )
        
        target = np.zeros((100, 100, 100))
        phases = hologram.optimize(target, iterations=100)
        
        assert isinstance(phases, np.ndarray)
        assert len(phases) == 256
        assert np.all(phases >= 0)
        assert np.all(phases <= 2 * np.pi)
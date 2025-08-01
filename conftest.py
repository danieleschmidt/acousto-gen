"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def mock_transducer():
    """Mock transducer array for testing."""
    return {
        "positions": np.random.rand(256, 3),
        "elements": 256,
        "frequency": 40000,
        "type": "mock_array"
    }


@pytest.fixture
def sample_pressure_field():
    """Sample 3D pressure field for testing."""
    return np.random.rand(100, 100, 100)


@pytest.fixture
def acoustic_hologram_params():
    """Standard parameters for AcousticHologram testing."""
    return {
        "frequency": 40000,
        "medium": "air",
        "optimization_method": "adam",
        "iterations": 100
    }
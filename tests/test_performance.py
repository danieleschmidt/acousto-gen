"""Performance benchmarking tests."""

import time
import pytest
import numpy as np

from acousto_gen.core import AcousticHologram


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for core algorithms."""
    
    def test_hologram_creation_performance(self, mock_transducer, benchmark=None):
        """Benchmark hologram creation speed."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        def create_focus():
            return hologram.create_focus_point(
                position=(0, 0, 0.1),
                pressure=4000
            )
        
        # Manual timing if benchmark fixture not available
        if benchmark:
            result = benchmark(create_focus)
        else:
            start_time = time.time()
            result = create_focus()
            end_time = time.time()
            
            # Performance assertion: should complete in < 1 second
            assert (end_time - start_time) < 1.0
        
        assert result is not None
    
    def test_optimization_performance(self, mock_transducer):
        """Benchmark optimization algorithm speed."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        target = np.zeros((50, 50, 50))  # Smaller for speed
        
        start_time = time.time()
        phases = hologram.optimize(target, iterations=10)
        end_time = time.time()
        
        # Should complete small optimization quickly
        assert (end_time - start_time) < 2.0
        assert len(phases) == 256
    
    @pytest.mark.slow
    def test_large_field_computation(self, mock_transducer):
        """Test performance with large pressure fields."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        # Large field test
        target = np.zeros((200, 200, 200))
        
        start_time = time.time()
        phases = hologram.optimize(target, iterations=5)
        end_time = time.time()
        
        # Should handle large fields reasonably
        assert (end_time - start_time) < 10.0
        assert len(phases) == 256
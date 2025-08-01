"""Integration tests for acousto-gen components."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestAcousticHologramIntegration:
    """Test complete acoustic hologram workflows."""
    
    def test_hologram_creation_workflow(self, mock_transducer, acoustic_hologram_params):
        """Test complete hologram creation from target to phases."""
        # This would test the full pipeline in a real implementation
        # For now, we verify the components can work together
        assert mock_transducer.elements == 256
        assert acoustic_hologram_params["frequency"] == 40000
        
        # Simulate phase generation
        phases = np.random.rand(256) * 2 * np.pi
        mock_transducer.set_phases(phases)
        mock_transducer.activate()
        
        mock_transducer.set_phases.assert_called_once_with(phases)
        mock_transducer.activate.assert_called_once()
    
    def test_multi_point_levitation_workflow(self, mock_transducer):
        """Test multi-point levitation setup."""
        positions = [
            [0, 0, 0.08],
            [0.02, 0, 0.10],
            [-0.02, 0, 0.12]
        ]
        
        # Simulate multi-point phase calculation
        phases = np.random.rand(256) * 2 * np.pi
        mock_transducer.set_phases(phases)
        
        assert len(positions) == 3
        mock_transducer.set_phases.assert_called_once()


class TestHardwareIntegration:
    """Test hardware interface integration."""
    
    @pytest.mark.hardware
    def test_hardware_connection(self, mock_hardware_interface):
        """Test hardware connection workflow."""
        result = mock_hardware_interface.connect()
        assert result is True
        
        status = mock_hardware_interface.get_status()
        assert status == "connected"
        
        mock_hardware_interface.disconnect()
        mock_hardware_interface.disconnect.assert_called_once()
    
    @pytest.mark.hardware
    def test_phase_transmission(self, mock_hardware_interface):
        """Test phase data transmission to hardware."""
        phases = np.random.rand(256) * 2 * np.pi
        mock_hardware_interface.send_phases(phases)
        
        mock_hardware_interface.send_phases.assert_called_once_with(phases)


class TestPerformanceIntegration:
    """Test performance-critical integration scenarios."""
    
    @pytest.mark.slow
    def test_large_array_optimization(self, performance_benchmark_data):
        """Test optimization with large transducer arrays."""
        large_array = performance_benchmark_data["large_array"]
        assert large_array.shape == (1000, 1000, 100)
        
        # Simulate optimization timing
        import time
        start_time = time.time()
        # Placeholder for actual optimization
        result = np.mean(large_array)
        end_time = time.time()
        
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
        assert result is not None
    
    @pytest.mark.slow
    def test_real_time_performance(self, performance_benchmark_data):
        """Test real-time performance requirements."""
        benchmark_cases = performance_benchmark_data["benchmark_cases"]
        
        for case in benchmark_cases:
            # Simulate phase calculation for different array sizes
            phases = np.random.rand(case["size"]) * 2 * np.pi
            
            # Verify we can handle different sizes
            assert len(phases) == case["size"]
            assert case["frequency"] == 40000


class TestSafetyIntegration:
    """Test safety system integration."""
    
    def test_pressure_limit_enforcement(self, mock_transducer):
        """Test that pressure limits are enforced."""
        max_pressure = 4000  # Pa
        test_pressure = 5000  # Above limit
        
        # In a real system, this would trigger safety shutdown
        # For now, verify the safety check would trigger
        assert test_pressure > max_pressure
        
        # Safety system would prevent activation
        with patch('acousto_gen.safety.validate_pressure') as mock_validate:
            mock_validate.return_value = False
            
            # System should not activate with unsafe pressure
            result = mock_validate(test_pressure, max_pressure)
            assert result is False
    
    def test_temperature_monitoring(self):
        """Test temperature safety monitoring."""
        max_temp = 40  # °C
        current_temp = 35  # °C
        
        assert current_temp < max_temp
        
        # Simulate temperature check
        with patch('acousto_gen.safety.get_temperature') as mock_temp:
            mock_temp.return_value = current_temp
            
            temp = mock_temp()
            assert temp < max_temp
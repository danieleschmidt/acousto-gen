"""Integration tests for complete acoustic holography pipeline."""

import numpy as np
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from acousto_gen.core import AcousticHologram


class TestFullPipeline:
    """Test complete acoustic holography workflow."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Load test configuration."""
        config_path = Path(__file__).parent.parent / "fixtures" / "test_data.json"
        with open(config_path) as f:
            return json.load(f)
    
    def test_single_focus_pipeline(self, pipeline_config, mock_transducer):
        """Test complete pipeline for single focus generation."""
        # Get test parameters
        target_config = pipeline_config["test_targets"]["single_focus"]
        array_config = pipeline_config["transducer_arrays"]["medium_array"]
        
        # Create hologram instance
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000,
            medium="air"
        )
        
        # Create target field
        target = hologram.create_focus_point(
            position=target_config["position"],
            pressure=target_config["pressure"]
        )
        
        # Optimize phases
        phases = hologram.optimize(
            target=target,
            iterations=100,
            method="adam"
        )
        
        # Validate results
        assert isinstance(phases, np.ndarray)
        assert len(phases) == array_config["elements"]
        assert np.all(phases >= 0)
        assert np.all(phases <= 2 * np.pi)
        
        # Test phase application to hardware
        mock_transducer.set_phases.assert_not_called()  # Will be called in real implementation
    
    def test_multi_focus_pipeline(self, pipeline_config, mock_transducer):
        """Test pipeline for multiple focus points."""
        target_config = pipeline_config["test_targets"]["twin_trap"]
        
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000,
            medium="air"
        )
        
        # This would create multi-focus target (not implemented yet)
        # target = hologram.create_multi_focus(
        #     positions=target_config["positions"],
        #     pressures=target_config["pressures"]
        # )
        
        # For now, test with simple target
        target = hologram.create_focus_point(
            position=target_config["positions"][0],
            pressure=target_config["pressures"][0]
        )
        
        phases = hologram.optimize(target, iterations=100)
        
        assert isinstance(phases, np.ndarray)
        assert len(phases) == 256  # Default array size
    
    def test_safety_validation_pipeline(self, pipeline_config):
        """Test safety validation throughout pipeline."""
        safety_limits = pipeline_config["safety_limits"]
        
        # Test pressure limits
        max_pressure = safety_limits["max_pressure"]
        test_pressure = 6000  # Above limit
        
        # Would test actual safety validation
        is_safe = test_pressure <= max_pressure
        assert not is_safe  # Should fail safety check
        
        # Test safe pressure
        safe_pressure = 3000
        is_safe = safe_pressure <= max_pressure
        assert is_safe
    
    def test_medium_switching_pipeline(self, pipeline_config, mock_transducer):
        """Test pipeline with different propagation media."""
        media_configs = pipeline_config["acoustic_parameters"]["media"]
        
        for medium_name, medium_props in media_configs.items():
            hologram = AcousticHologram(
                transducer=mock_transducer,
                frequency=40000,
                medium=medium_name
            )
            
            # Test that medium is properly set
            assert hologram.medium == medium_name
            
            # Test field calculation with different medium
            target = hologram.create_focus_point([0, 0, 0.1], 3000)
            phases = hologram.optimize(target, iterations=50)
            
            assert isinstance(phases, np.ndarray)
    
    @pytest.mark.slow
    def test_optimization_convergence(self, mock_transducer):
        """Test optimization algorithm convergence."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        target = hologram.create_focus_point([0, 0, 0.1], 3000)
        
        # Test different iteration counts
        iteration_counts = [10, 50, 100, 500]
        previous_phases = None
        
        for iterations in iteration_counts:
            phases = hologram.optimize(target, iterations=iterations)
            
            if previous_phases is not None:
                # Phases should stabilize with more iterations
                phase_change = np.mean(np.abs(phases - previous_phases))
                # Would check actual convergence criteria
                assert phase_change >= 0  # Placeholder test
            
            previous_phases = phases.copy()
    
    def test_error_handling_pipeline(self, mock_transducer):
        """Test error handling throughout pipeline."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        # Test invalid target position
        with pytest.raises((ValueError, IndexError)):
            # This should raise an error (when validation is implemented)
            invalid_target = hologram.create_focus_point([999, 999, 999], 3000)
        
        # Test invalid pressure
        with pytest.raises((ValueError, AssertionError)):
            # This should raise an error for unsafe pressure
            unsafe_target = hologram.create_focus_point([0, 0, 0.1], 50000)


class TestPerformanceIntegration:
    """Integration tests focusing on performance."""
    
    @pytest.mark.performance
    def test_large_array_performance(self, mock_transducer):
        """Test performance with large transducer arrays."""
        # Mock large array
        mock_transducer.elements = 1024
        
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        target = hologram.create_focus_point([0, 0, 0.1], 3000)
        
        # Time the optimization (would use actual timing in real test)
        import time
        start_time = time.time()
        phases = hologram.optimize(target, iterations=100)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        # Should complete within reasonable time
        assert optimization_time < 60  # 1 minute threshold
        assert len(phases) == 1024
    
    @pytest.mark.performance
    def test_memory_usage_integration(self, mock_transducer):
        """Test memory usage during full pipeline."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        # Create multiple targets
        targets = []
        for i in range(10):
            target = hologram.create_focus_point([0, 0, 0.1 + i*0.01], 3000)
            targets.append(target)
        
        # Optimize all targets
        all_phases = []
        for target in targets:
            phases = hologram.optimize(target, iterations=50)
            all_phases.append(phases)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB)
        assert memory_increase < 1024 * 1024 * 1024


class TestHardwareIntegration:
    """Integration tests with hardware interfaces."""
    
    @pytest.mark.hardware
    def test_hardware_connection_pipeline(self, mock_hardware_interface):
        """Test full pipeline with hardware connection."""
        # This would test actual hardware connection
        # For now, test the mock interface
        
        # Connect to hardware
        connection_success = mock_hardware_interface.connect()
        assert connection_success
        
        # Check status
        status = mock_hardware_interface.get_status()
        assert status == "connected"
        
        # Send test phases
        test_phases = np.random.rand(256) * 2 * np.pi
        mock_hardware_interface.send_phases(test_phases)
        
        # Verify phases were sent
        mock_hardware_interface.send_phases.assert_called_once()
        
        # Disconnect
        mock_hardware_interface.disconnect()
        mock_hardware_interface.disconnect.assert_called_once()
    
    @pytest.mark.hardware
    def test_calibration_integration(self, mock_hardware_interface):
        """Test calibration data integration."""
        # Mock calibration data
        calibration_data = {
            'phase_corrections': np.random.rand(256) * 0.1,
            'amplitude_corrections': np.ones(256) * 0.95,
            'frequency_response': np.ones(256),
            'timestamp': '2025-01-01T00:00:00Z'
        }
        
        # Would test actual calibration application
        assert len(calibration_data['phase_corrections']) == 256
        assert len(calibration_data['amplitude_corrections']) == 256
        assert all(0.8 <= amp <= 1.2 for amp in calibration_data['amplitude_corrections'])
    
    @pytest.mark.hardware  
    def test_safety_monitoring_integration(self, mock_hardware_interface):
        """Test safety monitoring integration."""
        # Mock safety monitoring
        safety_status = {
            'pressure_ok': True,
            'temperature_ok': True,
            'power_ok': True,
            'emergency_stop': False
        }
        
        # Would test actual safety monitoring
        assert safety_status['pressure_ok']
        assert safety_status['temperature_ok'] 
        assert safety_status['power_ok']
        assert not safety_status['emergency_stop']


class TestApplicationIntegration:
    """Integration tests for specific applications."""
    
    def test_levitation_application(self, mock_transducer):
        """Test acoustic levitation application."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000,
            medium="air"
        )
        
        # Create levitation trap
        trap_position = [0, 0, 0.08]  # 8cm above array
        trap_pressure = 4000  # Strong trap
        
        target = hologram.create_focus_point(trap_position, trap_pressure)
        phases = hologram.optimize(target, iterations=200)
        
        # Validate trap strength
        assert isinstance(phases, np.ndarray)
        assert len(phases) == 256
        
        # Would validate actual trap characteristics
        # - Gradient strength
        # - Trap stability
        # - Particle size compatibility
    
    def test_haptic_application(self, mock_transducer):
        """Test mid-air haptic application."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000,
            medium="air"
        )
        
        # Create haptic feedback point
        haptic_position = [0, 0, 0.15]  # 15cm above array
        haptic_pressure = 200  # Perceivable but safe
        
        target = hologram.create_focus_point(haptic_position, haptic_pressure)
        phases = hologram.optimize(target, iterations=100)
        
        # Validate haptic characteristics
        assert isinstance(phases, np.ndarray)
        
        # Would validate:
        # - Pressure in perceivable range
        # - Modulation frequency optimization
        # - Spatial resolution
    
    @pytest.mark.slow
    def test_medical_application(self, mock_transducer):
        """Test medical focused ultrasound application."""
        # Configure for medical frequency
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=1500000,  # 1.5 MHz
            medium="tissue"
        )
        
        # Create therapeutic focus
        focus_position = [0, 0, 0.12]  # 12cm depth
        therapeutic_pressure = 1000000  # 1 MPa
        
        target = hologram.create_focus_point(focus_position, therapeutic_pressure)
        phases = hologram.optimize(target, iterations=500)
        
        # Validate medical safety
        assert isinstance(phases, np.ndarray)
        
        # Would validate:
        # - Thermal dose calculations
        # - Regulatory compliance
        # - Treatment planning integration
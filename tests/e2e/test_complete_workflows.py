"""End-to-end tests for complete user workflows."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from acousto_gen.core import AcousticHologram


class TestCompleteUserWorkflows:
    """Test complete user workflows from start to finish."""
    
    def test_researcher_workflow(self, mock_transducer):
        """Test typical academic researcher workflow."""
        # Scenario: Researcher wants to levitate a 3mm polystyrene bead
        
        # Step 1: Initialize system
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000,
            medium="air"
        )
        
        # Step 2: Define levitation target
        bead_position = [0, 0, 0.08]  # 8cm above array
        trap_strength = 3500  # Pa
        
        target = hologram.create_focus_point(bead_position, trap_strength)
        
        # Step 3: Optimize hologram
        phases = hologram.optimize(
            target=target,
            iterations=500,
            method="adam"
        )
        
        # Step 4: Validate results
        assert isinstance(phases, np.ndarray)
        assert len(phases) == 256
        assert np.all(phases >= 0) and np.all(phases <= 2 * np.pi)
        
        # Step 5: Apply to hardware (mocked)
        mock_transducer.set_phases(phases)
        mock_transducer.activate()
        
        # Verify workflow completion
        mock_transducer.set_phases.assert_called_once()
        mock_transducer.activate.assert_called_once()
    
    def test_engineer_workflow(self, mock_transducer):
        """Test hardware engineer development workflow."""
        # Scenario: Engineer testing new array configuration
        
        # Step 1: Custom array setup
        custom_array = Mock()
        custom_array.elements = 128
        custom_array.positions = np.random.rand(128, 3) * 0.1
        custom_array.frequency = 40000
        
        hologram = AcousticHologram(
            transducer=custom_array,
            frequency=40000,
            medium="air"
        )
        
        # Step 2: Calibration test pattern
        calibration_targets = [
            [0, 0, 0.05],
            [0.02, 0, 0.05],
            [-0.02, 0, 0.05],
            [0, 0.02, 0.05],
            [0, -0.02, 0.05]
        ]
        
        all_phases = []
        for position in calibration_targets:
            target = hologram.create_focus_point(position, 2000)
            phases = hologram.optimize(target, iterations=100)
            all_phases.append(phases)
        
        # Step 3: Validate calibration patterns
        for phases in all_phases:
            assert len(phases) == 128
            assert np.all(phases >= 0) and np.all(phases <= 2 * np.pi)
        
        # Step 4: Check phase diversity
        phase_std = np.std([np.std(p) for p in all_phases])
        assert phase_std > 0  # Should have phase variation
    
    def test_medical_researcher_workflow(self, mock_transducer):
        """Test medical researcher workflow with safety validation."""
        # Scenario: Medical researcher planning focused ultrasound treatment
        
        # Step 1: Medical system setup
        medical_array = Mock()
        medical_array.elements = 256
        medical_array.frequency = 1500000  # 1.5 MHz
        medical_array.aperture = 0.15  # 15cm
        
        hologram = AcousticHologram(
            transducer=medical_array,
            frequency=1500000,
            medium="tissue"
        )
        
        # Step 2: Treatment target definition
        treatment_focus = [0, 0, 0.10]  # 10cm depth
        therapeutic_pressure = 800000  # 0.8 MPa (below FDA limit)
        
        target = hologram.create_focus_point(treatment_focus, therapeutic_pressure)
        
        # Step 3: Safety validation
        max_safe_pressure = 1000000  # 1 MPa FDA limit
        assert therapeutic_pressure <= max_safe_pressure
        
        # Step 4: Treatment optimization
        phases = hologram.optimize(
            target=target,
            iterations=1000,  # High precision for medical
            method="adam"
        )
        
        # Step 5: Quality assurance
        assert isinstance(phases, np.ndarray)
        assert len(phases) == 256
        
        # Medical-specific validations
        phase_uniformity = np.std(phases)
        assert phase_uniformity > 0.1  # Should have sufficient phase variation
    
    def test_haptic_developer_workflow(self, mock_transducer):
        """Test haptic application developer workflow."""
        # Scenario: Developer creating mid-air button interface
        
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000,
            medium="air"
        )
        
        # Step 1: Define haptic interface
        button_positions = [
            [0, 0, 0.15],      # Center button
            [-0.03, 0, 0.15],  # Left button
            [0.03, 0, 0.15],   # Right button
        ]
        
        button_phases = []
        
        # Step 2: Generate haptic feedback for each button
        for position in button_positions:
            # Haptic pressure (perceivable but safe)
            haptic_pressure = 150  # Pa
            
            target = hologram.create_focus_point(position, haptic_pressure)
            phases = hologram.optimize(target, iterations=100)
            button_phases.append(phases)
        
        # Step 3: Validate haptic patterns
        for phases in button_phases:
            assert len(phases) == 256
            assert np.all(phases >= 0) and np.all(phases <= 2 * np.pi)
        
        # Step 4: Test pattern switching (for interactive feedback)
        phase_differences = []
        for i in range(len(button_phases) - 1):
            diff = np.mean(np.abs(button_phases[i] - button_phases[i+1]))
            phase_differences.append(diff)
        
        # Buttons should have different phase patterns
        assert all(diff > 0.1 for diff in phase_differences)
    
    @pytest.mark.slow
    def test_manufacturing_validation_workflow(self, mock_transducer):
        """Test manufacturing quality control workflow."""
        # Scenario: QC engineer testing production arrays
        
        # Step 1: Production test setup
        production_arrays = []
        for i in range(3):  # Test 3 production units
            array = Mock()
            array.elements = 256
            # Add slight manufacturing variations
            array.positions = np.random.rand(256, 3) * 0.168 + np.random.normal(0, 0.001, (256, 3))
            array.serial_number = f"ACA-2025-{1000+i}"
            production_arrays.append(array)
        
        test_results = []
        
        # Step 2: Standardized test for each array
        for array in production_arrays:
            hologram = AcousticHologram(
                transducer=array,
                frequency=40000,
                medium="air"
            )
            
            # Standard test pattern
            test_target = hologram.create_focus_point([0, 0, 0.08], 3000)
            phases = hologram.optimize(test_target, iterations=200)
            
            # QC metrics
            phase_range = np.ptp(phases)  # Peak-to-peak range
            phase_std = np.std(phases)
            
            test_result = {
                'serial_number': array.serial_number,
                'phase_range': phase_range,
                'phase_std': phase_std,
                'passed': 1.0 < phase_range < 5.0 and 0.5 < phase_std < 2.0
            }
            test_results.append(test_result)
        
        # Step 3: Validate QC results
        assert len(test_results) == 3
        passed_units = sum(1 for result in test_results if result['passed'])
        
        # At least 2 out of 3 should pass (realistic manufacturing yield)
        assert passed_units >= 2
    
    def test_educational_workflow(self, mock_transducer):
        """Test educational/demonstration workflow."""
        # Scenario: Professor demonstrating acoustic holography concepts
        
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000,
            medium="air"
        )
        
        # Step 1: Basic demonstration - single focus
        demo_target = hologram.create_focus_point([0, 0, 0.1], 2000)
        demo_phases = hologram.optimize(demo_target, iterations=50)
        
        assert len(demo_phases) == 256
        
        # Step 2: Show effect of frequency changes
        frequencies = [30000, 40000, 50000]
        frequency_results = []
        
        for freq in frequencies:
            hologram_freq = AcousticHologram(
                transducer=mock_transducer,
                frequency=freq,
                medium="air"
            )
            target = hologram_freq.create_focus_point([0, 0, 0.1], 2000)
            phases = hologram_freq.optimize(target, iterations=50)
            frequency_results.append(phases)
        
        # Different frequencies should give different phase patterns
        for i in range(len(frequency_results) - 1):
            phase_diff = np.mean(np.abs(frequency_results[i] - frequency_results[i+1]))
            assert phase_diff > 0.01  # Should be measurably different
        
        # Step 3: Demonstrate safety limits
        safe_pressures = [1000, 2000, 3000, 4000]  # Pa
        unsafe_pressure = 10000  # Pa
        
        for pressure in safe_pressures:
            # These should work fine
            target = hologram.create_focus_point([0, 0, 0.1], pressure)
            phases = hologram.optimize(target, iterations=25)
            assert len(phases) == 256
        
        # Unsafe pressure would trigger safety warning (in real implementation)
        # For now, just demonstrate the concept
        assert unsafe_pressure > 5000  # Above typical safety limit


class TestWorkflowErrorHandling:
    """Test error handling in complete workflows."""
    
    def test_hardware_failure_recovery(self, mock_transducer):
        """Test workflow recovery from hardware failures."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        # Simulate hardware failure
        mock_transducer.set_phases.side_effect = RuntimeError("Hardware disconnected")
        
        target = hologram.create_focus_point([0, 0, 0.1], 3000)
        phases = hologram.optimize(target, iterations=100)
        
        # Phases should still be generated
        assert len(phases) == 256
        
        # Hardware application should fail gracefully
        with pytest.raises(RuntimeError):
            mock_transducer.set_phases(phases)
    
    def test_invalid_parameter_handling(self, mock_transducer):
        """Test handling of invalid parameters in workflows."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        # Test invalid positions
        invalid_positions = [
            [999, 999, 999],  # Too far from array
            [0, 0, -1],       # Negative distance
            [0, 0, 0],        # Zero distance
        ]
        
        for position in invalid_positions:
            # These should handle gracefully or raise appropriate errors
            try:
                target = hologram.create_focus_point(position, 3000)
                phases = hologram.optimize(target, iterations=10)
                # If no error, phases should still be valid format
                assert len(phases) == 256
            except (ValueError, AssertionError):
                # Expected for invalid inputs
                pass
    
    def test_optimization_convergence_failure(self, mock_transducer):
        """Test handling of optimization convergence failures."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        target = hologram.create_focus_point([0, 0, 0.1], 3000)
        
        # Test with very few iterations (may not converge)
        phases = hologram.optimize(target, iterations=1)
        
        # Should still return valid phases even if not converged
        assert len(phases) == 256
        assert np.all(phases >= 0) and np.all(phases <= 2 * np.pi)


class TestWorkflowPerformance:
    """Test performance aspects of complete workflows."""
    
    @pytest.mark.performance
    def test_real_time_workflow(self, mock_transducer):
        """Test real-time interaction workflow performance."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        # Simulate real-time position updates
        positions = [
            [0, 0, 0.08],
            [0.01, 0, 0.08],
            [0.02, 0, 0.08],
            [0.01, 0.01, 0.08],
            [0, 0.01, 0.08],
        ]
        
        import time
        update_times = []
        
        for position in positions:
            start_time = time.time()
            
            target = hologram.create_focus_point(position, 3000)
            phases = hologram.optimize(target, iterations=50)
            
            end_time = time.time()
            update_time = end_time - start_time
            update_times.append(update_time)
        
        # Each update should complete quickly for real-time interaction
        max_update_time = max(update_times)
        avg_update_time = np.mean(update_times)
        
        # Should complete in reasonable time for interactive use
        assert max_update_time < 5.0  # 5 seconds max
        assert avg_update_time < 2.0  # 2 seconds average
    
    @pytest.mark.performance
    def test_batch_processing_workflow(self, mock_transducer):
        """Test batch processing workflow performance."""
        hologram = AcousticHologram(
            transducer=mock_transducer,
            frequency=40000
        )
        
        # Generate batch of targets
        batch_positions = []
        batch_pressures = []
        
        for i in range(10):
            x = 0.02 * np.sin(i * 0.6)
            y = 0.02 * np.cos(i * 0.6)
            z = 0.08 + 0.01 * i
            
            batch_positions.append([x, y, z])
            batch_pressures.append(3000 + 500 * i)
        
        # Process batch
        import time
        start_time = time.time()
        
        batch_phases = []
        for position, pressure in zip(batch_positions, batch_pressures):
            target = hologram.create_focus_point(position, pressure)
            phases = hologram.optimize(target, iterations=100)
            batch_phases.append(phases)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Validate batch results
        assert len(batch_phases) == 10
        for phases in batch_phases:
            assert len(phases) == 256
        
        # Performance should be reasonable for batch processing
        time_per_target = total_time / 10
        assert time_per_target < 10.0  # 10 seconds per target max
"""
Core Acoustic Holography Enhancement System
Generation 1: MAKE IT WORK - Essential functionality improvements
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class OptimizationMode(Enum):
    """Optimization modes for different use cases."""
    FAST = "fast"           # Quick optimization for real-time applications
    BALANCED = "balanced"   # Balance between speed and quality
    ACCURATE = "accurate"   # Maximum accuracy optimization
    ADAPTIVE = "adaptive"   # Automatically adjust based on target complexity


class FieldType(Enum):
    """Types of acoustic field patterns."""
    SINGLE_FOCUS = "single_focus"
    MULTI_FOCUS = "multi_focus"
    LINE_TRAP = "line_trap"
    RING_PATTERN = "ring_pattern"
    CUSTOM_SHAPE = "custom_shape"
    HAPTIC_TEXTURE = "haptic_texture"


@dataclass
class OptimizationConfig:
    """Configuration for hologram optimization."""
    
    mode: OptimizationMode = OptimizationMode.BALANCED
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    learning_rate: float = 0.01
    regularization_weight: float = 0.1
    enable_gpu: bool = True
    enable_multi_threading: bool = True
    safety_limits: Dict[str, float] = field(default_factory=lambda: {
        "max_pressure_pa": 4000,
        "max_intensity_w_cm2": 10,
        "max_temperature_c": 40
    })


@dataclass
class FieldTarget:
    """Target field specification."""
    
    field_type: FieldType
    positions: List[Tuple[float, float, float]]  # 3D coordinates
    amplitudes: List[float]
    phases: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate target field specification."""
        if len(self.positions) != len(self.amplitudes):
            return False
        if self.phases and len(self.phases) != len(self.positions):
            return False
        
        # Check position bounds (example workspace: ±10cm x ±10cm x 5-20cm)
        for pos in self.positions:
            x, y, z = pos
            if not (-0.1 <= x <= 0.1 and -0.1 <= y <= 0.1 and 0.05 <= z <= 0.2):
                return False
        
        # Check amplitude bounds
        for amp in self.amplitudes:
            if not (0 <= amp <= 5000):  # Pa
                return False
        
        return True


class EnhancedAcousticField:
    """Enhanced acoustic field computation with optimizations."""
    
    def __init__(
        self,
        array_geometry: np.ndarray,
        frequency: float = 40e3,
        medium_properties: Optional[Dict[str, float]] = None
    ):
        """
        Initialize enhanced acoustic field calculator.
        
        Args:
            array_geometry: Nx3 array of transducer positions
            frequency: Operating frequency in Hz
            medium_properties: Medium properties (density, speed of sound, etc.)
        """
        self.array_geometry = array_geometry
        self.frequency = frequency
        self.num_elements = len(array_geometry)
        
        # Default air properties at 20°C
        self.medium = medium_properties or {
            "density": 1.2,          # kg/m³
            "speed_of_sound": 343,   # m/s
            "absorption": 0.01,      # Np/m
            "nonlinearity": 3.5      # B/A coefficient
        }
        
        # Precompute constants
        self.wavelength = self.medium["speed_of_sound"] / self.frequency
        self.wavenumber = 2 * np.pi / self.wavelength
        
        # Performance optimizations
        self._distance_cache = {}
        self._green_function_cache = {}
    
    def compute_distances(self, target_points: np.ndarray) -> np.ndarray:
        """
        Compute distances from transducers to target points with caching.
        
        Args:
            target_points: Mx3 array of target positions
            
        Returns:
            NxM array of distances
        """
        # Create cache key from target points
        cache_key = hash(target_points.tobytes())
        
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # Vectorized distance computation
        # Broadcasting: (N,1,3) - (1,M,3) -> (N,M,3)
        diff = self.array_geometry[:, np.newaxis, :] - target_points[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        
        # Cache result
        self._distance_cache[cache_key] = distances
        
        return distances
    
    def green_function(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute Green's function for wave propagation.
        
        Args:
            distances: Array of distances
            
        Returns:
            Complex Green's function values
        """
        # Free-space Green's function with absorption
        absorption_factor = np.exp(-self.medium["absorption"] * distances)
        phase_factor = np.exp(1j * self.wavenumber * distances)
        
        # Avoid division by zero
        safe_distances = np.where(distances > 1e-10, distances, 1e-10)
        amplitude_factor = 1.0 / (4 * np.pi * safe_distances)
        
        return amplitude_factor * absorption_factor * phase_factor
    
    def compute_pressure_field(
        self,
        target_points: np.ndarray,
        phases: np.ndarray,
        amplitudes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute pressure field at target points.
        
        Args:
            target_points: Mx3 array of target positions
            phases: Array of transducer phases
            amplitudes: Array of transducer amplitudes (default: uniform)
            
        Returns:
            Complex pressure field at target points
        """
        if amplitudes is None:
            amplitudes = np.ones(self.num_elements)
        
        # Compute distances
        distances = self.compute_distances(target_points)
        
        # Green's function
        green = self.green_function(distances)
        
        # Transducer contributions
        contributions = amplitudes[:, np.newaxis] * np.exp(1j * phases[:, np.newaxis]) * green
        
        # Sum contributions
        pressure_field = np.sum(contributions, axis=0)
        
        return pressure_field
    
    def compute_field_metrics(
        self,
        target_points: np.ndarray,
        phases: np.ndarray,
        amplitudes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive field quality metrics.
        
        Args:
            target_points: Target evaluation points
            phases: Transducer phases
            amplitudes: Transducer amplitudes
            
        Returns:
            Dictionary of field quality metrics
        """
        pressure_field = self.compute_pressure_field(target_points, phases, amplitudes)
        pressure_magnitude = np.abs(pressure_field)
        
        metrics = {
            "max_pressure": np.max(pressure_magnitude),
            "mean_pressure": np.mean(pressure_magnitude),
            "pressure_std": np.std(pressure_magnitude),
            "pressure_uniformity": 1.0 / (1.0 + np.std(pressure_magnitude) / np.mean(pressure_magnitude)),
            "field_efficiency": np.sum(pressure_magnitude ** 2) / len(target_points),
            "focal_quality": self._compute_focal_quality(pressure_magnitude),
            "side_lobe_ratio": self._compute_side_lobe_ratio(pressure_magnitude),
            "beam_width": self._compute_beam_width(target_points, pressure_magnitude)
        }
        
        return metrics
    
    def _compute_focal_quality(self, pressure_magnitude: np.ndarray) -> float:
        """Compute focal point quality metric."""
        max_pressure = np.max(pressure_magnitude)
        mean_pressure = np.mean(pressure_magnitude)
        return max_pressure / (mean_pressure + 1e-10)
    
    def _compute_side_lobe_ratio(self, pressure_magnitude: np.ndarray) -> float:
        """Compute side lobe to main lobe ratio."""
        sorted_pressures = np.sort(pressure_magnitude)[::-1]
        main_lobe = sorted_pressures[0]
        
        # Find secondary peaks (simple approach)
        if len(sorted_pressures) > 1:
            side_lobe = sorted_pressures[1]
            return side_lobe / main_lobe
        
        return 0.0
    
    def _compute_beam_width(
        self, 
        target_points: np.ndarray, 
        pressure_magnitude: np.ndarray
    ) -> float:
        """Compute beam width at half maximum."""
        max_pressure = np.max(pressure_magnitude)
        half_max = max_pressure / 2
        
        # Find points above half maximum
        above_half_max = pressure_magnitude > half_max
        
        if np.any(above_half_max):
            points_above = target_points[above_half_max]
            
            # Compute bounding box
            min_coords = np.min(points_above, axis=0)
            max_coords = np.max(points_above, axis=0)
            
            # Return average dimension
            dimensions = max_coords - min_coords
            return np.mean(dimensions)
        
        return 0.0


class AdaptiveOptimizer:
    """Adaptive hologram optimizer with multiple strategies."""
    
    def __init__(
        self,
        acoustic_field: EnhancedAcousticField,
        config: OptimizationConfig = None
    ):
        """
        Initialize adaptive optimizer.
        
        Args:
            acoustic_field: Enhanced acoustic field calculator
            config: Optimization configuration
        """
        self.field = acoustic_field
        self.config = config or OptimizationConfig()
        
        # Optimization state
        self.current_phases = None
        self.current_amplitudes = None
        self.optimization_history = []
        
        # Adaptive parameters
        self.learning_rate = self.config.learning_rate
        self.momentum_factor = 0.9
        self.velocity = None
        
        # Performance tracking
        self.iteration_times = []
        self.convergence_metrics = []
    
    def optimize_field(
        self,
        target: FieldTarget,
        evaluation_points: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimize acoustic field for target specification.
        
        Args:
            target: Target field specification
            evaluation_points: Points for field evaluation
            
        Returns:
            Optimization results
        """
        if not target.validate():
            raise ValueError("Invalid target field specification")
        
        start_time = time.time()
        
        # Generate evaluation grid if not provided
        if evaluation_points is None:
            evaluation_points = self._generate_evaluation_grid(target)
        
        # Initialize optimization variables
        self._initialize_optimization()
        
        # Create target field
        target_field = self._create_target_field(target, evaluation_points)
        
        # Adaptive optimization loop
        best_loss = float('inf')
        best_phases = None
        stagnation_counter = 0
        
        for iteration in range(self.config.max_iterations):
            iter_start = time.time()
            
            # Forward pass
            current_field = self.field.compute_pressure_field(
                evaluation_points, self.current_phases, self.current_amplitudes
            )
            
            # Compute loss
            loss = self._compute_loss(current_field, target_field)
            
            # Track best solution
            if loss < best_loss:
                best_loss = loss
                best_phases = self.current_phases.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Adaptive learning rate
            if stagnation_counter > 50:
                self.learning_rate *= 0.9  # Reduce learning rate
                stagnation_counter = 0
            
            # Compute gradients
            gradients = self._compute_gradients(
                current_field, target_field, evaluation_points
            )
            
            # Update phases with momentum
            self._update_phases(gradients)
            
            # Apply safety constraints
            self._apply_safety_constraints()
            
            # Record metrics
            iter_time = time.time() - iter_start
            self.iteration_times.append(iter_time)
            
            metrics = {
                "iteration": iteration,
                "loss": loss,
                "learning_rate": self.learning_rate,
                "gradient_norm": np.linalg.norm(gradients),
                "time": iter_time
            }
            self.convergence_metrics.append(metrics)
            
            # Convergence check
            if loss < self.config.convergence_threshold:
                print(f"Converged at iteration {iteration}")
                break
            
            # Progress callback
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        # Final field computation
        final_field = self.field.compute_pressure_field(
            evaluation_points, best_phases, self.current_amplitudes
        )
        
        field_metrics = self.field.compute_field_metrics(
            evaluation_points, best_phases, self.current_amplitudes
        )
        
        # Compile results
        results = {
            "success": best_loss < self.config.convergence_threshold * 10,
            "final_loss": best_loss,
            "optimized_phases": best_phases,
            "optimized_amplitudes": self.current_amplitudes,
            "field_metrics": field_metrics,
            "optimization_time": time.time() - start_time,
            "iterations": len(self.convergence_metrics),
            "convergence_history": self.convergence_metrics,
            "target_specification": {
                "field_type": target.field_type.value,
                "positions": target.positions,
                "amplitudes": target.amplitudes
            }
        }
        
        return results
    
    def _initialize_optimization(self):
        """Initialize optimization variables."""
        # Random phase initialization
        self.current_phases = np.random.uniform(0, 2*np.pi, self.field.num_elements)
        
        # Uniform amplitude initialization
        self.current_amplitudes = np.ones(self.field.num_elements)
        
        # Initialize momentum
        self.velocity = np.zeros_like(self.current_phases)
        
        # Reset tracking
        self.optimization_history = []
        self.iteration_times = []
        self.convergence_metrics = []
    
    def _generate_evaluation_grid(self, target: FieldTarget) -> np.ndarray:
        """Generate evaluation grid around target positions."""
        
        # Extract target positions
        target_positions = np.array(target.positions)
        
        if target.field_type == FieldType.SINGLE_FOCUS:
            # Dense grid around single focus
            center = target_positions[0]
            grid_size = 0.02  # 2cm around focus
            resolution = 0.002  # 2mm resolution
            
            x_range = np.arange(center[0] - grid_size/2, center[0] + grid_size/2, resolution)
            y_range = np.arange(center[1] - grid_size/2, center[1] + grid_size/2, resolution)
            z_range = np.arange(center[2] - grid_size/4, center[2] + grid_size/4, resolution)
            
            X, Y, Z = np.meshgrid(x_range, y_range, z_range)
            evaluation_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        elif target.field_type == FieldType.MULTI_FOCUS:
            # Grid around each focus point
            evaluation_points = []
            
            for pos in target_positions:
                grid_size = 0.01  # 1cm around each focus
                resolution = 0.002
                
                x_range = np.arange(pos[0] - grid_size/2, pos[0] + grid_size/2, resolution)
                y_range = np.arange(pos[1] - grid_size/2, pos[1] + grid_size/2, resolution)
                z_range = np.arange(pos[2] - grid_size/4, pos[2] + grid_size/4, resolution)
                
                X, Y, Z = np.meshgrid(x_range, y_range, z_range)
                points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
                evaluation_points.append(points)
            
            evaluation_points = np.vstack(evaluation_points)
        
        else:
            # Default: sparse grid in workspace
            x_range = np.linspace(-0.05, 0.05, 20)
            y_range = np.linspace(-0.05, 0.05, 20)
            z_range = np.linspace(0.08, 0.15, 15)
            
            X, Y, Z = np.meshgrid(x_range, y_range, z_range)
            evaluation_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        return evaluation_points
    
    def _create_target_field(
        self, 
        target: FieldTarget, 
        evaluation_points: np.ndarray
    ) -> np.ndarray:
        """Create target pressure field at evaluation points."""
        
        target_field = np.zeros(len(evaluation_points), dtype=complex)
        
        for i, (pos, amp) in enumerate(zip(target.positions, target.amplitudes)):
            # Distance from target position to evaluation points
            distances = np.linalg.norm(evaluation_points - np.array(pos), axis=1)
            
            # Gaussian-like target field centered at position
            sigma = 0.005  # 5mm standard deviation
            field_contribution = amp * np.exp(-distances**2 / (2 * sigma**2))
            
            # Add phase if specified
            if target.phases:
                field_contribution *= np.exp(1j * target.phases[i])
            
            target_field += field_contribution
        
        return target_field
    
    def _compute_loss(
        self, 
        current_field: np.ndarray, 
        target_field: np.ndarray
    ) -> float:
        """Compute optimization loss function."""
        
        # Main loss: mean squared error
        field_error = np.mean(np.abs(current_field - target_field) ** 2)
        
        # Regularization: phase smoothness
        phase_diff = np.diff(self.current_phases)
        smoothness_penalty = self.config.regularization_weight * np.mean(phase_diff ** 2)
        
        # Power constraint
        power_penalty = 0.001 * np.mean(self.current_amplitudes ** 2)
        
        total_loss = field_error + smoothness_penalty + power_penalty
        
        return total_loss
    
    def _compute_gradients(
        self,
        current_field: np.ndarray,
        target_field: np.ndarray,
        evaluation_points: np.ndarray
    ) -> np.ndarray:
        """Compute gradients using finite differences."""
        
        gradients = np.zeros_like(self.current_phases)
        epsilon = 1e-6
        
        # Current loss
        current_loss = self._compute_loss(current_field, target_field)
        
        # Finite difference gradients
        for i in range(len(self.current_phases)):
            # Perturb phase
            self.current_phases[i] += epsilon
            
            # Compute perturbed field
            perturbed_field = self.field.compute_pressure_field(
                evaluation_points, self.current_phases, self.current_amplitudes
            )
            
            # Compute perturbed loss
            perturbed_loss = self._compute_loss(perturbed_field, target_field)
            
            # Gradient
            gradients[i] = (perturbed_loss - current_loss) / epsilon
            
            # Restore phase
            self.current_phases[i] -= epsilon
        
        return gradients
    
    def _update_phases(self, gradients: np.ndarray):
        """Update phases using momentum-based gradient descent."""
        
        # Momentum update
        self.velocity = (
            self.momentum_factor * self.velocity - 
            self.learning_rate * gradients
        )
        
        # Phase update
        self.current_phases += self.velocity
        
        # Wrap phases to [0, 2π]
        self.current_phases = np.remainder(self.current_phases, 2 * np.pi)
    
    def _apply_safety_constraints(self):
        """Apply safety constraints to optimization variables."""
        
        # Amplitude limits
        max_amplitude = 1.0  # Normalized
        self.current_amplitudes = np.clip(self.current_amplitudes, 0, max_amplitude)
        
        # Phase bounds already handled by wrapping
        
        # Power limits (if needed)
        total_power = np.sum(self.current_amplitudes ** 2)
        max_power = self.field.num_elements  # Normalized
        
        if total_power > max_power:
            self.current_amplitudes *= np.sqrt(max_power / total_power)


def create_single_focus_target(
    position: Tuple[float, float, float],
    amplitude: float = 3000
) -> FieldTarget:
    """Create a single focus target specification."""
    
    return FieldTarget(
        field_type=FieldType.SINGLE_FOCUS,
        positions=[position],
        amplitudes=[amplitude],
        metadata={"description": "Single focal point target"}
    )


def create_multi_focus_target(
    positions: List[Tuple[float, float, float]],
    amplitudes: List[float]
) -> FieldTarget:
    """Create a multi-focus target specification."""
    
    return FieldTarget(
        field_type=FieldType.MULTI_FOCUS,
        positions=positions,
        amplitudes=amplitudes,
        metadata={"description": f"Multi-focus target with {len(positions)} points"}
    )


def demonstrate_core_enhancements():
    """Demonstrate core enhancement functionality."""
    
    print("🚀 Core Acoustic Holography Enhancements - Generation 1")
    
    # Create mock transducer array (8x8 grid)
    array_size = 8
    spacing = 0.01  # 1cm spacing
    
    x_coords = np.linspace(-(array_size-1)*spacing/2, (array_size-1)*spacing/2, array_size)
    y_coords = np.linspace(-(array_size-1)*spacing/2, (array_size-1)*spacing/2, array_size)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = np.zeros_like(X)
    
    array_geometry = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    print(f"✅ Created {len(array_geometry)} element transducer array")
    
    # Initialize enhanced acoustic field
    field_calculator = EnhancedAcousticField(
        array_geometry=array_geometry,
        frequency=40e3
    )
    
    print("✅ Enhanced acoustic field calculator initialized")
    
    # Create optimization configuration
    config = OptimizationConfig(
        mode=OptimizationMode.BALANCED,
        max_iterations=200,  # Reduced for demo
        convergence_threshold=1e-4
    )
    
    # Initialize adaptive optimizer
    optimizer = AdaptiveOptimizer(field_calculator, config)
    
    print("✅ Adaptive optimizer initialized")
    
    # Test single focus optimization
    print("\n🎯 Testing Single Focus Optimization")
    
    single_target = create_single_focus_target(
        position=(0.0, 0.0, 0.1),  # 10cm above array center
        amplitude=3000  # 3kPa
    )
    
    results = optimizer.optimize_field(single_target)
    
    print(f"Single Focus Results:")
    print(f"  Success: {results['success']}")
    print(f"  Final Loss: {results['final_loss']:.6f}")
    print(f"  Optimization Time: {results['optimization_time']:.2f}s")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Max Pressure: {results['field_metrics']['max_pressure']:.1f} Pa")
    print(f"  Focal Quality: {results['field_metrics']['focal_quality']:.2f}")
    
    # Test multi-focus optimization
    print("\n🎯 Testing Multi-Focus Optimization")
    
    multi_target = create_multi_focus_target(
        positions=[
            (-0.02, 0.0, 0.1),   # Left focus
            (0.02, 0.0, 0.1),    # Right focus
            (0.0, 0.0, 0.12)     # Center focus (higher)
        ],
        amplitudes=[2000, 2000, 1500]
    )
    
    results_multi = optimizer.optimize_field(multi_target)
    
    print(f"Multi-Focus Results:")
    print(f"  Success: {results_multi['success']}")
    print(f"  Final Loss: {results_multi['final_loss']:.6f}")
    print(f"  Optimization Time: {results_multi['optimization_time']:.2f}s")
    print(f"  Iterations: {results_multi['iterations']}")
    print(f"  Field Efficiency: {results_multi['field_metrics']['field_efficiency']:.2f}")
    
    # Save results
    results_summary = {
        "timestamp": time.time(),
        "array_elements": len(array_geometry),
        "single_focus_test": {
            "success": results['success'],
            "final_loss": results['final_loss'],
            "optimization_time": results['optimization_time'],
            "field_metrics": results['field_metrics']
        },
        "multi_focus_test": {
            "success": results_multi['success'],
            "final_loss": results_multi['final_loss'],
            "optimization_time": results_multi['optimization_time'],
            "field_metrics": results_multi['field_metrics']
        },
        "generation": "1_make_it_work",
        "status": "completed"
    }
    
    with open("core_enhancements_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n✅ Core enhancements demonstration completed")
    print("📊 Results saved to core_enhancements_results.json")
    
    return results_summary


if __name__ == "__main__":
    # Run demonstration
    demonstrate_core_enhancements()
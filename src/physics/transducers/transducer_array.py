"""
Transducer array models and configurations for acoustic holography.
Supports various commercial and custom array geometries.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json


@dataclass
class TransducerElement:
    """Individual transducer element properties."""
    position: np.ndarray  # 3D position [x, y, z]
    orientation: np.ndarray  # 3D orientation vector
    radius: float  # Element radius in meters
    efficiency: float = 1.0  # Conversion efficiency
    phase_offset: float = 0.0  # Calibration phase offset
    amplitude_factor: float = 1.0  # Calibration amplitude factor
    
    def get_directivity(self, angle: float, frequency: float) -> float:
        """
        Calculate directivity pattern for the transducer element.
        
        Args:
            angle: Angle from element normal in radians
            frequency: Operating frequency in Hz
            
        Returns:
            Directivity factor (0-1)
        """
        # Piston radiator directivity
        c = 343  # Speed of sound in air
        k = 2 * np.pi * frequency / c
        ka = k * self.radius
        
        if angle == 0:
            return 1.0
        
        # Bessel function approximation for directivity
        x = ka * np.sin(angle)
        if x < 1e-6:
            return 1.0
        
        # First-order Bessel function
        directivity = 2 * np.abs(np.sin(x) / x)
        return min(1.0, directivity)


class TransducerArray(ABC):
    """Base class for transducer arrays."""
    
    def __init__(
        self,
        frequency: float = 40e3,
        element_radius: float = 5e-3,
        name: str = "Generic Array"
    ):
        """
        Initialize transducer array.
        
        Args:
            frequency: Operating frequency in Hz
            element_radius: Radius of individual elements in meters
            name: Array identifier
        """
        self.frequency = frequency
        self.element_radius = element_radius
        self.name = name
        self.elements: List[TransducerElement] = []
        self.current_phases = None
        self.current_amplitudes = None
        
        # Initialize array geometry
        self._setup_array()
    
    @abstractmethod
    def _setup_array(self):
        """Setup array geometry - must be implemented by subclasses."""
        pass
    
    def get_positions(self) -> np.ndarray:
        """Get all element positions as Nx3 array."""
        return np.array([elem.position for elem in self.elements])
    
    def get_orientations(self) -> np.ndarray:
        """Get all element orientations as Nx3 array."""
        return np.array([elem.orientation for elem in self.elements])
    
    def set_phases(self, phases: np.ndarray):
        """
        Set phase values for all elements.
        
        Args:
            phases: Array of phase values in radians
        """
        if len(phases) != len(self.elements):
            raise ValueError(f"Expected {len(self.elements)} phases, got {len(phases)}")
        
        self.current_phases = np.array(phases)
    
    def set_amplitudes(self, amplitudes: np.ndarray):
        """
        Set amplitude values for all elements.
        
        Args:
            amplitudes: Array of amplitude values (0-1)
        """
        if len(amplitudes) != len(self.elements):
            raise ValueError(f"Expected {len(self.elements)} amplitudes, got {len(amplitudes)}")
        
        # Clip to valid range
        self.current_amplitudes = np.clip(amplitudes, 0, 1)
    
    def get_effective_amplitudes(self) -> np.ndarray:
        """Get effective amplitudes including calibration factors."""
        if self.current_amplitudes is None:
            self.current_amplitudes = np.ones(len(self.elements))
        
        factors = np.array([elem.amplitude_factor * elem.efficiency 
                          for elem in self.elements])
        return self.current_amplitudes * factors
    
    def get_effective_phases(self) -> np.ndarray:
        """Get effective phases including calibration offsets."""
        if self.current_phases is None:
            self.current_phases = np.zeros(len(self.elements))
        
        offsets = np.array([elem.phase_offset for elem in self.elements])
        return self.current_phases + offsets
    
    def apply_calibration(self, calibration_data: Dict[str, Any]):
        """
        Apply calibration data to array elements.
        
        Args:
            calibration_data: Dictionary with phase_offsets and amplitude_factors
        """
        if "phase_offsets" in calibration_data:
            offsets = calibration_data["phase_offsets"]
            for i, elem in enumerate(self.elements):
                if i < len(offsets):
                    elem.phase_offset = offsets[i]
        
        if "amplitude_factors" in calibration_data:
            factors = calibration_data["amplitude_factors"]
            for i, elem in enumerate(self.elements):
                if i < len(factors):
                    elem.amplitude_factor = factors[i]
    
    def save_configuration(self, filepath: str):
        """Save array configuration to JSON file."""
        config = {
            "name": self.name,
            "frequency": self.frequency,
            "element_radius": self.element_radius,
            "num_elements": len(self.elements),
            "elements": [
                {
                    "position": elem.position.tolist(),
                    "orientation": elem.orientation.tolist(),
                    "phase_offset": elem.phase_offset,
                    "amplitude_factor": elem.amplitude_factor
                }
                for elem in self.elements
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_configuration(self, filepath: str):
        """Load array configuration from JSON file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.name = config["name"]
        self.frequency = config["frequency"]
        self.element_radius = config["element_radius"]
        
        self.elements = []
        for elem_data in config["elements"]:
            elem = TransducerElement(
                position=np.array(elem_data["position"]),
                orientation=np.array(elem_data["orientation"]),
                radius=self.element_radius,
                phase_offset=elem_data.get("phase_offset", 0),
                amplitude_factor=elem_data.get("amplitude_factor", 1)
            )
            self.elements.append(elem)


class UltraLeap256(TransducerArray):
    """UltraLeap 256-element array (16x16 grid)."""
    
    def __init__(self):
        super().__init__(
            frequency=40e3,
            element_radius=5e-3,
            name="UltraLeap 256"
        )
    
    def _setup_array(self):
        """Setup 16x16 grid array geometry."""
        # Array parameters
        pitch = 10.5e-3  # Element pitch
        nx, ny = 16, 16
        
        # Create grid
        for i in range(nx):
            for j in range(ny):
                x = (i - nx/2 + 0.5) * pitch
                y = (j - ny/2 + 0.5) * pitch
                z = 0
                
                elem = TransducerElement(
                    position=np.array([x, y, z]),
                    orientation=np.array([0, 0, 1]),  # Pointing up
                    radius=self.element_radius
                )
                self.elements.append(elem)


class CircularArray(TransducerArray):
    """Circular/ring array configuration."""
    
    def __init__(
        self,
        radius: float = 0.1,
        num_elements: int = 64,
        frequency: float = 40e3
    ):
        self.array_radius = radius
        self.num_elements = num_elements
        super().__init__(
            frequency=frequency,
            element_radius=5e-3,
            name=f"Circular Array R={radius}m N={num_elements}"
        )
    
    def _setup_array(self):
        """Setup circular array geometry."""
        for i in range(self.num_elements):
            angle = 2 * np.pi * i / self.num_elements
            x = self.array_radius * np.cos(angle)
            y = self.array_radius * np.sin(angle)
            z = 0
            
            # Elements point toward center
            orientation = np.array([-np.cos(angle), -np.sin(angle), 0])
            
            elem = TransducerElement(
                position=np.array([x, y, z]),
                orientation=orientation,
                radius=self.element_radius
            )
            self.elements.append(elem)


class HemisphericalArray(TransducerArray):
    """Hemispherical array for focused ultrasound."""
    
    def __init__(
        self,
        radius: float = 0.15,
        num_rings: int = 10,
        elements_per_ring: int = 32,
        frequency: float = 1.5e6
    ):
        self.hemisphere_radius = radius
        self.num_rings = num_rings
        self.elements_per_ring = elements_per_ring
        super().__init__(
            frequency=frequency,
            element_radius=10e-3,
            name=f"Hemispherical Array R={radius}m"
        )
    
    def _setup_array(self):
        """Setup hemispherical array geometry."""
        # Distribute elements on hemisphere
        for ring in range(1, self.num_rings + 1):
            # Elevation angle for this ring
            theta = (np.pi / 2) * ring / self.num_rings
            ring_radius = self.hemisphere_radius * np.sin(theta)
            z = self.hemisphere_radius * np.cos(theta)
            
            for i in range(self.elements_per_ring):
                phi = 2 * np.pi * i / self.elements_per_ring
                x = ring_radius * np.cos(phi)
                y = ring_radius * np.sin(phi)
                
                # Normal points toward center
                position = np.array([x, y, z])
                orientation = -position / np.linalg.norm(position)
                
                elem = TransducerElement(
                    position=position,
                    orientation=orientation,
                    radius=self.element_radius
                )
                self.elements.append(elem)


class CustomArray(TransducerArray):
    """Custom array with user-defined geometry."""
    
    def __init__(
        self,
        positions: np.ndarray,
        orientations: Optional[np.ndarray] = None,
        frequency: float = 40e3,
        element_radius: float = 5e-3
    ):
        """
        Create custom array from positions and orientations.
        
        Args:
            positions: Nx3 array of element positions
            orientations: Optional Nx3 array of orientations (defaults to [0,0,1])
            frequency: Operating frequency
            element_radius: Element radius
        """
        self.custom_positions = positions
        self.custom_orientations = orientations
        super().__init__(
            frequency=frequency,
            element_radius=element_radius,
            name="Custom Array"
        )
    
    def _setup_array(self):
        """Setup custom array from provided positions."""
        num_elements = len(self.custom_positions)
        
        if self.custom_orientations is None:
            # Default to pointing up
            self.custom_orientations = np.tile([0, 0, 1], (num_elements, 1))
        
        for i in range(num_elements):
            elem = TransducerElement(
                position=self.custom_positions[i],
                orientation=self.custom_orientations[i],
                radius=self.element_radius
            )
            self.elements.append(elem)


class ArrayCalibrator:
    """Calibration system for transducer arrays."""
    
    def __init__(self, array: TransducerArray):
        """
        Initialize calibrator for a transducer array.
        
        Args:
            array: TransducerArray to calibrate
        """
        self.array = array
        self.calibration_data = {
            "phase_offsets": np.zeros(len(array.elements)),
            "amplitude_factors": np.ones(len(array.elements))
        }
    
    def holographic_calibration(
        self,
        measured_fields: List[np.ndarray],
        expected_fields: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Perform holographic calibration using field measurements.
        
        Args:
            measured_fields: List of measured complex fields
            expected_fields: List of expected complex fields
            
        Returns:
            Calibration data dictionary
        """
        num_elements = len(self.array.elements)
        phase_corrections = []
        amplitude_corrections = []
        
        for measured, expected in zip(measured_fields, expected_fields):
            # Complex field ratios
            ratio = expected / (measured + 1e-10)
            
            # Extract phase and amplitude corrections
            phase_corr = np.angle(ratio)
            amp_corr = np.abs(ratio)
            
            phase_corrections.append(phase_corr)
            amplitude_corrections.append(amp_corr)
        
        # Average corrections
        avg_phase = np.mean(phase_corrections, axis=0)
        avg_amplitude = np.mean(amplitude_corrections, axis=0)
        
        # Update calibration data
        self.calibration_data["phase_offsets"] = avg_phase
        self.calibration_data["amplitude_factors"] = avg_amplitude
        
        return self.calibration_data
    
    def iterative_calibration(
        self,
        target_field: np.ndarray,
        measurement_function,
        iterations: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Iterative calibration using feedback from measurements.
        
        Args:
            target_field: Desired field pattern
            measurement_function: Function to measure actual field
            iterations: Number of calibration iterations
            
        Returns:
            Optimized calibration data
        """
        for iteration in range(iterations):
            # Apply current calibration
            self.array.apply_calibration(self.calibration_data)
            
            # Measure field
            measured = measurement_function()
            
            # Compute error
            error = target_field - measured
            error_magnitude = np.mean(np.abs(error))
            
            print(f"Iteration {iteration + 1}: Error = {error_magnitude:.4f}")
            
            # Update corrections
            learning_rate = 0.1
            phase_update = learning_rate * np.angle(error)
            amp_update = learning_rate * np.abs(error) / (np.abs(measured) + 1e-10)
            
            self.calibration_data["phase_offsets"] += np.mean(phase_update)
            self.calibration_data["amplitude_factors"] *= (1 + np.mean(amp_update))
            
            # Clip amplitude factors to reasonable range
            self.calibration_data["amplitude_factors"] = np.clip(
                self.calibration_data["amplitude_factors"], 0.5, 2.0
            )
        
        return self.calibration_data
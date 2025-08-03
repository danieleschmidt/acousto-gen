"""
Acoustic field models and representations.
Core data structures for acoustic holography computations.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import h5py
import json


@dataclass
class FieldPoint:
    """Single point in an acoustic field."""
    position: np.ndarray  # 3D position [x, y, z]
    pressure: complex  # Complex pressure
    velocity: Optional[np.ndarray] = None  # 3D velocity vector
    intensity: Optional[float] = None  # Acoustic intensity
    
    def get_amplitude(self) -> float:
        """Get pressure amplitude."""
        return abs(self.pressure)
    
    def get_phase(self) -> float:
        """Get pressure phase in radians."""
        return np.angle(self.pressure)


@dataclass
class AcousticField:
    """
    3D acoustic pressure field representation.
    Stores complex pressure values on a regular grid.
    """
    data: np.ndarray  # Complex pressure field
    bounds: List[Tuple[float, float]]  # Physical bounds [(xmin,xmax), ...]
    resolution: float  # Spatial resolution
    frequency: float  # Operating frequency
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and setup field properties."""
        # Ensure data is complex
        if not np.iscomplexobj(self.data):
            self.data = self.data.astype(complex)
        
        # Calculate grid properties
        self.shape = self.data.shape
        self.nx, self.ny, self.nz = self.shape
        
        # Create coordinate arrays
        self._setup_coordinates()
    
    def _setup_coordinates(self):
        """Setup coordinate arrays for the field."""
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.nx)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.ny)
        z = np.linspace(self.bounds[2][0], self.bounds[2][1], self.nz)
        
        self.x_coords = x
        self.y_coords = y
        self.z_coords = z
        
        # Create meshgrids
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
    
    def get_amplitude_field(self) -> np.ndarray:
        """Get pressure amplitude field."""
        return np.abs(self.data)
    
    def get_phase_field(self) -> np.ndarray:
        """Get pressure phase field in radians."""
        return np.angle(self.data)
    
    def get_intensity_field(self, medium_impedance: float = 410) -> np.ndarray:
        """
        Calculate acoustic intensity field.
        
        Args:
            medium_impedance: Acoustic impedance of medium (default air)
            
        Returns:
            Time-averaged intensity in W/m²
        """
        amplitude = self.get_amplitude_field()
        intensity = amplitude**2 / (2 * medium_impedance)
        return intensity
    
    def get_slice(
        self,
        plane: str = "xy",
        position: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 2D slice through the field.
        
        Args:
            plane: Slice plane ('xy', 'xz', 'yz')
            position: Position along normal axis
            
        Returns:
            Tuple of (x_coords, y_coords, field_slice)
        """
        if plane == "xy":
            # Find closest z index
            z_idx = np.argmin(np.abs(self.z_coords - position))
            slice_data = self.data[:, :, z_idx]
            return self.x_coords, self.y_coords, slice_data
        
        elif plane == "xz":
            # Find closest y index
            y_idx = np.argmin(np.abs(self.y_coords - position))
            slice_data = self.data[:, y_idx, :]
            return self.x_coords, self.z_coords, slice_data
        
        elif plane == "yz":
            # Find closest x index
            x_idx = np.argmin(np.abs(self.x_coords - position))
            slice_data = self.data[x_idx, :, :]
            return self.y_coords, self.z_coords, slice_data
        
        else:
            raise ValueError(f"Unknown plane: {plane}")
    
    def interpolate_at_points(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate field values at arbitrary points.
        
        Args:
            points: Nx3 array of query points
            
        Returns:
            Complex pressure values at query points
        """
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolators for real and imaginary parts
        interp_real = RegularGridInterpolator(
            (self.x_coords, self.y_coords, self.z_coords),
            np.real(self.data),
            bounds_error=False,
            fill_value=0
        )
        
        interp_imag = RegularGridInterpolator(
            (self.x_coords, self.y_coords, self.z_coords),
            np.imag(self.data),
            bounds_error=False,
            fill_value=0
        )
        
        # Interpolate
        real_values = interp_real(points)
        imag_values = interp_imag(points)
        
        return real_values + 1j * imag_values
    
    def find_maxima(
        self,
        threshold: float = 0.5,
        min_distance: float = 0.01
    ) -> List[FieldPoint]:
        """
        Find local maxima in the pressure field.
        
        Args:
            threshold: Relative threshold (0-1) for maxima detection
            min_distance: Minimum distance between maxima in meters
            
        Returns:
            List of FieldPoint objects at maxima locations
        """
        from scipy.ndimage import maximum_filter
        from scipy.spatial.distance import cdist
        
        amplitude = self.get_amplitude_field()
        max_amplitude = np.max(amplitude)
        threshold_value = threshold * max_amplitude
        
        # Find local maxima
        local_max = maximum_filter(amplitude, size=3)
        maxima_mask = (amplitude == local_max) & (amplitude > threshold_value)
        
        # Get coordinates of maxima
        indices = np.where(maxima_mask)
        maxima_points = []
        
        for i, j, k in zip(indices[0], indices[1], indices[2]):
            position = np.array([
                self.x_coords[i],
                self.y_coords[j],
                self.z_coords[k]
            ])
            
            # Check minimum distance to existing maxima
            if maxima_points:
                distances = [np.linalg.norm(position - p.position) 
                           for p in maxima_points]
                if min(distances) < min_distance:
                    continue
            
            point = FieldPoint(
                position=position,
                pressure=self.data[i, j, k],
                intensity=amplitude[i, j, k]**2 / (2 * 410)  # Default air impedance
            )
            maxima_points.append(point)
        
        # Sort by amplitude
        maxima_points.sort(key=lambda p: p.get_amplitude(), reverse=True)
        
        return maxima_points
    
    def save(self, filepath: str):
        """
        Save field to HDF5 file.
        
        Args:
            filepath: Output file path
        """
        with h5py.File(filepath, 'w') as f:
            # Save field data
            f.create_dataset('field_real', data=np.real(self.data))
            f.create_dataset('field_imag', data=np.imag(self.data))
            
            # Save metadata
            f.attrs['bounds'] = json.dumps(self.bounds)
            f.attrs['resolution'] = self.resolution
            f.attrs['frequency'] = self.frequency
            f.attrs['shape'] = self.shape
            
            # Save additional metadata
            for key, value in self.metadata.items():
                if isinstance(value, (int, float, str)):
                    f.attrs[key] = value
                else:
                    f.attrs[key] = json.dumps(value)
    
    @classmethod
    def load(cls, filepath: str) -> 'AcousticField':
        """
        Load field from HDF5 file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded AcousticField object
        """
        with h5py.File(filepath, 'r') as f:
            # Load field data
            real_data = f['field_real'][:]
            imag_data = f['field_imag'][:]
            data = real_data + 1j * imag_data
            
            # Load metadata
            bounds = json.loads(f.attrs['bounds'])
            resolution = f.attrs['resolution']
            frequency = f.attrs['frequency']
            
            # Load additional metadata
            metadata = {}
            for key in f.attrs.keys():
                if key not in ['bounds', 'resolution', 'frequency', 'shape']:
                    value = f.attrs[key]
                    try:
                        metadata[key] = json.loads(value)
                    except:
                        metadata[key] = value
        
        return cls(
            data=data,
            bounds=bounds,
            resolution=resolution,
            frequency=frequency,
            metadata=metadata
        )


@dataclass
class TargetPattern:
    """
    Target pressure pattern specification for optimization.
    Defines desired focal points, null regions, and constraints.
    """
    focal_points: List[Dict[str, Any]]  # List of focal point specifications
    null_regions: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_field(
        self,
        shape: Tuple[int, int, int],
        bounds: List[Tuple[float, float]]
    ) -> AcousticField:
        """
        Convert target pattern to discrete field representation.
        
        Args:
            shape: Grid shape (nx, ny, nz)
            bounds: Physical bounds
            
        Returns:
            AcousticField representing the target
        """
        # Create coordinate grids
        x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
        y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
        z = np.linspace(bounds[2][0], bounds[2][1], shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize field
        field = np.zeros(shape, dtype=complex)
        
        # Add focal points
        for focal_point in self.focal_points:
            pos = focal_point['position']
            pressure = focal_point.get('pressure', 1000)
            width = focal_point.get('width', 0.01)
            
            # Gaussian focal point
            r_squared = ((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
            gaussian = pressure * np.exp(-r_squared / (2 * width**2))
            field += gaussian
        
        # Apply null regions
        for null_region in self.null_regions:
            if null_region['type'] == 'sphere':
                center = null_region['center']
                radius = null_region['radius']
                
                r = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
                mask = r < radius
                field[mask] = 0
            
            elif null_region['type'] == 'box':
                min_pos = null_region['min']
                max_pos = null_region['max']
                
                mask = ((X >= min_pos[0]) & (X <= max_pos[0]) &
                       (Y >= min_pos[1]) & (Y <= max_pos[1]) &
                       (Z >= min_pos[2]) & (Z <= max_pos[2]))
                field[mask] = 0
        
        return AcousticField(
            data=field,
            bounds=bounds,
            resolution=(bounds[0][1] - bounds[0][0]) / shape[0],
            frequency=40e3,  # Default frequency
            metadata={'pattern_type': 'target'}
        )


class FieldMetrics:
    """Metrics for evaluating acoustic field quality."""
    
    @staticmethod
    def calculate_focus_quality(
        field: AcousticField,
        target_position: np.ndarray,
        focal_radius: float = 0.005
    ) -> Dict[str, float]:
        """
        Calculate focus quality metrics.
        
        Args:
            field: Acoustic field
            target_position: Desired focal position
            focal_radius: Expected focal spot radius
            
        Returns:
            Dictionary of quality metrics
        """
        amplitude = field.get_amplitude_field()
        
        # Find actual focus
        maxima = field.find_maxima(threshold=0.9)
        
        if not maxima:
            return {
                'focus_error': np.inf,
                'peak_pressure': 0,
                'focus_size': np.inf,
                'contrast_ratio': 0
            }
        
        actual_focus = maxima[0]
        
        # Position error
        focus_error = np.linalg.norm(actual_focus.position - target_position)
        
        # Peak pressure
        peak_pressure = actual_focus.get_amplitude()
        
        # Focus size (FWHM)
        focus_region = field.interpolate_at_points(
            actual_focus.position + np.random.randn(100, 3) * focal_radius
        )
        focus_size = np.std(np.abs(focus_region))
        
        # Contrast ratio (peak to average)
        avg_pressure = np.mean(amplitude)
        contrast_ratio = peak_pressure / (avg_pressure + 1e-10)
        
        return {
            'focus_error': focus_error,
            'peak_pressure': peak_pressure,
            'focus_size': focus_size,
            'contrast_ratio': contrast_ratio
        }
    
    @staticmethod
    def calculate_uniformity(
        field: AcousticField,
        region: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate field uniformity in a region.
        
        Args:
            field: Acoustic field
            region: Optional region specification
            
        Returns:
            Uniformity metric (0-1, higher is more uniform)
        """
        amplitude = field.get_amplitude_field()
        
        if region:
            # Extract region
            if region['type'] == 'box':
                min_idx = [
                    np.argmin(np.abs(field.x_coords - region['min'][0])),
                    np.argmin(np.abs(field.y_coords - region['min'][1])),
                    np.argmin(np.abs(field.z_coords - region['min'][2]))
                ]
                max_idx = [
                    np.argmin(np.abs(field.x_coords - region['max'][0])),
                    np.argmin(np.abs(field.y_coords - region['max'][1])),
                    np.argmin(np.abs(field.z_coords - region['max'][2]))
                ]
                
                amplitude = amplitude[
                    min_idx[0]:max_idx[0],
                    min_idx[1]:max_idx[1],
                    min_idx[2]:max_idx[2]
                ]
        
        # Calculate uniformity as 1 - coefficient of variation
        mean_amp = np.mean(amplitude)
        std_amp = np.std(amplitude)
        
        if mean_amp > 0:
            cv = std_amp / mean_amp
            uniformity = 1 / (1 + cv)
        else:
            uniformity = 0
        
        return uniformity
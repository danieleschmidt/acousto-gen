"""
Wave propagation models for acoustic field calculation.
Implements accurate physics-based wave propagation with GPU acceleration support.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    # Try to use mock backend
    try:
        import sys
        from pathlib import Path
        mock_path = Path(__file__).parent.parent.parent.parent / "acousto_gen"
        sys.path.insert(0, str(mock_path))
        from mock_backend import setup_mock_dependencies
        setup_mock_dependencies()
        import torch
        import torch.nn.functional as F
    except ImportError:
        torch = None
        F = None


@dataclass
class MediumProperties:
    """Physical properties of the propagation medium."""
    density: float  # kg/m³
    speed_of_sound: float  # m/s
    absorption: float  # Np/m
    nonlinearity: float = 0.0  # Nonlinearity parameter
    temperature: float = 20.0  # °C
    
    def get_impedance(self) -> float:
        """Calculate acoustic impedance of the medium."""
        return self.density * self.speed_of_sound
    
    def get_wavelength(self, frequency: float) -> float:
        """Calculate wavelength for given frequency."""
        return self.speed_of_sound / frequency
    
    def get_wavenumber(self, frequency: float) -> complex:
        """Calculate complex wavenumber including absorption."""
        k = 2 * np.pi * frequency / self.speed_of_sound
        return complex(k, -self.absorption)


class WavePropagator:
    """
    Core wave propagation engine using angular spectrum method.
    Supports both CPU and GPU computation for performance optimization.
    """
    
    def __init__(
        self,
        resolution: float = 1e-3,
        bounds: List[Tuple[float, float]] = None,
        frequency: float = 40e3,
        medium: Optional[MediumProperties] = None,
        device: str = "cpu"
    ):
        """
        Initialize wave propagator with computational domain.
        
        Args:
            resolution: Spatial resolution in meters
            bounds: Computational domain bounds [(xmin,xmax), (ymin,ymax), (zmin,zmax)]
            frequency: Operating frequency in Hz
            medium: Medium properties (defaults to air)
            device: Computation device ('cpu' or 'cuda')
        """
        self.resolution = resolution
        self.bounds = bounds or [(-0.1, 0.1), (-0.1, 0.1), (0, 0.2)]
        self.frequency = frequency
        self.medium = medium or MediumProperties(
            density=1.2,
            speed_of_sound=343,
            absorption=0.01
        )
        if torch:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
        # Setup computational grid
        self._setup_grid()
        
        # Precompute propagation kernels
        self._compute_kernels()
    
    def _setup_grid(self):
        """Create computational grid based on domain bounds."""
        self.nx = int((self.bounds[0][1] - self.bounds[0][0]) / self.resolution)
        self.ny = int((self.bounds[1][1] - self.bounds[1][0]) / self.resolution)
        self.nz = int((self.bounds[2][1] - self.bounds[2][0]) / self.resolution)
        
        # Create coordinate meshgrids
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.nx)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.ny)
        z = np.linspace(self.bounds[2][0], self.bounds[2][1], self.nz)
        
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Convert to torch tensors if available
        if torch:
            self.X_tensor = torch.tensor(self.X, dtype=torch.float32, device=self.device)
            self.Y_tensor = torch.tensor(self.Y, dtype=torch.float32, device=self.device)
            self.Z_tensor = torch.tensor(self.Z, dtype=torch.float32, device=self.device)
        else:
            self.X_tensor = self.X
            self.Y_tensor = self.Y  
            self.Z_tensor = self.Z
    
    def _compute_kernels(self):
        """Precompute propagation kernels for angular spectrum method."""
        # Wavenumber and wavelength
        self.k = self.medium.get_wavenumber(self.frequency)
        self.wavelength = self.medium.get_wavelength(self.frequency)
        
        # Spatial frequency grids
        kx = np.fft.fftfreq(self.nx, d=self.resolution) * 2 * np.pi
        ky = np.fft.fftfreq(self.ny, d=self.resolution) * 2 * np.pi
        
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        
        # Angular spectrum propagation kernel
        k_real = np.real(self.k)
        self.kz = np.sqrt(k_real**2 - self.KX**2 - self.KY**2 + 0j)
        
        # Convert to torch if available
        if torch:
            self.kz_tensor = torch.tensor(self.kz, dtype=torch.complex64, device=self.device)
        else:
            self.kz_tensor = self.kz
    
    def compute_field_from_sources(
        self,
        source_positions: np.ndarray,
        source_amplitudes: np.ndarray,
        source_phases: np.ndarray,
        target_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute acoustic field from array of point sources.
        
        Args:
            source_positions: Nx3 array of source positions
            source_amplitudes: N array of source amplitudes
            source_phases: N array of source phases in radians
            target_points: Optional Mx3 array of evaluation points
            
        Returns:
            Complex pressure field at target points or full grid
        """
        # Convert to tensors if torch available
        if torch:
            positions = torch.tensor(source_positions, dtype=torch.float32, device=self.device)
            amplitudes = torch.tensor(source_amplitudes, dtype=torch.float32, device=self.device)
            phases = torch.tensor(source_phases, dtype=torch.float32, device=self.device)
        else:
            positions = source_positions
            amplitudes = source_amplitudes
            phases = source_phases
        
        if target_points is None:
            # Compute on full grid
            if torch:
                field = torch.zeros(
                    (self.nx, self.ny, self.nz),
                    dtype=torch.complex64,
                    device=self.device
                )
            else:
                field = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
            
            # Rayleigh-Sommerfeld integral
            for i in range(len(positions)):
                pos = positions[i]
                amp = amplitudes[i]
                phase = phases[i]
                
                # Distance from source to each grid point
                r = torch.sqrt(
                    (self.X_tensor - pos[0])**2 +
                    (self.Y_tensor - pos[1])**2 +
                    (self.Z_tensor - pos[2])**2
                )
                
                # Avoid singularity at source
                r = torch.maximum(r, torch.tensor(self.wavelength/10, device=self.device))
                
                # Green's function (point source)
                green = amp * torch.exp(1j * (self.k.real * r + phase)) / (4 * np.pi * r)
                
                # Add attenuation
                green *= torch.exp(-self.k.imag * r)
                
                field += green
            
            return field.cpu().numpy()
        
        else:
            # Compute at specific points
            targets = torch.tensor(target_points, dtype=torch.float32, device=self.device)
            field = torch.zeros(len(targets), dtype=torch.complex64, device=self.device)
            
            for i in range(len(positions)):
                pos = positions[i]
                amp = amplitudes[i]
                phase = phases[i]
                
                # Distance from source to each target
                r = torch.sqrt(
                    (targets[:, 0] - pos[0])**2 +
                    (targets[:, 1] - pos[1])**2 +
                    (targets[:, 2] - pos[2])**2
                )
                
                r = torch.maximum(r, torch.tensor(self.wavelength/10, device=self.device))
                
                # Green's function
                green = amp * torch.exp(1j * (self.k.real * r + phase)) / (4 * np.pi * r)
                green *= torch.exp(-self.k.imag * r)
                
                field += green
            
            return field.cpu().numpy()
    
    def angular_spectrum_propagation(
        self,
        initial_field: np.ndarray,
        z_distance: float
    ) -> np.ndarray:
        """
        Propagate field using angular spectrum method.
        
        Args:
            initial_field: Complex field at initial plane
            z_distance: Propagation distance in z direction
            
        Returns:
            Propagated complex field
        """
        # Convert to tensor
        field = torch.tensor(initial_field, dtype=torch.complex64, device=self.device)
        
        # FFT to spatial frequency domain
        field_fft = torch.fft.fft2(field)
        
        # Apply propagation kernel
        propagator = torch.exp(1j * self.kz_tensor * z_distance)
        field_fft *= propagator
        
        # Inverse FFT back to spatial domain
        propagated = torch.fft.ifft2(field_fft)
        
        return propagated.cpu().numpy()
    
    def compute_pressure_gradient(
        self,
        field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spatial gradients of pressure field.
        
        Args:
            field: Complex pressure field
            
        Returns:
            Gradient components (dP/dx, dP/dy, dP/dz)
        """
        field_tensor = torch.tensor(field, dtype=torch.complex64, device=self.device)
        
        # Compute gradients using finite differences
        dx = self.resolution
        
        # Padding for gradient computation
        padded = F.pad(field_tensor.unsqueeze(0).unsqueeze(0), 
                      (1, 1, 1, 1, 1, 1), mode='replicate')
        padded = padded.squeeze()
        
        grad_x = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) / (2 * dx)
        grad_y = (padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]) / (2 * dx)
        grad_z = (padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]) / (2 * dx)
        
        return (
            grad_x.cpu().numpy(),
            grad_y.cpu().numpy(),
            grad_z.cpu().numpy()
        )
    
    def compute_acoustic_radiation_force(
        self,
        field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute acoustic radiation force using Gor'kov potential.
        
        Args:
            field: Complex pressure field
            
        Returns:
            Force components (Fx, Fy, Fz) in Newtons
        """
        # Pressure amplitude
        p = np.abs(field)
        
        # Velocity from pressure gradient
        grad_x, grad_y, grad_z = self.compute_pressure_gradient(field)
        
        omega = 2 * np.pi * self.frequency
        rho = self.medium.density
        c = self.medium.speed_of_sound
        
        # Velocity components
        vx = -1j * grad_x / (omega * rho)
        vy = -1j * grad_y / (omega * rho)
        vz = -1j * grad_z / (omega * rho)
        
        v_squared = np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2
        
        # Gor'kov potential
        kappa = 1 / (rho * c**2)  # Compressibility
        U = 2 * np.pi * (kappa * p**2 / 4 - 3 * rho * v_squared / 4)
        
        # Force is negative gradient of potential
        U_tensor = torch.tensor(U, dtype=torch.float32, device=self.device)
        
        # Compute gradients
        padded = F.pad(U_tensor.unsqueeze(0).unsqueeze(0), 
                      (1, 1, 1, 1, 1, 1), mode='replicate')
        padded = padded.squeeze()
        
        dx = self.resolution
        Fx = -(padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) / (2 * dx)
        Fy = -(padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]) / (2 * dx)
        Fz = -(padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]) / (2 * dx)
        
        return (
            Fx.cpu().numpy(),
            Fy.cpu().numpy(),
            Fz.cpu().numpy()
        )
    
    def compute_acoustic_intensity(
        self,
        field: np.ndarray
    ) -> np.ndarray:
        """
        Compute time-averaged acoustic intensity.
        
        Args:
            field: Complex pressure field
            
        Returns:
            Acoustic intensity in W/m²
        """
        p = np.abs(field)
        rho = self.medium.density
        c = self.medium.speed_of_sound
        
        # Time-averaged intensity
        intensity = p**2 / (2 * rho * c)
        
        return intensity
    
    def find_focus_points(
        self,
        field: np.ndarray,
        threshold: float = 0.8
    ) -> List[Tuple[float, float, float, float]]:
        """
        Find focal points in the acoustic field.
        
        Args:
            field: Complex pressure field
            threshold: Relative threshold for focus detection (0-1)
            
        Returns:
            List of (x, y, z, pressure) tuples for focal points
        """
        p = np.abs(field)
        max_pressure = np.max(p)
        threshold_pressure = threshold * max_pressure
        
        # Find local maxima above threshold
        from scipy.ndimage import maximum_filter
        
        # Local maxima detection
        local_max = maximum_filter(p, size=3)
        maxima = (p == local_max) & (p > threshold_pressure)
        
        # Extract coordinates
        indices = np.where(maxima)
        focal_points = []
        
        for i, j, k in zip(indices[0], indices[1], indices[2]):
            x = self.X[i, j, k]
            y = self.Y[i, j, k]
            z = self.Z[i, j, k]
            pressure = p[i, j, k]
            focal_points.append((x, y, z, pressure))
        
        # Sort by pressure
        focal_points.sort(key=lambda x: x[3], reverse=True)
        
        return focal_points
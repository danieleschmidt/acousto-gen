"""Core acoustic holography functionality."""

from typing import Optional, Tuple, Union, List, Dict, Any
from pathlib import Path
import sys

# Robust dependency handling with graceful degradation
try:
    from .mock_backend import check_and_setup
    MOCK_MODE = not check_and_setup()
except ImportError:
    MOCK_MODE = False

# Import with fallback to mocks
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    if not MOCK_MODE:
        print("⚠️  Dependencies not found, enabling mock mode...")
        from .mock_backend import setup_mock_dependencies
        setup_mock_dependencies()
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from physics.propagation.wave_propagator import WavePropagator, MediumProperties
from models.acoustic_field import AcousticField, create_focus_target
from physics.transducers.transducer_array import TransducerArray


class AcousticHologram:
    """Core acoustic hologram generator and optimizer.
    
    This class provides the fundamental interface for creating and optimizing
    acoustic holograms for various applications including levitation, haptics,
    and medical focused ultrasound.
    
    Parameters
    ----------
    transducer : TransducerArray
        Transducer array configuration
    frequency : float
        Operating frequency in Hz
    medium : str, default="air"
        Propagation medium ("air", "water", "tissue")
    resolution : float, default=1e-3
        Spatial resolution in meters
    bounds : list, optional
        Computational bounds [(xmin,xmax), (ymin,ymax), (zmin,zmax)]
    device : str, default="cpu"
        Computation device ("cpu" or "cuda")
    
    Examples
    --------
    >>> from acousto_gen import AcousticHologram
    >>> hologram = AcousticHologram(transducer=array, frequency=40e3)
    >>> phases = hologram.optimize(target_field)
    """
    
    def __init__(
        self,
        transducer: 'TransducerArray',
        frequency: float,
        medium: str = "air",
        resolution: float = 1e-3,
        bounds: Optional[List[Tuple[float, float]]] = None,
        device: str = "cpu"
    ) -> None:
        self.transducer = transducer
        self.frequency = frequency
        self.medium_name = medium
        self.resolution = resolution
        self.bounds = bounds or [(-0.1, 0.1), (-0.1, 0.1), (0, 0.2)]
        self.device = device
        
        # Initialize medium properties based on type
        self.medium = self._get_medium_properties(medium)
        
        # Initialize wave propagator
        self.propagator = WavePropagator(
            resolution=resolution,
            bounds=self.bounds,
            frequency=frequency,
            medium=self.medium,
            device=device
        )
        
        # Optimization state
        self.current_phases = None
        self.optimization_history = []
        
    def _get_medium_properties(self, medium_name: str) -> MediumProperties:
        """Get medium properties for common media."""
        medium_db = {
            "air": MediumProperties(
                density=1.2,
                speed_of_sound=343,
                absorption=0.01,
                temperature=20.0
            ),
            "water": MediumProperties(
                density=1000,
                speed_of_sound=1500,
                absorption=0.002,
                temperature=20.0
            ),
            "tissue": MediumProperties(
                density=1050,
                speed_of_sound=1540,
                absorption=0.5,
                temperature=37.0
            )
        }
        
        if medium_name not in medium_db:
            raise ValueError(f"Unknown medium: {medium_name}. Available: {list(medium_db.keys())}")
        
        return medium_db[medium_name]
        
    def create_focus_point(
        self,
        position: Tuple[float, float, float],
        pressure: float = 3000,
        width: float = 0.005
    ) -> AcousticField:
        """Create a target pressure field with a single focus point.
        
        Parameters
        ----------
        position : tuple of float
            Focus point coordinates (x, y, z) in meters
        pressure : float, default=3000
            Target pressure in Pa
        width : float, default=0.005
            Focus width (standard deviation) in meters
            
        Returns
        -------
        AcousticField
            Target pressure field
        """
        # Calculate grid shape from bounds and resolution
        shape = (
            int((self.bounds[0][1] - self.bounds[0][0]) / self.resolution),
            int((self.bounds[1][1] - self.bounds[1][0]) / self.resolution),
            int((self.bounds[2][1] - self.bounds[2][0]) / self.resolution)
        )
        
        return create_focus_target(
            position=list(position),
            pressure=pressure,
            width=width,
            shape=shape,
            bounds=self.bounds
        )
    
    def create_multi_focus_target(
        self,
        focal_points: List[Dict[str, Any]],
        width: float = 0.005
    ) -> AcousticField:
        """Create target field with multiple focus points.
        
        Parameters
        ----------
        focal_points : list of dict
            List of focal point specifications with 'position' and 'pressure' keys
        width : float, default=0.005
            Focus width in meters
            
        Returns
        -------
        AcousticField
            Target field with multiple foci
        """
        shape = (
            int((self.bounds[0][1] - self.bounds[0][0]) / self.resolution),
            int((self.bounds[1][1] - self.bounds[1][0]) / self.resolution),
            int((self.bounds[2][1] - self.bounds[2][0]) / self.resolution)
        )
        
        # Create coordinate grids
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], shape[0])
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], shape[1])
        z = np.linspace(self.bounds[2][0], self.bounds[2][1], shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize field
        field_data = np.zeros(shape, dtype=complex)
        
        # Add each focus point
        for point in focal_points:
            pos = point['position']
            pressure = point.get('pressure', 3000)
            
            r_squared = ((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
            gaussian = pressure * np.exp(-r_squared / (2 * width**2))
            field_data += gaussian
        
        return AcousticField(
            data=field_data,
            bounds=self.bounds,
            resolution=self.resolution,
            frequency=self.frequency,
            metadata={'type': 'multi_focus', 'focal_points': focal_points}
        )
        
    def optimize(
        self,
        target: Union[AcousticField, np.ndarray],
        iterations: int = 1000,
        method: str = "adam",
        learning_rate: float = 0.01,
        lambda_smooth: float = 0.1,
        callback: Optional[callable] = None
    ) -> np.ndarray:
        """Optimize transducer phases to generate target pressure field.
        
        Parameters
        ----------
        target : AcousticField or np.ndarray
            Target pressure field
        iterations : int, default=1000
            Number of optimization iterations
        method : str, default="adam"
            Optimization method ("adam", "sgd", "lbfgs")
        learning_rate : float, default=0.01
            Learning rate for optimizer
        lambda_smooth : float, default=0.1
            Smoothness regularization weight
        callback : callable, optional
            Callback function called after each iteration with (iteration, loss, phases)
            
        Returns
        -------
        np.ndarray
            Optimized phase array for transducers
        """
        # Convert target to tensor
        if isinstance(target, AcousticField):
            target_field = torch.tensor(target.data, dtype=torch.complex64, device=self.device)
        else:
            target_field = torch.tensor(target, dtype=torch.complex64, device=self.device)
        
        # Initialize phases randomly
        num_transducers = len(self.transducer.positions)
        phases = torch.randn(num_transducers, requires_grad=True, device=self.device)
        
        # Fixed amplitudes (can be made optimizable later)
        amplitudes = torch.ones(num_transducers, device=self.device)
        
        # Choose optimizer
        if method == "adam":
            optimizer = optim.Adam([phases], lr=learning_rate)
        elif method == "sgd":
            optimizer = optim.SGD([phases], lr=learning_rate)
        elif method == "lbfgs":
            optimizer = optim.LBFGS([phases], lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Optimization history
        self.optimization_history = []
        
        def closure():
            optimizer.zero_grad()
            
            # Forward pass: compute field from current phases
            generated_field = self._forward_model(phases, amplitudes)
            
            # Loss function
            loss = self._compute_loss(generated_field, target_field, phases, lambda_smooth)
            
            # Backward pass
            loss.backward()
            
            return loss
        
        # Optimization loop
        for iteration in range(iterations):
            if method == "lbfgs":
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                generated_field = self._forward_model(phases, amplitudes)
                loss = self._compute_loss(generated_field, target_field, phases, lambda_smooth)
                loss.backward()
                optimizer.step()
            
            # Record history
            loss_value = loss.item()
            self.optimization_history.append(loss_value)
            
            # Callback
            if callback is not None:
                callback(iteration, loss_value, phases.detach().cpu().numpy())
            
            # Progress reporting
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss_value:.6f}")
        
        # Store optimized phases
        self.current_phases = phases.detach().cpu().numpy()
        
        return self.current_phases
    
    def _forward_model(self, phases, amplitudes):
        """Forward model: compute field from phases and amplitudes."""
        # Convert to numpy for propagator
        phases_np = phases.detach().cpu().numpy()
        amplitudes_np = amplitudes.detach().cpu().numpy()
        positions_np = self.transducer.positions
        
        # Compute field using wave propagator
        field_np = self.propagator.compute_field_from_sources(
            source_positions=positions_np,
            source_amplitudes=amplitudes_np,
            source_phases=phases_np
        )
        
        # Convert back to tensor
        return torch.tensor(field_np, dtype=torch.complex64, device=self.device)
    
    def _compute_loss(
        self,
        generated,
        target,
        phases,
        lambda_smooth: float
    ):
        """Compute optimization loss function."""
        # Field matching loss (MSE on complex field)
        field_loss = torch.mean(torch.abs(generated - target)**2)
        
        # Phase smoothness regularization
        if len(phases) > 1:
            phase_diff = phases[1:] - phases[:-1]
            smooth_loss = torch.mean(phase_diff**2)
        else:
            smooth_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = field_loss + lambda_smooth * smooth_loss
        
        return total_loss
    
    def compute_field(
        self,
        phases: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None
    ) -> AcousticField:
        """Compute acoustic field for given phases and amplitudes.
        
        Parameters
        ----------
        phases : np.ndarray, optional
            Phase array (uses current_phases if None)
        amplitudes : np.ndarray, optional
            Amplitude array (uses unit amplitudes if None)
            
        Returns
        -------
        AcousticField
            Computed acoustic field
        """
        if phases is None:
            if self.current_phases is None:
                raise ValueError("No phases available. Run optimization first or provide phases.")
            phases = self.current_phases
        
        if amplitudes is None:
            amplitudes = np.ones(len(phases))
        
        # Compute field
        field_data = self.propagator.compute_field_from_sources(
            source_positions=self.transducer.positions,
            source_amplitudes=amplitudes,
            source_phases=phases
        )
        
        return AcousticField(
            data=field_data,
            bounds=self.bounds,
            resolution=self.resolution,
            frequency=self.frequency,
            metadata={'phases': phases, 'amplitudes': amplitudes}
        )
    
    def evaluate_field_quality(
        self,
        field: Optional[AcousticField] = None,
        target_position: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate acoustic field quality metrics.
        
        Parameters
        ----------
        field : AcousticField, optional
            Field to evaluate (computes current field if None)
        target_position : np.ndarray, optional
            Target focus position for evaluation
            
        Returns
        -------
        dict
            Dictionary of quality metrics
        """
        if field is None:
            field = self.compute_field()
        
        from models.acoustic_field import FieldMetrics
        
        if target_position is not None:
            # Focus quality metrics
            metrics = FieldMetrics.calculate_focus_quality(
                field, target_position
            )
        else:
            # General field statistics
            metrics = FieldMetrics.calculate_field_statistics(field)
        
        return metrics
    
    def save_configuration(self, filepath: str):
        """Save hologram configuration to file."""
        config = {
            'frequency': self.frequency,
            'medium': self.medium_name,
            'resolution': self.resolution,
            'bounds': self.bounds,
            'transducer': {
                'positions': self.transducer.positions.tolist(),
                'num_elements': len(self.transducer.positions)
            },
            'current_phases': self.current_phases.tolist() if self.current_phases is not None else None,
            'optimization_history': self.optimization_history
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_configuration(cls, filepath: str, transducer: 'TransducerArray') -> 'AcousticHologram':
        """Load hologram configuration from file."""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        hologram = cls(
            transducer=transducer,
            frequency=config['frequency'],
            medium=config['medium'],
            resolution=config['resolution'],
            bounds=config['bounds']
        )
        
        if config['current_phases'] is not None:
            hologram.current_phases = np.array(config['current_phases'])
        hologram.optimization_history = config['optimization_history']
        
        return hologram
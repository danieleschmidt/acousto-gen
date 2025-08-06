"""
Mid-air haptic feedback renderer using acoustic radiation pressure.
Provides tactile sensations at arbitrary points in space without contact.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import threading
from collections import deque

from ..levitation.acoustic_levitator import AcousticLevitator
from ...physics.transducers.transducer_array import TransducerArray
from ...physics.propagation.wave_propagator import WavePropagator
from ...optimization.hologram_optimizer import GradientOptimizer
from ...models.acoustic_field import AcousticField, create_focus_target


@dataclass
class HapticShape:
    """Defines a haptic shape for rendering."""
    shape_type: str  # 'circle', 'line', 'square', 'custom'
    center: np.ndarray  # 3D center position
    parameters: Dict[str, Any]  # Shape-specific parameters
    pressure: float = 200  # Pa (perceivable threshold ~1 Pa)
    modulation_freq: float = 200  # Hz (optimal tactile perception)
    duration: Optional[float] = None  # Rendering duration in seconds


@dataclass
class HandPosition:
    """Hand position and tracking data."""
    position: np.ndarray  # 3D position [x, y, z]
    velocity: np.ndarray  # 3D velocity [vx, vy, vz]
    confidence: float = 1.0  # Tracking confidence (0-1)
    timestamp: float = 0.0  # Unix timestamp


class HapticRenderer:
    """
    Mid-air haptic feedback renderer using focused ultrasound.
    
    Provides precise tactile sensations at arbitrary points in 3D space
    without any physical contact. Uses amplitude modulation to create
    perceivable tactile sensations.
    """
    
    def __init__(
        self,
        transducer_array: TransducerArray,
        wave_propagator: Optional[WavePropagator] = None,
        optimizer: Optional[GradientOptimizer] = None,
        update_rate: int = 1000,  # Hz
        max_pressure: float = 1000,  # Pa (safety limit)
        workspace_bounds: List[Tuple[float, float]] = None
    ):
        """
        Initialize haptic renderer.
        
        Args:
            transducer_array: Ultrasonic transducer array
            wave_propagator: Wave propagation model
            optimizer: Phase optimization algorithm  
            update_rate: Update rate in Hz for real-time rendering
            max_pressure: Maximum allowable pressure for safety
            workspace_bounds: Workspace bounds [(xmin,xmax), ...]
        """
        self.array = transducer_array
        self.propagator = wave_propagator or WavePropagator(
            frequency=transducer_array.frequency,
            resolution=1e-3
        )
        self.optimizer = optimizer or GradientOptimizer(
            num_elements=len(transducer_array.elements)
        )
        
        self.update_rate = update_rate
        self.update_period = 1.0 / update_rate
        self.max_pressure = max_pressure
        
        # Workspace definition
        self.workspace_bounds = workspace_bounds or [
            (-0.1, 0.1), (-0.1, 0.1), (0.05, 0.2)  # 20x20x15cm above array
        ]
        
        # Haptic rendering state
        self.active_shapes: Dict[str, HapticShape] = {}
        self.current_phases = np.zeros(len(self.array.elements))
        self.is_rendering = False
        
        # Hand tracking
        self.hand_positions: Dict[str, HandPosition] = {}  # hand_id -> position
        self.position_history: Dict[str, deque] = {}  # For velocity calculation
        
        # Real-time rendering
        self.render_thread = None
        self.stop_rendering = False
        
        # Callback functions
        self.hand_callbacks: List[Callable] = []
        self.shape_callbacks: List[Callable] = []
        
        # Performance metrics
        self.render_stats = {
            'fps': 0,
            'avg_optimization_time': 0,
            'pressure_violations': 0,
            'shapes_rendered': 0
        }
    
    def add_shape(self, shape_id: str, shape: HapticShape) -> bool:
        """
        Add haptic shape for rendering.
        
        Args:
            shape_id: Unique identifier for the shape
            shape: HapticShape object to render
            
        Returns:
            True if shape was added successfully
        """
        # Validate shape is within workspace
        if not self._is_position_valid(shape.center):
            print(f"Warning: Shape {shape_id} center outside workspace")
            return False
        
        # Safety check on pressure
        if shape.pressure > self.max_pressure:
            print(f"Warning: Clipping pressure for shape {shape_id}")
            shape.pressure = self.max_pressure
        
        self.active_shapes[shape_id] = shape
        
        # Trigger callback
        for callback in self.shape_callbacks:
            callback('shape_added', shape_id, shape)
        
        return True
    
    def remove_shape(self, shape_id: str) -> bool:
        """
        Remove haptic shape from rendering.
        
        Args:
            shape_id: Identifier of shape to remove
            
        Returns:
            True if shape was removed
        """
        if shape_id in self.active_shapes:
            shape = self.active_shapes.pop(shape_id)
            
            # Trigger callback
            for callback in self.shape_callbacks:
                callback('shape_removed', shape_id, shape)
            
            return True
        return False
    
    def create_shape(
        self,
        shape_type: str,
        center: List[float],
        pressure: float = 200,
        **kwargs
    ) -> HapticShape:
        """
        Create a haptic shape with specified parameters.
        
        Args:
            shape_type: Type of shape ('circle', 'line', 'square', etc.)
            center: 3D center position [x, y, z]
            pressure: Pressure amplitude in Pa
            **kwargs: Shape-specific parameters
            
        Returns:
            HapticShape object
        """
        center_array = np.array(center)
        
        if shape_type == 'circle':
            radius = kwargs.get('radius', 0.02)  # 2cm default
            parameters = {
                'radius': radius,
                'num_points': kwargs.get('num_points', 8)
            }
        
        elif shape_type == 'line':
            start = kwargs.get('start', center_array - np.array([0.02, 0, 0]))
            end = kwargs.get('end', center_array + np.array([0.02, 0, 0]))
            parameters = {
                'start': start,
                'end': end,
                'width': kwargs.get('width', 0.005)  # 5mm width
            }
        
        elif shape_type == 'square':
            size = kwargs.get('size', 0.03)  # 3cm default
            parameters = {
                'size': size,
                'num_points': kwargs.get('num_points', 12)
            }
        
        elif shape_type == 'point':
            parameters = {
                'focal_width': kwargs.get('width', 0.005)  # 5mm focus
            }
        
        else:
            # Custom shape
            parameters = kwargs
        
        return HapticShape(
            shape_type=shape_type,
            center=center_array,
            parameters=parameters,
            pressure=pressure,
            modulation_freq=kwargs.get('modulation_freq', 200),
            duration=kwargs.get('duration', None)
        )
    
    def update_hand_position(
        self,
        hand_id: str,
        position: List[float],
        confidence: float = 1.0
    ) -> None:
        """
        Update tracked hand position.
        
        Args:
            hand_id: Unique hand identifier
            position: 3D position [x, y, z] in meters
            confidence: Tracking confidence (0-1)
        """
        pos_array = np.array(position)
        timestamp = time.time()
        
        # Calculate velocity from history
        if hand_id not in self.position_history:
            self.position_history[hand_id] = deque(maxlen=5)
        
        history = self.position_history[hand_id]
        
        if len(history) > 0:
            dt = timestamp - history[-1][1]
            if dt > 0:
                velocity = (pos_array - history[-1][0]) / dt
            else:
                velocity = np.zeros(3)
        else:
            velocity = np.zeros(3)
        
        # Update position
        hand_pos = HandPosition(
            position=pos_array,
            velocity=velocity,
            confidence=confidence,
            timestamp=timestamp
        )
        
        self.hand_positions[hand_id] = hand_pos
        history.append((pos_array.copy(), timestamp))
        
        # Trigger callbacks
        for callback in self.hand_callbacks:
            callback(hand_id, hand_pos)
    
    def on_hand_position(self, callback: Callable) -> None:
        """Register callback for hand position updates."""
        self.hand_callbacks.append(callback)
    
    def on_shape_event(self, callback: Callable) -> None:
        """Register callback for shape events."""
        self.shape_callbacks.append(callback)
    
    def start_rendering(self) -> None:
        """Start real-time haptic rendering."""
        if self.is_rendering:
            return
        
        self.is_rendering = True
        self.stop_rendering = False
        self.render_thread = threading.Thread(target=self._render_loop)
        self.render_thread.daemon = True
        self.render_thread.start()
        
        print(f"Haptic rendering started at {self.update_rate} Hz")
    
    def stop_rendering_loop(self) -> None:
        """Stop real-time haptic rendering."""
        if not self.is_rendering:
            return
        
        self.stop_rendering = True
        if self.render_thread:
            self.render_thread.join()
        
        self.is_rendering = False
        print("Haptic rendering stopped")
    
    def _render_loop(self) -> None:
        """Main rendering loop running in separate thread."""
        frame_times = deque(maxlen=100)
        
        while not self.stop_rendering:
            frame_start = time.time()
            
            try:
                # Render current frame
                self._render_frame()
                
                # Update statistics
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                
                if len(frame_times) >= 10:
                    self.render_stats['fps'] = 1.0 / np.mean(frame_times)
                
                # Sleep to maintain update rate
                sleep_time = self.update_period - frame_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Rendering error: {e}")
                time.sleep(0.001)  # Brief pause before retry
    
    def _render_frame(self) -> None:
        """Render single haptic frame."""
        if not self.active_shapes:
            # No shapes to render, turn off array
            self.current_phases = np.zeros(len(self.array.elements))
            self.array.set_phases(self.current_phases)
            return
        
        optimization_start = time.time()
        
        # Create combined target field from all active shapes
        target_field = self._create_combined_target()
        
        # Optimize phases for target field
        if target_field is not None:
            # Simple gradient optimization for real-time performance
            def forward_model(phases):
                return self._compute_field_from_phases(phases.detach().cpu().numpy())
            
            try:
                result = self.optimizer.optimize(
                    forward_model=forward_model,
                    target_field=torch.tensor(target_field.data, dtype=torch.complex64),
                    iterations=20  # Limited for real-time performance
                )
                
                self.current_phases = result.phases
                
                # Apply temporal modulation for tactile perception
                modulated_phases = self._apply_modulation(self.current_phases)
                
                # Safety check
                if self._check_safety(modulated_phases):
                    self.array.set_phases(modulated_phases)
                else:
                    self.render_stats['pressure_violations'] += 1
                
            except Exception as e:
                print(f"Optimization error: {e}")
        
        # Update optimization time statistics
        opt_time = time.time() - optimization_start
        if hasattr(self.render_stats, 'opt_times'):
            self.render_stats['opt_times'].append(opt_time)
            if len(self.render_stats['opt_times']) > 100:
                self.render_stats['opt_times'].pop(0)
            self.render_stats['avg_optimization_time'] = np.mean(self.render_stats['opt_times'])
        else:
            self.render_stats['opt_times'] = [opt_time]
        
        self.render_stats['shapes_rendered'] = len(self.active_shapes)
    
    def _create_combined_target(self) -> Optional[AcousticField]:
        """Create combined target field from all active shapes."""
        if not self.active_shapes:
            return None
        
        # Use small field for real-time performance
        shape = (25, 25, 25)
        bounds = self.workspace_bounds
        
        # Create coordinate grids
        x = np.linspace(bounds[0][0], bounds[0][1], shape[0])
        y = np.linspace(bounds[1][0], bounds[1][1], shape[1])
        z = np.linspace(bounds[2][0], bounds[2][1], shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize combined field
        combined_field = np.zeros(shape, dtype=complex)
        
        # Add each shape
        for shape_id, haptic_shape in self.active_shapes.items():
            field_data = self._render_shape(haptic_shape, X, Y, Z)
            combined_field += field_data
        
        return AcousticField(
            data=combined_field,
            bounds=bounds,
            resolution=(bounds[0][1] - bounds[0][0]) / shape[0],
            frequency=self.array.frequency
        )
    
    def _render_shape(
        self,
        shape: HapticShape,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray
    ) -> np.ndarray:
        """Render individual haptic shape."""
        field = np.zeros_like(X, dtype=complex)
        
        if shape.shape_type == 'point':
            # Single focus point
            r_squared = ((X - shape.center[0])**2 + 
                        (Y - shape.center[1])**2 + 
                        (Z - shape.center[2])**2)
            width = shape.parameters.get('focal_width', 0.005)
            field = shape.pressure * np.exp(-r_squared / (2 * width**2))
        
        elif shape.shape_type == 'circle':
            # Circular pattern
            radius = shape.parameters['radius']
            num_points = shape.parameters.get('num_points', 8)
            
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                pos = shape.center + np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0
                ])
                
                r_squared = ((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
                field += shape.pressure * np.exp(-r_squared / (2 * 0.005**2))
        
        elif shape.shape_type == 'line':
            # Line pattern
            start = shape.parameters['start']
            end = shape.parameters['end']
            width = shape.parameters.get('width', 0.005)
            
            # Sample points along line
            num_points = int(np.linalg.norm(end - start) / 0.01) + 1
            for i in range(num_points):
                t = i / max(1, num_points - 1)
                pos = start + t * (end - start)
                
                r_squared = ((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
                field += shape.pressure * np.exp(-r_squared / (2 * width**2))
        
        elif shape.shape_type == 'square':
            # Square perimeter
            size = shape.parameters['size']
            num_points = shape.parameters.get('num_points', 12)
            
            # Sample points around square perimeter
            perimeter_length = 4 * size
            for i in range(num_points):
                t = i / num_points * perimeter_length
                
                if t < size:  # Bottom edge
                    pos = shape.center + np.array([t - size/2, -size/2, 0])
                elif t < 2 * size:  # Right edge
                    pos = shape.center + np.array([size/2, (t - size) - size/2, 0])
                elif t < 3 * size:  # Top edge
                    pos = shape.center + np.array([(3*size - t) - size/2, size/2, 0])
                else:  # Left edge
                    pos = shape.center + np.array([-size/2, (4*size - t) - size/2, 0])
                
                r_squared = ((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
                field += shape.pressure * np.exp(-r_squared / (2 * 0.005**2))
        
        return field
    
    def _compute_field_from_phases(self, phases: np.ndarray) -> np.ndarray:
        """Compute field from phase array using propagator."""
        amplitudes = np.ones(len(phases))
        positions = self.array.get_positions()
        
        # Use reduced resolution for real-time computation
        small_propagator = WavePropagator(
            resolution=2e-3,  # 2mm resolution for speed
            bounds=self.workspace_bounds,
            frequency=self.array.frequency,
            device="cpu"  # Keep on CPU for threading
        )
        
        field = small_propagator.compute_field_from_sources(
            source_positions=positions,
            source_amplitudes=amplitudes,
            source_phases=phases
        )
        
        return field
    
    def _apply_modulation(self, phases: np.ndarray) -> np.ndarray:
        """Apply temporal modulation for tactile perception."""
        # Get current time modulation
        t = time.time()
        
        # Apply modulation to active shapes
        modulated_phases = phases.copy()
        
        for shape_id, shape in self.active_shapes.items():
            if shape.modulation_freq > 0:
                # Sinusoidal modulation for tactile perception
                modulation = 0.5 * (1 + np.sin(2 * np.pi * shape.modulation_freq * t))
                
                # Apply modulation to phases (simple approach)
                modulated_phases *= modulation
        
        return modulated_phases
    
    def _check_safety(self, phases: np.ndarray) -> bool:
        """Check if phase configuration is safe."""
        # Estimate maximum pressure (simplified)
        max_amplitude = np.sum(np.ones(len(phases)))  # All elements at max
        estimated_pressure = max_amplitude * 100  # Rough estimate
        
        return estimated_pressure <= self.max_pressure
    
    def _is_position_valid(self, position: np.ndarray) -> bool:
        """Check if position is within workspace bounds."""
        for i, (min_val, max_val) in enumerate(self.workspace_bounds):
            if position[i] < min_val or position[i] > max_val:
                return False
        return True
    
    def pulse(self, position: List[float], duration: float = 0.1, pressure: float = 300) -> None:
        """
        Create a brief haptic pulse at specified position.
        
        Args:
            position: 3D position for pulse
            duration: Pulse duration in seconds
            pressure: Pulse pressure in Pa
        """
        pulse_id = f"pulse_{time.time()}"
        
        pulse_shape = self.create_shape(
            shape_type='point',
            center=position,
            pressure=pressure,
            duration=duration
        )
        
        self.add_shape(pulse_id, pulse_shape)
        
        # Remove pulse after duration
        def remove_pulse():
            time.sleep(duration)
            self.remove_shape(pulse_id)
        
        pulse_thread = threading.Thread(target=remove_pulse)
        pulse_thread.daemon = True
        pulse_thread.start()
    
    def render(
        self,
        shapes: Dict[str, HapticShape],
        duration: Optional[float] = None,
        modulation: str = "sinusoidal",
        frequency: float = 200
    ) -> None:
        """
        Render haptic shapes for specified duration.
        
        Args:
            shapes: Dictionary of shapes to render
            duration: Rendering duration (None for indefinite)
            modulation: Modulation type for tactile perception
            frequency: Modulation frequency in Hz
        """
        # Clear existing shapes
        self.active_shapes.clear()
        
        # Add new shapes with modulation settings
        for shape_id, shape in shapes.items():
            shape.modulation_freq = frequency
            self.add_shape(shape_id, shape)
        
        # Start rendering
        if not self.is_rendering:
            self.start_rendering()
        
        # Stop after duration if specified
        if duration is not None:
            def stop_after_duration():
                time.sleep(duration)
                self.stop_rendering_loop()
                self.active_shapes.clear()
            
            stop_thread = threading.Thread(target=stop_after_duration)
            stop_thread.daemon = True
            stop_thread.start()
    
    def get_render_stats(self) -> Dict[str, Any]:
        """Get current rendering statistics."""
        return self.render_stats.copy()
    
    def calibrate_tactile_threshold(
        self,
        position: List[float],
        max_pressure: float = 500,
        step_size: float = 10
    ) -> float:
        """
        Calibrate tactile perception threshold at position.
        
        Args:
            position: Test position
            max_pressure: Maximum test pressure
            step_size: Pressure increment step
            
        Returns:
            Detected threshold pressure in Pa
        """
        print("Starting tactile threshold calibration...")
        print("Press Enter when you feel the sensation")
        
        threshold = None
        
        for pressure in np.arange(step_size, max_pressure + step_size, step_size):
            print(f"Testing pressure: {pressure} Pa")
            
            # Create test pulse
            test_shape = self.create_shape(
                shape_type='point',
                center=position,
                pressure=pressure,
                modulation_freq=200
            )
            
            self.add_shape('calibration', test_shape)
            
            # Wait for user input
            input("Press Enter when sensation is felt (or 'n' for not felt): ")
            response = input().lower()
            
            self.remove_shape('calibration')
            
            if response != 'n':
                threshold = pressure
                break
            
            time.sleep(0.5)  # Brief pause between tests
        
        if threshold:
            print(f"Tactile threshold detected at {threshold} Pa")
        else:
            print("No tactile sensation detected up to maximum pressure")
            threshold = max_pressure
        
        return threshold
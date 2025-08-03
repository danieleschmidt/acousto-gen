"""
Acoustic levitation application module.
Implements particle trapping, manipulation, and choreography.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time
from scipy.optimize import minimize
from scipy.interpolate import interp1d


@dataclass
class Particle:
    """Levitated particle properties."""
    position: np.ndarray  # Current 3D position
    velocity: np.ndarray  # Current 3D velocity
    radius: float  # Particle radius in meters
    density: float  # Particle density in kg/m³
    mass: float = None  # Calculated mass
    id: int = None  # Unique particle ID
    
    def __post_init__(self):
        """Calculate derived properties."""
        if self.mass is None:
            volume = (4/3) * np.pi * self.radius**3
            self.mass = self.density * volume
    
    def get_drag_coefficient(self, medium_viscosity: float = 1.8e-5) -> float:
        """
        Calculate Stokes drag coefficient.
        
        Args:
            medium_viscosity: Dynamic viscosity of medium (default air)
            
        Returns:
            Drag coefficient
        """
        return 6 * np.pi * medium_viscosity * self.radius


class AcousticLevitator:
    """
    Acoustic levitation controller for particle manipulation.
    Manages trap generation, particle tracking, and trajectory control.
    """
    
    def __init__(
        self,
        transducer_array,
        wave_propagator,
        optimizer,
        workspace_bounds: List[Tuple[float, float]] = None
    ):
        """
        Initialize acoustic levitator.
        
        Args:
            transducer_array: Transducer array object
            wave_propagator: Wave propagation engine
            optimizer: Hologram optimizer
            workspace_bounds: Safe working volume bounds
        """
        self.array = transducer_array
        self.propagator = wave_propagator
        self.optimizer = optimizer
        self.workspace_bounds = workspace_bounds or [
            (-0.05, 0.05), (-0.05, 0.05), (0.05, 0.15)
        ]
        
        # Particle management
        self.particles: List[Particle] = []
        self.next_particle_id = 0
        
        # Trap configuration
        self.trap_phases = None
        self.trap_positions = []
        self.trap_strength = 1.0
        
        # Physics parameters
        self.gravity = np.array([0, 0, -9.81])
        self.medium_density = 1.2  # Air density kg/m³
        
        # Control parameters
        self.update_rate = 100  # Hz
        self.position_tolerance = 1e-3  # meters
        
    def add_particle(
        self,
        position: List[float],
        radius: float = 1e-3,
        density: float = 25
    ) -> Particle:
        """
        Add a particle to the levitation system.
        
        Args:
            position: Initial 3D position
            radius: Particle radius in meters
            density: Particle density in kg/m³
            
        Returns:
            Created Particle object
        """
        particle = Particle(
            position=np.array(position),
            velocity=np.zeros(3),
            radius=radius,
            density=density,
            id=self.next_particle_id
        )
        
        self.next_particle_id += 1
        self.particles.append(particle)
        
        # Update trap configuration
        self._update_traps()
        
        return particle
    
    def remove_particle(self, particle: Particle):
        """Remove a particle from the system."""
        if particle in self.particles:
            self.particles.remove(particle)
            self._update_traps()
    
    def create_trap(
        self,
        position: List[float],
        trap_type: str = "twin_trap",
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Create acoustic trap at specified position.
        
        Args:
            position: 3D trap position
            trap_type: Type of trap ('twin_trap', 'vortex', 'bottle')
            strength: Trap strength factor
            
        Returns:
            Optimized phase pattern
        """
        position = np.array(position)
        
        # Check workspace bounds
        if not self._in_workspace(position):
            raise ValueError(f"Position {position} outside workspace bounds")
        
        if trap_type == "twin_trap":
            phases = self._create_twin_trap(position, strength)
        elif trap_type == "vortex":
            phases = self._create_vortex_trap(position, strength)
        elif trap_type == "bottle":
            phases = self._create_bottle_trap(position, strength)
        else:
            raise ValueError(f"Unknown trap type: {trap_type}")
        
        return phases
    
    def _create_twin_trap(
        self,
        position: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Create twin-trap configuration for stable levitation.
        
        Args:
            position: Trap center position
            strength: Trap strength
            
        Returns:
            Phase pattern
        """
        # Twin trap has two focal points above and below particle
        wavelength = self.propagator.medium.get_wavelength(self.propagator.frequency)
        separation = wavelength / 4  # Quarter wavelength separation
        
        # Define target field with two foci
        focus1 = position + np.array([0, 0, separation/2])
        focus2 = position - np.array([0, 0, separation/2])
        
        # Create target pressure field
        target_field = self._create_focus_field(focus1, strength)
        target_field += self._create_focus_field(focus2, strength)
        
        # Add null at particle position for stability
        target_field += self._create_null_field(position, wavelength/8)
        
        # Optimize phases
        def forward_model(phases):
            return self.propagator.compute_field_from_sources(
                self.array.get_positions(),
                np.ones(len(self.array.elements)) * strength,
                phases,
                target_points=self._get_optimization_points(position)
            )
        
        result = self.optimizer.optimize(
            forward_model=forward_model,
            target_field=target_field,
            iterations=500
        )
        
        return result.phases
    
    def _create_vortex_trap(
        self,
        position: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Create vortex beam trap with orbital angular momentum.
        
        Args:
            position: Trap center position
            strength: Trap strength
            
        Returns:
            Phase pattern
        """
        # Vortex beam has helical phase front
        transducer_positions = self.array.get_positions()
        phases = np.zeros(len(transducer_positions))
        
        for i, trans_pos in enumerate(transducer_positions):
            # Vector from transducer to target
            r = position - trans_pos
            
            # Azimuthal angle
            phi = np.arctan2(r[1], r[0])
            
            # Helical phase with topological charge
            topological_charge = 1  # Can be varied for different vortex orders
            helical_phase = topological_charge * phi
            
            # Focusing phase
            distance = np.linalg.norm(r)
            k = 2 * np.pi * self.propagator.frequency / self.propagator.medium.speed_of_sound
            focusing_phase = k * distance
            
            phases[i] = (helical_phase + focusing_phase) % (2 * np.pi)
        
        return phases * strength
    
    def _create_bottle_trap(
        self,
        position: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Create bottle beam trap with 3D confinement.
        
        Args:
            position: Trap center position
            strength: Trap strength
            
        Returns:
            Phase pattern
        """
        # Bottle beam combines Bessel beam with additional modulation
        transducer_positions = self.array.get_positions()
        phases = np.zeros(len(transducer_positions))
        
        # Bessel beam parameters
        k = 2 * np.pi * self.propagator.frequency / self.propagator.medium.speed_of_sound
        kr = k * 0.8  # Radial wave vector component
        kz = np.sqrt(k**2 - kr**2)  # Axial component
        
        for i, trans_pos in enumerate(transducer_positions):
            r = position - trans_pos
            rho = np.sqrt(r[0]**2 + r[1]**2)  # Radial distance
            
            # Bessel beam phase
            bessel_phase = kr * rho + kz * r[2]
            
            # Add axial modulation for bottle shape
            axial_mod = np.cos(2 * np.pi * r[2] / (2 * self.propagator.wavelength))
            
            phases[i] = (bessel_phase * axial_mod) % (2 * np.pi)
        
        return phases * strength
    
    def create_multi_trap(
        self,
        positions: List[List[float]],
        trap_type: str = "twin_trap"
    ) -> np.ndarray:
        """
        Create multiple traps simultaneously.
        
        Args:
            positions: List of trap positions
            trap_type: Type of traps to create
            
        Returns:
            Combined phase pattern
        """
        if len(positions) > 10:
            raise ValueError("Too many simultaneous traps (max 10)")
        
        # Use weighted Gerchberg-Saxton algorithm
        target_points = []
        target_amplitudes = []
        
        for pos in positions:
            points = self._get_optimization_points(pos)
            target_points.extend(points)
            target_amplitudes.extend([1.0] * len(points))
        
        target_points = np.array(target_points)
        target_amplitudes = np.array(target_amplitudes)
        
        # Iterative optimization
        phases = np.random.uniform(0, 2*np.pi, len(self.array.elements))
        
        for iteration in range(100):
            # Forward propagation
            field = self.propagator.compute_field_from_sources(
                self.array.get_positions(),
                np.ones(len(self.array.elements)),
                phases,
                target_points=target_points
            )
            
            # Apply target amplitude constraint
            field_phase = np.angle(field)
            field = target_amplitudes * np.exp(1j * field_phase)
            
            # Inverse propagation (simplified)
            # This would ideally use proper inverse propagation
            phase_corrections = np.angle(field)
            phases = (phases + 0.1 * np.mean(phase_corrections)) % (2 * np.pi)
        
        return phases
    
    def move_particle(
        self,
        particle: Particle,
        target_position: List[float],
        speed: float = 0.05,
        method: str = "direct"
    ):
        """
        Move particle to target position.
        
        Args:
            particle: Particle to move
            target_position: Target 3D position
            speed: Movement speed in m/s
            method: Movement method ('direct', 'smooth', 'minimum_jerk')
        """
        target_position = np.array(target_position)
        
        if not self._in_workspace(target_position):
            raise ValueError("Target position outside workspace")
        
        if method == "direct":
            trajectory = self._plan_direct_trajectory(
                particle.position, target_position, speed
            )
        elif method == "smooth":
            trajectory = self._plan_smooth_trajectory(
                particle.position, target_position, speed
            )
        elif method == "minimum_jerk":
            trajectory = self._plan_minimum_jerk_trajectory(
                particle.position, target_position, speed
            )
        else:
            raise ValueError(f"Unknown movement method: {method}")
        
        # Execute trajectory
        self._execute_trajectory(particle, trajectory)
    
    def _plan_direct_trajectory(
        self,
        start: np.ndarray,
        end: np.ndarray,
        speed: float
    ) -> List[Dict[str, Any]]:
        """Plan direct linear trajectory."""
        distance = np.linalg.norm(end - start)
        duration = distance / speed
        num_points = int(duration * self.update_rate)
        
        trajectory = []
        for i in range(num_points + 1):
            t = i / num_points
            position = start + t * (end - start)
            trajectory.append({
                'time': t * duration,
                'position': position,
                'velocity': (end - start) / duration if i < num_points else np.zeros(3)
            })
        
        return trajectory
    
    def _plan_smooth_trajectory(
        self,
        start: np.ndarray,
        end: np.ndarray,
        speed: float
    ) -> List[Dict[str, Any]]:
        """Plan smooth trajectory with acceleration limits."""
        distance = np.linalg.norm(end - start)
        max_acceleration = 2.0  # m/s²
        
        # Trapezoidal velocity profile
        accel_time = speed / max_acceleration
        accel_distance = 0.5 * max_acceleration * accel_time**2
        
        if 2 * accel_distance > distance:
            # Triangle profile (no constant velocity phase)
            accel_time = np.sqrt(distance / max_acceleration)
            cruise_time = 0
        else:
            cruise_distance = distance - 2 * accel_distance
            cruise_time = cruise_distance / speed
        
        total_time = 2 * accel_time + cruise_time
        num_points = int(total_time * self.update_rate)
        
        trajectory = []
        direction = (end - start) / distance
        
        for i in range(num_points + 1):
            t = i * total_time / num_points
            
            if t < accel_time:
                # Acceleration phase
                s = 0.5 * max_acceleration * t**2
                v = max_acceleration * t
            elif t < accel_time + cruise_time:
                # Cruise phase
                s = accel_distance + speed * (t - accel_time)
                v = speed
            else:
                # Deceleration phase
                t_decel = t - accel_time - cruise_time
                s = distance - 0.5 * max_acceleration * (accel_time - t_decel)**2
                v = speed - max_acceleration * t_decel
            
            position = start + s * direction
            velocity = v * direction
            
            trajectory.append({
                'time': t,
                'position': position,
                'velocity': velocity
            })
        
        return trajectory
    
    def _plan_minimum_jerk_trajectory(
        self,
        start: np.ndarray,
        end: np.ndarray,
        speed: float
    ) -> List[Dict[str, Any]]:
        """Plan minimum jerk trajectory for smooth motion."""
        distance = np.linalg.norm(end - start)
        duration = distance / speed * 1.5  # Slightly longer for smoothness
        num_points = int(duration * self.update_rate)
        
        trajectory = []
        
        for i in range(num_points + 1):
            t = i / num_points
            tau = t  # Normalized time
            
            # Minimum jerk polynomial
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
            
            position = start + s * (end - start)
            velocity = s_dot * (end - start)
            
            trajectory.append({
                'time': t * duration,
                'position': position,
                'velocity': velocity
            })
        
        return trajectory
    
    def _execute_trajectory(
        self,
        particle: Particle,
        trajectory: List[Dict[str, Any]]
    ):
        """Execute planned trajectory."""
        for waypoint in trajectory:
            # Update trap position
            self.trap_positions = [waypoint['position']]
            self._update_traps()
            
            # Update particle state
            particle.position = waypoint['position'].copy()
            particle.velocity = waypoint['velocity'].copy()
            
            # Apply phases to hardware
            if self.trap_phases is not None:
                self.array.set_phases(self.trap_phases)
            
            # Wait for next update
            time.sleep(1.0 / self.update_rate)
    
    def create_choreography(
        self,
        particles: List[Particle],
        formation_sequence: List[str],
        transition_time: float = 2.0
    ) -> List[List[Dict[str, Any]]]:
        """
        Create choreographed movement for multiple particles.
        
        Args:
            particles: List of particles to choreograph
            formation_sequence: Sequence of formation names
            transition_time: Time for each transition
            
        Returns:
            List of trajectories for each particle
        """
        choreography = []
        
        for formation_name in formation_sequence:
            formation_positions = self._get_formation_positions(
                formation_name, len(particles)
            )
            
            # Plan synchronized trajectories
            particle_trajectories = []
            for particle, target_pos in zip(particles, formation_positions):
                trajectory = self._plan_smooth_trajectory(
                    particle.position,
                    target_pos,
                    speed=np.linalg.norm(target_pos - particle.position) / transition_time
                )
                particle_trajectories.append(trajectory)
            
            choreography.append(particle_trajectories)
        
        return choreography
    
    def _get_formation_positions(
        self,
        formation_name: str,
        num_particles: int
    ) -> List[np.ndarray]:
        """Get particle positions for named formation."""
        center = np.array([0, 0, 0.1])  # Default center
        
        if formation_name == "line":
            spacing = 0.02
            positions = []
            for i in range(num_particles):
                offset = (i - num_particles/2 + 0.5) * spacing
                positions.append(center + np.array([offset, 0, 0]))
            
        elif formation_name == "circle":
            radius = 0.03
            positions = []
            for i in range(num_particles):
                angle = 2 * np.pi * i / num_particles
                pos = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
                positions.append(pos)
            
        elif formation_name == "helix":
            radius = 0.02
            pitch = 0.05
            positions = []
            for i in range(num_particles):
                angle = 2 * np.pi * i / num_particles * 2  # Two turns
                z = pitch * i / num_particles
                pos = center + np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    z - pitch/2
                ])
                positions.append(pos)
            
        elif formation_name == "cube":
            # Arrange in cubic lattice
            n = int(np.ceil(num_particles**(1/3)))
            spacing = 0.02
            positions = []
            count = 0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        if count >= num_particles:
                            break
                        pos = center + spacing * np.array([
                            i - n/2 + 0.5,
                            j - n/2 + 0.5,
                            k - n/2 + 0.5
                        ])
                        positions.append(pos)
                        count += 1
        
        elif formation_name == "sphere":
            # Distribute on sphere surface
            positions = []
            golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle
            
            for i in range(num_particles):
                theta = golden_angle * i
                y = 1 - (i / (num_particles - 1)) * 2  # -1 to 1
                radius_at_y = np.sqrt(1 - y * y)
                
                pos = center + 0.03 * np.array([
                    np.cos(theta) * radius_at_y,
                    y,
                    np.sin(theta) * radius_at_y
                ])
                positions.append(pos)
        
        else:
            raise ValueError(f"Unknown formation: {formation_name}")
        
        return positions
    
    def _update_traps(self):
        """Update trap configuration based on current particles."""
        if not self.particles:
            self.trap_phases = None
            return
        
        # Get particle positions
        positions = [p.position.tolist() for p in self.particles]
        
        # Create multi-trap pattern
        self.trap_phases = self.create_multi_trap(positions)
        self.trap_positions = positions
    
    def _in_workspace(self, position: np.ndarray) -> bool:
        """Check if position is within workspace bounds."""
        for i, (min_val, max_val) in enumerate(self.workspace_bounds):
            if position[i] < min_val or position[i] > max_val:
                return False
        return True
    
    def _create_focus_field(
        self,
        position: np.ndarray,
        amplitude: float
    ) -> np.ndarray:
        """Create focused pressure field at position."""
        points = self._get_optimization_points(position)
        field = np.zeros(len(points), dtype=complex)
        
        for i, point in enumerate(points):
            distance = np.linalg.norm(point - position)
            if distance < 0.01:  # Within 1cm
                field[i] = amplitude * np.exp(-distance**2 / (2 * 0.005**2))
        
        return field
    
    def _create_null_field(
        self,
        position: np.ndarray,
        radius: float
    ) -> np.ndarray:
        """Create null pressure region."""
        points = self._get_optimization_points(position)
        field = np.zeros(len(points), dtype=complex)
        
        for i, point in enumerate(points):
            distance = np.linalg.norm(point - position)
            if distance < radius:
                field[i] = -1.0  # Negative to create null
        
        return field
    
    def _get_optimization_points(
        self,
        center: np.ndarray,
        radius: float = 0.02
    ) -> np.ndarray:
        """Get sample points for field optimization."""
        # Create grid of points around center
        n_points = 10
        x = np.linspace(center[0] - radius, center[0] + radius, n_points)
        y = np.linspace(center[1] - radius, center[1] + radius, n_points)
        z = np.linspace(center[2] - radius, center[2] + radius, n_points)
        
        points = []
        for xi in x:
            for yi in y:
                for zi in z:
                    points.append([xi, yi, zi])
        
        return np.array(points)
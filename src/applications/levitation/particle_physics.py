"""
Particle physics simulation for acoustic levitation.
Implements force calculations, trajectory simulation, and stability analysis.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import time

from ..models.acoustic_field import AcousticField


@dataclass
class Particle:
    """Physical particle for acoustic levitation."""
    id: int
    position: np.ndarray  # 3D position [x, y, z] in meters
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s²
    radius: float = 1e-3  # meters
    density: float = 25  # kg/m³ (polystyrene foam)
    mass: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived properties."""
        volume = (4/3) * np.pi * self.radius**3
        self.mass = self.density * volume
    
    def update_position(self, dt: float):
        """Update particle position using Verlet integration."""
        # Verlet integration for stable numerical simulation
        new_position = (self.position + self.velocity * dt + 
                       0.5 * self.acceleration * dt**2)
        
        new_acceleration = self.acceleration  # Will be updated by force calculation
        new_velocity = (self.velocity + 0.5 * (self.acceleration + new_acceleration) * dt)
        
        self.position = new_position
        self.velocity = new_velocity
        self.acceleration = new_acceleration


@dataclass
class ForceComponents:
    """Components of forces acting on a particle."""
    acoustic_radiation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    drag: np.ndarray = field(default_factory=lambda: np.zeros(3))
    total: np.ndarray = field(default_factory=lambda: np.zeros(3))


class AcousticForceCalculator:
    """Calculate forces on particles in acoustic fields."""
    
    def __init__(
        self,
        medium_density: float = 1.2,  # kg/m³ (air)
        medium_speed: float = 343,    # m/s
        medium_viscosity: float = 1.8e-5  # Pa·s
    ):
        """
        Initialize force calculator.
        
        Args:
            medium_density: Density of surrounding medium
            medium_speed: Speed of sound in medium
            medium_viscosity: Dynamic viscosity of medium
        """
        self.medium_density = medium_density
        self.medium_speed = medium_speed
        self.medium_viscosity = medium_viscosity
        self.medium_impedance = medium_density * medium_speed
    
    def calculate_acoustic_radiation_force(
        self,
        particle: Particle,
        acoustic_field: AcousticField
    ) -> np.ndarray:
        """
        Calculate acoustic radiation force using Gor'kov potential.
        
        Args:
            particle: Particle object
            acoustic_field: Acoustic field
            
        Returns:
            Force vector in Newtons
        """
        # Get field values at particle position
        field_at_particle = acoustic_field.interpolate_at_points(
            particle.position.reshape(1, -1)
        )[0]
        
        pressure_amplitude = abs(field_at_particle)
        
        # Calculate velocity field from pressure gradient
        velocity_field = self._calculate_velocity_from_pressure(
            acoustic_field, particle.position
        )
        velocity_amplitude = np.linalg.norm(velocity_field)
        
        # Gor'kov potential coefficients
        f1, f2 = self._calculate_gorkov_coefficients(particle)
        
        # Monopole contribution (f1 term)
        monopole_potential = f1 * pressure_amplitude**2 / (4 * self.medium_density * self.medium_speed**2)
        
        # Dipole contribution (f2 term)
        dipole_potential = f2 * (3/4) * self.medium_density * velocity_amplitude**2
        
        # Total Gor'kov potential
        total_potential = monopole_potential + dipole_potential
        
        # Force is negative gradient of potential
        force = -self._calculate_potential_gradient(
            acoustic_field, particle.position, particle.radius, f1, f2
        )
        
        return force
    
    def _calculate_gorkov_coefficients(self, particle: Particle) -> Tuple[float, float]:
        """Calculate Gor'kov potential coefficients f1 and f2."""
        # Material properties (polystyrene in air)
        particle_density = particle.density
        particle_speed = 2350  # m/s (speed of sound in polystyrene)
        
        # Monopole coefficient f1
        kappa_particle = 1 / (particle_density * particle_speed**2)
        kappa_medium = 1 / (self.medium_density * self.medium_speed**2)
        f1 = 1 - kappa_particle / kappa_medium
        
        # Dipole coefficient f2
        rho_ratio = particle_density / self.medium_density
        f2 = (2 * (rho_ratio - 1)) / (2 * rho_ratio + 1)
        
        return f1, f2
    
    def _calculate_velocity_from_pressure(
        self,
        acoustic_field: AcousticField,
        position: np.ndarray
    ) -> np.ndarray:
        """Calculate velocity field from pressure gradient."""
        # Create small perturbations for gradient calculation
        h = acoustic_field.resolution * 0.1
        positions = np.array([
            position + [h, 0, 0],
            position - [h, 0, 0],
            position + [0, h, 0],
            position - [0, h, 0],
            position + [0, 0, h],
            position - [0, 0, h]
        ])
        
        # Interpolate pressure at all positions
        pressures = acoustic_field.interpolate_at_points(positions)
        
        # Calculate pressure gradients
        dp_dx = (pressures[0] - pressures[1]) / (2 * h)
        dp_dy = (pressures[2] - pressures[3]) / (2 * h)
        dp_dz = (pressures[4] - pressures[5]) / (2 * h)
        
        # Convert to velocity (Euler equation)
        omega = 2 * np.pi * acoustic_field.frequency
        velocity = np.array([dp_dx, dp_dy, dp_dz]) / (1j * omega * self.medium_density)
        
        return np.real(velocity)
    
    def _calculate_potential_gradient(
        self,
        acoustic_field: AcousticField,
        position: np.ndarray,
        radius: float,
        f1: float,
        f2: float
    ) -> np.ndarray:
        """Calculate gradient of Gor'kov potential."""
        # This is a simplified calculation
        # In practice, would need to compute field gradients more accurately
        h = acoustic_field.resolution * 0.1
        
        forces = []
        for i in range(3):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += h
            pos_minus[i] -= h
            
            # Calculate potential at both positions
            potential_plus = self._calculate_gorkov_potential_at_point(
                acoustic_field, pos_plus, radius, f1, f2
            )
            potential_minus = self._calculate_gorkov_potential_at_point(
                acoustic_field, pos_minus, radius, f1, f2
            )
            
            # Numerical gradient
            force_component = -(potential_plus - potential_minus) / (2 * h)
            forces.append(force_component)
        
        return np.array(forces)
    
    def _calculate_gorkov_potential_at_point(
        self,
        acoustic_field: AcousticField,
        position: np.ndarray,
        radius: float,
        f1: float,
        f2: float
    ) -> float:
        """Calculate Gor'kov potential at a specific point."""
        try:
            # Get field at position
            field_value = acoustic_field.interpolate_at_points(position.reshape(1, -1))[0]
            pressure_amplitude = abs(field_value)
            
            # Get velocity
            velocity = self._calculate_velocity_from_pressure(acoustic_field, position)
            velocity_amplitude = np.linalg.norm(velocity)
            
            # Particle volume
            volume = (4/3) * np.pi * radius**3
            
            # Gor'kov potential
            monopole_term = f1 * pressure_amplitude**2 / (4 * self.medium_density * self.medium_speed**2)
            dipole_term = f2 * (3/4) * self.medium_density * velocity_amplitude**2
            
            potential = volume * (monopole_term + dipole_term)
            
            return potential
            
        except:
            return 0.0
    
    def calculate_gravity_force(self, particle: Particle) -> np.ndarray:
        """Calculate gravitational force."""
        g = 9.81  # m/s²
        return np.array([0, 0, -particle.mass * g])
    
    def calculate_drag_force(
        self,
        particle: Particle,
        ambient_velocity: np.ndarray = None
    ) -> np.ndarray:
        """
        Calculate viscous drag force.
        
        Args:
            particle: Particle object
            ambient_velocity: Velocity of surrounding medium
            
        Returns:
            Drag force vector
        """
        if ambient_velocity is None:
            ambient_velocity = np.zeros(3)
        
        relative_velocity = particle.velocity - ambient_velocity
        speed = np.linalg.norm(relative_velocity)
        
        if speed < 1e-10:
            return np.zeros(3)
        
        # Reynolds number
        Re = (self.medium_density * speed * 2 * particle.radius) / self.medium_viscosity
        
        # Drag coefficient (sphere)
        if Re < 0.1:
            # Stokes flow
            Cd = 24 / Re
        elif Re < 1000:
            # Intermediate regime
            Cd = 24 / Re + 6 / (1 + np.sqrt(Re)) + 0.4
        else:
            # Turbulent
            Cd = 0.44
        
        # Drag force
        cross_sectional_area = np.pi * particle.radius**2
        drag_magnitude = 0.5 * self.medium_density * speed**2 * Cd * cross_sectional_area
        
        # Direction opposite to velocity
        drag_direction = -relative_velocity / speed
        
        return drag_magnitude * drag_direction
    
    def calculate_total_force(
        self,
        particle: Particle,
        acoustic_field: AcousticField,
        ambient_velocity: np.ndarray = None
    ) -> ForceComponents:
        """Calculate all forces acting on particle."""
        forces = ForceComponents()
        
        # Individual force components
        forces.acoustic_radiation = self.calculate_acoustic_radiation_force(
            particle, acoustic_field
        )
        forces.gravity = self.calculate_gravity_force(particle)
        forces.drag = self.calculate_drag_force(particle, ambient_velocity)
        
        # Total force
        forces.total = forces.acoustic_radiation + forces.gravity + forces.drag
        
        return forces


class ParticleSimulator:
    """Simulate particle motion in acoustic fields."""
    
    def __init__(
        self,
        force_calculator: AcousticForceCalculator = None,
        dt: float = 1e-4  # 0.1 ms time step
    ):
        """
        Initialize particle simulator.
        
        Args:
            force_calculator: Force calculation engine
            dt: Time step for simulation
        """
        self.force_calculator = force_calculator or AcousticForceCalculator()
        self.dt = dt
        self.particles: List[Particle] = []
        self.time = 0.0
        self.history: List[Dict[str, Any]] = []
    
    def add_particle(
        self,
        position: List[float],
        radius: float = 1e-3,
        density: float = 25,
        velocity: List[float] = None
    ) -> Particle:
        """Add particle to simulation."""
        particle_id = len(self.particles)
        
        particle = Particle(
            id=particle_id,
            position=np.array(position),
            velocity=np.array(velocity or [0, 0, 0]),
            radius=radius,
            density=density
        )
        
        self.particles.append(particle)
        return particle
    
    def step(self, acoustic_field: AcousticField):
        """Advance simulation by one time step."""
        for particle in self.particles:
            # Calculate forces
            forces = self.force_calculator.calculate_total_force(
                particle, acoustic_field
            )
            
            # Update acceleration
            particle.acceleration = forces.total / particle.mass
            
            # Update position and velocity
            particle.update_position(self.dt)
        
        # Update time
        self.time += self.dt
        
        # Record history
        self._record_state()
    
    def simulate(
        self,
        acoustic_field: AcousticField,
        duration: float,
        record_interval: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Run simulation for specified duration.
        
        Args:
            acoustic_field: Acoustic field
            duration: Simulation duration in seconds
            record_interval: How often to record state
            
        Returns:
            Simulation results
        """
        start_time = time.time()
        steps = int(duration / self.dt)
        record_every = int(record_interval / self.dt)
        
        for step in range(steps):
            self.step(acoustic_field)
            
            # Record state at intervals
            if step % record_every == 0:
                self._record_state()
        
        elapsed_time = time.time() - start_time
        
        return {
            'simulation_time': duration,
            'real_time': elapsed_time,
            'time_steps': steps,
            'particles': len(self.particles),
            'history': self.history
        }
    
    def _record_state(self):
        """Record current state of all particles."""
        state = {
            'time': self.time,
            'particles': []
        }
        
        for particle in self.particles:
            particle_state = {
                'id': particle.id,
                'position': particle.position.copy(),
                'velocity': particle.velocity.copy(),
                'acceleration': particle.acceleration.copy(),
                'kinetic_energy': 0.5 * particle.mass * np.sum(particle.velocity**2),
                'speed': np.linalg.norm(particle.velocity)
            }
            state['particles'].append(particle_state)
        
        self.history.append(state)
    
    def get_trajectory(self, particle_id: int) -> Dict[str, np.ndarray]:
        """Get trajectory data for specific particle."""
        positions = []
        velocities = []
        times = []
        
        for state in self.history:
            if particle_id < len(state['particles']):
                particle_state = state['particles'][particle_id]
                positions.append(particle_state['position'])
                velocities.append(particle_state['velocity'])
                times.append(state['time'])
        
        return {
            'times': np.array(times),
            'positions': np.array(positions),
            'velocities': np.array(velocities)
        }
    
    def check_stability(
        self,
        particle_id: int,
        time_window: float = 0.1
    ) -> Dict[str, float]:
        """
        Analyze particle stability over recent time window.
        
        Args:
            particle_id: ID of particle to analyze
            time_window: Time window for analysis
            
        Returns:
            Stability metrics
        """
        trajectory = self.get_trajectory(particle_id)
        
        if len(trajectory['times']) < 2:
            return {'stability': 0.0, 'position_variance': np.inf}
        
        # Find recent data within time window
        current_time = trajectory['times'][-1]
        mask = trajectory['times'] >= (current_time - time_window)
        
        recent_positions = trajectory['positions'][mask]
        recent_velocities = trajectory['velocities'][mask]
        
        if len(recent_positions) < 2:
            return {'stability': 0.0, 'position_variance': np.inf}
        
        # Calculate stability metrics
        position_variance = np.mean(np.var(recent_positions, axis=0))
        velocity_variance = np.mean(np.var(recent_velocities, axis=0))
        
        # Position drift
        position_drift = np.linalg.norm(recent_positions[-1] - recent_positions[0])
        
        # Stability score (0-1, higher is more stable)
        stability = 1 / (1 + position_variance * 1000 + velocity_variance * 100)
        
        return {
            'stability': stability,
            'position_variance': position_variance,
            'velocity_variance': velocity_variance,
            'position_drift': position_drift,
            'time_window': time_window
        }


class TrajectoryPlanner:
    """Plan and generate particle trajectories."""
    
    @staticmethod
    def linear_path(
        start: np.ndarray,
        end: np.ndarray,
        duration: float,
        num_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """Generate linear trajectory between two points."""
        times = np.linspace(0, duration, num_points)
        positions = np.linspace(start, end, num_points)
        
        # Calculate velocities
        velocities = np.gradient(positions, times, axis=0)
        
        return {
            'times': times,
            'positions': positions,
            'velocities': velocities
        }
    
    @staticmethod
    def circular_path(
        center: np.ndarray,
        radius: float,
        duration: float,
        num_points: int = 100,
        axis: str = 'xy'
    ) -> Dict[str, np.ndarray]:
        """Generate circular trajectory."""
        times = np.linspace(0, duration, num_points)
        angles = 2 * np.pi * times / duration
        
        positions = np.zeros((num_points, 3))
        
        if axis == 'xy':
            positions[:, 0] = center[0] + radius * np.cos(angles)
            positions[:, 1] = center[1] + radius * np.sin(angles)
            positions[:, 2] = center[2]
        elif axis == 'xz':
            positions[:, 0] = center[0] + radius * np.cos(angles)
            positions[:, 1] = center[1]
            positions[:, 2] = center[2] + radius * np.sin(angles)
        elif axis == 'yz':
            positions[:, 0] = center[0]
            positions[:, 1] = center[1] + radius * np.cos(angles)
            positions[:, 2] = center[2] + radius * np.sin(angles)
        
        velocities = np.gradient(positions, times, axis=0)
        
        return {
            'times': times,
            'positions': positions,
            'velocities': velocities
        }
    
    @staticmethod
    def figure_8_path(
        center: np.ndarray,
        size: float,
        duration: float,
        num_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """Generate figure-8 trajectory."""
        times = np.linspace(0, duration, num_points)
        t = 2 * np.pi * times / duration
        
        positions = np.zeros((num_points, 3))
        positions[:, 0] = center[0] + size * np.sin(t)
        positions[:, 1] = center[1] + size * np.sin(t) * np.cos(t)
        positions[:, 2] = center[2]
        
        velocities = np.gradient(positions, times, axis=0)
        
        return {
            'times': times,
            'positions': positions,
            'velocities': velocities
        }
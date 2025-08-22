"""
SQLAlchemy models for Acousto-Gen database.
Defines tables for storing optimization results, field data, and experiment tracking.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean, 
    LargeBinary, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class TimestampMixin:
    """Mixin for created/updated timestamps."""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class OptimizationResult(Base, TimestampMixin):
    """Store optimization results and parameters."""
    __tablename__ = "optimization_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    
    # Optimization parameters
    method = Column(String(50), nullable=False)  # gradient, genetic, neural
    target_type = Column(String(50), nullable=False)  # single_focus, multi_focus, shaped
    iterations = Column(Integer, nullable=False)
    convergence_threshold = Column(Float, nullable=True)
    
    # Results
    final_loss = Column(Float, nullable=False)
    converged = Column(Boolean, default=False)
    time_elapsed = Column(Float, nullable=False)  # seconds
    
    # Phase data (JSON for small arrays, binary for large)
    phases_json = Column(JSON, nullable=True)  # For arrays < 1000 elements
    phases_binary = Column(LargeBinary, nullable=True)  # For large arrays
    num_elements = Column(Integer, nullable=False)
    
    # Target specification
    target_specification = Column(JSON, nullable=False)
    
    # Convergence history (compressed JSON)
    convergence_history = Column(Text, nullable=True)  # JSON string
    
    # Quality metrics
    focus_error = Column(Float, nullable=True)
    peak_pressure = Column(Float, nullable=True)
    contrast_ratio = Column(Float, nullable=True)
    efficiency = Column(Float, nullable=True)
    
    # System information
    device_used = Column(String(20), nullable=False)  # cpu, cuda:0, etc.
    memory_usage_mb = Column(Float, nullable=True)
    
    # Relationships
    experiment_run_id = Column(Integer, ForeignKey("experiment_runs.id"), nullable=True)
    experiment_run = relationship("ExperimentRun", back_populates="optimization_results")
    
    field_data = relationship("AcousticFieldData", back_populates="optimization_result", uselist=False)
    
    def get_phases(self) -> Optional[List[float]]:
        """Get phase array from storage."""
        if self.phases_json is not None:
            return self.phases_json
        elif self.phases_binary is not None:
            import numpy as np
            return np.frombuffer(self.phases_binary, dtype=np.float32).tolist()
        return None
    
    def set_phases(self, phases: List[float]):
        """Store phase array efficiently."""
        import numpy as np
        phases_array = np.array(phases, dtype=np.float32)
        
        if len(phases) < 1000:
            self.phases_json = phases
            self.phases_binary = None
        else:
            self.phases_json = None
            self.phases_binary = phases_array.tobytes()
    
    def get_convergence_history(self) -> Optional[List[float]]:
        """Get convergence history."""
        if self.convergence_history:
            return json.loads(self.convergence_history)
        return None
    
    def set_convergence_history(self, history: List[float]):
        """Store convergence history."""
        # Compress by sampling if too long
        if len(history) > 1000:
            step = len(history) // 1000
            history = history[::step]
        self.convergence_history = json.dumps(history)
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_optimization_method", "method"),
        Index("idx_optimization_target_type", "target_type"),
        Index("idx_optimization_created_at", "created_at"),
        Index("idx_optimization_final_loss", "final_loss"),
    )


class AcousticFieldData(Base, TimestampMixin):
    """Store acoustic field data."""
    __tablename__ = "acoustic_field_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    field_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    
    # Field metadata
    field_type = Column(String(50), nullable=False)  # target, generated, measured
    shape_x = Column(Integer, nullable=False)
    shape_y = Column(Integer, nullable=False)
    shape_z = Column(Integer, nullable=False)
    resolution = Column(Float, nullable=False)
    frequency = Column(Float, nullable=False)
    
    # Physical bounds
    bounds_json = Column(JSON, nullable=False)  # [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
    
    # Field data storage
    field_data_path = Column(String(500), nullable=True)  # Path to HDF5 file
    field_statistics = Column(JSON, nullable=True)  # Pre-computed statistics
    
    # Quality metrics
    max_pressure = Column(Float, nullable=True)
    mean_pressure = Column(Float, nullable=True)
    rms_pressure = Column(Float, nullable=True)
    dynamic_range_db = Column(Float, nullable=True)
    
    # Relationships
    optimization_result_id = Column(Integer, ForeignKey("optimization_results.id"), nullable=True)
    optimization_result = relationship("OptimizationResult", back_populates="field_data")
    
    def get_bounds(self) -> List[List[float]]:
        """Get field bounds."""
        return self.bounds_json
    
    def set_bounds(self, bounds: List[List[float]]):
        """Set field bounds."""
        self.bounds_json = bounds
    
    __table_args__ = (
        Index("idx_field_type", "field_type"),
        Index("idx_field_frequency", "frequency"),
        Index("idx_field_created_at", "created_at"),
    )


class ArrayConfiguration(Base, TimestampMixin):
    """Store transducer array configurations."""
    __tablename__ = "array_configurations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    config_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    
    # Array metadata
    name = Column(String(100), nullable=False)
    array_type = Column(String(50), nullable=False)  # ultraleap, circular, custom
    num_elements = Column(Integer, nullable=False)
    frequency = Column(Float, nullable=False)
    
    # Geometry
    positions_json = Column(JSON, nullable=False)  # Nx3 array of positions
    orientations_json = Column(JSON, nullable=True)  # Nx3 array of orientations
    
    # Calibration
    phase_offsets = Column(JSON, nullable=True)
    amplitude_factors = Column(JSON, nullable=True)
    calibration_date = Column(DateTime, nullable=True)
    calibration_quality = Column(Float, nullable=True)  # 0-1 score
    
    # Hardware information
    hardware_id = Column(String(100), nullable=True)
    driver_version = Column(String(20), nullable=True)
    
    # Configuration metadata
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    
    def get_positions(self) -> List[List[float]]:
        """Get element positions."""
        return self.positions_json
    
    def set_positions(self, positions: List[List[float]]):
        """Set element positions."""
        self.positions_json = positions
    
    def get_orientations(self) -> Optional[List[List[float]]]:
        """Get element orientations."""
        return self.orientations_json
    
    def set_orientations(self, orientations: List[List[float]]):
        """Set element orientations."""
        self.orientations_json = orientations
    
    __table_args__ = (
        Index("idx_array_type", "array_type"),
        Index("idx_array_active", "is_active"),
        Index("idx_array_num_elements", "num_elements"),
    )


class ExperimentRun(Base, TimestampMixin):
    """Track experimental runs and parameter sweeps."""
    __tablename__ = "experiment_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    
    # Experiment metadata
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    experiment_type = Column(String(50), nullable=False)  # optimization, sweep, validation
    
    # Parameters
    parameters = Column(JSON, nullable=False)
    
    # Status
    status = Column(String(20), default="running")  # running, completed, failed
    progress = Column(Float, default=0.0)  # 0-100
    
    # Results summary
    total_runs = Column(Integer, default=0)
    successful_runs = Column(Integer, default=0)
    best_result_id = Column(Integer, ForeignKey("optimization_results.id"), nullable=True)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # User information
    user_id = Column(String(100), nullable=True)
    
    # Relationships
    optimization_results = relationship(
        "OptimizationResult", 
        back_populates="experiment_run",
        foreign_keys="OptimizationResult.experiment_run_id"
    )
    best_result = relationship(
        "OptimizationResult", 
        foreign_keys=[best_result_id],
        post_update=True
    )
    
    def mark_completed(self):
        """Mark experiment as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        self.progress = 100.0
    
    def mark_failed(self, error_message: str = None):
        """Mark experiment as failed."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()
        if error_message:
            self.description = f"{self.description}\nError: {error_message}"
    
    __table_args__ = (
        Index("idx_experiment_status", "status"),
        Index("idx_experiment_type", "experiment_type"),
        Index("idx_experiment_user", "user_id"),
        Index("idx_experiment_started", "started_at"),
    )


class UserSession(Base, TimestampMixin):
    """Track user sessions and activity."""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    
    # User information
    user_id = Column(String(100), nullable=False)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    
    # Session data
    is_active = Column(Boolean, default=True)
    last_activity = Column(DateTime, default=datetime.utcnow)
    session_data = Column(JSON, nullable=True)
    
    # Activity tracking
    api_calls = Column(Integer, default=0)
    optimizations_run = Column(Integer, default=0)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
        self.api_calls += 1
    
    def end_session(self):
        """End the session."""
        self.is_active = False
        self.last_activity = datetime.utcnow()
    
    __table_args__ = (
        Index("idx_session_user", "user_id"),
        Index("idx_session_active", "is_active"),
        Index("idx_session_activity", "last_activity"),
    )


class SystemMetric(Base):
    """Store system performance metrics."""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # System metrics
    cpu_usage = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    gpu_usage = Column(Float, nullable=True)
    gpu_memory_mb = Column(Float, nullable=True)
    
    # Application metrics
    active_sessions = Column(Integer, default=0)
    optimizations_per_hour = Column(Float, default=0.0)
    average_optimization_time = Column(Float, default=0.0)
    
    # Hardware metrics
    hardware_connected = Column(Boolean, default=False)
    hardware_status = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index("idx_metrics_timestamp", "timestamp"),
    )
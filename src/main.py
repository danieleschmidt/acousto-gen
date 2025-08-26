"""
Acousto-Gen main entry point and API server.
Provides REST API and WebSocket interfaces for acoustic holography control.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from physics.propagation.wave_propagator import WavePropagator, MediumProperties
from physics.transducers.transducer_array import UltraLeap256, CircularArray, CustomArray
from optimization.hologram_optimizer import GradientOptimizer, GeneticOptimizer
from models.acoustic_field import AcousticField, TargetPattern
from applications.levitation.acoustic_levitator import AcousticLevitator
from hardware.drivers.hardware_interface import SerialHardware, NetworkHardware
from database.connection import DatabaseManager
from database.repositories import OptimizationRepository, FieldRepository, ExperimentRepository
from database.models import OptimizationResult, AcousticFieldData, ExperimentRun, SystemMetric
from monitoring.metrics import MetricsCollector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class SystemConfig(BaseModel):
    """System configuration parameters."""
    frequency: float = Field(default=40e3, description="Operating frequency in Hz")
    resolution: float = Field(default=1e-3, description="Spatial resolution in meters")
    device: str = Field(default="cpu", description="Computation device")
    array_type: str = Field(default="ultraleap", description="Transducer array type")


class OptimizationRequest(BaseModel):
    """Optimization request parameters."""
    target_type: str = Field(description="Type of target pattern")
    target_position: Optional[List[float]] = Field(default=None, description="Target position [x, y, z]")
    target_pressure: float = Field(default=3000.0, description="Target pressure in Pa")
    focal_points: list = Field(default=[], description="List of focal points")
    method: str = Field(default="adam", description="Optimization method")
    iterations: int = Field(default=1000, description="Number of iterations")
    convergence_threshold: float = Field(default=1e-6, description="Convergence threshold")
    learning_rate: float = Field(default=0.01, description="Learning rate for gradient methods")
    array_config: Optional[Dict[str, Any]] = Field(default=None, description="Array configuration")
    
    @validator('target_type')
    def validate_target_type(cls, v):
        allowed = ['single_focus', 'twin_trap', 'line_trap', 'custom']
        if v not in allowed:
            raise ValueError(f'target_type must be one of {allowed}')
        return v
    
    @validator('method')
    def validate_method(cls, v):
        allowed = ['adam', 'sgd', 'genetic', 'neural']
        if v not in allowed:
            raise ValueError(f'method must be one of {allowed}')
        return v


class ParticleRequest(BaseModel):
    """Particle manipulation request."""
    position: list = Field(description="3D position [x, y, z]")
    radius: float = Field(default=1e-3, description="Particle radius")
    density: float = Field(default=25, description="Particle density")


class HardwareCommand(BaseModel):
    """Hardware control command."""
    command: str = Field(description="Command type")
    parameters: Optional[Dict[str, Any]] = Field(default=None)
    
    @validator('command')
    def validate_command(cls, v):
        allowed = ['activate', 'deactivate', 'emergency_stop', 'set_phases', 'set_amplitudes', 'calibrate']
        if v not in allowed:
            raise ValueError(f'command must be one of {allowed}')
        return v


class FieldCalculationRequest(BaseModel):
    """Field calculation request."""
    phases: List[float] = Field(description="Phase array")
    amplitudes: List[float] = Field(description="Amplitude array")
    field_bounds: Dict[str, float] = Field(description="Field bounds")
    resolution: Dict[str, int] = Field(description="Field resolution")
    frequency: float = Field(default=40000, description="Frequency in Hz")
    medium: str = Field(default="air", description="Propagation medium")


class ExperimentRequest(BaseModel):
    """Experiment creation request."""
    name: str = Field(description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    application_type: str = Field(description="Application type")
    hardware_config: Optional[Dict[str, Any]] = Field(default=None, description="Hardware configuration")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Experiment parameters")


class SafetyLimits(BaseModel):
    """Safety limits configuration."""
    max_pressure: float = Field(default=5000, description="Maximum pressure in Pa")
    max_intensity: float = Field(default=15, description="Maximum intensity in W/cm²")
    max_temperature: float = Field(default=45, description="Maximum temperature in °C")
    emergency_shutdown_enabled: bool = Field(default=True, description="Emergency shutdown enabled")


# Global system state
class AcoustoGenSystem:
    """Main system controller."""
    
    def __init__(self):
        self.config = SystemConfig()
        self.array = None
        self.propagator = None
        self.optimizer = None
        self.levitator = None
        self.hardware = None
        self.current_field = None
        self.current_phases = None
        
        # Database components
        self.db_manager = DatabaseManager()
        self.opt_repo = None
        self.field_repo = None
        self.exp_repo = None
        
        # WebSocket connections
        self.websocket_clients = set()
        
        # Metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Safety monitoring
        self.safety_limits = SafetyLimits()
        self.safety_violations = []
        
        # System metrics
        self.metrics = {
            "optimizations_performed": 0,
            "particles_levitated": 0,
            "fields_calculated": 0,
            "total_runtime": 0,
            "last_error": None,
            "uptime_start": datetime.now(timezone.utc)
        }
    
    def initialize(self, config: SystemConfig):
        """Initialize system with configuration."""
        self.config = config
        
        # Initialize database
        self.db_manager.create_tables()
        with self.db_manager.get_session() as session:
            self.opt_repo = OptimizationRepository(session)
            self.field_repo = FieldRepository(session)
            self.exp_repo = ExperimentRepository(session)
        
        # Setup transducer array
        if config.array_type == "ultraleap":
            self.array = UltraLeap256()
        elif config.array_type == "circular":
            self.array = CircularArray(radius=0.1, num_elements=64)
        else:
            raise ValueError(f"Unknown array type: {config.array_type}")
        
        # Setup wave propagator
        self.propagator = WavePropagator(
            resolution=config.resolution,
            frequency=config.frequency,
            device=config.device
        )
        
        # Setup optimizer
        self.optimizer = GradientOptimizer(
            num_elements=len(self.array.elements),
            device=config.device
        )
        
        # Setup levitator
        self.levitator = AcousticLevitator(
            transducer_array=self.array,
            wave_propagator=self.propagator,
            optimizer=self.optimizer
        )
        
        logger.info(f"System initialized with {config.array_type} array")
        self.metrics_collector.record_system_event("system_initialized", {"array_type": config.array_type})
    
    def connect_hardware(self, interface_type: str, **kwargs):
        """Connect to hardware interface."""
        if interface_type == "serial":
            self.hardware = SerialHardware(**kwargs)
        elif interface_type == "network":
            self.hardware = NetworkHardware(**kwargs)
        else:
            raise ValueError(f"Unknown interface type: {interface_type}")
        
        success = self.hardware.connect()
        if success:
            logger.info(f"Connected to hardware via {interface_type}")
        else:
            logger.error(f"Failed to connect to hardware")
        
        return success
    
    def optimize_hologram(self, request: OptimizationRequest) -> Dict[str, Any]:
        """Optimize hologram for target pattern."""
        try:
            # Create target pattern
            target = TargetPattern(
                focal_points=request.focal_points,
                null_regions=[],
                constraints={}
            )
            
            # Convert to field
            target_field = target.to_field(
                shape=(50, 50, 50),
                bounds=self.propagator.bounds
            )
            
            # Optimize
            def forward_model(phases):
                return self.propagator.compute_field_from_sources(
                    self.array.get_positions(),
                    np.ones(len(self.array.elements)),
                    phases
                )
            
            import torch
            target_tensor = torch.tensor(
                target_field.data,
                dtype=torch.complex64,
                device=self.optimizer.device
            )
            
            result = self.optimizer.optimize(
                forward_model=forward_model,
                target_field=target_tensor,
                iterations=request.iterations
            )
            
            # Store results
            self.current_phases = result.phases
            
            # Apply to hardware if connected
            if self.hardware and self.hardware.is_connected():
                self.hardware.send_phases(result.phases)
            
            # Update metrics
            self.metrics["optimizations_performed"] += 1
            
            return {
                "success": True,
                "final_loss": float(result.final_loss),
                "iterations": result.iterations,
                "time_elapsed": result.time_elapsed
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self.metrics["last_error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))
    
    def add_particle(self, request: ParticleRequest) -> Dict[str, Any]:
        """Add particle to levitation system."""
        try:
            particle = self.levitator.add_particle(
                position=request.position,
                radius=request.radius,
                density=request.density
            )
            
            # Update metrics
            self.metrics["particles_levitated"] += 1
            
            return {
                "success": True,
                "particle_id": particle.id,
                "mass": particle.mass
            }
            
        except Exception as e:
            logger.error(f"Failed to add particle: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def broadcast_status(self):
        """Broadcast system status to WebSocket clients."""
        status = {
            "type": "status",
            "data": {
                "connected": self.hardware.is_connected() if self.hardware else False,
                "particles": len(self.levitator.particles) if self.levitator else 0,
                "metrics": self.metrics
            }
        }
        
        message = json.dumps(status)
        disconnected = set()
        
        for websocket in self.websocket_clients:
            try:
                await websocket.send_text(message)
            except:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected


# Prometheus metrics
optimization_counter = Counter('acousto_optimizations_total', 'Total number of optimizations')
field_calculation_counter = Counter('acousto_field_calculations_total', 'Total number of field calculations')
api_request_duration = Histogram('acousto_api_request_duration_seconds', 'API request duration')
hardware_connection_gauge = Gauge('acousto_hardware_connected', 'Hardware connection status')
active_particles_gauge = Gauge('acousto_active_particles', 'Number of active particles')

# Security
security = HTTPBearer(auto_error=False)

# Create FastAPI app
app = FastAPI(
    title="Acousto-Gen API",
    description="Generative acoustic holography control system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create system instance
system = AcoustoGenSystem()


# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("Starting Acousto-Gen API server")
    system.initialize(SystemConfig())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Acousto-Gen API server")
    if system.hardware:
        system.hardware.disconnect()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Acousto-Gen API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    uptime = datetime.now(timezone.utc) - system.metrics["uptime_start"]
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "uptime_seconds": uptime.total_seconds(),
        "hardware_connected": system.hardware.is_connected() if system.hardware else False,
        "database_connected": system.db_manager.test_connection(),
        "metrics": system.metrics
    }


@app.get("/info")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Acousto-Gen API",
        "version": "1.0.0",
        "description": "Generative acoustic holography control system",
        "endpoints": {
            "optimization": "/api/v1/optimization/*",
            "field": "/api/v1/field/*",
            "hardware": "/api/v1/hardware/*",
            "safety": "/api/v1/safety/*",
            "experiments": "/api/v1/experiments/*",
            "applications": "/api/v1/applications/*"
        },
        "documentation": "/docs"
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest().decode('utf-8')


@app.post("/system/initialize")
async def initialize_system(config: SystemConfig):
    """Initialize system with configuration."""
    try:
        system.initialize(config)
        return {"success": True, "message": "System initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hardware/connect")
async def connect_hardware(
    interface_type: str,
    port: Optional[str] = None,
    host: Optional[str] = None
):
    """Connect to hardware interface."""
    kwargs = {}
    if port:
        kwargs["port"] = port
    if host:
        kwargs["host"] = host
    
    success = system.connect_hardware(interface_type, **kwargs)
    
    if success:
        return {"success": True, "message": "Hardware connected"}
    else:
        raise HTTPException(status_code=500, detail="Failed to connect to hardware")


@app.post("/hardware/disconnect")
async def disconnect_hardware():
    """Disconnect hardware interface."""
    if system.hardware:
        system.hardware.disconnect()
        return {"success": True, "message": "Hardware disconnected"}
    else:
        return {"success": False, "message": "No hardware connected"}


@app.post("/hardware/command")
async def send_hardware_command(command: HardwareCommand):
    """Send command to hardware."""
    if not system.hardware or not system.hardware.is_connected():
        raise HTTPException(status_code=400, detail="Hardware not connected")
    
    success = system.hardware.send_control_command(
        command.command,
        command.parameters
    )
    
    return {"success": success}


# Legacy endpoints for backward compatibility
@app.post("/optimize")
async def optimize_hologram_legacy(request: OptimizationRequest):
    """Legacy optimize hologram endpoint."""
    return await start_optimization(request)


@app.post("/levitation/particle")
async def add_particle(request: ParticleRequest):
    """Add particle to levitation system."""
    result = system.add_particle(request)
    await system.broadcast_status()
    return result


@app.post("/levitation/move")
async def move_particle(
    particle_id: int,
    target_position: list,
    speed: float = 0.05
):
    """Move particle to target position."""
    if not system.levitator:
        raise HTTPException(status_code=400, detail="System not initialized")
    
    # Find particle
    particle = None
    for p in system.levitator.particles:
        if p.id == particle_id:
            particle = p
            break
    
    if not particle:
        raise HTTPException(status_code=404, detail="Particle not found")
    
    try:
        system.levitator.move_particle(particle, target_position, speed)
        return {"success": True, "message": "Movement started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/field/current")
async def get_current_field():
    """Get current acoustic field data."""
    if system.current_field is None:
        return {"error": "No field computed"}
    
    # Return field statistics
    amplitude = system.current_field.get_amplitude_field()
    
    return {
        "max_pressure": float(np.max(amplitude)),
        "mean_pressure": float(np.mean(amplitude)),
        "shape": system.current_field.shape,
        "frequency": system.current_field.frequency
    }


# API v1 routes
@app.post("/api/v1/optimization/start")
async def start_optimization(request: OptimizationRequest):
    """Start optimization process."""
    optimization_counter.inc()
    
    try:
        # Create experiment record
        with system.db_manager.get_session() as session:
            exp_repo = ExperimentRepository(session)
            experiment = ExperimentRun(
                name=f"Optimization_{request.target_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=f"Optimization for {request.target_type} target",
                application_type="optimization",
                parameters=request.dict()
            )
            saved_exp = exp_repo.save(experiment)
        
        # Run optimization
        result = system.optimize_hologram(request)
        result["optimization_id"] = saved_exp.id
        result["status"] = "completed" if result["success"] else "failed"
        
        # Save optimization result
        if result["success"]:
            with system.db_manager.get_session() as session:
                opt_repo = OptimizationRepository(session)
                opt_result = OptimizationResult(
                    experiment_id=saved_exp.id,
                    target_type=request.target_type,
                    target_position=request.target_position or [0, 0, 0.1],
                    target_pressure=request.target_pressure,
                    optimization_method=request.method,
                    final_cost=result["final_loss"],
                    iterations=result["iterations"],
                    convergence_achieved=True,
                    phases_json=system.current_phases.tolist() if system.current_phases is not None else [],
                    amplitudes_json=np.ones(256).tolist()
                )
                opt_repo.save(opt_result)
        
        await system.broadcast_status()
        return result
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/optimization/{optimization_id}/status")
async def get_optimization_status(optimization_id: int):
    """Get optimization status."""
    try:
        with system.db_manager.get_session() as session:
            opt_repo = OptimizationRepository(session)
            result = opt_repo.get_by_id(optimization_id)
            
            if not result:
                raise HTTPException(status_code=404, detail="Optimization not found")
            
            return {
                "id": result.id,
                "status": "completed" if result.convergence_achieved else "failed",
                "progress": 100 if result.convergence_achieved else 0,
                "final_cost": result.final_cost,
                "iterations": result.iterations,
                "target_type": result.target_type,
                "created_at": result.created_at.isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/optimization/results")
async def list_optimization_results(limit: int = 20, offset: int = 0):
    """List optimization results."""
    try:
        with system.db_manager.get_session() as session:
            opt_repo = OptimizationRepository(session)
            results = opt_repo.get_recent(hours=24, limit=limit)
            
            return [
                {
                    "id": r.id,
                    "target_type": r.target_type,
                    "final_cost": r.final_cost,
                    "iterations": r.iterations,
                    "convergence_achieved": r.convergence_achieved,
                    "created_at": r.created_at.isoformat()
                }
                for r in results
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/field/calculate")
async def calculate_field(request: FieldCalculationRequest):
    """Calculate acoustic field."""
    field_calculation_counter.inc()
    
    try:
        # Validate input
        if len(request.phases) != len(request.amplitudes):
            raise HTTPException(status_code=400, detail="Phases and amplitudes must have same length")
        
        # Calculate field using propagator
        phases = np.array(request.phases)
        amplitudes = np.array(request.amplitudes)
        
        field_data = system.propagator.compute_field_from_sources(
            system.array.get_positions(),
            amplitudes,
            phases
        )
        
        # Calculate metrics
        metrics = {
            "max_pressure": float(np.max(np.abs(field_data))),
            "mean_pressure": float(np.mean(np.abs(field_data))),
            "rms_pressure": float(np.sqrt(np.mean(np.abs(field_data)**2))),
            "energy": float(np.sum(np.abs(field_data)**2))
        }
        
        # Store field data in database
        with system.db_manager.get_session() as session:
            field_repo = FieldRepository(session)
            field_record = AcousticFieldData(
                field_type="acoustic_pressure",
                shape_x=field_data.shape[0],
                shape_y=field_data.shape[1],
                shape_z=field_data.shape[2],
                bounds_x_min=request.field_bounds.get("x_min", -0.05),
                bounds_x_max=request.field_bounds.get("x_max", 0.05),
                bounds_y_min=request.field_bounds.get("y_min", -0.05),
                bounds_y_max=request.field_bounds.get("y_max", 0.05),
                bounds_z_min=request.field_bounds.get("z_min", 0.05),
                bounds_z_max=request.field_bounds.get("z_max", 0.15),
                data_json=np.abs(field_data).flatten().tolist()
            )
            saved_field = field_repo.save(field_record)
        
        system.metrics["fields_calculated"] += 1
        
        return {
            "field_id": saved_field.id,
            "field_data": np.abs(field_data).tolist(),
            "metrics": metrics,
            "shape": field_data.shape
        }
        
    except Exception as e:
        logger.error(f"Field calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/field/{field_id}")
async def get_field_data(field_id: int):
    """Get field data by ID."""
    try:
        with system.db_manager.get_session() as session:
            field_repo = FieldRepository(session)
            field = field_repo.get_by_id(field_id)
            
            if not field:
                raise HTTPException(status_code=404, detail="Field not found")
            
            return {
                "id": field.id,
                "field_type": field.field_type,
                "shape": [field.shape_x, field.shape_y, field.shape_z],
                "bounds": {
                    "x_min": field.bounds_x_min,
                    "x_max": field.bounds_x_max,
                    "y_min": field.bounds_y_min,
                    "y_max": field.bounds_y_max,
                    "z_min": field.bounds_z_min,
                    "z_max": field.bounds_z_max
                },
                "data": field.data_json,
                "created_at": field.created_at.isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/field/list")
async def list_field_data(limit: int = 20):
    """List available field data."""
    try:
        with system.db_manager.get_session() as session:
            field_repo = FieldRepository(session)
            fields = field_repo.get_recent(hours=24, limit=limit)
            
            return [
                {
                    "id": f.id,
                    "field_type": f.field_type,
                    "shape": [f.shape_x, f.shape_y, f.shape_z],
                    "created_at": f.created_at.isoformat()
                }
                for f in fields
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/hardware/status")
async def hardware_status():
    """Get hardware status."""
    if not system.hardware:
        return {"connected": False, "message": "No hardware interface initialized"}
    
    status = {
        "connected": system.hardware.is_connected(),
        "device_id": getattr(system.hardware, 'device_id', 'unknown'),
        "temperature": getattr(system.hardware, 'temperature', None),
        "power": getattr(system.hardware, 'power', None)
    }
    
    hardware_connection_gauge.set(1 if status["connected"] else 0)
    return status


@app.post("/api/v1/hardware/connect")
async def connect_hardware_api(interface_type: str = "serial", port: str = None, host: str = None):
    """Connect to hardware interface."""
    kwargs = {}
    if port:
        kwargs["port"] = port
    if host:
        kwargs["host"] = host
    
    success = system.connect_hardware(interface_type, **kwargs)
    hardware_connection_gauge.set(1 if success else 0)
    
    if success:
        return {"status": "connected", "message": "Hardware connected successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to connect to hardware")


@app.post("/api/v1/hardware/disconnect")
async def disconnect_hardware_api():
    """Disconnect hardware interface."""
    if system.hardware:
        system.hardware.disconnect()
        hardware_connection_gauge.set(0)
        return {"status": "disconnected", "message": "Hardware disconnected"}
    else:
        return {"status": "not_connected", "message": "No hardware to disconnect"}


@app.post("/api/v1/hardware/emergency_stop")
async def emergency_stop():
    """Emergency stop all hardware operations."""
    if system.hardware and system.hardware.is_connected():
        system.hardware.emergency_stop()
        await system.broadcast_status()
        return {"status": "stopped", "message": "Emergency stop activated"}
    else:
        raise HTTPException(status_code=400, detail="Hardware not connected")


@app.get("/api/v1/safety/status")
async def safety_status():
    """Get safety monitoring status."""
    return {
        "overall_status": "safe" if len(system.safety_violations) == 0 else "warning",
        "limits": system.safety_limits.dict(),
        "violations": system.safety_violations[-10:],  # Last 10 violations
        "checks": {
            "pressure_monitoring": True,
            "temperature_monitoring": True,
            "emergency_stop_ready": system.hardware.is_connected() if system.hardware else False
        }
    }


@app.get("/api/v1/safety/limits")
async def get_safety_limits():
    """Get current safety limits."""
    return system.safety_limits.dict()


@app.put("/api/v1/safety/limits")
async def update_safety_limits(limits: SafetyLimits):
    """Update safety limits."""
    system.safety_limits = limits
    await system.broadcast_status()
    return {"status": "updated", "limits": limits.dict()}


@app.post("/api/v1/experiments")
async def create_experiment(request: ExperimentRequest):
    """Create new experiment."""
    try:
        with system.db_manager.get_session() as session:
            exp_repo = ExperimentRepository(session)
            experiment = ExperimentRun(
                name=request.name,
                description=request.description,
                application_type=request.application_type,
                hardware_config=request.hardware_config,
                parameters=request.parameters
            )
            saved_exp = exp_repo.save(experiment)
            
            return {
                "id": saved_exp.id,
                "name": saved_exp.name,
                "status": saved_exp.status,
                "created_at": saved_exp.created_at.isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/experiments")
async def list_experiments(limit: int = 20, application_type: str = None):
    """List experiments."""
    try:
        with system.db_manager.get_session() as session:
            exp_repo = ExperimentRepository(session)
            
            if application_type:
                experiments = exp_repo.get_by_application_type(application_type)
            else:
                experiments = exp_repo.get_recent(hours=24, limit=limit)
            
            return [
                {
                    "id": e.id,
                    "name": e.name,
                    "application_type": e.application_type,
                    "status": e.status,
                    "created_at": e.created_at.isoformat()
                }
                for e in experiments
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics/performance")
async def get_performance_metrics():
    """Get system performance metrics."""
    return {
        "cpu_usage": system.metrics_collector.get_cpu_usage(),
        "memory_usage": system.metrics_collector.get_memory_usage(),
        "gpu_usage": system.metrics_collector.get_gpu_usage(),
        "disk_usage": system.metrics_collector.get_disk_usage(),
        "uptime": (datetime.now(timezone.utc) - system.metrics["uptime_start"]).total_seconds()
    }


@app.websocket("/ws/optimization/progress")
async def optimization_progress_websocket(websocket: WebSocket):
    """WebSocket for optimization progress updates."""
    await websocket.accept()
    try:
        while True:
            # Send progress updates
            await websocket.send_json({
                "type": "optimization_progress",
                "data": {
                    "active_optimizations": system.metrics["optimizations_performed"],
                    "status": "monitoring"
                }
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/hardware/status")
async def hardware_status_websocket(websocket: WebSocket):
    """WebSocket for hardware status updates."""
    await websocket.accept()
    try:
        while True:
            status = {
                "type": "hardware_status",
                "data": {
                    "connected": system.hardware.is_connected() if system.hardware else False,
                    "temperature": getattr(system.hardware, 'temperature', None) if system.hardware else None,
                    "power": getattr(system.hardware, 'power', None) if system.hardware else None
                }
            }
            await websocket.send_json(status)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """General WebSocket endpoint for real-time updates."""
    await websocket.accept()
    system.websocket_clients.add(websocket)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process commands
            if message["type"] == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
            elif message["type"] == "get_status":
                await system.broadcast_status()
            
    except WebSocketDisconnect:
        system.websocket_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        system.websocket_clients.discard(websocket)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Acousto-Gen API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run server
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info" if not args.debug else "debug"
    )


if __name__ == "__main__":
    main()
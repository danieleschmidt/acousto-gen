"""
Acousto-Gen main entry point and API server.
Provides REST API and WebSocket interfaces for acoustic holography control.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from physics.propagation.wave_propagator import WavePropagator, MediumProperties
from physics.transducers.transducer_array import UltraLeap256, CircularArray, CustomArray
from optimization.hologram_optimizer import GradientOptimizer, GeneticOptimizer
from models.acoustic_field import AcousticField, TargetPattern
from applications.levitation.acoustic_levitator import AcousticLevitator
from hardware.drivers.hardware_interface import SerialHardware, NetworkHardware


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
    focal_points: list = Field(default=[], description="List of focal points")
    method: str = Field(default="gradient", description="Optimization method")
    iterations: int = Field(default=1000, description="Number of iterations")


class ParticleRequest(BaseModel):
    """Particle manipulation request."""
    position: list = Field(description="3D position [x, y, z]")
    radius: float = Field(default=1e-3, description="Particle radius")
    density: float = Field(default=25, description="Particle density")


class HardwareCommand(BaseModel):
    """Hardware control command."""
    command: str = Field(description="Command type")
    parameters: Optional[Dict[str, Any]] = Field(default=None)


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
        
        # WebSocket connections
        self.websocket_clients = set()
        
        # System metrics
        self.metrics = {
            "optimizations_performed": 0,
            "particles_levitated": 0,
            "total_runtime": 0,
            "last_error": None
        }
    
    def initialize(self, config: SystemConfig):
        """Initialize system with configuration."""
        self.config = config
        
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


# Create FastAPI app
app = FastAPI(
    title="Acousto-Gen API",
    description="Generative acoustic holography control system",
    version="1.0.0"
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
    return {
        "status": "healthy",
        "hardware_connected": system.hardware.is_connected() if system.hardware else False,
        "metrics": system.metrics
    }


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


@app.post("/optimize")
async def optimize_hologram(request: OptimizationRequest):
    """Optimize hologram for target pattern."""
    result = system.optimize_hologram(request)
    await system.broadcast_status()
    return result


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
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
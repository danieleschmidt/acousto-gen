"""Unit tests for FastAPI server functionality."""

import pytest
import json
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Import the FastAPI app
from src.main import app
from src.database.connection import DatabaseManager
from src.database.models import Base


class TestAPIBasics:
    """Test basic API functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_api_info(self, client):
        """Test API info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
    
    def test_openapi_docs(self, client):
        """Test OpenAPI documentation endpoints."""
        # Test docs endpoint
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test OpenAPI spec
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert "openapi" in spec
        assert "info" in spec


class TestOptimizationAPI:
    """Test optimization-related API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_optimization_request(self):
        """Sample optimization request data."""
        return {
            "target_type": "single_focus",
            "target_position": [0.0, 0.0, 0.1],
            "target_pressure": 3000.0,
            "optimization_method": "adam",
            "max_iterations": 100,
            "convergence_threshold": 1e-6,
            "learning_rate": 0.01,
            "array_config": {
                "type": "ultraleap_stratos",
                "frequency": 40000,
                "elements": 256
            }
        }
    
    @patch('src.optimization.hologram_optimizer.HologramOptimizer')
    def test_start_optimization(self, mock_optimizer, client, sample_optimization_request):
        """Test starting an optimization."""
        # Mock optimizer
        mock_instance = Mock()
        mock_instance.optimize.return_value = {
            "phases": np.random.rand(256).tolist(),
            "amplitudes": np.ones(256).tolist(),
            "final_cost": 0.05,
            "iterations": 50,
            "converged": True
        }
        mock_optimizer.return_value = mock_instance
        
        response = client.post("/api/v1/optimization/start", 
                              json=sample_optimization_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "optimization_id" in data
        assert "status" in data
        assert data["status"] == "running"
    
    @patch('src.optimization.hologram_optimizer.HologramOptimizer')
    def test_get_optimization_status(self, mock_optimizer, client):
        """Test getting optimization status."""
        # First start an optimization
        optimization_request = {
            "target_type": "single_focus",
            "target_position": [0.0, 0.0, 0.1],
            "optimization_method": "adam"
        }
        
        mock_instance = Mock()
        mock_instance.optimize.return_value = {
            "phases": np.random.rand(256).tolist(),
            "final_cost": 0.05,
            "converged": True
        }
        mock_optimizer.return_value = mock_instance
        
        start_response = client.post("/api/v1/optimization/start", 
                                   json=optimization_request)
        optimization_id = start_response.json()["optimization_id"]
        
        # Get status
        response = client.get(f"/api/v1/optimization/{optimization_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "progress" in data
    
    def test_list_optimization_results(self, client):
        """Test listing optimization results."""
        response = client.get("/api/v1/optimization/results")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_optimization_result(self, client):
        """Test getting specific optimization result."""
        # This would normally require a real optimization result ID
        # For now, test the endpoint structure
        response = client.get("/api/v1/optimization/results/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_optimization_request(self, client):
        """Test handling of invalid optimization requests."""
        invalid_request = {
            "target_type": "invalid_type",
            "target_position": "invalid_position"
        }
        
        response = client.post("/api/v1/optimization/start", 
                              json=invalid_request)
        assert response.status_code == 422  # Validation error


class TestFieldAPI:
    """Test acoustic field-related API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_field_request(self):
        """Sample field calculation request."""
        return {
            "phases": (np.random.rand(256) * 2 * np.pi).tolist(),
            "amplitudes": np.ones(256).tolist(),
            "field_bounds": {
                "x_min": -0.05, "x_max": 0.05,
                "y_min": -0.05, "y_max": 0.05,
                "z_min": 0.05, "z_max": 0.15
            },
            "resolution": {"x": 64, "y": 64, "z": 32},
            "frequency": 40000,
            "medium": "air"
        }
    
    @patch('src.physics.propagation.wave_propagator.WavePropagator')
    def test_calculate_field(self, mock_propagator, client, sample_field_request):
        """Test acoustic field calculation."""
        # Mock propagator
        mock_instance = Mock()
        field_shape = (64, 64, 32)
        mock_field = np.random.rand(*field_shape) * 1000
        mock_instance.propagate_field.return_value = mock_field
        mock_propagator.return_value = mock_instance
        
        response = client.post("/api/v1/field/calculate", 
                              json=sample_field_request)
        
        assert response.status_code == 200
        data = response.json()
        assert "field_id" in data
        assert "field_data" in data
        assert "metrics" in data
    
    def test_get_field_data(self, client):
        """Test retrieving field data."""
        response = client.get("/api/v1/field/nonexistent")
        assert response.status_code == 404
    
    def test_list_field_data(self, client):
        """Test listing available field data."""
        response = client.get("/api/v1/field/list")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_field_metrics(self, client):
        """Test field metrics calculation."""
        field_data = {
            "field_data": np.random.rand(32, 32, 16).flatten().tolist(),
            "shape": [32, 32, 16],
            "bounds": {
                "x_min": -0.02, "x_max": 0.02,
                "y_min": -0.02, "y_max": 0.02,
                "z_min": 0.08, "z_max": 0.12
            }
        }
        
        response = client.post("/api/v1/field/metrics", json=field_data)
        assert response.status_code == 200
        data = response.json()
        assert "max_pressure" in data
        assert "focus_metrics" in data


class TestHardwareAPI:
    """Test hardware interface API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @patch('src.hardware.drivers.hardware_interface.HardwareInterface')
    def test_hardware_status(self, mock_hardware, client):
        """Test hardware status endpoint."""
        mock_instance = Mock()
        mock_instance.get_status.return_value = {
            "connected": True,
            "device_id": "TEST-001",
            "temperature": 25.0,
            "power": 10.0
        }
        mock_hardware.return_value = mock_instance
        
        response = client.get("/api/v1/hardware/status")
        assert response.status_code == 200
        data = response.json()
        assert "connected" in data
        assert "device_id" in data
    
    @patch('src.hardware.drivers.hardware_interface.HardwareInterface')
    def test_hardware_connect(self, mock_hardware, client):
        """Test hardware connection."""
        mock_instance = Mock()
        mock_instance.connect.return_value = True
        mock_hardware.return_value = mock_instance
        
        response = client.post("/api/v1/hardware/connect")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    @patch('src.hardware.drivers.hardware_interface.HardwareInterface')
    def test_hardware_disconnect(self, mock_hardware, client):
        """Test hardware disconnection."""
        mock_instance = Mock()
        mock_instance.disconnect.return_value = True
        mock_hardware.return_value = mock_instance
        
        response = client.post("/api/v1/hardware/disconnect")
        assert response.status_code == 200
    
    @patch('src.hardware.drivers.hardware_interface.HardwareInterface')
    def test_send_phases(self, mock_hardware, client):
        """Test sending phases to hardware."""
        mock_instance = Mock()
        mock_instance.send_phases.return_value = True
        mock_hardware.return_value = mock_instance
        
        phase_data = {
            "phases": (np.random.rand(256) * 2 * np.pi).tolist(),
            "amplitudes": np.ones(256).tolist()
        }
        
        response = client.post("/api/v1/hardware/send_phases", json=phase_data)
        assert response.status_code == 200
    
    @patch('src.hardware.drivers.hardware_interface.HardwareInterface')
    def test_emergency_stop(self, mock_hardware, client):
        """Test emergency stop functionality."""
        mock_instance = Mock()
        mock_instance.emergency_stop.return_value = True
        mock_hardware.return_value = mock_instance
        
        response = client.post("/api/v1/hardware/emergency_stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"


class TestSafetyAPI:
    """Test safety monitoring API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_safety_status(self, client):
        """Test safety status endpoint."""
        response = client.get("/api/v1/safety/status")
        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        assert "checks" in data
    
    def test_safety_limits(self, client):
        """Test safety limits endpoint."""
        response = client.get("/api/v1/safety/limits")
        assert response.status_code == 200
        data = response.json()
        assert "max_pressure" in data
        assert "max_intensity" in data
        assert "max_temperature" in data
    
    def test_update_safety_limits(self, client):
        """Test updating safety limits."""
        new_limits = {
            "max_pressure": 4500,
            "max_intensity": 12,
            "max_temperature": 42
        }
        
        response = client.put("/api/v1/safety/limits", json=new_limits)
        assert response.status_code == 200
    
    def test_safety_violation_reporting(self, client):
        """Test safety violation reporting."""
        violation_data = {
            "violation_type": "pressure_exceeded",
            "measured_value": 5500,
            "limit_value": 5000,
            "timestamp": "2025-01-01T12:00:00Z",
            "severity": "high"
        }
        
        response = client.post("/api/v1/safety/violations", json=violation_data)
        assert response.status_code == 200


class TestApplicationAPI:
    """Test application-specific API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_levitation_presets(self, client):
        """Test levitation presets endpoint."""
        response = client.get("/api/v1/applications/levitation/presets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_haptic_patterns(self, client):
        """Test haptic patterns endpoint."""
        response = client.get("/api/v1/applications/haptic/patterns")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('src.applications.levitation.acoustic_levitator.AcousticLevitator')
    def test_start_levitation(self, mock_levitator, client):
        """Test starting levitation."""
        mock_instance = Mock()
        mock_instance.start_levitation.return_value = {
            "status": "active",
            "trap_positions": [[0.0, 0.0, 0.1]],
            "trap_strength": 2500
        }
        mock_levitator.return_value = mock_instance
        
        levitation_request = {
            "particle_properties": {
                "radius": 0.001,
                "density": 25.0
            },
            "trap_position": [0.0, 0.0, 0.1],
            "trap_strength": 2500
        }
        
        response = client.post("/api/v1/applications/levitation/start", 
                              json=levitation_request)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestMetricsAPI:
    """Test metrics and monitoring API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_prometheus_metrics(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Check for basic Prometheus format
        text = response.text
        assert "# HELP" in text or "# TYPE" in text
    
    def test_performance_metrics(self, client):
        """Test performance metrics endpoint."""
        response = client.get("/api/v1/metrics/performance")
        assert response.status_code == 200
        data = response.json()
        assert "cpu_usage" in data or "memory_usage" in data
    
    def test_optimization_metrics(self, client):
        """Test optimization metrics endpoint."""
        response = client.get("/api/v1/metrics/optimization")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client with temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Override database URL for testing
            app.dependency_overrides[DatabaseManager] = lambda: DatabaseManager(
                f"sqlite:///{tmp.name}"
            )
            
            client = TestClient(app)
            yield client
            
            # Cleanup
            Path(tmp.name).unlink()
            app.dependency_overrides.clear()
    
    def test_full_optimization_workflow(self, client):
        """Test complete optimization workflow through API."""
        # 1. Start optimization
        optimization_request = {
            "target_type": "single_focus",
            "target_position": [0.0, 0.0, 0.1],
            "optimization_method": "adam",
            "max_iterations": 50
        }
        
        with patch('src.optimization.hologram_optimizer.HologramOptimizer') as mock_opt:
            mock_instance = Mock()
            mock_instance.optimize.return_value = {
                "phases": np.random.rand(256).tolist(),
                "amplitudes": np.ones(256).tolist(),
                "final_cost": 0.05,
                "iterations": 30,
                "converged": True
            }
            mock_opt.return_value = mock_instance
            
            start_response = client.post("/api/v1/optimization/start", 
                                       json=optimization_request)
            assert start_response.status_code == 200
            optimization_id = start_response.json()["optimization_id"]
            
            # 2. Check status
            status_response = client.get(f"/api/v1/optimization/{optimization_id}/status")
            assert status_response.status_code == 200
            
            # 3. Get results
            results_response = client.get("/api/v1/optimization/results")
            assert results_response.status_code == 200
    
    def test_field_calculation_workflow(self, client):
        """Test field calculation workflow."""
        field_request = {
            "phases": (np.random.rand(256) * 2 * np.pi).tolist(),
            "amplitudes": np.ones(256).tolist(),
            "resolution": {"x": 32, "y": 32, "z": 16}
        }
        
        with patch('src.physics.propagation.wave_propagator.WavePropagator') as mock_prop:
            mock_instance = Mock()
            mock_instance.propagate_field.return_value = np.random.rand(32, 32, 16) * 1000
            mock_prop.return_value = mock_instance
            
            # Calculate field
            calc_response = client.post("/api/v1/field/calculate", json=field_request)
            assert calc_response.status_code == 200
            field_id = calc_response.json()["field_id"]
            
            # Retrieve field data
            get_response = client.get(f"/api/v1/field/{field_id}")
            # Note: This might return 404 if not properly stored
    
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test 404 errors
        response = client.get("/api/v1/nonexistent/endpoint")
        assert response.status_code == 404
        
        # Test validation errors
        response = client.post("/api/v1/optimization/start", json={})
        assert response.status_code == 422
        
        # Test malformed JSON
        response = client.post("/api/v1/optimization/start", 
                              data="invalid json", 
                              headers={"Content-Type": "application/json"})
        assert response.status_code == 422


class TestWebSocketAPI:
    """Test WebSocket API for real-time updates."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_optimization_progress_websocket(self, client):
        """Test optimization progress WebSocket."""
        with client.websocket_connect("/ws/optimization/progress") as websocket:
            # This would test real-time optimization progress updates
            # For now, just test connection
            data = websocket.receive_json()
            assert "type" in data or "status" in data
    
    def test_hardware_status_websocket(self, client):
        """Test hardware status WebSocket."""
        with client.websocket_connect("/ws/hardware/status") as websocket:
            # Test real-time hardware status updates
            data = websocket.receive_json()
            assert "type" in data or "status" in data


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_api_response_times(self, client):
        """Test API response times."""
        import time
        
        endpoints = [
            "/health",
            "/info",
            "/api/v1/optimization/results",
            "/api/v1/field/list"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            end_time = time.time()
            
            assert response.status_code in [200, 404]  # 404 is ok for empty lists
            assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import time
        
        def make_request():
            return client.get("/health")
        
        # Test 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
            end_time = time.time()
            
            # All requests should succeed
            assert all(r.status_code == 200 for r in responses)
            # Should complete within reasonable time
            assert (end_time - start_time) < 5.0
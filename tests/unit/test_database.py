"""Unit tests for database layer functionality."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import DatabaseManager
from src.database.models import (
    Base, OptimizationResult, FieldData, Experiment, 
    PerformanceMetrics, CalibrationData
)
from src.database.repositories import (
    OptimizationRepository, FieldDataRepository, 
    ExperimentRepository
)


class TestDatabaseConnection:
    """Test database connection management."""
    
    def test_sqlite_connection(self):
        """Test SQLite database connection."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            
            assert db_manager.database_url == db_url
            assert db_manager.engine is not None
            
            # Test connection
            with db_manager.get_session() as session:
                assert session is not None
                
            # Cleanup
            Path(tmp.name).unlink()
    
    def test_schema_creation(self):
        """Test database schema creation."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            db_manager.create_tables()
            
            # Verify tables exist
            inspector = db_manager.engine.dialect.get_schema_names(
                db_manager.engine.connect()
            )
            
            # Cleanup
            Path(tmp.name).unlink()
    
    def test_connection_pool(self):
        """Test connection pooling."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            
            # Test multiple concurrent sessions
            sessions = []
            for _ in range(3):
                session = db_manager.get_session()
                sessions.append(session)
            
            for session in sessions:
                session.close()
                
            # Cleanup
            Path(tmp.name).unlink()


class TestDatabaseModels:
    """Test database model definitions."""
    
    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
    
    def test_optimization_result_model(self, db_session):
        """Test OptimizationResult model."""
        # Create test optimization result
        phases = np.random.rand(256) * 2 * np.pi
        amplitudes = np.ones(256)
        
        result = OptimizationResult(
            target_type="single_focus",
            target_position=[0.0, 0.0, 0.1],
            target_pressure=3000.0,
            optimization_method="adam",
            final_cost=0.05,
            iterations=150,
            convergence_achieved=True,
            phases_json=phases.tolist(),
            amplitudes_json=amplitudes.tolist()
        )
        
        db_session.add(result)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(OptimizationResult).first()
        assert retrieved.target_type == "single_focus"
        assert len(retrieved.phases_json) == 256
        assert retrieved.convergence_achieved is True
        
        # Test phase/amplitude property access
        retrieved_phases = np.array(retrieved.phases_json)
        assert np.allclose(retrieved_phases, phases)
    
    def test_field_data_model(self, db_session):
        """Test FieldData model."""
        # Create test field data
        field_shape = (50, 50, 25)
        pressure_data = np.random.rand(*field_shape) * 1000
        
        field = FieldData(
            field_type="acoustic_pressure",
            shape_x=field_shape[0],
            shape_y=field_shape[1], 
            shape_z=field_shape[2],
            bounds_x_min=-0.05,
            bounds_x_max=0.05,
            bounds_y_min=-0.05,
            bounds_y_max=0.05,
            bounds_z_min=0.05,
            bounds_z_max=0.15,
            data_json=pressure_data.flatten().tolist()
        )
        
        db_session.add(field)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(FieldData).first()
        assert retrieved.field_type == "acoustic_pressure"
        assert (retrieved.shape_x, retrieved.shape_y, retrieved.shape_z) == field_shape
        
        # Test data reconstruction
        retrieved_data = np.array(retrieved.data_json).reshape(field_shape)
        assert np.allclose(retrieved_data, pressure_data)
    
    def test_experiment_model(self, db_session):
        """Test Experiment model."""
        experiment = Experiment(
            name="Test Levitation",
            description="Testing particle levitation stability",
            application_type="levitation",
            hardware_config={
                "array_type": "ultraleap_stratos",
                "frequency": 40000,
                "elements": 256
            },
            parameters={
                "particle_radius": 0.001,
                "target_height": 0.08
            },
            status="completed"
        )
        
        db_session.add(experiment)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(Experiment).first()
        assert retrieved.name == "Test Levitation"
        assert retrieved.application_type == "levitation"
        assert retrieved.hardware_config["frequency"] == 40000
    
    def test_performance_metrics_model(self, db_session):
        """Test PerformanceMetrics model."""
        metrics = PerformanceMetrics(
            operation_type="optimization",
            duration_seconds=45.2,
            memory_usage_mb=150.5,
            gpu_utilization=0.85,
            field_resolution=(64, 64, 32),
            optimization_iterations=200,
            convergence_achieved=True,
            final_cost=0.03
        )
        
        db_session.add(metrics)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(PerformanceMetrics).first()
        assert retrieved.operation_type == "optimization"
        assert retrieved.duration_seconds == 45.2
        assert retrieved.gpu_utilization == 0.85
    
    def test_calibration_data_model(self, db_session):
        """Test CalibrationData model."""
        calibration = CalibrationData(
            array_serial="TEST-001",
            calibration_version="1.0",
            phase_corrections=np.random.rand(256).tolist(),
            amplitude_corrections=(np.ones(256) * 0.95).tolist(),
            element_status=[True] * 256,
            quality_score=0.92,
            environmental_temperature=22.5,
            environmental_humidity=45.0
        )
        
        db_session.add(calibration)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(CalibrationData).first()
        assert retrieved.array_serial == "TEST-001"
        assert len(retrieved.phase_corrections) == 256
        assert retrieved.quality_score == 0.92


class TestRepositories:
    """Test repository pattern implementations."""
    
    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
    
    def test_optimization_repository(self, db_session):
        """Test OptimizationRepository functionality."""
        repo = OptimizationRepository(db_session)
        
        # Create test data
        phases = np.random.rand(256) * 2 * np.pi
        result1 = OptimizationResult(
            target_type="single_focus",
            target_position=[0.0, 0.0, 0.1],
            optimization_method="adam",
            final_cost=0.05,
            iterations=100,
            convergence_achieved=True,
            phases_json=phases.tolist(),
            amplitudes_json=np.ones(256).tolist()
        )
        
        result2 = OptimizationResult(
            target_type="twin_trap", 
            target_position=[0.02, 0.0, 0.1],
            optimization_method="genetic",
            final_cost=0.08,
            iterations=150,
            convergence_achieved=True,
            phases_json=phases.tolist(),
            amplitudes_json=np.ones(256).tolist()
        )
        
        # Test save functionality
        saved1 = repo.save(result1)
        saved2 = repo.save(result2)
        
        assert saved1.id is not None
        assert saved2.id is not None
        
        # Test get_best_results
        best_results = repo.get_best_results(limit=1)
        assert len(best_results) == 1
        assert best_results[0].final_cost == 0.05  # Lower cost = better
        
        # Test get_by_target_type
        focus_results = repo.get_by_target_type("single_focus")
        assert len(focus_results) == 1
        assert focus_results[0].target_type == "single_focus"
        
        # Test get_recent
        recent_results = repo.get_recent(hours=24)
        assert len(recent_results) == 2
    
    def test_field_data_repository(self, db_session):
        """Test FieldDataRepository functionality."""
        repo = FieldDataRepository(db_session)
        
        # Create test field data
        field_shape = (32, 32, 16)
        pressure_data = np.random.rand(*field_shape) * 1000
        
        field = FieldData(
            field_type="acoustic_pressure",
            shape_x=field_shape[0],
            shape_y=field_shape[1],
            shape_z=field_shape[2],
            bounds_x_min=-0.02,
            bounds_x_max=0.02,
            bounds_y_min=-0.02,
            bounds_y_max=0.02,
            bounds_z_min=0.08,
            bounds_z_max=0.12,
            data_json=pressure_data.flatten().tolist()
        )
        
        # Test save and retrieve
        saved_field = repo.save(field)
        assert saved_field.id is not None
        
        retrieved_field = repo.get_by_id(saved_field.id)
        assert retrieved_field is not None
        assert retrieved_field.field_type == "acoustic_pressure"
        
        # Test get_by_bounds
        fields_in_bounds = repo.get_by_bounds(
            x_range=(-0.03, 0.03),
            y_range=(-0.03, 0.03),
            z_range=(0.07, 0.13)
        )
        assert len(fields_in_bounds) == 1
    
    def test_experiment_repository(self, db_session):
        """Test ExperimentRepository functionality."""
        repo = ExperimentRepository(db_session)
        
        # Create test experiment
        experiment = Experiment(
            name="Test Experiment",
            description="Testing repository functionality",
            application_type="levitation",
            status="running"
        )
        
        # Test save and status update
        saved_exp = repo.save(experiment)
        assert saved_exp.id is not None
        
        # Test status update
        updated_exp = repo.update_status(saved_exp.id, "completed")
        assert updated_exp.status == "completed"
        
        # Test get_by_application_type
        levitation_experiments = repo.get_by_application_type("levitation")
        assert len(levitation_experiments) == 1
        
        # Test get_active
        active_experiments = repo.get_active()
        assert len(active_experiments) == 0  # Changed to completed
        
        # Test add_performance_metrics
        metrics = PerformanceMetrics(
            operation_type="test",
            duration_seconds=10.0,
            memory_usage_mb=50.0
        )
        repo.add_performance_metrics(saved_exp.id, metrics)
        
        # Verify metrics were added
        retrieved_exp = repo.get_by_id(saved_exp.id)
        assert len(retrieved_exp.performance_metrics) == 1


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database layer."""
    
    def test_full_workflow_persistence(self):
        """Test complete workflow data persistence."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            db_manager.create_tables()
            
            # Create repositories
            with db_manager.get_session() as session:
                opt_repo = OptimizationRepository(session)
                field_repo = FieldDataRepository(session)
                exp_repo = ExperimentRepository(session)
                
                # Create experiment
                experiment = Experiment(
                    name="Integration Test Workflow",
                    description="Full workflow test",
                    application_type="levitation"
                )
                saved_exp = exp_repo.save(experiment)
                
                # Create optimization result
                phases = np.random.rand(256) * 2 * np.pi
                optimization = OptimizationResult(
                    experiment_id=saved_exp.id,
                    target_type="single_focus",
                    optimization_method="adam",
                    final_cost=0.03,
                    phases_json=phases.tolist(),
                    amplitudes_json=np.ones(256).tolist()
                )
                saved_opt = opt_repo.save(optimization)
                
                # Create field data
                field_data = np.random.rand(32, 32, 16) * 1000
                field = FieldData(
                    experiment_id=saved_exp.id,
                    optimization_result_id=saved_opt.id,
                    field_type="acoustic_pressure",
                    shape_x=32, shape_y=32, shape_z=16,
                    data_json=field_data.flatten().tolist()
                )
                saved_field = field_repo.save(field)
                
                # Verify relationships
                retrieved_exp = exp_repo.get_by_id(saved_exp.id)
                assert len(retrieved_exp.optimization_results) == 1
                assert len(retrieved_exp.field_data) == 1
                
                retrieved_opt = opt_repo.get_by_id(saved_opt.id)
                assert retrieved_opt.experiment_id == saved_exp.id
                
            # Cleanup
            Path(tmp.name).unlink()
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            db_manager.create_tables()
            
            with db_manager.get_session() as session:
                exp_repo = ExperimentRepository(session)
                
                # Create experiment
                experiment = Experiment(
                    name="Performance Test",
                    application_type="performance_testing"
                )
                saved_exp = exp_repo.save(experiment)
                
                # Add multiple performance metrics
                for i in range(3):
                    metrics = PerformanceMetrics(
                        operation_type="optimization",
                        duration_seconds=10.0 + i,
                        memory_usage_mb=100.0 + i * 10,
                        gpu_utilization=0.8 + i * 0.05
                    )
                    exp_repo.add_performance_metrics(saved_exp.id, metrics)
                
                # Verify metrics
                retrieved_exp = exp_repo.get_by_id(saved_exp.id)
                assert len(retrieved_exp.performance_metrics) == 3
                
                # Test performance analysis
                avg_duration = sum(
                    m.duration_seconds for m in retrieved_exp.performance_metrics
                ) / len(retrieved_exp.performance_metrics)
                assert avg_duration == 11.0  # (10 + 11 + 12) / 3
                
            # Cleanup
            Path(tmp.name).unlink()
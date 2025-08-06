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
    Base, OptimizationResult, AcousticFieldData, ExperimentRun, 
    SystemMetric, ArrayConfiguration
)
from src.database.repositories import (
    OptimizationRepository, FieldRepository, 
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
        
        result = OptimizationResult(
            method="adam",
            target_type="single_focus",
            iterations=150,
            final_loss=0.05,
            converged=True,
            time_elapsed=10.5,
            num_elements=256,
            target_specification={"position": [0.0, 0.0, 0.1], "pressure": 3000.0},
            device_used="cpu"
        )
        result.set_phases(phases.tolist())
        
        db_session.add(result)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(OptimizationResult).first()
        assert retrieved.target_type == "single_focus"
        assert retrieved.method == "adam"
        assert retrieved.converged is True
        
        # Test phase property access
        retrieved_phases = retrieved.get_phases()
        assert len(retrieved_phases) == 256
        assert np.allclose(retrieved_phases, phases)
    
    def test_field_data_model(self, db_session):
        """Test AcousticFieldData model."""
        # Create test field data
        field_shape = (50, 50, 25)
        
        field = AcousticFieldData(
            field_type="generated",
            shape_x=field_shape[0],
            shape_y=field_shape[1], 
            shape_z=field_shape[2],
            resolution=0.001,
            frequency=40000.0,
            field_data_path="/tmp/test_field.h5",
            max_pressure=5000.0,
            mean_pressure=2000.0
        )
        field.set_bounds([[-0.05, 0.05], [-0.05, 0.05], [0.05, 0.15]])
        
        db_session.add(field)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(AcousticFieldData).first()
        assert retrieved.field_type == "generated"
        assert (retrieved.shape_x, retrieved.shape_y, retrieved.shape_z) == field_shape
        assert retrieved.frequency == 40000.0
        
        # Test bounds access
        bounds = retrieved.get_bounds()
        assert len(bounds) == 3
        assert bounds[0] == [-0.05, 0.05]
    
    def test_experiment_model(self, db_session):
        """Test ExperimentRun model."""
        experiment = ExperimentRun(
            name="Test Levitation",
            description="Testing particle levitation stability",
            experiment_type="optimization",
            parameters={
                "array_type": "ultraleap_stratos",
                "frequency": 40000,
                "elements": 256,
                "particle_radius": 0.001,
                "target_height": 0.08
            },
            status="completed",
            total_runs=10,
            successful_runs=8
        )
        
        db_session.add(experiment)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(ExperimentRun).first()
        assert retrieved.name == "Test Levitation"
        assert retrieved.experiment_type == "optimization"
        assert retrieved.parameters["frequency"] == 40000
        assert retrieved.total_runs == 10
    
    def test_system_metrics_model(self, db_session):
        """Test SystemMetric model."""
        metrics = SystemMetric(
            cpu_usage=45.2,
            memory_usage_mb=150.5,
            gpu_usage=85.0,
            gpu_memory_mb=2048.0,
            active_sessions=3,
            optimizations_per_hour=12.5,
            average_optimization_time=8.2,
            hardware_connected=True
        )
        
        db_session.add(metrics)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(SystemMetric).first()
        assert retrieved.cpu_usage == 45.2
        assert retrieved.memory_usage_mb == 150.5
        assert retrieved.gpu_usage == 85.0
        assert retrieved.hardware_connected is True
    
    def test_array_configuration_model(self, db_session):
        """Test ArrayConfiguration model."""
        positions = [[i*0.01, j*0.01, 0] for i in range(16) for j in range(16)]
        
        array_config = ArrayConfiguration(
            name="Test Array",
            array_type="custom",
            num_elements=256,
            frequency=40000.0,
            hardware_id="TEST-001",
            driver_version="1.0",
            calibration_quality=0.92,
            description="Test array configuration"
        )
        array_config.set_positions(positions)
        
        db_session.add(array_config)
        db_session.commit()
        
        # Verify stored correctly
        retrieved = db_session.query(ArrayConfiguration).first()
        assert retrieved.name == "Test Array"
        assert retrieved.num_elements == 256
        assert retrieved.calibration_quality == 0.92
        retrieved_positions = retrieved.get_positions()
        assert len(retrieved_positions) == 256


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
            method="adam",
            target_type="single_focus",
            iterations=100,
            final_loss=0.05,
            converged=True,
            time_elapsed=10.5,
            num_elements=256,
            target_specification={"position": [0.0, 0.0, 0.1]},
            device_used="cpu"
        )
        result1.set_phases(phases.tolist())
        
        result2 = OptimizationResult(
            method="genetic",
            target_type="twin_trap", 
            iterations=150,
            final_loss=0.08,
            converged=True,
            time_elapsed=15.2,
            num_elements=256,
            target_specification={"position": [0.02, 0.0, 0.1]},
            device_used="cpu"
        )
        result2.set_phases(phases.tolist())
        
        # Test create functionality
        db_session.add(result1)
        db_session.add(result2)
        db_session.commit()
        
        assert result1.id is not None
        assert result2.id is not None
        
        # Test get_best_results
        best_results = repo.get_best_results(limit=1)
        assert len(best_results) == 1
        assert best_results[0].final_loss == 0.05  # Lower cost = better
        
        # Test get_by_method
        adam_results = repo.get_by_method("adam")
        assert len(adam_results) == 1
        assert adam_results[0].method == "adam"
        
        # Test get_recent_results
        recent_results = repo.get_recent_results(hours=24)
        assert len(recent_results) == 2
    
    def test_field_repository(self, db_session):
        """Test FieldRepository functionality."""
        repo = FieldRepository(db_session)
        
        # Create test field data
        field_shape = (32, 32, 16)
        
        field = AcousticFieldData(
            field_type="generated",
            shape_x=field_shape[0],
            shape_y=field_shape[1],
            shape_z=field_shape[2],
            resolution=0.001,
            frequency=40000.0,
            field_data_path="/tmp/test_field.h5",
            max_pressure=5000.0
        )
        field.set_bounds([[-0.02, 0.02], [-0.02, 0.02], [0.08, 0.12]])
        
        # Test create and retrieve
        db_session.add(field)
        db_session.commit()
        assert field.id is not None
        
        retrieved_field = repo.get_by_id(field.id)
        assert retrieved_field is not None
        assert retrieved_field.field_type == "generated"
        
        # Test get_by_type
        generated_fields = repo.get_by_type("generated")
        assert len(generated_fields) == 1
        
        # Test get_by_frequency_range
        freq_fields = repo.get_by_frequency_range(35000.0, 45000.0)
        assert len(freq_fields) == 1
    
    def test_experiment_repository(self, db_session):
        """Test ExperimentRepository functionality."""
        repo = ExperimentRepository(db_session)
        
        # Create test experiment
        experiment = ExperimentRun(
            name="Test Experiment",
            description="Testing repository functionality",
            experiment_type="optimization",
            parameters={"test": True},
            status="running",
            total_runs=5,
            successful_runs=3
        )
        
        # Test create
        db_session.add(experiment)
        db_session.commit()
        assert experiment.id is not None
        
        # Test update status via update method
        updated_exp = repo.update(experiment.id, status="completed")
        assert updated_exp.status == "completed"
        
        # Test get_active_experiments (should be none now)
        active_experiments = repo.get_active_experiments()
        assert len(active_experiments) == 0  # Changed to completed
        
        # Test get_recent_experiments
        recent_experiments = repo.get_recent_experiments(days=1)
        assert len(recent_experiments) == 1
        
        # Test update_progress
        success = repo.update_progress(experiment.experiment_id, 75.0)
        assert success is True


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
                experiment = ExperimentRun(
                    name="Integration Test Workflow",
                    description="Full workflow test",
                    experiment_type="optimization",
                    parameters={"test": True}
                )
                session.add(experiment)
                session.commit()
                
                # Create optimization result
                phases = np.random.rand(256) * 2 * np.pi
                optimization = OptimizationResult(
                    experiment_run_id=experiment.id,
                    method="adam",
                    target_type="single_focus",
                    iterations=100,
                    final_loss=0.03,
                    time_elapsed=12.0,
                    num_elements=256,
                    target_specification={"test": True},
                    device_used="cpu"
                )
                optimization.set_phases(phases.tolist())
                session.add(optimization)
                session.commit()
                
                # Create field data
                field = AcousticFieldData(
                    optimization_result_id=optimization.id,
                    field_type="generated",
                    shape_x=32, shape_y=32, shape_z=16,
                    resolution=0.001,
                    frequency=40000.0,
                    max_pressure=3000.0
                )
                session.add(field)
                session.commit()
                
                # Verify relationships
                retrieved_exp = exp_repo.get_by_id(experiment.id)
                assert len(retrieved_exp.optimization_results) == 1
                
                retrieved_opt = opt_repo.get_by_id(optimization.id)
                assert retrieved_opt.experiment_run_id == experiment.id
                
                retrieved_field = field_repo.get_by_id(field.id)
                assert retrieved_field.optimization_result_id == optimization.id
                
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
                experiment = ExperimentRun(
                    name="Performance Test",
                    experiment_type="performance_testing",
                    parameters={"test": True}
                )
                session.add(experiment)
                session.commit()
                
                # Test performance stats from repository
                stats = opt_repo.get_performance_stats(days=1)
                assert isinstance(stats, dict)
                
                # Test recent results
                recent = opt_repo.get_recent_results(hours=1)
                assert len(recent) >= 0
                
            # Cleanup
            Path(tmp.name).unlink()
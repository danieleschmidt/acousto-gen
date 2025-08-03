"""
Repository patterns for data access in Acousto-Gen.
Provides high-level interfaces for database operations.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, and_, or_

from .models import (
    OptimizationResult,
    AcousticFieldData,
    ArrayConfiguration,
    ExperimentRun,
    UserSession,
    SystemMetric
)


class BaseRepository:
    """Base repository with common operations."""
    
    def __init__(self, session: Session, model_class):
        self.session = session
        self.model_class = model_class
    
    def get_by_id(self, id: int):
        """Get record by ID."""
        return self.session.query(self.model_class).filter(self.model_class.id == id).first()
    
    def get_all(self, limit: int = 100, offset: int = 0):
        """Get all records with pagination."""
        return (self.session.query(self.model_class)
                .offset(offset)
                .limit(limit)
                .all())
    
    def create(self, **kwargs):
        """Create new record."""
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        self.session.flush()  # Get ID without committing
        return instance
    
    def update(self, id: int, **kwargs):
        """Update record by ID."""
        instance = self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            self.session.flush()
        return instance
    
    def delete(self, id: int) -> bool:
        """Delete record by ID."""
        instance = self.get_by_id(id)
        if instance:
            self.session.delete(instance)
            self.session.flush()
            return True
        return False
    
    def count(self) -> int:
        """Count total records."""
        return self.session.query(self.model_class).count()


class OptimizationRepository(BaseRepository):
    """Repository for optimization results."""
    
    def __init__(self, session: Session):
        super().__init__(session, OptimizationResult)
    
    def get_by_run_id(self, run_id: str) -> Optional[OptimizationResult]:
        """Get optimization result by run ID."""
        return (self.session.query(OptimizationResult)
                .filter(OptimizationResult.run_id == run_id)
                .first())
    
    def get_by_method(self, method: str, limit: int = 50) -> List[OptimizationResult]:
        """Get optimization results by method."""
        return (self.session.query(OptimizationResult)
                .filter(OptimizationResult.method == method)
                .order_by(desc(OptimizationResult.created_at))
                .limit(limit)
                .all())
    
    def get_best_results(
        self,
        target_type: Optional[str] = None,
        method: Optional[str] = None,
        limit: int = 10
    ) -> List[OptimizationResult]:
        """Get best optimization results by final loss."""
        query = self.session.query(OptimizationResult)
        
        if target_type:
            query = query.filter(OptimizationResult.target_type == target_type)
        if method:
            query = query.filter(OptimizationResult.method == method)
        
        return (query
                .filter(OptimizationResult.converged == True)
                .order_by(asc(OptimizationResult.final_loss))
                .limit(limit)
                .all())
    
    def get_recent_results(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[OptimizationResult]:
        """Get optimization results from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return (self.session.query(OptimizationResult)
                .filter(OptimizationResult.created_at >= cutoff)
                .order_by(desc(OptimizationResult.created_at))
                .limit(limit)
                .all())
    
    def get_performance_stats(
        self,
        method: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get performance statistics for optimization method."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = self.session.query(OptimizationResult).filter(
            OptimizationResult.created_at >= cutoff
        )
        
        if method:
            query = query.filter(OptimizationResult.method == method)
        
        results = query.all()
        
        if not results:
            return {}
        
        converged = [r for r in results if r.converged]
        
        stats = {
            'total_runs': len(results),
            'successful_runs': len(converged),
            'success_rate': len(converged) / len(results) if results else 0,
            'avg_time': sum(r.time_elapsed for r in results) / len(results),
            'avg_iterations': sum(r.iterations for r in results) / len(results),
        }
        
        if converged:
            stats.update({
                'avg_final_loss': sum(r.final_loss for r in converged) / len(converged),
                'best_loss': min(r.final_loss for r in converged),
                'worst_loss': max(r.final_loss for r in converged),
                'avg_focus_error': sum(r.focus_error for r in converged if r.focus_error) / len([r for r in converged if r.focus_error]),
                'avg_efficiency': sum(r.efficiency for r in converged if r.efficiency) / len([r for r in converged if r.efficiency]),
            })
        
        return stats
    
    def search(
        self,
        method: Optional[str] = None,
        target_type: Optional[str] = None,
        min_loss: Optional[float] = None,
        max_loss: Optional[float] = None,
        converged_only: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> List[OptimizationResult]:
        """Search optimization results with filters."""
        query = self.session.query(OptimizationResult)
        
        if method:
            query = query.filter(OptimizationResult.method == method)
        if target_type:
            query = query.filter(OptimizationResult.target_type == target_type)
        if min_loss is not None:
            query = query.filter(OptimizationResult.final_loss >= min_loss)
        if max_loss is not None:
            query = query.filter(OptimizationResult.final_loss <= max_loss)
        if converged_only:
            query = query.filter(OptimizationResult.converged == True)
        
        return (query
                .order_by(desc(OptimizationResult.created_at))
                .offset(offset)
                .limit(limit)
                .all())


class FieldRepository(BaseRepository):
    """Repository for acoustic field data."""
    
    def __init__(self, session: Session):
        super().__init__(session, AcousticFieldData)
    
    def get_by_field_id(self, field_id: str) -> Optional[AcousticFieldData]:
        """Get field data by field ID."""
        return (self.session.query(AcousticFieldData)
                .filter(AcousticFieldData.field_id == field_id)
                .first())
    
    def get_by_type(self, field_type: str, limit: int = 50) -> List[AcousticFieldData]:
        """Get field data by type."""
        return (self.session.query(AcousticFieldData)
                .filter(AcousticFieldData.field_type == field_type)
                .order_by(desc(AcousticFieldData.created_at))
                .limit(limit)
                .all())
    
    def get_by_frequency_range(
        self,
        min_freq: float,
        max_freq: float,
        limit: int = 50
    ) -> List[AcousticFieldData]:
        """Get field data by frequency range."""
        return (self.session.query(AcousticFieldData)
                .filter(and_(
                    AcousticFieldData.frequency >= min_freq,
                    AcousticFieldData.frequency <= max_freq
                ))
                .order_by(desc(AcousticFieldData.created_at))
                .limit(limit)
                .all())
    
    def get_for_optimization(self, optimization_id: int) -> Optional[AcousticFieldData]:
        """Get field data for specific optimization."""
        return (self.session.query(AcousticFieldData)
                .filter(AcousticFieldData.optimization_result_id == optimization_id)
                .first())


class ArrayRepository(BaseRepository):
    """Repository for array configurations."""
    
    def __init__(self, session: Session):
        super().__init__(session, ArrayConfiguration)
    
    def get_by_config_id(self, config_id: str) -> Optional[ArrayConfiguration]:
        """Get array configuration by config ID."""
        return (self.session.query(ArrayConfiguration)
                .filter(ArrayConfiguration.config_id == config_id)
                .first())
    
    def get_active_arrays(self) -> List[ArrayConfiguration]:
        """Get all active array configurations."""
        return (self.session.query(ArrayConfiguration)
                .filter(ArrayConfiguration.is_active == True)
                .order_by(desc(ArrayConfiguration.created_at))
                .all())
    
    def get_by_type(self, array_type: str) -> List[ArrayConfiguration]:
        """Get arrays by type."""
        return (self.session.query(ArrayConfiguration)
                .filter(ArrayConfiguration.array_type == array_type)
                .filter(ArrayConfiguration.is_active == True)
                .order_by(desc(ArrayConfiguration.created_at))
                .all())
    
    def get_calibrated_arrays(self, max_age_days: int = 30) -> List[ArrayConfiguration]:
        """Get arrays with recent calibration."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        return (self.session.query(ArrayConfiguration)
                .filter(ArrayConfiguration.calibration_date >= cutoff)
                .filter(ArrayConfiguration.is_active == True)
                .order_by(desc(ArrayConfiguration.calibration_date))
                .all())
    
    def deactivate_array(self, config_id: str) -> bool:
        """Deactivate an array configuration."""
        array_config = self.get_by_config_id(config_id)
        if array_config:
            array_config.is_active = False
            self.session.flush()
            return True
        return False


class ExperimentRepository(BaseRepository):
    """Repository for experiment runs."""
    
    def __init__(self, session: Session):
        super().__init__(session, ExperimentRun)
    
    def get_by_experiment_id(self, experiment_id: str) -> Optional[ExperimentRun]:
        """Get experiment by experiment ID."""
        return (self.session.query(ExperimentRun)
                .filter(ExperimentRun.experiment_id == experiment_id)
                .first())
    
    def get_by_user(self, user_id: str, limit: int = 50) -> List[ExperimentRun]:
        """Get experiments by user."""
        return (self.session.query(ExperimentRun)
                .filter(ExperimentRun.user_id == user_id)
                .order_by(desc(ExperimentRun.started_at))
                .limit(limit)
                .all())
    
    def get_active_experiments(self) -> List[ExperimentRun]:
        """Get currently running experiments."""
        return (self.session.query(ExperimentRun)
                .filter(ExperimentRun.status == "running")
                .order_by(desc(ExperimentRun.started_at))
                .all())
    
    def get_recent_experiments(
        self,
        days: int = 7,
        experiment_type: Optional[str] = None
    ) -> List[ExperimentRun]:
        """Get recent experiments."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = self.session.query(ExperimentRun).filter(
            ExperimentRun.started_at >= cutoff
        )
        
        if experiment_type:
            query = query.filter(ExperimentRun.experiment_type == experiment_type)
        
        return (query
                .order_by(desc(ExperimentRun.started_at))
                .all())
    
    def update_progress(self, experiment_id: str, progress: float) -> bool:
        """Update experiment progress."""
        experiment = self.get_by_experiment_id(experiment_id)
        if experiment:
            experiment.progress = progress
            self.session.flush()
            return True
        return False


class SessionRepository(BaseRepository):
    """Repository for user sessions."""
    
    def __init__(self, session: Session):
        super().__init__(session, UserSession)
    
    def get_by_session_id(self, session_id: str) -> Optional[UserSession]:
        """Get session by session ID."""
        return (self.session.query(UserSession)
                .filter(UserSession.session_id == session_id)
                .first())
    
    def get_active_sessions(self) -> List[UserSession]:
        """Get all active sessions."""
        return (self.session.query(UserSession)
                .filter(UserSession.is_active == True)
                .order_by(desc(UserSession.last_activity))
                .all())
    
    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[UserSession]:
        """Get sessions for a specific user."""
        query = self.session.query(UserSession).filter(UserSession.user_id == user_id)
        
        if active_only:
            query = query.filter(UserSession.is_active == True)
        
        return (query
                .order_by(desc(UserSession.last_activity))
                .all())
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old inactive sessions."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        count = (self.session.query(UserSession)
                .filter(and_(
                    UserSession.last_activity < cutoff,
                    UserSession.is_active == False
                ))
                .count())
        
        (self.session.query(UserSession)
         .filter(and_(
             UserSession.last_activity < cutoff,
             UserSession.is_active == False
         ))
         .delete())
        
        self.session.flush()
        return count


class MetricsRepository(BaseRepository):
    """Repository for system metrics."""
    
    def __init__(self, session: Session):
        super().__init__(session, SystemMetric)
    
    def get_recent_metrics(
        self,
        hours: int = 24,
        limit: int = 1000
    ) -> List[SystemMetric]:
        """Get recent system metrics."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return (self.session.query(SystemMetric)
                .filter(SystemMetric.timestamp >= cutoff)
                .order_by(desc(SystemMetric.timestamp))
                .limit(limit)
                .all())
    
    def get_latest_metric(self) -> Optional[SystemMetric]:
        """Get the most recent system metric."""
        return (self.session.query(SystemMetric)
                .order_by(desc(SystemMetric.timestamp))
                .first())
    
    def get_average_metrics(
        self,
        hours: int = 24
    ) -> Dict[str, float]:
        """Get average metrics over time period."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        result = (self.session.query(
            func.avg(SystemMetric.cpu_usage).label('avg_cpu'),
            func.avg(SystemMetric.memory_usage_mb).label('avg_memory'),
            func.avg(SystemMetric.gpu_usage).label('avg_gpu'),
            func.avg(SystemMetric.gpu_memory_mb).label('avg_gpu_memory'),
            func.avg(SystemMetric.active_sessions).label('avg_sessions'),
            func.avg(SystemMetric.optimizations_per_hour).label('avg_opt_rate'),
            func.avg(SystemMetric.average_optimization_time).label('avg_opt_time')
        )
        .filter(SystemMetric.timestamp >= cutoff)
        .first())
        
        return {
            'cpu_usage': result.avg_cpu or 0,
            'memory_usage_mb': result.avg_memory or 0,
            'gpu_usage': result.avg_gpu or 0,
            'gpu_memory_mb': result.avg_gpu_memory or 0,
            'active_sessions': result.avg_sessions or 0,
            'optimizations_per_hour': result.avg_opt_rate or 0,
            'average_optimization_time': result.avg_opt_time or 0
        }
    
    def cleanup_old_metrics(self, max_age_days: int = 30) -> int:
        """Clean up old metrics data."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        
        count = (self.session.query(SystemMetric)
                .filter(SystemMetric.timestamp < cutoff)
                .count())
        
        (self.session.query(SystemMetric)
         .filter(SystemMetric.timestamp < cutoff)
         .delete())
        
        self.session.flush()
        return count
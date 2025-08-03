"""
Database layer for Acousto-Gen.
Handles persistence of optimization results, field data, and system configurations.
"""

from .connection import DatabaseManager, get_db_session
from .models import (
    Base,
    OptimizationResult,
    AcousticFieldData,
    ArrayConfiguration,
    ExperimentRun,
    UserSession
)
from .repositories import (
    OptimizationRepository,
    FieldRepository,
    ArrayRepository,
    ExperimentRepository
)

__all__ = [
    "DatabaseManager",
    "get_db_session", 
    "Base",
    "OptimizationResult",
    "AcousticFieldData", 
    "ArrayConfiguration",
    "ExperimentRun",
    "UserSession",
    "OptimizationRepository",
    "FieldRepository",
    "ArrayRepository", 
    "ExperimentRepository"
]
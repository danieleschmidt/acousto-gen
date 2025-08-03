"""
Database connection management for Acousto-Gen.
Supports SQLite for development and PostgreSQL for production.
"""

import os
import logging
from typing import Generator, Optional
from contextlib import contextmanager
from urllib.parse import urlparse

from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url or self._get_default_database_url()
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        
        self._setup_engine()
        self._setup_session_factory()
    
    def _get_default_database_url(self) -> str:
        """Get default database URL from environment."""
        env = os.getenv("ACOUSTO_GEN_ENV", "development")
        
        if env == "development":
            return "sqlite:///data/acousto-gen-dev.db"
        elif env == "testing":
            return "sqlite:///:memory:"
        else:
            # Production - require explicit configuration
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL environment variable required for production")
            return db_url
    
    def _setup_engine(self):
        """Setup SQLAlchemy engine with appropriate configuration."""
        parsed = urlparse(self.database_url)
        
        if parsed.scheme == "sqlite":
            # SQLite configuration
            connect_args = {"check_same_thread": False}
            
            if parsed.path == "/:memory:":
                # In-memory database for testing
                connect_args.update({
                    "poolclass": StaticPool,
                    "pool_pre_ping": True
                })
            else:
                # File-based SQLite
                os.makedirs(os.path.dirname(parsed.path), exist_ok=True)
            
            self.engine = create_engine(
                self.database_url,
                connect_args=connect_args,
                echo=os.getenv("DATABASE_ECHO", "false").lower() == "true"
            )
            
            # Enable WAL mode for better concurrency
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=1000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
        
        elif parsed.scheme.startswith("postgresql"):
            # PostgreSQL configuration
            self.engine = create_engine(
                self.database_url,
                pool_size=int(os.getenv("DATABASE_POOL_SIZE", "20")),
                max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "30")),
                pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
                pool_pre_ping=True,
                echo=os.getenv("DATABASE_ECHO", "false").lower() == "true"
            )
        
        else:
            raise ValueError(f"Unsupported database scheme: {parsed.scheme}")
        
        logger.info(f"Database engine created for: {parsed.scheme}")
    
    def _setup_session_factory(self):
        """Setup session factory."""
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables."""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self):
        """Drop all database tables."""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around database operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.session_scope() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {
            "url": self.database_url,
            "engine": str(self.engine.url).split("@")[-1] if self.engine else None,
            "pool_size": getattr(self.engine.pool, "size", None) if self.engine else None,
            "checked_out_connections": getattr(self.engine.pool, "checkedout", None) if self.engine else None,
        }
        
        # Get table counts
        try:
            with self.session_scope() as session:
                from .models import OptimizationResult, AcousticFieldData, ExperimentRun
                
                stats.update({
                    "optimization_results": session.query(OptimizationResult).count(),
                    "field_data_records": session.query(AcousticFieldData).count(),
                    "experiment_runs": session.query(ExperimentRun).count(),
                })
        except Exception as e:
            logger.warning(f"Could not get table statistics: {e}")
            stats["table_stats_error"] = str(e)
        
        return stats


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    """
    Get database session for dependency injection.
    Used with FastAPI Depends().
    """
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def initialize_database(database_url: Optional[str] = None, create_tables: bool = True):
    """
    Initialize database with specific URL.
    
    Args:
        database_url: Database connection URL
        create_tables: Whether to create tables
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    
    if create_tables:
        _db_manager.create_tables()
    
    logger.info("Database initialized successfully")
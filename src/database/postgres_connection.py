"""
PostgreSQL database connection and session management for Quantum Trading Matrix™
Handles database initialization, connections, and session management
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Database configuration
class DatabaseConfig:
    """Database configuration"""
    
    def __init__(self):
        self.url = settings.database.url
        self.pool_size = settings.database.pool_size
        self.max_overflow = settings.database.max_overflow
        self.echo = settings.debug

# Global database components
engine = None
SessionLocal = None
Base = declarative_base()
metadata = MetaData()

def init_db_connection():
    """Initialize database connection and session factory"""
    global engine, SessionLocal
    
    try:
        config = DatabaseConfig()
        
        # Create engine
        engine = create_engine(
            config.url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            echo=config.echo,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
        )
        
        # Create session factory
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        
        logger.info("Database connection initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {e}")
        raise

def close_db():
    """Close database connection"""
    global engine
    
    if engine:
        engine.dispose()
        logger.info("Database connection closed")

def get_db() -> Generator[Session, None, None]:
    """
    Get database session
    
    Yields:
        Session: SQLAlchemy database session
    """
    if SessionLocal is None:
        init_db_connection()
    
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get database session as context manager
    
    Yields:
        Session: SQLAlchemy database session
    """
    if SessionLocal is None:
        init_db_connection()
    
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    if engine is None:
        init_db_connection()
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

def drop_tables():
    """Drop all database tables (use with caution!)"""
    if engine is None:
        init_db_connection()
    
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise

def test_connection() -> bool:
    """
    Test database connection
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        if engine is None:
            init_db_connection()
        
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        logger.info("Database connection test successful")
        return True
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

# Utility functions for testing
def create_test_engine():
    """Create test database engine"""
    test_url = settings.testing.database_url
    
    return create_engine(
        test_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )

def get_test_db():
    """Get test database session"""
    test_engine = create_test_engine()
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    # Create tables
    Base.metadata.create_all(bind=test_engine)
    
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Clean up
        Base.metadata.drop_all(bind=test_engine) 

# Add alias for backwards compatibility
get_db_connection = get_db 
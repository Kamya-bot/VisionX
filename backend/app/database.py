"""
Database configuration and connection management for VisionX.

Supports both SQLite (development) and PostgreSQL (production).
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from typing import Generator

# Get database URL from environment or use SQLite for development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./visionx.db"  # Default to SQLite for easy development
)

# Handle PostgreSQL URL format from some hosting providers (postgres:// -> postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configure engine based on database type
if DATABASE_URL.startswith("sqlite"):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL query logging
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        echo=False  # Set to True for SQL query logging
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator:
    """
    Dependency for getting database sessions.
    
    Usage in FastAPI endpoints:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            items = db.query(Item).all()
            return items
    
    Yields:
        Database session that will be automatically closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    import models  # noqa - ensures all models are registered
    Base.metadata.create_all(bind=engine, checkfirst=True)


def get_db_info() -> dict:
    """
    Get information about current database connection.
    
    Returns:
        Dictionary with database type, URL, and connection status.
    """
    return {
        "database_type": "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite",
        "database_url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
        "engine": str(engine.url),
        "pool_size": engine.pool.size() if hasattr(engine.pool, 'size') else "N/A"
    }

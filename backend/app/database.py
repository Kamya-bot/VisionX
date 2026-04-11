"""
Database configuration and connection management for VisionX.
Supports both SQLite (development) and PostgreSQL (production).
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from typing import Generator

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////tmp/visionx.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    import models  # noqa
    Base.metadata.create_all(bind=engine, checkfirst=True)
    # Safe column migrations — adds missing columns without dropping existing data
    with engine.connect() as conn:
        for col, coltype in [
            ("avatar_url",      "TEXT"),
            ("oauth_provider",  "TEXT"),
            ("oauth_sub",       "TEXT"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col} {coltype}"))
                conn.commit()
            except Exception:
                pass  # column already exists, safe to ignore


def get_db_info() -> dict:
    return {
        "database_type": "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite",
        "database_url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
        "engine": str(engine.url),
        "pool_size": engine.pool.size() if hasattr(engine.pool, 'size') else "N/A"
    }
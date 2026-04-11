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
        migrations = [
            ("users",           "avatar_url",              "TEXT"),
            ("users",           "oauth_provider",          "TEXT"),
            ("users",           "oauth_sub",               "TEXT"),
            ("predictions_log", "domain_detected",         "TEXT"),
            ("predictions_log", "options_count",           "INTEGER"),
            ("predictions_log", "recommended_option_id",   "TEXT"),
            ("predictions_log", "recommended_option_name", "TEXT"),
            ("predictions_log", "recommendation",          "TEXT"),
            ("predictions_log", "shap_values",             "JSONB"),
            ("predictions_log", "universal_features",      "JSONB"),
            ("predictions_log", "model_version",           "TEXT"),
            ("predictions_log", "prediction_time_ms",      "FLOAT"),
        ]
        for table, col, coltype in migrations:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"))
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
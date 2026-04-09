"""
VisionX ML Backend Configuration
All cluster labels and characteristics are generated dynamically
from the trained model — nothing is hardcoded.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):

    # Application
    APP_NAME: str = "VisionX ML Backend"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() != "false"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = 4

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5500",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8080",
        "https://visionx-topaz.vercel.app",
    ]

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    MODEL_DIR: str = os.path.join(BASE_DIR, "trained_models")
    LOG_DIR: str = os.path.join(BASE_DIR, "logs")
    MLFLOW_DIR: str = os.path.join(BASE_DIR, "mlruns")

    # Model Files
    CLUSTERING_MODEL_PATH: str = os.path.join(MODEL_DIR, "clustering.pkl")
    PREDICTION_MODEL_PATH: str = os.path.join(MODEL_DIR, "prediction.pkl")
    SCALER_PATH: str = os.path.join(MODEL_DIR, "scaler.pkl")
    CLUSTER_PROFILES_PATH: str = os.path.join(MODEL_DIR, "cluster_profiles.json")

    # ML Configuration
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    N_CLUSTERS: int = 4
    MAX_ITERATIONS: int = 300

    # Performance
    MODEL_CACHE_SIZE: int = 100
    PREDICTION_TIMEOUT: int = 5
    MAX_BATCH_SIZE: int = 1000

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # API
    API_V1_PREFIX: str = "/api/v1"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"

    # Rate Limiting
    RATE_LIMIT_PREDICT: str = "10/minute"       # per user on /ml/predict
    RATE_LIMIT_AUTH: str = "5/minute"           # per IP on login/register
    RATE_LIMIT_GENERAL: str = "100/minute"      # everything else

    # JWT Auth
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "visionx-change-this-in-production-use-openssl-rand-hex-32"
    )
    REFRESH_SECRET_KEY: str = os.getenv(
        "REFRESH_SECRET_KEY",
        "visionx-refresh-change-this-in-production-use-openssl-rand-hex-32"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15           # 15 minutes
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7              # 7 days

    # OAuth — Google
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    GOOGLE_REDIRECT_URI: str = os.getenv(
        "GOOGLE_REDIRECT_URI",
        "http://localhost:3000/auth/callback/google"
    )

    # OAuth — GitHub
    GITHUB_CLIENT_ID: str = os.getenv("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET", "")
    GITHUB_REDIRECT_URI: str = os.getenv(
        "GITHUB_REDIRECT_URI",
        "http://localhost:3000/auth/callback/github"
    )

    # Feature columns (universal 6-feature space)
    FEATURE_COLUMNS: List[str] = [
        "value_score",
        "quality_score",
        "growth_score",
        "risk_score",
        "fit_score",
        "speed_score",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


def create_directories():
    for d in [
        settings.DATA_DIR,
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.MODEL_DIR,
        settings.LOG_DIR,
        settings.MLFLOW_DIR,
    ]:
        os.makedirs(d, exist_ok=True)
    print("✅ All directories created successfully")
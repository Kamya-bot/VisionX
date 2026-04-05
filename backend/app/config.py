"""
VisionX ML Backend Configuration
Centralized configuration management for the ML backend system
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Application
    APP_NAME: str = "VisionX ML Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5500",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8080",
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
    RECOMMENDER_MODEL_PATH: str = os.path.join(MODEL_DIR, "recommender.pkl")
    SCALER_PATH: str = os.path.join(MODEL_DIR, "scaler.pkl")
    ENCODER_PATH: str = os.path.join(MODEL_DIR, "encoder.pkl")
    
    # ML Configuration
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    N_CLUSTERS: int = 4
    MAX_ITERATIONS: int = 300
    
    # Performance
    MODEL_CACHE_SIZE: int = 100
    PREDICTION_TIMEOUT: int = 5  # seconds
    MAX_BATCH_SIZE: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = f"file://{MLFLOW_DIR}"
    MLFLOW_EXPERIMENT_NAME: str = "visionx-ml-experiments"
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100

    # JWT Auth
    SECRET_KEY: str = "visionx-change-this-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Feature Engineering
    FEATURE_COLUMNS: List[str] = [
        "session_time",
        "clicks",
        "scroll_depth",
        "comparison_count",
        "product_views",
        "decision_time",
        "price_sensitivity",
        "feature_interest_score",
        "engagement_score",
        "purchase_intent_score"
    ]
    
    # Cluster Labels
    CLUSTER_LABELS: dict = {
        0: "Casual User",
        1: "Analytical Researcher",
        2: "High Intent Buyer",
        3: "Power Decision Maker"
    }
    
    # Cluster Characteristics
    CLUSTER_CHARACTERISTICS: dict = {
        0: ["casual", "browsing", "exploratory"],
        1: ["analytical", "thorough", "data-driven"],
        2: ["focused", "intent-driven", "decisive"],
        3: ["experienced", "confident", "efficient"]
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.DATA_DIR,
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.MODEL_DIR,
        settings.LOG_DIR,
        settings.MLFLOW_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ All directories created successfully")


if __name__ == "__main__":
    create_directories()
    print(f"Configuration loaded for {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Base directory: {settings.BASE_DIR}")
    print(f"Model directory: {settings.MODEL_DIR}")
    print(f"Data directory: {settings.DATA_DIR}")

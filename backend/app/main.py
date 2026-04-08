"""
VisionX ML Backend - FastAPI Application
Production-grade ML API service
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import joblib
import os
import sys
import logging
from datetime import datetime

from middleware.request_tracking import RequestTrackingMiddleware, setup_production_logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings, create_directories
from database import init_db, get_db_info
from api.routes_health import router as health_router
from api.routes_ml import router as ml_router
from api.routes_advanced_ml import router as advanced_ml_router
from api.routes_analytics import router as analytics_router
from api.routes_drift import router as drift_router
from api.routes_model_version import router as model_version_router
from api.routes_auth import router as auth_router
from api.routes_predictions import router as predictions_router

setup_production_logging(log_dir=settings.LOG_DIR)
logger = logging.getLogger(__name__)


class ModelStore:
    clustering_model = None
    prediction_model = None
    scaler = None
    feature_columns = None
    models_loaded = False


model_store = ModelStore()


def load_models():
    logger.info("🔄 Loading ML models...")
    try:
        create_directories()

        if not os.path.exists(settings.CLUSTERING_MODEL_PATH):
            logger.warning(f"⚠️  Clustering model not found: {settings.CLUSTERING_MODEL_PATH}")
            return False
        if not os.path.exists(settings.PREDICTION_MODEL_PATH):
            logger.warning(f"⚠️  Prediction model not found: {settings.PREDICTION_MODEL_PATH}")
            return False

        model_store.clustering_model = joblib.load(settings.CLUSTERING_MODEL_PATH)
        logger.info(f"✅ Clustering model loaded")

        model_store.prediction_model = joblib.load(settings.PREDICTION_MODEL_PATH)
        logger.info(f"✅ Prediction model loaded")

        if os.path.exists(settings.SCALER_PATH):
            model_store.scaler = joblib.load(settings.SCALER_PATH)
            logger.info(f"✅ Scaler loaded")

        feature_cols_path = os.path.join(settings.MODEL_DIR, "feature_columns.pkl")
        if os.path.exists(feature_cols_path):
            model_store.feature_columns = joblib.load(feature_cols_path)
            logger.info(f"✅ Feature columns loaded")

        model_store.models_loaded = True
        logger.info("✅ All models loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        return False


def load_cluster_profiles():
    """
    Auto-generate dynamic cluster labels from the trained KMeans model.
    No hardcoded labels anywhere.
    """
    try:
        from ml.cluster_profiler import get_or_generate_profiles
        profiles = get_or_generate_profiles(
            model_path=settings.CLUSTERING_MODEL_PATH,
            scaler_path=settings.SCALER_PATH,
            profiles_path=settings.CLUSTER_PROFILES_PATH,
        )
        logger.info(f"✅ Cluster profiles loaded: {[p['label'] for p in profiles.values()]}")
        return profiles
    except Exception as e:
        logger.warning(f"⚠️  Could not generate cluster profiles: {e}")
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 70)

    # Database
    try:
        logger.info("🗄️  Initializing database...")
        init_db()
        db_info = get_db_info()
        logger.info(f"✅ Database: {db_info['database_type']} @ {db_info['database_url']}")
    except Exception as e:
        logger.error(f"❌ Database init failed: {str(e)}")
        logger.warning("   Continuing without database (some features disabled)")

    # Models
    models_loaded = load_models()
    if not models_loaded:
        logger.warning("⚠️  Models not loaded. Run the training pipeline.")
    else:
        # Load dynamic cluster profiles (no hardcoded labels)
        app.state.cluster_profiles = load_cluster_profiles()

    logger.info(f"🌐 Server starting on {settings.HOST}:{settings.PORT}")
    logger.info(f"📚 API Docs: http://{settings.HOST}:{settings.PORT}/docs")

    yield

    logger.info("🛑 Shutting down VisionX ML Backend...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade ML backend for VisionX decision intelligence platform",
    version=settings.APP_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    lifespan=lifespan,
)

app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)} - {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
        },
    )


app.include_router(health_router, tags=["Health"])
app.include_router(auth_router, prefix=settings.API_V1_PREFIX, tags=["Authentication"])
app.include_router(predictions_router, prefix=settings.API_V1_PREFIX, tags=["Prediction History"])
app.include_router(ml_router, prefix=settings.API_V1_PREFIX, tags=["Machine Learning"])
app.include_router(advanced_ml_router, prefix=settings.API_V1_PREFIX, tags=["Advanced ML"])
app.include_router(analytics_router, prefix=settings.API_V1_PREFIX, tags=["Analytics"])
app.include_router(drift_router, prefix=settings.API_V1_PREFIX, tags=["Drift Detection"])
app.include_router(model_version_router, prefix=settings.API_V1_PREFIX, tags=["Model Versioning"])


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "api": settings.API_V1_PREFIX,
        "health": "/health",
        "timestamp": datetime.now().isoformat(),
    }


app.state.model_store = model_store


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
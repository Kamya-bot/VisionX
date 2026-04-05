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

# Add app directory to path
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


# Configure production-grade logging
setup_production_logging(log_dir=settings.LOG_DIR)

logger = logging.getLogger(__name__)


# Global model storage
class ModelStore:
    """Global storage for loaded ML models"""
    clustering_model = None
    prediction_model = None
    scaler = None
    feature_columns = None
    models_loaded = False


model_store = ModelStore()


def load_models():
    """Load all ML models at startup"""
    
    logger.info("🔄 Loading ML models...")
    
    try:
        # Create directories if they don't exist
        create_directories()
        
        # Check if models exist
        if not os.path.exists(settings.CLUSTERING_MODEL_PATH):
            logger.warning(f"⚠️  Clustering model not found at: {settings.CLUSTERING_MODEL_PATH}")
            logger.warning("Please run: python training/train_models.py")
            return False
        
        if not os.path.exists(settings.PREDICTION_MODEL_PATH):
            logger.warning(f"⚠️  Prediction model not found at: {settings.PREDICTION_MODEL_PATH}")
            logger.warning("Please run: python training/train_models.py")
            return False
        
        # Load clustering model
        model_store.clustering_model = joblib.load(settings.CLUSTERING_MODEL_PATH)
        logger.info(f"✅ Clustering model loaded from: {settings.CLUSTERING_MODEL_PATH}")
        
        # Load prediction model
        model_store.prediction_model = joblib.load(settings.PREDICTION_MODEL_PATH)
        logger.info(f"✅ Prediction model loaded from: {settings.PREDICTION_MODEL_PATH}")
        
        # Load scaler
        if os.path.exists(settings.SCALER_PATH):
            model_store.scaler = joblib.load(settings.SCALER_PATH)
            logger.info(f"✅ Scaler loaded from: {settings.SCALER_PATH}")
        
        # Load feature columns
        feature_cols_path = os.path.join(settings.MODEL_DIR, 'feature_columns.pkl')
        if os.path.exists(feature_cols_path):
            model_store.feature_columns = joblib.load(feature_cols_path)
            logger.info(f"✅ Feature columns loaded from: {feature_cols_path}")
        
        model_store.models_loaded = True
        logger.info("✅ All models loaded successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    
    # Startup
    logger.info("=" * 70)
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 70)
    
    # Initialize database
    try:
        logger.info("🗄️  Initializing database...")
        init_db()
        db_info = get_db_info()
        logger.info(f"✅ Database initialized: {db_info['database_type']}")
        logger.info(f"   Connection: {db_info['database_url']}")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        logger.warning("   Continuing without database (some features disabled)")
    
    # Load models
    models_loaded = load_models()
    
    if not models_loaded:
        logger.warning("⚠️  Models not loaded. Some endpoints will not work.")
        logger.warning("Run: python training/generate_dataset.py")
        logger.warning("Run: python training/train_models.py")
    
    logger.info(f"🌐 Server starting on {settings.HOST}:{settings.PORT}")
    logger.info(f"📚 API Docs: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"📖 ReDoc: http://{settings.HOST}:{settings.PORT}/redoc")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down VisionX ML Backend...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade ML backend for VisionX decision intelligence platform",
    version=settings.APP_VERSION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    lifespan=lifespan
)


# Add request tracking middleware (BEFORE CORS)
app.add_middleware(RequestTrackingMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    
    logger.error(f"Unhandled exception: {str(exc)} - {request.url.path}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(auth_router, prefix=settings.API_V1_PREFIX, tags=["Authentication"])
app.include_router(predictions_router, prefix=settings.API_V1_PREFIX, tags=["Prediction History"])
app.include_router(ml_router, prefix=settings.API_V1_PREFIX, tags=["Machine Learning"])
app.include_router(advanced_ml_router, prefix=settings.API_V1_PREFIX, tags=["Advanced ML - Explainability & Simulation"])
app.include_router(analytics_router, prefix=settings.API_V1_PREFIX, tags=["Analytics - KPIs, A/B Testing, Business Insights"])
app.include_router(drift_router, prefix=settings.API_V1_PREFIX, tags=["Drift Detection - Production ML Monitoring"])
app.include_router(model_version_router, prefix=settings.API_V1_PREFIX, tags=["Model Versioning - Experiment Tracking"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": f"/docs",
        "api": f"{settings.API_V1_PREFIX}",
        "health": "/health",
        "timestamp": datetime.now().isoformat()
    }


# Make model store accessible
app.state.model_store = model_store


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

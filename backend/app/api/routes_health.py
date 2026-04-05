"""
Health Check Routes
Monitor API and model status
"""

from fastapi import APIRouter, Request
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings


router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint
    
    Returns:
        - status: API status
        - models_loaded: Whether ML models are loaded
        - timestamp: Current server time
    """
    
    model_store = request.app.state.model_store
    
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "models_loaded": model_store.models_loaded,
        "models": {
            "clustering": model_store.clustering_model is not None,
            "prediction": model_store.prediction_model is not None,
            "scaler": model_store.scaler is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health/detailed")
async def detailed_health_check(request: Request):
    """
    Detailed health check with system information
    
    Returns comprehensive system and model status
    """
    
    model_store = request.app.state.model_store
    
    # Check model file existence
    model_files = {
        "clustering_model": os.path.exists(settings.CLUSTERING_MODEL_PATH),
        "prediction_model": os.path.exists(settings.PREDICTION_MODEL_PATH),
        "scaler": os.path.exists(settings.SCALER_PATH),
    }
    
    return {
        "status": "healthy",
        "app_info": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "debug_mode": settings.DEBUG
        },
        "models": {
            "loaded_in_memory": {
                "clustering": model_store.clustering_model is not None,
                "prediction": model_store.prediction_model is not None,
                "scaler": model_store.scaler is not None
            },
            "files_exist": model_files,
            "all_loaded": model_store.models_loaded
        },
        "configuration": {
            "n_clusters": settings.N_CLUSTERS,
            "random_seed": settings.RANDOM_SEED,
            "api_version": settings.API_V1_PREFIX
        },
        "directories": {
            "model_dir": settings.MODEL_DIR,
            "data_dir": settings.DATA_DIR,
            "log_dir": settings.LOG_DIR
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health/ready")
async def readiness_check(request: Request):
    """
    Kubernetes-style readiness probe
    
    Returns:
        200 if models are loaded and ready
        503 if models are not loaded
    """
    
    model_store = request.app.state.model_store
    
    if model_store.models_loaded:
        return {
            "ready": True,
            "message": "Service is ready to accept requests"
        }
    else:
        return {
            "ready": False,
            "message": "Models not loaded. Service not ready."
        }


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe
    
    Always returns 200 if the API is running
    """
    
    return {
        "alive": True,
        "message": "Service is alive"
    }

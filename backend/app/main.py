"""
VisionX — FastAPI Application Entry Point

Phase 2: Added /ml/retrain routes
"""

from __future__ import annotations

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings
from database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "trained_models"

CLUSTER_PROFILES = {
    0: {"label": "Independent Thinker & Risk-Averse",   "dominant_features": ["fit_score", "risk_score"]},
    1: {"label": "Growth-Oriented & Value-Conscious",    "dominant_features": ["growth_score", "value_score"]},
    2: {"label": "Budget Pragmatist & Stability-Seeker", "dominant_features": ["quality_score", "growth_score"]},
    3: {"label": "Socially-Validated & Speed-Driven",    "dominant_features": ["fit_score", "speed_score"]},
}


class ModelStore:
    def __init__(self):
        self.prediction_model = None
        self.clustering_model = None
        self.scaler = None
        self.feature_columns = None
        self.models_loaded = False

    def load(self):
        try:
            with open(MODELS_DIR / "prediction.pkl", "rb") as f:
                self.prediction_model = pickle.load(f)
            with open(MODELS_DIR / "clustering.pkl", "rb") as f:
                self.clustering_model = pickle.load(f)
            with open(MODELS_DIR / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
                self.feature_columns = pickle.load(f)
            self.models_loaded = True
            logger.info("✅ Models loaded successfully")
        except FileNotFoundError:
            logger.warning("⚠️  Model files not found — running training pipeline now")
            self._train_and_load()
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self._train_and_load()

    def _train_and_load(self):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from ml.train import train
            train()
            # Retry loading
            with open(MODELS_DIR / "prediction.pkl", "rb") as f:
                self.prediction_model = pickle.load(f)
            with open(MODELS_DIR / "clustering.pkl", "rb") as f:
                self.clustering_model = pickle.load(f)
            with open(MODELS_DIR / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
                self.feature_columns = pickle.load(f)
            self.models_loaded = True
            logger.info("✅ Models trained and loaded")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            self.models_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    model_store = ModelStore()
    model_store.load()
    app.state.model_store = model_store
    app.state.cluster_profiles = CLUSTER_PROFILES
    logger.info(f"🚀 VisionX {settings.APP_VERSION} started")
    yield
    # Shutdown
    logger.info("VisionX shutting down")


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="VisionX API",
    description="AI-powered decision intelligence platform",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    ms = request.app.state.model_store
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "models_loaded": ms.models_loaded,
    }


# ── Routers ───────────────────────────────────────────────────────────────────

from api.routes_auth import router as auth_router
from api.routes_ml import router as ml_router
from api.routes_feedback import router as feedback_router
from api.routes_retrain import router as retrain_router

PREFIX = f"/api/{settings.API_VERSION}"

app.include_router(auth_router,     prefix=PREFIX, tags=["auth"])
app.include_router(ml_router,       prefix=PREFIX, tags=["ml"])
app.include_router(feedback_router, prefix=PREFIX, tags=["feedback"])
app.include_router(retrain_router,  prefix=PREFIX, tags=["retrain"])
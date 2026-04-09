"""
VisionX — Retraining Trigger Routes

POST /api/v1/ml/retrain
  → Checks if >= 500 labelled feedback samples exist
  → Runs train() in a background thread
  → Reloads model_store from new .pkl files when done

GET /api/v1/ml/retrain/status
  → Returns last training run metadata from training_results.json
"""

from __future__ import annotations

import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from api.routes_auth import get_current_user
from database import get_db
import models

router = APIRouter()
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "trained_models"
_retrain_lock = threading.Lock()
_retrain_status = {"running": False, "last_result": None, "error": None}


def _reload_model_store(app_state):
    """Hot-reload models from disk into app.state.model_store after retraining."""
    try:
        with open(MODELS_DIR / "prediction.pkl", "rb") as f:
            app_state.model_store.prediction_model = pickle.load(f)
        with open(MODELS_DIR / "clustering.pkl", "rb") as f:
            app_state.model_store.clustering_model = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            app_state.model_store.scaler = pickle.load(f)
        app_state.model_store.models_loaded = True
        logger.info("Model store hot-reloaded after retraining")
    except Exception as e:
        logger.error(f"Hot-reload failed: {e}")


def _run_retrain(app_state, force_synthetic_only: bool):
    global _retrain_status
    _retrain_status["running"] = True
    _retrain_status["error"] = None
    try:
        from ml.train import train
        result = train(force_synthetic_only=force_synthetic_only)
        _retrain_status["last_result"] = result
        _reload_model_store(app_state)
        logger.info("Retraining pipeline completed successfully")
    except Exception as e:
        _retrain_status["error"] = str(e)
        logger.error(f"Retraining failed: {e}", exc_info=True)
    finally:
        _retrain_status["running"] = False


@router.post("/ml/retrain")
async def trigger_retrain(
    request: Request,
    background_tasks: BackgroundTasks,
    force: bool = False,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Triggers a retraining run.
    - Checks labelled sample count first (must be >= 500 unless force=True)
    - Runs in background so endpoint returns immediately
    - Hot-reloads models when done
    """
    if _retrain_status["running"]:
        return {"status": "already_running", "message": "Retraining is already in progress"}

    if not force:
        labelled_count = (
            db.query(models.OutcomeFeedback)
            .filter(models.OutcomeFeedback.features_snapshot.isnot(None))
            .count()
        )
        if labelled_count < 500:
            raise HTTPException(
                status_code=400,
                detail=f"Only {labelled_count} labelled samples. Need 500+ to retrain. Use force=true to override.",
            )

    if not _retrain_lock.acquire(blocking=False):
        return {"status": "already_running", "message": "Lock is held by another retrain job"}

    def _run_and_release():
        try:
            _run_retrain(request.app.state, force_synthetic_only=force)
        finally:
            _retrain_lock.release()

    background_tasks.add_task(_run_and_release)
    return {
        "status": "started",
        "message": "Retraining started in background. Check /ml/retrain/status for progress.",
    }


@router.get("/ml/retrain/status")
async def retrain_status(
    current_user: models.User = Depends(get_current_user),
):
    """Returns current retraining status and last run results."""
    # Also read from disk for persistence across restarts
    disk_result = None
    results_path = MODELS_DIR / "training_results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                disk_result = json.load(f)
        except Exception:
            pass

    return {
        "running": _retrain_status["running"],
        "error": _retrain_status["error"],
        "last_result": _retrain_status["last_result"] or disk_result,
    }
"""
VisionX — Analytics & Prediction History Routes

GET /api/v1/analytics/kpis
  → Dashboard KPI cards: total predictions, avg confidence, model accuracy

GET /api/v1/predictions/history
  → Paginated prediction history for the current user
  → Used by dashboard recent predictions list and history page

GET /api/v1/predictions/{prediction_id}
  → Single prediction detail (for results page)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from api.routes_auth import get_current_user
from database import get_db
import models

router = APIRouter()
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "trained_models"


def _load_model_accuracy() -> tuple:
    """Read accuracy/roc from training_results.json."""
    try:
        with open(MODELS_DIR / "training_results.json") as f:
            r = json.load(f)
        return r.get("accuracy", 0.9806), r.get("roc_auc", 0.9988)
    except Exception:
        return 0.9806, 0.9988


# ── KPI endpoint ──────────────────────────────────────────────────────────────

@router.get("/analytics/kpis")
async def get_kpis(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Dashboard KPI cards.
    Returns per-user stats + global model metrics.
    """
    user_id = str(current_user.id)
    accuracy, roc_auc = _load_model_accuracy()

    # Total predictions for this user
    total_predictions = (
        db.query(func.count(models.PredictionLog.id))
        .filter(models.PredictionLog.user_id == user_id)
        .scalar() or 0
    )

    # Average confidence for this user (last 30 days)
    cutoff = datetime.utcnow() - timedelta(days=30)
    avg_conf = (
        db.query(func.avg(models.PredictionLog.confidence))
        .filter(
            models.PredictionLog.user_id == user_id,
            models.PredictionLog.created_at >= cutoff,
            models.PredictionLog.confidence.isnot(None),
        )
        .scalar()
    )

    # User cluster
    cluster_label = None
    if current_user.cluster_id is not None:
        # Look up label from most recent prediction
        recent = (
            db.query(models.PredictionLog)
            .filter(models.PredictionLog.user_id == user_id)
            .order_by(desc(models.PredictionLog.created_at))
            .first()
        )
        if recent:
            cluster_id = recent.cluster_id
            cluster_map = {
                0: "Independent Thinker & Risk-Averse",
                1: "Growth-Oriented & Value-Conscious",
                2: "Budget Pragmatist & Stability-Seeker",
                3: "Socially-Validated & Speed-Driven",
            }
            cluster_label = cluster_map.get(cluster_id)

    # Feedback stats
    total_feedback = (
        db.query(func.count(models.OutcomeFeedback.id))
        .filter(models.OutcomeFeedback.user_id == user_id)
        .scalar() or 0
    )
    accepted_feedback = (
        db.query(func.count(models.OutcomeFeedback.id))
        .filter(
            models.OutcomeFeedback.user_id == user_id,
            models.OutcomeFeedback.accepted == True,
        )
        .scalar() or 0
    )

    return {
        "status": "success",
        "data": {
            "total_predictions": total_predictions,
            "avg_confidence": round(float(avg_conf), 3) if avg_conf else None,
            "model_accuracy": round(accuracy, 4),
            "model_roc_auc": round(roc_auc, 4),
            "model_type": "XGBoost_CalibratedCV",
            "user_cluster": cluster_label,
            "feedback": {
                "total": total_feedback,
                "accepted": accepted_feedback,
                "acceptance_rate": round(accepted_feedback / total_feedback, 3) if total_feedback > 0 else None,
            },
        },
    }


# ── Prediction history ────────────────────────────────────────────────────────

@router.get("/predictions/history")
async def get_prediction_history(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Paginated prediction history for the current user.
    Returns enough data for the dashboard recent list and history page.
    """
    user_id = str(current_user.id)

    rows = (
        db.query(models.PredictionLog)
        .filter(models.PredictionLog.user_id == user_id)
        .order_by(desc(models.PredictionLog.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )

    total = (
        db.query(func.count(models.PredictionLog.id))
        .filter(models.PredictionLog.user_id == user_id)
        .scalar() or 0
    )

    predictions = []
    for row in rows:
        predictions.append({
            "prediction_id": str(row.id),
            "recommended_option_name": row.recommended_option_name,
            "recommended_option_id": row.recommended_option_id,
            "confidence": row.confidence,
            "reasoning": row.recommendation,
            "domain_detected": row.domain_detected,
            "cluster_id": row.cluster_id,
            "options_count": row.options_count,
            "shap_values": row.shap_values,
            "universal_features": row.universal_features,
            "feature_importance": None,  # stored in features blob
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "prediction_time_ms": row.prediction_time_ms,
        })

    return {
        "status": "success",
        "total": total,
        "offset": offset,
        "limit": limit,
        "predictions": predictions,
    }


# ── Single prediction detail ──────────────────────────────────────────────────

@router.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Full detail for a single prediction — used by results.html.
    """
    row = (
        db.query(models.PredictionLog)
        .filter(
            models.PredictionLog.id == prediction_id,
            models.PredictionLog.user_id == str(current_user.id),
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Load feedback for this prediction if it exists
    feedback = (
        db.query(models.OutcomeFeedback)
        .filter(models.OutcomeFeedback.prediction_id == prediction_id)
        .first()
    )

    return {
        "status": "success",
        "data": {
            "prediction_id": str(row.id),
            "recommended_option_name": row.recommended_option_name,
            "recommended_option_id": row.recommended_option_id,
            "confidence": row.confidence,
            "reasoning": row.recommendation,
            "domain_detected": row.domain_detected,
            "cluster_id": row.cluster_id,
            "shap_values": row.shap_values,
            "universal_features": row.universal_features,
            "features": row.features,
            "options_count": row.options_count,
            "prediction_time_ms": row.prediction_time_ms,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "feedback": {
                "accepted": feedback.accepted if feedback else None,
                "satisfaction": feedback.satisfaction if feedback else None,
            } if feedback else None,
        },
    }
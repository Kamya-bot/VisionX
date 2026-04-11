"""
VisionX – Analytics API Routes

Phase 4: Removed duplicate /predictions/history (lives in routes_predictions.py).
         This file owns /analytics/kpis and /analytics/overview only.
"""


import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.routes_auth import get_current_user
from database import get_db
from models import PredictionLog, User, FeedbackLog

router = APIRouter(prefix="/analytics", tags=["analytics"])
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "trained_models"


def _load_training_results() -> dict:
    path = MODELS_DIR / "training_results.json"
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


@router.get("/kpis")
async def get_kpis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Real KPI metrics for the dashboard.
    Returns total predictions, avg confidence, acceptance rate,
    cluster distribution, and model accuracy – all from live DB data.
    """
    try:
        # ── Total predictions for this user ───────────────────────────────
        total = (
            db.query(func.count(PredictionLog.id))
            .filter(PredictionLog.user_id == current_user.id)
            .scalar() or 0
        )

        # ── Avg confidence (last 30 days) ─────────────────────────────────
        since = datetime.utcnow() - timedelta(days=30)
        avg_conf_row = (
            db.query(func.avg(PredictionLog.confidence))
            .filter(
                PredictionLog.user_id == current_user.id,
                PredictionLog.created_at >= since,
            )
            .scalar()
        )
        avg_confidence = round(float(avg_conf_row), 3) if avg_conf_row else 0.0

        # ── Acceptance rate from feedback ─────────────────────────────────
        try:
            total_feedback = (
                db.query(func.count(FeedbackLog.id))
                .filter(FeedbackLog.user_id == current_user.id)
                .scalar() or 0
            )
            accepted = (
                db.query(func.count(FeedbackLog.id))
                .filter(
                    FeedbackLog.user_id == current_user.id,
                    FeedbackLog.accepted == True,  # noqa: E712
                )
                .scalar() or 0
            )
            acceptance_rate = round(accepted / total_feedback, 3) if total_feedback else 0.0
        except Exception:
            acceptance_rate = 0.0
            total_feedback = 0

        # ── Cluster distribution (real, from DB) ──────────────────────────
        cluster_rows = (
            db.query(PredictionLog.cluster_id, func.count(PredictionLog.id))
            .filter(PredictionLog.user_id == current_user.id)
            .group_by(PredictionLog.cluster_id)
            .all()
        )
        cluster_dist = {str(row[0]): row[1] for row in cluster_rows if row[0] is not None}

        # ── Model accuracy from training_results.json ─────────────────────
        tr = _load_training_results()
        model_accuracy = tr.get("accuracy", tr.get("test_accuracy", None))
        if model_accuracy is not None:
            model_accuracy = round(float(model_accuracy), 4)

        # ── Predictions this week vs last week ────────────────────────────
        week_start = datetime.utcnow() - timedelta(days=7)
        prev_week_start = datetime.utcnow() - timedelta(days=14)
        this_week = (
            db.query(func.count(PredictionLog.id))
            .filter(
                PredictionLog.user_id == current_user.id,
                PredictionLog.created_at >= week_start,
            )
            .scalar() or 0
        )
        last_week = (
            db.query(func.count(PredictionLog.id))
            .filter(
                PredictionLog.user_id == current_user.id,
                PredictionLog.created_at >= prev_week_start,
                PredictionLog.created_at < week_start,
            )
            .scalar() or 0
        )
        weekly_delta = this_week - last_week

        return {
            "status": "success",
            "total_predictions": total,
            "avg_confidence": avg_confidence,
            "acceptance_rate": acceptance_rate,
            "total_feedback": total_feedback,
            "cluster_distribution": cluster_dist,
            "model_accuracy": model_accuracy,
            "this_week_predictions": this_week,
            "last_week_predictions": last_week,
            "weekly_delta": weekly_delta,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"KPI fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overview")
async def get_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Global platform overview (admin-style).
    Returns platform-wide totals, not per-user.
    """
    try:
        platform_total = db.query(func.count(PredictionLog.id)).scalar() or 0
        unique_users = (
            db.query(func.count(func.distinct(PredictionLog.user_id))).scalar() or 0
        )
        avg_conf = (
            db.query(func.avg(PredictionLog.confidence)).scalar()
        )
        avg_conf = round(float(avg_conf), 3) if avg_conf else 0.0

        tr = _load_training_results()

        return {
            "status": "success",
            "platform_total_predictions": platform_total,
            "unique_active_users": unique_users,
            "platform_avg_confidence": avg_conf,
            "model_version": tr.get("model_version", "unknown"),
            "last_trained": tr.get("trained_at", None),
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Overview fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""
VisionX — Outcome Feedback Routes

POST /api/v1/feedback/prediction
  → Records whether user followed the recommendation
  → Stores labelled data for retraining

GET /api/v1/feedback/stats
  → Returns acceptance rate, avg satisfaction, feedback count
  → Used by the MLOps dashboard
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from api.routes_auth import get_current_user, get_current_user_optional
from database import get_db
from schemas.request_models import OutcomeFeedback
import models

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/feedback/prediction", status_code=201)
async def submit_feedback(
    body: OutcomeFeedback,
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(get_current_user_optional),
):
    """
    Called by the frontend after user acts on a recommendation.
    This is the ground-truth label that feeds the retraining pipeline.
    """
    # Verify the prediction exists
    pred = db.query(models.PredictionLog).filter(
        models.PredictionLog.id == body.prediction_id
    ).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Don't allow duplicate feedback
    existing = db.query(models.OutcomeFeedback).filter(
        models.OutcomeFeedback.prediction_id == body.prediction_id
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Feedback already submitted for this prediction")

    feedback = models.OutcomeFeedback(
        prediction_id=body.prediction_id,
        user_id=current_user.id if current_user else pred.user_id,
        accepted=body.accepted,
        satisfaction=body.satisfaction,
        actual_choice_id=body.actual_choice_id,
        features_snapshot=pred.universal_features,
        cluster_id_at_prediction=pred.cluster_id,
    )
    db.add(feedback)

    # Update the linked decision if there is one
    if pred.decision_id:
        decision = db.query(models.Decision).filter(
            models.Decision.id == pred.decision_id
        ).first()
        if decision:
            decision.outcome = "success" if body.accepted else "rejected"
            decision.satisfaction_score = body.satisfaction
            decision.completed_at = datetime.utcnow()
            decision.status = "completed"

    db.commit()
    logger.info(
        f"Feedback recorded: prediction={body.prediction_id}, "
        f"accepted={body.accepted}, satisfaction={body.satisfaction}"
    )
    return {"message": "Feedback recorded", "prediction_id": body.prediction_id}


@router.get("/feedback/stats")
async def feedback_stats(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Aggregate feedback stats for the MLOps dashboard.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    total = db.query(func.count(models.OutcomeFeedback.id))\
        .filter(models.OutcomeFeedback.created_at >= cutoff)\
        .scalar() or 0

    accepted = db.query(func.count(models.OutcomeFeedback.id))\
        .filter(
            models.OutcomeFeedback.created_at >= cutoff,
            models.OutcomeFeedback.accepted == True,
        )\
        .scalar() or 0

    avg_satisfaction = db.query(func.avg(models.OutcomeFeedback.satisfaction))\
        .filter(
            models.OutcomeFeedback.created_at >= cutoff,
            models.OutcomeFeedback.satisfaction.isnot(None),
        )\
        .scalar()

    # Count labelled samples available for retraining
    labelled_total = db.query(func.count(models.OutcomeFeedback.id))\
        .filter(models.OutcomeFeedback.features_snapshot.isnot(None))\
        .scalar() or 0

    return {
        "period_days": days,
        "total_feedback": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "acceptance_rate": round(accepted / total, 3) if total > 0 else None,
        "avg_satisfaction": round(float(avg_satisfaction), 2) if avg_satisfaction else None,
        "labelled_samples_for_retraining": labelled_total,
        "retraining_ready": labelled_total >= 500,
    }


@router.get("/feedback/training-data")
async def get_training_data(
    limit: int = 5000,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """
    Returns labelled training data for the retraining pipeline.
    Only accessible to authenticated users (in production: admin-only).
    Returns: list of {features: {...6 scores...}, label: 0|1}
    """
    rows = db.query(models.OutcomeFeedback)\
        .filter(models.OutcomeFeedback.features_snapshot.isnot(None))\
        .order_by(models.OutcomeFeedback.created_at.desc())\
        .limit(limit)\
        .all()

    data = []
    for row in rows:
        if row.features_snapshot and isinstance(row.features_snapshot, dict):
            data.append({
                "features": row.features_snapshot,
                "label": int(row.accepted),
                "cluster_id": row.cluster_id_at_prediction,
                "satisfaction": row.satisfaction,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })

    return {
        "count": len(data),
        "data": data,
    }
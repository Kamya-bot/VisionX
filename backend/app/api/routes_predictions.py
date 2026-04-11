"""
VisionX - Prediction History API Routes
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
import logging

from database import get_db
from api.routes_auth import get_current_user
from models import User, PredictionLog

router = APIRouter()
logger = logging.getLogger(__name__)

CLUSTER_LABELS = {
    0: "Casual User",
    1: "Analytical Researcher",
    2: "High Intent Buyer",
    3: "Power Decision Maker"
}


@router.get("/predictions/history")
async def get_prediction_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    rows = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id == current_user.id)
        .order_by(PredictionLog.created_at.desc())
        .offset(offset).limit(limit).all()
    )
    total = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id == current_user.id)
        .count()
    )

    predictions = []
    for row in rows:
        option_name = row.recommended_option_name if row.recommended_option_name and len(row.recommended_option_name) < 100 else None
        predictions.append({
            "id":                      row.id,
            "prediction_id":           row.id,
            "created_at":              row.created_at.isoformat() if row.created_at else None,
            "cluster_id":              row.cluster_id,
            "cluster_label":           CLUSTER_LABELS.get(row.cluster_id, f"Cluster {row.cluster_id}"),
            "confidence":              round(float(row.confidence), 3) if row.confidence else 0,
            "recommendation":          row.recommendation,
            "recommended_option_name": option_name or "Option A",
            "title":                   option_name or "Prediction",
            "features":                row.features,
            "shap_values":             row.shap_values,
            "model_version":           row.model_version,
            "domain_detected":         row.domain_detected,
            "prediction_time_ms":      row.prediction_time_ms,
        })

    return {
        "status":      "success",
        "total":       total,
        "offset":      offset,
        "limit":       limit,
        "predictions": predictions,
        "data":        predictions,
    }


@router.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    row = (
        db.query(PredictionLog)
        .filter(
            PredictionLog.id == prediction_id,
            PredictionLog.user_id == current_user.id
        ).first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(row)
    db.commit()
    return {"status": "success", "deleted": prediction_id}


@router.get("/predictions/{prediction_id}")
async def get_prediction_by_id(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    row = (
        db.query(PredictionLog)
        .filter(PredictionLog.id == prediction_id, PredictionLog.user_id == current_user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")

    option_name = row.recommended_option_name if row.recommended_option_name and len(row.recommended_option_name) < 100 else None

    return {
        "status":                  "success",
        "id":                      row.id,
        "prediction_id":           row.id,
        "created_at":              row.created_at.isoformat() if row.created_at else None,
        "cluster_id":              row.cluster_id,
        "cluster_label":           CLUSTER_LABELS.get(row.cluster_id, f"Cluster {row.cluster_id}"),
        "confidence":              round(float(row.confidence), 3) if row.confidence else 0,
        "recommendation":          row.recommendation,
        "recommended_option_name": option_name or "Option A",
        "reasoning":               row.recommendation,
        "features":                row.features,
        "shap_values":             row.shap_values,
        "universal_features":      row.universal_features,
        "model_version":           row.model_version,
        "domain_detected":         row.domain_detected,
        "prediction_time_ms":      row.prediction_time_ms,
    }
"""
VisionX - Drift Detection API Routes
Real-time monitoring endpoints for model reliability
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
import numpy as np
from datetime import datetime, timedelta
import logging

from database import get_db
from services.drift_detection import DriftMonitor, check_model_drift
from crud import get_recent_predictions

router = APIRouter(prefix="/drift", tags=["Drift Detection"])
logger = logging.getLogger(__name__)


@router.get("/check")
async def get_drift_status(
    lookback_days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """
    🔍 Check for data and prediction drift
    
    **Production ML Feature:** Detects when model performance degrades
    
    **What it does:**
    - Compares recent predictions vs historical baseline
    - Uses Kolmogorov-Smirnov test for feature distributions
    - Calculates Population Stability Index (PSI) for predictions
    - Alerts when drift exceeds thresholds
    
    **Returns:**
    - Feature drift analysis (per feature)
    - Prediction drift (PSI score)
    - Overall status (stable/monitor/alert)
    - Recommended actions
    
    **Interview Talking Point:**
    "I implemented real-time drift detection using KS tests and PSI to monitor
    model reliability in production, ensuring predictions remain accurate over time."
    """
    try:
        # Get recent predictions from database
        recent_preds = get_recent_predictions(db, days=lookback_days, limit=100)
        
        if len(recent_preds) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 10 predictions in last {lookback_days} days. Found: {len(recent_preds)}",
                "recommendations": [
                    "Create more predictions to enable drift detection",
                    "Use comparison.html to generate predictions"
                ]
            }
        
        # Extract features and predictions
        feature_names = ['price', 'quality_score', 'satisfaction_score', 'risk_score']
        
        # Convert predictions to numpy arrays
        current_features = []
        current_predictions = []
        
        for pred in recent_preds:
            # Extract features from stored prediction
            features = pred.features if hasattr(pred, 'features') else {}
            
            # Build feature vector
            feature_vector = [
                features.get('price', 0.0),
                features.get('quality_score', 5.0),
                features.get('satisfaction_score', 5.0),
                features.get('risk_score', 0.5)
            ]
            current_features.append(feature_vector)
            
            # Get prediction confidence
            confidence = pred.confidence if hasattr(pred, 'confidence') else 0.5
            current_predictions.append(confidence)
        
        current_features = np.array(current_features)
        current_predictions = np.array(current_predictions)
        
        # Run drift detection
        drift_report = check_model_drift(
            db,
            current_features,
            current_predictions,
            feature_names
        )
        
        # Add metadata
        drift_report['metadata'] = {
            'check_time': datetime.utcnow().isoformat(),
            'samples_analyzed': len(recent_preds),
            'lookback_days': lookback_days
        }
        
        logger.info(f"Drift check completed: {drift_report['summary']['overall_status']}")
        
        return drift_report
        
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift detection error: {str(e)}")


@router.get("/summary")
async def get_drift_summary(
    db: Session = Depends(get_db)
):
    """
    📊 Get quick drift summary (last 7 days)
    
    **Quick health check for model stability**
    
    Returns:
    - Overall drift status
    - Number of drifted features
    - PSI score
    - Recommended action
    """
    try:
        # Get drift status
        drift_data = await get_drift_status(lookback_days=7, db=db)
        
        if drift_data.get('status') == 'insufficient_data':
            return drift_data
        
        summary = drift_data.get('summary', {})
        prediction_drift = drift_data.get('prediction_drift', {})
        
        return {
            'status': summary.get('overall_status', 'unknown'),
            'drift_detected': summary.get('feature_drift_count', 0) > 0 or summary.get('has_prediction_drift', False),
            'drifted_features_count': summary.get('feature_drift_count', 0),
            'prediction_psi': round(summary.get('prediction_psi', 0.0), 4),
            'recommended_action': summary.get('recommended_action', 'monitor'),
            'message': summary.get('summary', 'No drift data available'),
            'last_checked': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Drift summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{feature_name}")
async def get_feature_drift_detail(
    feature_name: str,
    lookback_days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """
    🔬 Get detailed drift analysis for a specific feature
    
    **Deep dive into why a specific feature is drifting**
    
    Returns:
    - KS statistic
    - P-value
    - Mean shift
    - Std deviation shift
    - Baseline vs current distributions
    """
    try:
        drift_data = await get_drift_status(lookback_days=lookback_days, db=db)
        
        if drift_data.get('status') == 'insufficient_data':
            return drift_data
        
        feature_drift = drift_data.get('feature_drift', {})
        
        if feature_name not in feature_drift:
            raise HTTPException(
                status_code=404,
                detail=f"Feature '{feature_name}' not found. Available: {list(feature_drift.keys())}"
            )
        
        detail = feature_drift[feature_name]
        
        return {
            'feature_name': feature_name,
            'drift_detected': detail['is_drift'],
            'severity': detail['severity'],
            'statistics': {
                'ks_statistic': round(detail['ks_statistic'], 4),
                'p_value': round(detail['p_value'], 6),
                'mean_shift': round(detail['mean_shift'], 4),
                'std_shift': round(detail['std_shift'], 4)
            },
            'baseline': {
                'mean': round(detail['baseline_mean'], 4),
                'std': round(detail['baseline_std'], 4)
            },
            'current': {
                'mean': round(detail['current_mean'], 4),
                'std': round(detail['current_std'], 4)
            },
            'interpretation': _interpret_drift(detail),
            'checked_at': datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature drift detail failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_drift_alerts(
    severity: Optional[str] = Query(default=None, pattern="^(minor|moderate|severe)$"),
    db: Session = Depends(get_db)
):
    """
    🚨 Get active drift alerts
    
    **Alert system for production monitoring**
    
    Returns list of features/predictions with detected drift
    Filter by severity: minor, moderate, severe
    """
    try:
        drift_data = await get_drift_status(lookback_days=7, db=db)
        
        if drift_data.get('status') == 'insufficient_data':
            return {
                'alerts': [],
                'message': drift_data.get('message')
            }
        
        feature_drift = drift_data.get('feature_drift', {})
        prediction_drift = drift_data.get('prediction_drift', {})
        
        alerts = []
        
        # Feature drift alerts
        for feature_name, detail in feature_drift.items():
            if detail['is_drift']:
                if severity is None or detail['severity'] == severity:
                    alerts.append({
                        'type': 'feature_drift',
                        'feature': feature_name,
                        'severity': detail['severity'],
                        'p_value': round(detail['p_value'], 6),
                        'mean_shift': round(detail['mean_shift'], 4),
                        'message': f"Feature '{feature_name}' shows {detail['severity']} drift (p={detail['p_value']:.4f})"
                    })
        
        # Prediction drift alert
        if prediction_drift.get('is_drift'):
            alerts.append({
                'type': 'prediction_drift',
                'feature': 'predictions',
                'severity': prediction_drift['severity'],
                'psi_score': round(prediction_drift['psi_score'], 4),
                'message': f"Prediction distribution shows {prediction_drift['severity']} drift (PSI={prediction_drift['psi_score']:.4f})"
            })
        
        return {
            'alert_count': len(alerts),
            'alerts': alerts,
            'overall_status': drift_data['summary']['overall_status'],
            'checked_at': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Drift alerts failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _interpret_drift(detail: dict) -> str:
    """Generate human-readable interpretation of drift"""
    if not detail['is_drift']:
        return "No significant drift detected. Feature distribution is stable."
    
    mean_shift = detail['mean_shift']
    severity = detail['severity']
    
    if severity == 'severe':
        return (
            f"SEVERE drift detected! Mean shifted by {mean_shift:.2f}. "
            "Immediate investigation and model retraining recommended."
        )
    elif severity == 'moderate':
        return (
            f"MODERATE drift detected. Mean shifted by {mean_shift:.2f}. "
            "Monitor closely and consider retraining if trend continues."
        )
    else:
        return (
            f"MINOR drift detected. Mean shifted by {mean_shift:.2f}. "
            "Continue monitoring."
        )

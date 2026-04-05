"""
CRUD (Create, Read, Update, Delete) operations for database models.

These functions provide a clean interface for database operations.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json

import models


# ============================================================================
# USER OPERATIONS
# ============================================================================

def create_user(db: Session, email: str, username: str, hashed_password: str, 
                full_name: Optional[str] = None) -> models.User:
    """Create a new user."""
    user = models.User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        full_name=full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    """Get user by email."""
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[models.User]:
    """Get user by ID."""
    return db.query(models.User).filter(models.User.id == user_id).first()


def update_user_cluster(db: Session, user_id: str, cluster_id: int) -> models.User:
    """Update user's behavioral cluster."""
    user = get_user_by_id(db, user_id)
    if user:
        user.cluster_id = cluster_id
        db.commit()
        db.refresh(user)
    return user


# ============================================================================
# DECISION OPERATIONS
# ============================================================================

def create_decision(db: Session, user_id: str, decision_type: str = "prediction") -> models.Decision:
    """Create a new decision record."""
    decision = models.Decision(
        user_id=user_id,
        decision_type=decision_type,
        status="pending"
    )
    db.add(decision)
    db.commit()
    db.refresh(decision)
    return decision


def update_decision_outcome(db: Session, decision_id: str, outcome: str, 
                           satisfaction_score: Optional[float] = None) -> models.Decision:
    """Update decision outcome and satisfaction."""
    decision = db.query(models.Decision).filter(models.Decision.id == decision_id).first()
    if decision:
        decision.outcome = outcome
        decision.satisfaction_score = satisfaction_score
        decision.completed_at = datetime.utcnow()
        decision.status = "completed"
        db.commit()
        db.refresh(decision)
    return decision


def get_user_decisions(db: Session, user_id: str, limit: int = 50) -> List[models.Decision]:
    """Get user's decision history."""
    return db.query(models.Decision)\
        .filter(models.Decision.user_id == user_id)\
        .order_by(desc(models.Decision.created_at))\
        .limit(limit)\
        .all()


# ============================================================================
# PREDICTION LOG OPERATIONS
# ============================================================================

def create_prediction_log(db: Session, user_id: Optional[str], decision_id: Optional[str],
                         features: dict, cluster_id: int, confidence: float,
                         recommendation: Optional[str] = None,
                         shap_values: Optional[dict] = None,
                         model_version: Optional[str] = None,
                         prediction_time_ms: Optional[float] = None) -> models.PredictionLog:
    """Log a prediction."""
    prediction = models.PredictionLog(
        user_id=user_id,
        decision_id=decision_id,
        features=features,
        cluster_id=cluster_id,
        confidence=confidence,
        recommendation=recommendation,
        shap_values=shap_values,
        model_version=model_version,
        prediction_time_ms=prediction_time_ms
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_recent_predictions(db: Session, limit: int = 100) -> List[models.PredictionLog]:
    """Get recent predictions for monitoring."""
    return db.query(models.PredictionLog)\
        .order_by(desc(models.PredictionLog.created_at))\
        .limit(limit)\
        .all()


def get_predictions_by_cluster(db: Session, cluster_id: int, days: int = 7) -> List[models.PredictionLog]:
    """Get predictions for a specific cluster."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    return db.query(models.PredictionLog)\
        .filter(
            models.PredictionLog.cluster_id == cluster_id,
            models.PredictionLog.created_at >= cutoff
        )\
        .all()


# ============================================================================
# SIMULATION LOG OPERATIONS
# ============================================================================

def create_simulation_log(db: Session, user_id: Optional[str], scenario: str,
                         base_features: dict, modified_features: dict,
                         base_prediction: int, modified_prediction: int,
                         base_confidence: float, modified_confidence: float) -> models.SimulationLog:
    """Log a simulation run."""
    simulation = models.SimulationLog(
        user_id=user_id,
        scenario=scenario,
        base_features=base_features,
        modified_features=modified_features,
        base_prediction=base_prediction,
        modified_prediction=modified_prediction,
        base_confidence=base_confidence,
        modified_confidence=modified_confidence
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)
    return simulation


def get_user_simulations(db: Session, user_id: str, limit: int = 50) -> List[models.SimulationLog]:
    """Get user's simulation history."""
    return db.query(models.SimulationLog)\
        .filter(models.SimulationLog.user_id == user_id)\
        .order_by(desc(models.SimulationLog.created_at))\
        .limit(limit)\
        .all()


# ============================================================================
# A/B EXPERIMENT OPERATIONS
# ============================================================================

def create_experiment(db: Session, name: str, description: str,
                     control_variant: str = "baseline",
                     treatment_variant: str = "ml_system",
                     traffic_allocation: float = 0.5) -> models.ABExperiment:
    """Create a new A/B experiment."""
    experiment = models.ABExperiment(
        name=name,
        description=description,
        control_variant=control_variant,
        treatment_variant=treatment_variant,
        traffic_allocation=traffic_allocation,
        status="draft"
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def get_active_experiments(db: Session) -> List[models.ABExperiment]:
    """Get all running experiments."""
    return db.query(models.ABExperiment)\
        .filter(models.ABExperiment.status == "running")\
        .all()


def assign_user_to_experiment(db: Session, experiment_id: str, user_id: str, 
                             variant: str) -> models.ExperimentAssignment:
    """Assign a user to an experiment variant."""
    # Check if assignment already exists
    existing = db.query(models.ExperimentAssignment)\
        .filter(
            models.ExperimentAssignment.experiment_id == experiment_id,
            models.ExperimentAssignment.user_id == user_id
        )\
        .first()
    
    if existing:
        return existing
    
    assignment = models.ExperimentAssignment(
        experiment_id=experiment_id,
        user_id=user_id,
        variant=variant
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment


def record_experiment_conversion(db: Session, experiment_id: str, user_id: str,
                                satisfaction_score: Optional[float] = None):
    """Record a conversion for an experiment."""
    assignment = db.query(models.ExperimentAssignment)\
        .filter(
            models.ExperimentAssignment.experiment_id == experiment_id,
            models.ExperimentAssignment.user_id == user_id
        )\
        .first()
    
    if assignment:
        assignment.converted = True
        assignment.conversion_at = datetime.utcnow()
        assignment.satisfaction_score = satisfaction_score
        db.commit()
        db.refresh(assignment)
    
    return assignment


# ============================================================================
# MODEL METRICS OPERATIONS
# ============================================================================

def record_metric(db: Session, metric_name: str, metric_value: float,
                 model_version: Optional[str] = None,
                 data_window: Optional[str] = None,
                 metadata: Optional[dict] = None) -> models.ModelMetrics:
    """Record a model performance metric."""
    metric = models.ModelMetrics(
        metric_name=metric_name,
        metric_value=metric_value,
        model_version=model_version,
        data_window=data_window,
        metadata=metadata
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def get_latest_metric(db: Session, metric_name: str) -> Optional[models.ModelMetrics]:
    """Get the most recent value for a metric."""
    return db.query(models.ModelMetrics)\
        .filter(models.ModelMetrics.metric_name == metric_name)\
        .order_by(desc(models.ModelMetrics.recorded_at))\
        .first()


def get_metric_history(db: Session, metric_name: str, days: int = 7) -> List[models.ModelMetrics]:
    """Get metric history over time."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    return db.query(models.ModelMetrics)\
        .filter(
            models.ModelMetrics.metric_name == metric_name,
            models.ModelMetrics.recorded_at >= cutoff
        )\
        .order_by(models.ModelMetrics.recorded_at)\
        .all()


# ============================================================================
# ANALYTICS QUERIES
# ============================================================================

def get_decision_success_rate(db: Session, days: int = 30) -> float:
    """Calculate decision success rate."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    total = db.query(func.count(models.Decision.id))\
        .filter(
            models.Decision.completed_at >= cutoff,
            models.Decision.outcome.isnot(None)
        )\
        .scalar()
    
    if total == 0:
        return 0.0
    
    successful = db.query(func.count(models.Decision.id))\
        .filter(
            models.Decision.completed_at >= cutoff,
            models.Decision.outcome == "success"
        )\
        .scalar()
    
    return (successful / total) * 100.0


def get_average_satisfaction(db: Session, days: int = 30) -> float:
    """Calculate average user satisfaction."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    avg = db.query(func.avg(models.Decision.satisfaction_score))\
        .filter(
            models.Decision.completed_at >= cutoff,
            models.Decision.satisfaction_score.isnot(None)
        )\
        .scalar()
    
    return float(avg) if avg else 0.0


def get_active_users_count(db: Session, days: int = 30) -> int:
    """Count active users."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    return db.query(func.count(func.distinct(models.Decision.user_id)))\
        .filter(models.Decision.created_at >= cutoff)\
        .scalar()


def get_conversion_rate(db: Session, days: int = 30) -> float:
    """Calculate conversion rate."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    total_decisions = db.query(func.count(models.Decision.id))\
        .filter(models.Decision.created_at >= cutoff)\
        .scalar()
    
    if total_decisions == 0:
        return 0.0
    
    completed = db.query(func.count(models.Decision.id))\
        .filter(
            models.Decision.created_at >= cutoff,
            models.Decision.status == "completed"
        )\
        .scalar()
    
    return (completed / total_decisions) * 100.0


def get_predictions_count(db: Session, days: int = 1) -> int:
    """Count predictions in time window."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    return db.query(func.count(models.PredictionLog.id))\
        .filter(models.PredictionLog.created_at >= cutoff)\
        .scalar()


def get_average_confidence(db: Session, days: int = 7) -> float:
    """Calculate average prediction confidence."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    avg = db.query(func.avg(models.PredictionLog.confidence))\
        .filter(models.PredictionLog.created_at >= cutoff)\
        .scalar()
    
    return float(avg) if avg else 0.0


def get_cluster_distribution(db: Session, days: int = 7) -> Dict[int, int]:
    """Get distribution of predictions across clusters."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    results = db.query(
        models.PredictionLog.cluster_id,
        func.count(models.PredictionLog.id)
    )\
        .filter(models.PredictionLog.created_at >= cutoff)\
        .group_by(models.PredictionLog.cluster_id)\
        .all()
    
    return {cluster_id: count for cluster_id, count in results}

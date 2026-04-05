"""
SQLAlchemy models for VisionX database.

Tables:
- users: User accounts
- decisions: User decision history
- predictions_log: ML prediction tracking
- ab_experiments: A/B test experiments
- experiment_assignments: User assignments to experiments
- model_metrics: Model performance tracking
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from database import Base


def generate_uuid():
    """Generate UUID for primary keys."""
    return str(uuid.uuid4())


class User(Base):
    """User accounts and profiles."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    cluster_id = Column(Integer, nullable=True)  # Behavioral cluster (0-3)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    decisions = relationship("Decision", back_populates="user")
    predictions = relationship("PredictionLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, cluster={self.cluster_id})>"


class Decision(Base):
    """User decision history and outcomes."""
    __tablename__ = "decisions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Decision metadata
    decision_type = Column(String, nullable=True)  # e.g., "purchase", "comparison"
    status = Column(String, default="pending")  # pending, accepted, rejected, completed
    
    # Outcome tracking
    outcome = Column(String, nullable=True)  # success, failure, abandoned
    satisfaction_score = Column(Float, nullable=True)  # 1-5 rating
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), index=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="decisions")
    predictions = relationship("PredictionLog", back_populates="decision")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),
        Index('idx_status_created', 'status', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Decision(id={self.id}, user_id={self.user_id}, status={self.status})>"


class PredictionLog(Base):
    """ML prediction tracking and monitoring."""
    __tablename__ = "predictions_log"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    decision_id = Column(String, ForeignKey("decisions.id"), nullable=True, index=True)
    
    # Input features (stored as JSON)
    features = Column(JSON, nullable=False)  # {price, quality_score, delivery_time, etc.}
    
    # Prediction outputs
    cluster_id = Column(Integer, nullable=False)  # Predicted cluster (0-3)
    confidence = Column(Float, nullable=False)  # Prediction confidence (0-1)
    recommendation = Column(Text, nullable=True)  # Generated recommendation text
    
    # SHAP values (for explainability)
    shap_values = Column(JSON, nullable=True)  # Feature importance scores
    
    # Model metadata
    model_version = Column(String, nullable=True)  # Track which model version was used
    prediction_time_ms = Column(Float, nullable=True)  # Latency tracking
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", back_populates="predictions")
    decision = relationship("Decision", back_populates="predictions")
    
    # Indexes for analytics queries
    __table_args__ = (
        Index('idx_cluster_created', 'cluster_id', 'created_at'),
        Index('idx_confidence_created', 'confidence', 'created_at'),
    )
    
    def __repr__(self):
        return f"<PredictionLog(id={self.id}, cluster={self.cluster_id}, confidence={self.confidence:.2f})>"


class ABExperiment(Base):
    """A/B test experiment definitions."""
    __tablename__ = "ab_experiments"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    
    # Experiment configuration
    control_variant = Column(String, default="baseline")  # Control group name
    treatment_variant = Column(String, default="ml_system")  # Treatment group name
    traffic_allocation = Column(Float, default=0.5)  # % of traffic to treatment (0-1)
    
    # Status
    status = Column(String, default="draft")  # draft, running, paused, completed
    
    # Timestamps
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    assignments = relationship("ExperimentAssignment", back_populates="experiment")
    
    def __repr__(self):
        return f"<ABExperiment(id={self.id}, name={self.name}, status={self.status})>"


class ExperimentAssignment(Base):
    """User assignments to A/B test experiments."""
    __tablename__ = "experiment_assignments"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    experiment_id = Column(String, ForeignKey("ab_experiments.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    
    # Assignment details
    variant = Column(String, nullable=False)  # control or treatment
    
    # Outcome tracking
    converted = Column(Boolean, default=False)  # Did user complete desired action?
    satisfaction_score = Column(Float, nullable=True)  # Post-experiment rating
    
    # Timestamps
    assigned_at = Column(DateTime, server_default=func.now())
    conversion_at = Column(DateTime, nullable=True)
    
    # Relationships
    experiment = relationship("ABExperiment", back_populates="assignments")
    
    # Unique constraint: one assignment per user per experiment
    __table_args__ = (
        Index('idx_experiment_user', 'experiment_id', 'user_id', unique=True),
        Index('idx_experiment_variant', 'experiment_id', 'variant'),
    )
    
    def __repr__(self):
        return f"<ExperimentAssignment(experiment={self.experiment_id}, variant={self.variant})>"


class ModelMetrics(Base):
    """Model performance metrics tracking."""
    __tablename__ = "model_metrics"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    
    # Metric details
    metric_name = Column(String, nullable=False, index=True)  # accuracy, drift_score, etc.
    metric_value = Column(Float, nullable=False)
    
    # Context
    model_version = Column(String, nullable=True)
    data_window = Column(String, nullable=True)  # e.g., "last_24h", "last_7d"
    
    # Metadata
    extra_data = Column(JSON, nullable=True)  # Additional context (sample_size, etc.)
    
    # Timestamp
    recorded_at = Column(DateTime, server_default=func.now(), index=True)
    
    # Indexes for time-series queries
    __table_args__ = (
        Index('idx_metric_recorded', 'metric_name', 'recorded_at'),
    )
    
    def __repr__(self):
        return f"<ModelMetrics(metric={self.metric_name}, value={self.metric_value:.3f})>"


class SimulationLog(Base):
    """Decision simulation tracking."""
    __tablename__ = "simulation_logs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    
    # Simulation details
    scenario = Column(String, nullable=False)  # scenario name (increase_budget, etc.)
    base_features = Column(JSON, nullable=False)  # Original features
    modified_features = Column(JSON, nullable=False)  # Modified features
    
    # Results
    base_prediction = Column(Integer, nullable=False)  # Original cluster
    modified_prediction = Column(Integer, nullable=False)  # New cluster
    base_confidence = Column(Float, nullable=False)
    modified_confidence = Column(Float, nullable=False)
    
    # Timestamp
    created_at = Column(DateTime, server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<SimulationLog(scenario={self.scenario}, {self.base_prediction}→{self.modified_prediction})>"


# Create indexes after all models are defined
def create_indexes():
    """Create additional indexes for performance."""
    # This function can be called after init_db() if needed
    pass

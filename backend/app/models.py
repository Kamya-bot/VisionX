"""
SQLAlchemy models for VisionX database.

Tables:
- users               — User accounts
- refresh_tokens      — JWT refresh token store (rotation on use)
- decisions           — User decision history
- predictions_log     — ML prediction tracking
- outcome_feedback    — User feedback on recommendations (feeds retraining)
- ab_experiments      — A/B test experiments
- experiment_assignments
- model_metrics       — Model performance tracking
- simulation_logs
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, JSON, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


# ── Users ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=True)   # nullable for OAuth-only users
    full_name = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)        # from OAuth provider
    oauth_provider = Column(String, nullable=True)    # "google" | "github" | None
    oauth_sub = Column(String, nullable=True, index=True)  # provider's user ID
    cluster_id = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    decisions = relationship("Decision", back_populates="user")
    predictions = relationship("PredictionLog", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    outcome_feedbacks = relationship("OutcomeFeedback", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, cluster={self.cluster_id})>"


# ── Refresh tokens ────────────────────────────────────────────────────────────

class RefreshToken(Base):
    """
    Stores hashed refresh tokens for rotation-on-use strategy.
    On each /auth/refresh call:
      1. Look up token by jti (JWT ID)
      2. Verify it hasn't been used or revoked
      3. Mark it used=True
      4. Issue new access token + new refresh token
    This detects refresh token theft — if a stolen token is replayed,
    the legitimate user's new token will not match, and we revoke the family.
    """
    __tablename__ = "refresh_tokens"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    jti = Column(String, unique=True, nullable=False, index=True)   # JWT ID claim
    token_hash = Column(String, nullable=False)                     # SHA-256 of raw token
    family_id = Column(String, nullable=False, index=True)          # rotation family
    used = Column(Boolean, default=False, nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    used_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="refresh_tokens")

    __table_args__ = (
        Index("idx_rt_jti", "jti"),
        Index("idx_rt_family", "family_id"),
        Index("idx_rt_user_active", "user_id", "used", "revoked"),
    )

    def __repr__(self):
        return f"<RefreshToken(user={self.user_id}, used={self.used}, revoked={self.revoked})>"


# ── Decisions ─────────────────────────────────────────────────────────────────

class Decision(Base):
    __tablename__ = "decisions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    decision_type = Column(String, nullable=True)
    status = Column(String, default="pending")
    outcome = Column(String, nullable=True)
    satisfaction_score = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="decisions")
    predictions = relationship("PredictionLog", back_populates="decision")

    __table_args__ = (
        Index("idx_user_created", "user_id", "created_at"),
        Index("idx_status_created", "status", "created_at"),
    )

    def __repr__(self):
        return f"<Decision(id={self.id}, user_id={self.user_id}, status={self.status})>"


# ── Prediction log ────────────────────────────────────────────────────────────

class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    decision_id = Column(String, ForeignKey("decisions.id"), nullable=True, index=True)

    # Input
    features = Column(JSON, nullable=False)
    domain_detected = Column(String, nullable=True)      # products/jobs/education/housing
    options_count = Column(Integer, nullable=True)

    # Output
    recommended_option_id = Column(String, nullable=True)
    recommended_option_name = Column(String, nullable=True)
    cluster_id = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    recommendation = Column(Text, nullable=True)

    # Explainability
    shap_values = Column(JSON, nullable=True)
    universal_features = Column(JSON, nullable=True)    # the 6-dim vector used

    # MLOps
    model_version = Column(String, nullable=True)
    prediction_time_ms = Column(Float, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), index=True)

    user = relationship("User", back_populates="predictions")
    decision = relationship("Decision", back_populates="predictions")
    feedback = relationship("OutcomeFeedback", back_populates="prediction", uselist=False)

    __table_args__ = (
        Index("idx_cluster_created", "cluster_id", "created_at"),
        Index("idx_confidence_created", "confidence", "created_at"),
        Index("idx_domain_created", "domain_detected", "created_at"),
    )

    def __repr__(self):
        return f"<PredictionLog(id={self.id}, cluster={self.cluster_id}, conf={self.confidence:.2f})>"


# ── Outcome feedback ──────────────────────────────────────────────────────────

class OutcomeFeedback(Base):
    """
    Stores whether the user accepted the ML recommendation.
    This is the ground truth label used for model retraining.
    Schema: prediction_id → (accepted, satisfaction, actual_choice_id)
    """
    __tablename__ = "outcome_feedback"

    id = Column(String, primary_key=True, default=generate_uuid)
    prediction_id = Column(String, ForeignKey("predictions_log.id"), nullable=False, unique=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)

    accepted = Column(Boolean, nullable=False)               # followed recommendation?
    satisfaction = Column(Float, nullable=True)              # 1–5 stars
    actual_choice_id = Column(String, nullable=True)         # what they actually picked

    # Denormalised for fast retraining queries
    features_snapshot = Column(JSON, nullable=True)          # copy of universal_features
    cluster_id_at_prediction = Column(Integer, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), index=True)

    prediction = relationship("PredictionLog", back_populates="feedback")
    user = relationship("User", back_populates="outcome_feedbacks")

    __table_args__ = (
        Index("idx_feedback_accepted", "accepted", "created_at"),
        Index("idx_feedback_user", "user_id", "created_at"),
    )

    def __repr__(self):
        return f"<OutcomeFeedback(prediction={self.prediction_id}, accepted={self.accepted})>"


# ── A/B Experiments ───────────────────────────────────────────────────────────

class ABExperiment(Base):
    __tablename__ = "ab_experiments"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    control_variant = Column(String, default="baseline")
    treatment_variant = Column(String, default="ml_system")
    traffic_allocation = Column(Float, default=0.5)
    status = Column(String, default="draft")
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    assignments = relationship("ExperimentAssignment", back_populates="experiment")

    def __repr__(self):
        return f"<ABExperiment(name={self.name}, status={self.status})>"


class ExperimentAssignment(Base):
    __tablename__ = "experiment_assignments"

    id = Column(String, primary_key=True, default=generate_uuid)
    experiment_id = Column(String, ForeignKey("ab_experiments.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    variant = Column(String, nullable=False)
    converted = Column(Boolean, default=False)
    satisfaction_score = Column(Float, nullable=True)
    assigned_at = Column(DateTime, server_default=func.now())
    conversion_at = Column(DateTime, nullable=True)

    experiment = relationship("ABExperiment", back_populates="assignments")

    __table_args__ = (
        Index("idx_experiment_user", "experiment_id", "user_id", unique=True),
        Index("idx_experiment_variant", "experiment_id", "variant"),
    )


# ── Model metrics ─────────────────────────────────────────────────────────────

class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(String, primary_key=True, default=generate_uuid)
    metric_name = Column(String, nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    model_version = Column(String, nullable=True)
    data_window = Column(String, nullable=True)
    extra_data = Column(JSON, nullable=True)
    recorded_at = Column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index("idx_metric_recorded", "metric_name", "recorded_at"),
    )

    def __repr__(self):
        return f"<ModelMetrics(metric={self.metric_name}, value={self.metric_value:.3f})>"


# ── Simulation logs ───────────────────────────────────────────────────────────

class SimulationLog(Base):
    __tablename__ = "simulation_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    scenario = Column(String, nullable=False)
    base_features = Column(JSON, nullable=False)
    modified_features = Column(JSON, nullable=False)
    base_prediction = Column(Integer, nullable=False)
    modified_prediction = Column(Integer, nullable=False)
    base_confidence = Column(Float, nullable=False)
    modified_confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    def __repr__(self):
        return f"<SimulationLog(scenario={self.scenario}, {self.base_prediction}→{self.modified_prediction})>"
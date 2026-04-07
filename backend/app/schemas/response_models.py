"""
Response Models (Pydantic Schemas)
Define output structure for API endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ClusterResponse(BaseModel):
    """User cluster response"""

    user_id: str = Field(..., description="User identifier")
    cluster_id: int = Field(..., description="Cluster ID (0-3)")
    cluster_label: str = Field(..., description="Human-readable cluster label")
    confidence: float = Field(..., description="Prediction confidence score")
    characteristics: List[str] = Field(..., description="Cluster characteristics")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FeatureImportance(BaseModel):
    """Feature importance for predictions"""
    feature_name: str
    importance: float


class PredictionResponse(BaseModel):
    """Prediction response"""

    recommended_option_id: str
    recommended_option_name: str
    confidence: float
    reasoning: str
    alternative_options: List[Dict[str, Any]] = Field(default_factory=list)
    feature_importance: List[FeatureImportance] = Field(default_factory=list)
    user_cluster: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RecommendationItem(BaseModel):
    """Single recommendation item"""
    option_id: str
    option_name: str
    similarity_score: float
    reason: str
    estimated_satisfaction: float


class RecommendationResponse(BaseModel):
    """Recommendation response"""
    user_id: str
    current_option_id: str
    recommendations: List[RecommendationItem]
    total_analyzed: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class InsightItem(BaseModel):
    """Single insight item"""
    insight_id: str
    title: str
    description: str
    impact_score: float = Field(..., ge=0.0, le=1.0)
    category: str
    actionable: bool = Field(default=True)


class InsightsResponse(BaseModel):
    """AI-generated insights response"""
    user_id: str
    insights: List[InsightItem]
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class PatternItem(BaseModel):
    """Decision pattern item"""
    pattern_name: str
    frequency: float = Field(..., ge=0.0, le=1.0)
    description: str


class PatternsResponse(BaseModel):
    """Decision patterns response"""
    user_id: str
    patterns: List[PatternItem]
    analyzed_decisions: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    status_code: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = None


# ── Single unified analytics/KPI/AB response models ──────────────────────────

class AnalyticsResponse(BaseModel):
    """Generic analytics response — used by both ml/analytics and analytics/* routes"""
    status: str = "success"
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class KPIResponse(BaseModel):
    """KPI tracking response"""
    status: str = "success"
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ABTestResponse(BaseModel):
    """A/B testing response"""
    status: str = "success"
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
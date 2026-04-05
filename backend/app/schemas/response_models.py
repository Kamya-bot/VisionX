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
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "cluster_id": 2,
                "cluster_label": "High Intent Buyer",
                "confidence": 0.91,
                "characteristics": ["focused", "intent-driven", "decisive"],
                "timestamp": "2024-03-14T10:30:00Z"
            }
        }


class FeatureImportance(BaseModel):
    """Feature importance for predictions"""
    
    feature_name: str
    importance: float


class PredictionResponse(BaseModel):
    """Prediction response"""
    
    recommended_option_id: str = Field(..., description="ID of recommended option")
    recommended_option_name: str = Field(..., description="Name of recommended option")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    reasoning: str = Field(..., description="Explanation of recommendation")
    alternative_options: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative options with scores")
    feature_importance: List[FeatureImportance] = Field(default_factory=list, description="Feature importance")
    user_cluster: Optional[str] = Field(None, description="User's behavioral cluster")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "recommended_option_id": "opt_2",
                "recommended_option_name": "Option B - Pro Package",
                "confidence": 0.87,
                "reasoning": "Best overall score based on quality and features",
                "alternative_options": [
                    {
                        "id": "opt_1",
                        "name": "Option A",
                        "score": 0.82,
                        "reason": "Lower price but slightly lower quality"
                    }
                ],
                "feature_importance": [
                    {"feature_name": "quality_score", "importance": 0.35},
                    {"feature_name": "features", "importance": 0.29},
                    {"feature_name": "price", "importance": 0.23},
                    {"feature_name": "brand", "importance": 0.13}
                ],
                "user_cluster": "High Intent Buyer",
                "timestamp": "2024-03-14T10:30:00Z"
            }
        }


class RecommendationItem(BaseModel):
    """Single recommendation item"""
    
    option_id: str = Field(..., description="Option identifier")
    option_name: str = Field(..., description="Option name")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    reason: str = Field(..., description="Reason for recommendation")
    estimated_satisfaction: float = Field(..., description="Estimated user satisfaction (0-1)")


class RecommendationResponse(BaseModel):
    """Recommendation response"""
    
    user_id: str = Field(..., description="User identifier")
    current_option_id: str = Field(..., description="Current option ID")
    recommendations: List[RecommendationItem] = Field(..., description="Recommended alternatives")
    total_analyzed: int = Field(..., description="Total options analyzed")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "current_option_id": "opt_1",
                "recommendations": [
                    {
                        "option_id": "opt_3",
                        "option_name": "Option C",
                        "similarity_score": 0.92,
                        "reason": "Similar features with better quality",
                        "estimated_satisfaction": 0.88
                    },
                    {
                        "option_id": "opt_5",
                        "option_name": "Option E",
                        "similarity_score": 0.87,
                        "reason": "Lower price, comparable quality",
                        "estimated_satisfaction": 0.84
                    }
                ],
                "total_analyzed": 8,
                "timestamp": "2024-03-14T10:30:00Z"
            }
        }


class AnalyticsResponse(BaseModel):
    """Analytics and insights response"""
    
    user_cluster_distribution: Dict[str, float] = Field(..., description="Distribution of user clusters")
    average_decision_time: float = Field(..., description="Average decision time in seconds")
    popular_categories: List[str] = Field(..., description="Most popular categories")
    conversion_rate: float = Field(..., description="Overall conversion rate")
    engagement_metrics: Dict[str, float] = Field(..., description="Engagement metrics")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_cluster_distribution": {
                    "Casual User": 0.30,
                    "Analytical Researcher": 0.25,
                    "High Intent Buyer": 0.25,
                    "Power Decision Maker": 0.20
                },
                "average_decision_time": 720.5,
                "popular_categories": ["Electronics", "Software", "Services"],
                "conversion_rate": 0.68,
                "engagement_metrics": {
                    "avg_scroll_depth": 0.72,
                    "avg_clicks": 15.3,
                    "avg_comparisons": 3.8
                },
                "timestamp": "2024-03-14T10:30:00Z"
            }
        }


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
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "insights": [
                    {
                        "insight_id": "insight_001",
                        "title": "Price Sensitivity Detected",
                        "description": "User shows high price sensitivity. Consider showing value-focused options first.",
                        "impact_score": 0.82,
                        "category": "pricing",
                        "actionable": True
                    },
                    {
                        "insight_id": "insight_002",
                        "title": "Feature-Focused Behavior",
                        "description": "User spends more time reviewing features than other factors.",
                        "impact_score": 0.75,
                        "category": "behavior",
                        "actionable": True
                    }
                ],
                "generated_at": "2024-03-14T10:30:00Z"
            }
        }


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
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "patterns": [
                    {
                        "pattern_name": "Quality-First",
                        "frequency": 0.65,
                        "description": "Prioritizes quality over price in most decisions"
                    },
                    {
                        "pattern_name": "Feature-Rich",
                        "frequency": 0.55,
                        "description": "Prefers options with more features"
                    },
                    {
                        "pattern_name": "Brand-Conscious",
                        "frequency": 0.42,
                        "description": "Shows preference for known brands"
                    }
                ],
                "analyzed_decisions": 24,
                "timestamp": "2024-03-14T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# ========== Analytics Response Models ==========

class AnalyticsResponse(BaseModel):
    """Generic analytics response"""
    status: str
    data: dict
    timestamp: str


class KPIResponse(BaseModel):
    """KPI tracking response"""
    status: str
    data: dict
    timestamp: str


class ABTestResponse(BaseModel):
    """A/B testing response"""
    status: str
    data: dict
    timestamp: str

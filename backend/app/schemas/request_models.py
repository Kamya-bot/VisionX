"""
Request Models (Pydantic Schemas)
Define input validation for API endpoints.
All feature fields have strict min/max bounds — garbage in = 422 error, not silent garbage out.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class DeviceType(str, Enum):
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"


class UserBehaviorInput(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    session_time: int = Field(..., ge=0, le=86400, description="Session time in seconds (max 24h)")
    clicks: int = Field(..., ge=0, le=10000)
    scroll_depth: float = Field(..., ge=0.0, le=1.0)
    categories_viewed: int = Field(..., ge=0, le=100)
    comparison_count: int = Field(..., ge=0, le=1000)
    product_views: int = Field(..., ge=0, le=10000)
    decision_time: int = Field(..., ge=0, le=86400)
    price_sensitivity: float = Field(..., ge=0.0, le=1.0)
    feature_interest_score: float = Field(..., ge=0.0, le=1.0)
    device_type: DeviceType
    previous_decisions: int = Field(..., ge=0, le=100000)
    engagement_score: float = Field(..., ge=0.0, le=1.0)
    purchase_intent_score: float = Field(..., ge=0.0, le=1.0)


class OptionFeatures(BaseModel):
    """
    Features for any domain option.
    All values are strictly bounded — any out-of-range input returns HTTP 422
    before reaching the model.
    """
    # Product / generic fields
    price: Optional[float] = Field(None, gt=0, le=100_000_000, description="Price (must be positive)")
    quality_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    feature_count: Optional[int] = Field(None, ge=0, le=10000)
    brand_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    availability: Optional[float] = Field(None, ge=0.0, le=1.0)
    delivery_time: Optional[float] = Field(None, ge=0.0, le=365.0, description="Days")
    rating: Optional[float] = Field(None, ge=0.0, le=10.0)

    # Job fields
    salary: Optional[float] = Field(None, ge=0.0, le=10_000_000)
    max_salary: Optional[float] = Field(None, ge=0.0, le=10_000_000)
    min_salary: Optional[float] = Field(None, ge=0.0, le=10_000_000)
    company_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    seniority_level: Optional[str] = Field(None, max_length=50)
    company_size: Optional[str] = Field(None, max_length=50)
    remote: Optional[str] = Field(None, max_length=20)
    benefits_score: Optional[float] = Field(None, ge=0.0, le=10.0)

    # Education fields
    tuition: Optional[float] = Field(None, ge=0.0, le=1_000_000)
    ranking: Optional[float] = Field(None, ge=1.0, le=10000.0)
    world_rank: Optional[float] = Field(None, ge=1.0, le=10000.0)
    research_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    teaching_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    acceptance_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    citations: Optional[float] = Field(None, ge=0.0, le=100.0)
    total_score: Optional[float] = Field(None, ge=0.0, le=100.0)

    # Housing fields
    area: Optional[float] = Field(None, ge=0.0, le=1_000_000, description="Square feet")
    sqft: Optional[float] = Field(None, ge=0.0, le=1_000_000)
    bedrooms: Optional[int] = Field(None, ge=0, le=100)
    bathrooms: Optional[float] = Field(None, ge=0.0, le=100.0)
    year_built: Optional[int] = Field(None, ge=1800, le=2030)
    garage: Optional[float] = Field(None, ge=0.0, le=20.0)
    overall_quality: Optional[float] = Field(None, ge=0.0, le=10.0)
    neighborhood_score: Optional[float] = Field(None, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def at_least_one_feature(self) -> "OptionFeatures":
        values = self.model_dump(exclude_none=True)
        if not values:
            raise ValueError("At least one feature must be provided")
        return self

    class Config:
        extra = "allow"   # allow domain-specific keys not listed above


class ComparisonOption(BaseModel):
    id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    features: OptionFeatures

    @field_validator("id")
    @classmethod
    def id_no_sql(cls, v: str) -> str:
        # Basic injection guard
        forbidden = ["'", '"', ";", "--", "/*", "*/", "xp_"]
        for f in forbidden:
            if f in v:
                raise ValueError(f"Invalid character in option id: {f}")
        return v


class PredictionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=200)
    options: List[ComparisonOption] = Field(..., min_length=2, max_length=10)
    user_behavior: Optional[UserBehaviorInput] = None

    @field_validator("options")
    @classmethod
    def unique_option_ids(cls, v: List[ComparisonOption]) -> List[ComparisonOption]:
        ids = [o.id for o in v]
        if len(ids) != len(set(ids)):
            raise ValueError("All option IDs must be unique")
        return v


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=200)
    current_option_id: str = Field(..., min_length=1, max_length=100)
    available_options: List[ComparisonOption] = Field(..., min_length=1, max_length=20)
    top_k: int = Field(default=5, ge=1, le=10)


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(..., min_length=1, max_length=100)


# ── Outcome feedback (for retraining loop) ───────────────────────────────────

class OutcomeFeedback(BaseModel):
    """
    Posted by the frontend after user acts on a recommendation.
    This is what feeds the retraining pipeline.
    """
    prediction_id: str = Field(..., description="ID of the PredictionLog row")
    accepted: bool = Field(..., description="Did user follow the recommendation?")
    satisfaction: Optional[float] = Field(
        None, ge=1.0, le=5.0,
        description="Optional 1–5 star rating"
    )
    actual_choice_id: Optional[str] = Field(
        None, max_length=100,
        description="Which option the user actually chose (may differ from recommendation)"
    )
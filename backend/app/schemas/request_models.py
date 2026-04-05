"""
Request Models (Pydantic Schemas)
Define input validation for API endpoints
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class DeviceType(str, Enum):
    """Device type enum"""
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"


class UserBehaviorInput(BaseModel):
    """User behavioral data input"""
    
    user_id: str = Field(..., description="Unique user identifier")
    session_time: int = Field(..., ge=0, description="Session time in seconds")
    clicks: int = Field(..., ge=0, description="Number of clicks")
    scroll_depth: float = Field(..., ge=0.0, le=1.0, description="Scroll depth percentage")
    categories_viewed: int = Field(..., ge=0, description="Number of categories viewed")
    comparison_count: int = Field(..., ge=0, description="Number of comparisons made")
    product_views: int = Field(..., ge=0, description="Number of products viewed")
    decision_time: int = Field(..., ge=0, description="Decision time in seconds")
    price_sensitivity: float = Field(..., ge=0.0, le=1.0, description="Price sensitivity score")
    feature_interest_score: float = Field(..., ge=0.0, le=1.0, description="Feature interest score")
    device_type: DeviceType = Field(..., description="Device type")
    previous_decisions: int = Field(..., ge=0, description="Number of previous decisions")
    engagement_score: float = Field(..., ge=0.0, le=1.0, description="Engagement score")
    purchase_intent_score: float = Field(..., ge=0.0, le=1.0, description="Purchase intent score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "session_time": 450,
                "clicks": 18,
                "scroll_depth": 0.75,
                "categories_viewed": 3,
                "comparison_count": 4,
                "product_views": 10,
                "decision_time": 600,
                "price_sensitivity": 0.65,
                "feature_interest_score": 0.82,
                "device_type": "desktop",
                "previous_decisions": 12,
                "engagement_score": 0.78,
                "purchase_intent_score": 0.85
            }
        }


class OptionFeatures(BaseModel):
    """Features of a comparison option"""
    
    price: float = Field(..., gt=0, description="Price of the option")
    quality_score: float = Field(..., ge=0.0, le=10.0, description="Quality score (0-10)")
    feature_count: int = Field(..., ge=0, description="Number of features")
    brand_score: float = Field(default=5.0, ge=0.0, le=10.0, description="Brand reputation score")
    availability: float = Field(default=1.0, ge=0.0, le=1.0, description="Availability score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "price": 299.99,
                "quality_score": 8.5,
                "feature_count": 12,
                "brand_score": 7.8,
                "availability": 0.95
            }
        }


class ComparisonOption(BaseModel):
    """A single comparison option"""
    
    id: str = Field(..., description="Option identifier")
    name: str = Field(..., description="Option name")
    features: OptionFeatures = Field(..., description="Option features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "opt_1",
                "name": "Option A - Premium Package",
                "features": {
                    "price": 299.99,
                    "quality_score": 8.5,
                    "feature_count": 12,
                    "brand_score": 7.8,
                    "availability": 0.95
                }
            }
        }


class PredictionRequest(BaseModel):
    """Request for prediction"""
    
    user_id: str = Field(..., description="User identifier")
    options: List[ComparisonOption] = Field(..., min_length=2, description="Options to compare")
    user_behavior: Optional[UserBehaviorInput] = Field(None, description="Optional user behavior data")
    
    @validator('options')
    def validate_options(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 options required for comparison")
        if len(v) > 10:
            raise ValueError("Maximum 10 options allowed")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "options": [
                    {
                        "id": "opt_1",
                        "name": "Option A",
                        "features": {
                            "price": 299.99,
                            "quality_score": 8.5,
                            "feature_count": 12,
                            "brand_score": 7.8,
                            "availability": 0.95
                        }
                    },
                    {
                        "id": "opt_2",
                        "name": "Option B",
                        "features": {
                            "price": 399.99,
                            "quality_score": 9.2,
                            "feature_count": 15,
                            "brand_score": 8.5,
                            "availability": 0.88
                        }
                    }
                ]
            }
        }


class RecommendationRequest(BaseModel):
    """Request for recommendations"""
    
    user_id: str = Field(..., description="User identifier")
    current_option_id: str = Field(..., description="Current option being considered")
    available_options: List[ComparisonOption] = Field(..., description="Available alternatives")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of recommendations to return")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "current_option_id": "opt_1",
                "available_options": [
                    {
                        "id": "opt_2",
                        "name": "Option B",
                        "features": {
                            "price": 399.99,
                            "quality_score": 9.2,
                            "feature_count": 15
                        }
                    }
                ],
                "top_k": 3
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    
    predictions: List[PredictionRequest] = Field(..., max_length=100, description="Batch of prediction requests")
    
    @validator('predictions')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum batch size is 100")
        return v

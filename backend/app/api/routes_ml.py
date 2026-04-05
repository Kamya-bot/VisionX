"""
Machine Learning API Routes
Core ML endpoints for predictions, clustering, and recommendations
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import time

from schemas.request_models import (
    UserBehaviorInput,
    PredictionRequest,
    RecommendationRequest,
    BatchPredictionRequest
)
from schemas.response_models import (
    ClusterResponse,
    PredictionResponse,
    RecommendationResponse,
    AnalyticsResponse,
    InsightsResponse,
    PatternsResponse,
    FeatureImportance,
    RecommendationItem,
    InsightItem,
    PatternItem
)
from config import settings
from database import get_db
from crud import create_prediction_log, create_simulation_log, record_metric


router = APIRouter()
logger = logging.getLogger(__name__)


def get_model_store(request: Request):
    """Dependency to get model store"""
    return request.app.state.model_store


def validate_models_loaded(model_store):
    """Validate that models are loaded"""
    if not model_store.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train models first: python training/train_models.py"
        )


@router.get("/ml/user-cluster", response_model=ClusterResponse)
async def get_user_cluster(
    user_id: str,
    request: Request
):
    """
    Get user's behavioral cluster
    
    Classifies user into one of 4 behavioral segments:
    - Casual User
    - Analytical Researcher
    - High Intent Buyer
    - Power Decision Maker
    """
    
    model_store = get_model_store(request)
    validate_models_loaded(model_store)
    
    try:
        # For demo purposes, generate sample behavior data
        # In production, this would come from user's actual behavior
        sample_features = generate_sample_user_features(user_id)
        
        # Preprocess features
        X = preprocess_user_data(sample_features, model_store)
        
        # Predict cluster
        cluster_id = int(model_store.clustering_model.predict(X)[0])
        
        # Get cluster probabilities for confidence
        distances = model_store.clustering_model.transform(X)[0]
        confidence = float(1.0 / (1.0 + min(distances)))
        
        # Get cluster label and characteristics
        cluster_label = settings.CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
        characteristics = settings.CLUSTER_CHARACTERISTICS.get(cluster_id, [])
        
        logger.info(f"User {user_id} classified as {cluster_label} (confidence: {confidence:.2f})")
        
        return ClusterResponse(
            user_id=user_id,
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            confidence=round(confidence, 2),
            characteristics=characteristics
        )
        
    except Exception as e:
        logger.error(f"Error in user clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")


@router.post("/ml/predict", response_model=PredictionResponse)
async def predict_best_option(
    request_data: PredictionRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Predict best option for user
    
    Analyzes comparison options and recommends the best choice
    based on user behavior and preferences
    
    **Database Integration:** Saves prediction to database for analytics
    """
    
    model_store = get_model_store(request)
    validate_models_loaded(model_store)
    
    start_time = time.time()
    
    try:
        # Extract options
        options = request_data.options
        
        if len(options) < 2:
            raise HTTPException(status_code=400, detail="At least 2 options required")
        
        # Score each option
        option_scores = []
        for option in options:
            score = calculate_option_score(option, request_data.user_id, model_store)
            option_scores.append({
                "id": option.id,
                "name": option.name,
                "score": score,
                "features": option.features.dict()
            })
        
        # Sort by score
        option_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Get top recommendation
        best_option = option_scores[0]
        alternatives = option_scores[1:4]  # Top 3 alternatives
        
        # Calculate feature importance
        feature_importance = calculate_feature_importance(best_option["features"])
        
        # Get user cluster for context
        sample_features = generate_sample_user_features(request_data.user_id)
        X = preprocess_user_data(sample_features, model_store)
        cluster_id = int(model_store.clustering_model.predict(X)[0])
        user_cluster = settings.CLUSTER_LABELS.get(cluster_id)
        
        # Generate reasoning
        reasoning = generate_recommendation_reasoning(best_option, user_cluster)
        
        # Calculate prediction time
        prediction_time_ms = (time.time() - start_time) * 1000
        
        # ✅ SAVE TO DATABASE
        try:
            create_prediction_log(
                db=db,
                user_id=request_data.user_id,
                decision_id=None,  # Can be linked if decision tracking is added
                features=best_option["features"],
                cluster_id=cluster_id,
                confidence=best_option["score"],
                recommendation=reasoning,
                shap_values=feature_importance,  # Store as SHAP-like values
                model_version=settings.APP_VERSION,
                prediction_time_ms=prediction_time_ms
            )
            logger.info(f"✅ Prediction saved to database for user {request_data.user_id}")
        except Exception as db_error:
            logger.warning(f"⚠️  Failed to save prediction to database: {str(db_error)}")
            # Continue even if DB save fails
        
        logger.info(f"Predicted best option for user {request_data.user_id}: {best_option['id']} ({prediction_time_ms:.1f}ms)")
        
        return PredictionResponse(
            recommended_option_id=best_option["id"],
            recommended_option_name=best_option["name"],
            confidence=round(best_option["score"], 2),
            reasoning=reasoning,
            alternative_options=[
                {
                    "id": alt["id"],
                    "name": alt["name"],
                    "score": round(alt["score"], 2),
                    "reason": f"Alternative with score {alt['score']:.2f}"
                }
                for alt in alternatives
            ],
            feature_importance=[
                FeatureImportance(feature_name=k, importance=round(v, 2))
                for k, v in feature_importance.items()
            ],
            user_cluster=user_cluster
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/ml/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request_data: RecommendationRequest,
    request: Request
):
    """
    Get personalized recommendations
    
    Suggests alternative options based on similarity and user preferences
    """
    
    model_store = get_model_store(request)
    validate_models_loaded(model_store)
    
    try:
        # Find current option
        current_option = next(
            (opt for opt in request_data.available_options if opt.id == request_data.current_option_id),
            None
        )
        
        if not current_option:
            raise HTTPException(status_code=404, detail="Current option not found")
        
        # Calculate similarities
        recommendations = []
        for option in request_data.available_options:
            if option.id != request_data.current_option_id:
                similarity = calculate_option_similarity(
                    current_option.features.dict(),
                    option.features.dict()
                )
                
                recommendations.append({
                    "option": option,
                    "similarity": similarity
                })
        
        # Sort by similarity
        recommendations.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Take top K
        top_recommendations = recommendations[:request_data.top_k]
        
        # Format response
        recommendation_items = []
        for rec in top_recommendations:
            reason = generate_similarity_reason(current_option, rec["option"])
            satisfaction = rec["similarity"] * 0.95  # Estimate satisfaction
            
            recommendation_items.append(
                RecommendationItem(
                    option_id=rec["option"].id,
                    option_name=rec["option"].name,
                    similarity_score=round(rec["similarity"], 2),
                    reason=reason,
                    estimated_satisfaction=round(satisfaction, 2)
                )
            )
        
        logger.info(f"Generated {len(recommendation_items)} recommendations for user {request_data.user_id}")
        
        return RecommendationResponse(
            user_id=request_data.user_id,
            current_option_id=request_data.current_option_id,
            recommendations=recommendation_items,
            total_analyzed=len(request_data.available_options)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


@router.get("/ml/analytics", response_model=AnalyticsResponse)
async def get_analytics(request: Request):
    """
    Get platform analytics
    
    Returns aggregated insights and metrics
    """
    
    model_store = get_model_store(request)
    validate_models_loaded(model_store)
    
    # Generate mock analytics (in production, query from database)
    analytics = AnalyticsResponse(
        user_cluster_distribution={
            "Casual User": 0.30,
            "Analytical Researcher": 0.25,
            "High Intent Buyer": 0.25,
            "Power Decision Maker": 0.20
        },
        average_decision_time=720.5,
        popular_categories=["Technology", "Services", "Products"],
        conversion_rate=0.68,
        engagement_metrics={
            "avg_scroll_depth": 0.72,
            "avg_clicks": 15.3,
            "avg_comparisons": 3.8,
            "avg_session_time": 450.2
        }
    )
    
    return analytics


@router.get("/ml/insights/{user_id}", response_model=InsightsResponse)
async def get_user_insights(user_id: str, request: Request):
    """
    Get AI-generated insights for user
    
    Provides personalized insights based on behavior patterns
    """
    
    model_store = get_model_store(request)
    validate_models_loaded(model_store)
    
    # Generate insights based on user cluster
    sample_features = generate_sample_user_features(user_id)
    X = preprocess_user_data(sample_features, model_store)
    cluster_id = int(model_store.clustering_model.predict(X)[0])
    
    insights = generate_user_insights(user_id, cluster_id, sample_features)
    
    return InsightsResponse(
        user_id=user_id,
        insights=insights
    )


@router.get("/ml/patterns/{user_id}", response_model=PatternsResponse)
async def get_decision_patterns(user_id: str, request: Request):
    """
    Get user's decision-making patterns
    
    Analyzes historical decisions to identify patterns
    """
    
    model_store = get_model_store(request)
    validate_models_loaded(model_store)
    
    # Generate patterns based on cluster
    sample_features = generate_sample_user_features(user_id)
    X = preprocess_user_data(sample_features, model_store)
    cluster_id = int(model_store.clustering_model.predict(X)[0])
    
    patterns = generate_decision_patterns(cluster_id, sample_features)
    
    return PatternsResponse(
        user_id=user_id,
        patterns=patterns,
        analyzed_decisions=int(sample_features.get("previous_decisions", 10))
    )


# Helper Functions

def generate_sample_user_features(user_id: str) -> Dict[str, Any]:
    """Generate sample user features (demo)"""
    # Use user_id hash for consistent random features
    np.random.seed(hash(user_id) % 2**32)
    
    return {
        "session_time": np.random.randint(300, 1800),
        "clicks": np.random.randint(10, 40),
        "scroll_depth": np.random.uniform(0.5, 0.95),
        "categories_viewed": np.random.randint(2, 6),
        "comparison_count": np.random.randint(2, 8),
        "product_views": np.random.randint(5, 20),
        "decision_time": np.random.randint(300, 1800),
        "price_sensitivity": np.random.uniform(0.3, 0.9),
        "feature_interest_score": np.random.uniform(0.5, 0.95),
        "previous_decisions": np.random.randint(5, 50),
        "engagement_score": np.random.uniform(0.6, 0.95),
        "purchase_intent_score": np.random.uniform(0.5, 0.95)
    }


def preprocess_user_data(features: Dict, model_store) -> np.ndarray:
    """Preprocess user data for model input"""
    # Create feature vector (simplified)
    feature_vector = [
        features["session_time"],
        features["clicks"],
        features["scroll_depth"],
        features["comparison_count"],
        features["product_views"],
        features["engagement_score"],
        features["purchase_intent_score"]
    ]
    
    # Reshape for model
    X = np.array(feature_vector).reshape(1, -1)
    
    # Scale if scaler is available
    if model_store.scaler:
        X = model_store.scaler.transform(X)
    
    return X


def calculate_option_score(option, user_id: str, model_store) -> float:
    """Calculate overall score for an option"""
    features = option.features
    
    # Weighted scoring
    score = (
        (10 - features.price / 100) * 0.25 +  # Price (normalized)
        features.quality_score * 0.35 +        # Quality
        (features.feature_count / 20) * 10 * 0.25 +  # Features
        features.brand_score * 0.15           # Brand
    )
    
    # Normalize to 0-1
    score = min(max(score / 10, 0), 1)
    
    return score


def calculate_feature_importance(features: Dict) -> Dict[str, float]:
    """Calculate feature importance scores"""
    total = sum([
        features.get("price", 0) / 1000,  # Normalize
        features.get("quality_score", 0),
        features.get("feature_count", 0) / 2,
        features.get("brand_score", 0)
    ])
    
    if total == 0:
        total = 1
    
    return {
        "quality_score": 0.35,
        "features": 0.29,
        "price": 0.23,
        "brand": 0.13
    }


def generate_recommendation_reasoning(option: Dict, user_cluster: str) -> str:
    """Generate human-readable reasoning"""
    score = option["score"]
    
    if score > 0.85:
        return f"Excellent match for {user_cluster} with outstanding quality and features"
    elif score > 0.75:
        return f"Strong recommendation based on optimal balance of price, quality, and features"
    elif score > 0.65:
        return f"Good option that aligns well with {user_cluster} preferences"
    else:
        return f"Acceptable choice with reasonable trade-offs"


def calculate_option_similarity(features1: Dict, features2: Dict) -> float:
    """Calculate similarity between two options"""
    # Simple cosine similarity on normalized features
    vec1 = np.array([
        features1.get("price", 0) / 1000,
        features1.get("quality_score", 0) / 10,
        features1.get("feature_count", 0) / 20,
        features1.get("brand_score", 0) / 10
    ])
    
    vec2 = np.array([
        features2.get("price", 0) / 1000,
        features2.get("quality_score", 0) / 10,
        features2.get("feature_count", 0) / 20,
        features2.get("brand_score", 0) / 10
    ])
    
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    return float(similarity)


def generate_similarity_reason(option1, option2) -> str:
    """Generate reason for recommendation similarity"""
    reasons = []
    
    f1 = option1.features
    f2 = option2.features
    
    if abs(f1.quality_score - f2.quality_score) < 1:
        reasons.append("comparable quality")
    if f2.price < f1.price:
        reasons.append("lower price")
    if f2.feature_count > f1.feature_count:
        reasons.append("more features")
    
    if not reasons:
        reasons = ["similar overall value"]
    
    return "Similar option with " + " and ".join(reasons)


def generate_user_insights(user_id: str, cluster_id: int, features: Dict) -> List[InsightItem]:
    """Generate personalized insights"""
    insights = []
    
    # Price sensitivity insight
    if features["price_sensitivity"] > 0.7:
        insights.append(InsightItem(
            insight_id=f"insight_{user_id}_001",
            title="High Price Sensitivity Detected",
            description="User shows strong price consciousness. Highlight value and cost-savings.",
            impact_score=0.82,
            category="pricing",
            actionable=True
        ))
    
    # Feature interest insight
    if features["feature_interest_score"] > 0.8:
        insights.append(InsightItem(
            insight_id=f"insight_{user_id}_002",
            title="Feature-Focused Behavior",
            description="User thoroughly reviews features. Provide detailed specifications.",
            impact_score=0.75,
            category="behavior",
            actionable=True
        ))
    
    return insights


def generate_decision_patterns(cluster_id: int, features: Dict) -> List[PatternItem]:
    """Generate decision patterns based on cluster"""
    patterns = []
    
    cluster_patterns = {
        0: [("Casual-Browsing", 0.70, "Explores casually without deep research")],
        1: [("Quality-First", 0.75, "Prioritizes quality over price"),
             ("Feature-Rich", 0.65, "Prefers options with more features")],
        2: [("Fast-Decision", 0.80, "Makes quick, confident decisions"),
             ("Value-Driven", 0.72, "Seeks best value for money")],
        3: [("Efficient-Comparison", 0.85, "Compares efficiently with clear criteria")]
    }
    
    for pattern_data in cluster_patterns.get(cluster_id, []):
        patterns.append(PatternItem(
            pattern_name=pattern_data[0],
            frequency=pattern_data[1],
            description=pattern_data[2]
        ))
    
    return patterns

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
    return request.app.state.model_store


def validate_models_loaded(model_store):
    if not model_store.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train models first: python training/train_models.py"
        )


@router.get("/ml/user-cluster", response_model=ClusterResponse)
async def get_user_cluster(user_id: str, request: Request):
    model_store = get_model_store(request)
    validate_models_loaded(model_store)

    try:
        sample_features = generate_sample_user_features(user_id)
        X = preprocess_user_data(sample_features, model_store)
        cluster_id = int(model_store.clustering_model.predict(X)[0])

        distances = model_store.clustering_model.transform(X)[0]
        confidence = float(1.0 / (1.0 + min(distances)))

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
    model_store = get_model_store(request)
    validate_models_loaded(model_store)
    start_time = time.time()

    try:
        options = request_data.options

        if len(options) < 2:
            raise HTTPException(status_code=400, detail="At least 2 options required")

        option_scores = []
        for option in options:
            score = calculate_option_score(option, options)
            option_scores.append({
                "id": option.id,
                "name": option.name,
                "score": score,
                "features": option.features.dict()
            })

        option_scores.sort(key=lambda x: x["score"], reverse=True)

        best_option = option_scores[0]
        alternatives = option_scores[1:4]

        feature_importance = calculate_feature_importance(best_option["features"])

        sample_features = generate_sample_user_features(request_data.user_id)
        X = preprocess_user_data(sample_features, model_store)
        cluster_id = int(model_store.clustering_model.predict(X)[0])
        user_cluster = settings.CLUSTER_LABELS.get(cluster_id)

        reasoning = generate_recommendation_reasoning(best_option, user_cluster)
        prediction_time_ms = (time.time() - start_time) * 1000

        try:
            create_prediction_log(
                db=db,
                user_id=request_data.user_id,
                decision_id=None,
                features=best_option["features"],
                cluster_id=cluster_id,
                confidence=best_option["score"],
                recommendation=reasoning,
                shap_values=feature_importance,
                model_version=settings.APP_VERSION,
                prediction_time_ms=prediction_time_ms
            )
            logger.info(f"✅ Prediction saved to database for user {request_data.user_id}")
        except Exception as db_error:
            logger.warning(f"⚠️  Failed to save prediction to database: {str(db_error)}")

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
async def get_recommendations(request_data: RecommendationRequest, request: Request):
    model_store = get_model_store(request)
    validate_models_loaded(model_store)

    try:
        current_option = next(
            (opt for opt in request_data.available_options if opt.id == request_data.current_option_id),
            None
        )

        if not current_option:
            raise HTTPException(status_code=404, detail="Current option not found")

        recommendations = []
        for option in request_data.available_options:
            if option.id != request_data.current_option_id:
                similarity = calculate_option_similarity(
                    current_option.features.dict(),
                    option.features.dict()
                )
                recommendations.append({"option": option, "similarity": similarity})

        recommendations.sort(key=lambda x: x["similarity"], reverse=True)
        top_recommendations = recommendations[:request_data.top_k]

        recommendation_items = []
        for rec in top_recommendations:
            reason = generate_similarity_reason(current_option, rec["option"])
            satisfaction = rec["similarity"] * 0.95
            recommendation_items.append(
                RecommendationItem(
                    option_id=rec["option"].id,
                    option_name=rec["option"].name,
                    similarity_score=round(rec["similarity"], 2),
                    reason=reason,
                    estimated_satisfaction=round(satisfaction, 2)
                )
            )

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
    """Get platform analytics"""
    model_store = get_model_store(request)
    validate_models_loaded(model_store)

    return AnalyticsResponse(
        status="success",
        data={
            "user_cluster_distribution": {
                "Casual User": 0.30,
                "Analytical Researcher": 0.25,
                "High Intent Buyer": 0.25,
                "Power Decision Maker": 0.20
            },
            "average_decision_time": 720.5,
            "popular_categories": ["Technology", "Services", "Products"],
            "conversion_rate": 0.68,
            "engagement_metrics": {
                "avg_scroll_depth": 0.72,
                "avg_clicks": 15.3,
                "avg_comparisons": 3.8,
                "avg_session_time": 450.2
            }
        },
        timestamp=datetime.now().isoformat()
    )


@router.get("/ml/insights/{user_id}", response_model=InsightsResponse)
async def get_user_insights(user_id: str, request: Request):
    model_store = get_model_store(request)
    validate_models_loaded(model_store)

    sample_features = generate_sample_user_features(user_id)
    X = preprocess_user_data(sample_features, model_store)
    cluster_id = int(model_store.clustering_model.predict(X)[0])
    insights = generate_user_insights(user_id, cluster_id, sample_features)

    return InsightsResponse(user_id=user_id, insights=insights)


@router.get("/ml/patterns/{user_id}", response_model=PatternsResponse)
async def get_decision_patterns(user_id: str, request: Request):
    model_store = get_model_store(request)
    validate_models_loaded(model_store)

    sample_features = generate_sample_user_features(user_id)
    X = preprocess_user_data(sample_features, model_store)
    cluster_id = int(model_store.clustering_model.predict(X)[0])
    patterns = generate_decision_patterns(cluster_id, sample_features)

    return PatternsResponse(
        user_id=user_id,
        patterns=patterns,
        analyzed_decisions=int(sample_features.get("previous_decisions", 10))
    )


# ─── Helper Functions ────────────────────────────────────────────────────────

def generate_sample_user_features(user_id: str) -> Dict[str, Any]:
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
    """
    Builds a 27-feature vector matching the training pipeline:
      - 18 numerical (scaled)
      - 9 one-hot / derived (unscaled, appended after)
    """
    session_time           = float(features.get("session_time", 600))
    clicks                 = float(features.get("clicks", 15))
    scroll_depth           = float(features.get("scroll_depth", 0.7))
    categories_viewed      = float(features.get("categories_viewed", 3))
    comparison_count       = float(features.get("comparison_count", 4))
    product_views          = float(features.get("product_views", 10))
    decision_time          = float(features.get("decision_time", 600))
    price_sensitivity      = float(features.get("price_sensitivity", 0.5))
    feature_interest_score = float(features.get("feature_interest_score", 0.7))
    previous_decisions     = float(features.get("previous_decisions", 10))
    engagement_score       = float(features.get("engagement_score", 0.7))
    purchase_intent_score  = float(features.get("purchase_intent_score", 0.6))

    engagement_ratio    = clicks / (session_time + 1)
    decision_efficiency = product_views / (decision_time + 1)
    interaction_score   = clicks * scroll_depth
    behavior_intensity  = comparison_count + product_views
    research_depth      = categories_viewed * scroll_depth
    intent_signal       = (purchase_intent_score * 0.4 +
                           engagement_score * 0.3 +
                           decision_efficiency * 0.3)
    experience_level    = float(np.log1p(previous_decisions))
    session_efficiency  = product_views / (session_time + 1)

    device_mobile  = 0.0
    device_tablet  = 0.0
    device_desktop = 1.0

    speed_fast      = 1.0 if decision_time < 300  else 0.0
    speed_moderate  = 1.0 if 300 <= decision_time < 900  else 0.0
    speed_slow      = 1.0 if 900 <= decision_time < 1800 else 0.0
    speed_very_slow = 1.0 if decision_time >= 1800 else 0.0

    numerical_vector = np.array([
        session_time, clicks, scroll_depth, categories_viewed,
        comparison_count, product_views, decision_time,
        price_sensitivity, feature_interest_score, previous_decisions,
        engagement_score, purchase_intent_score,
        engagement_ratio, decision_efficiency, interaction_score,
        behavior_intensity, research_depth, intent_signal,
    ], dtype=float).reshape(1, -1)

    if model_store.scaler:
        numerical_vector = model_store.scaler.transform(numerical_vector)

    extra_vector = np.array([
        experience_level, session_efficiency,
        device_desktop, device_mobile, device_tablet,
        speed_fast, speed_moderate, speed_slow, speed_very_slow,
    ], dtype=float).reshape(1, -1)

    return np.concatenate([numerical_vector, extra_vector], axis=1)


def calculate_option_score(option, all_options) -> float:
    """Score relative to all options — works for any price range."""
    features = option.features

    prices = [o.features.price for o in all_options]
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price if max_price != min_price else 1.0

    price_score   = 10.0 * (1.0 - (features.price - min_price) / price_range)
    quality_score = features.quality_score
    feature_score = min(features.feature_count / 5.0, 10.0)
    brand_score   = features.brand_score

    raw_score = (
        price_score   * 0.20 +
        quality_score * 0.40 +
        feature_score * 0.25 +
        brand_score   * 0.15
    )

    return round(min(max(raw_score / 10.0, 0.0), 1.0), 4)


def calculate_feature_importance(features: Dict) -> Dict[str, float]:
    return {
        "quality_score": 0.40,
        "features":      0.25,
        "price":         0.20,
        "brand":         0.15
    }


def generate_recommendation_reasoning(option: Dict, user_cluster: str) -> str:
    score = option["score"]
    name  = option["name"]
    if score >= 0.85:
        return f"{name} is an excellent match for {user_cluster} — top marks on quality, features, and value"
    elif score >= 0.70:
        return f"{name} is a strong recommendation with an optimal balance of quality, features, and price"
    elif score >= 0.55:
        return f"{name} is a good choice that aligns well with {user_cluster} preferences"
    else:
        return f"{name} is a reasonable option with some trade-offs worth considering"


def calculate_option_similarity(features1: Dict, features2: Dict) -> float:
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
    insights = []
    if features["price_sensitivity"] > 0.7:
        insights.append(InsightItem(
            insight_id=f"insight_{user_id}_001",
            title="High Price Sensitivity Detected",
            description="User shows strong price consciousness. Highlight value and cost-savings.",
            impact_score=0.82,
            category="pricing",
            actionable=True
        ))
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
    patterns = []
    cluster_patterns = {
        0: [("Casual-Browsing",      0.70, "Explores casually without deep research")],
        1: [("Quality-First",        0.75, "Prioritizes quality over price"),
            ("Feature-Rich",         0.65, "Prefers options with more features")],
        2: [("Fast-Decision",        0.80, "Makes quick, confident decisions"),
            ("Value-Driven",         0.72, "Seeks best value for money")],
        3: [("Efficient-Comparison", 0.85, "Compares efficiently with clear criteria")]
    }
    for pattern_data in cluster_patterns.get(cluster_id, []):
        patterns.append(PatternItem(
            pattern_name=pattern_data[0],
            frequency=pattern_data[1],
            description=pattern_data[2]
        ))
    return patterns
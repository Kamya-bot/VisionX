"""
VisionX Integration Tests - Shared Fixtures and Configuration
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """FastAPI test client for making API requests"""
    return TestClient(app)


@pytest.fixture
def sample_user_behavior():
    """Sample user behavior data for testing"""
    return {
        "session_time": 300,
        "clicks": 45,
        "scroll_depth": 0.75,
        "categories_viewed": ["electronics", "computers", "accessories"],
        "comparison_count": 3,
        "product_views": 8,
        "decision_time": 180,
        "price_sensitivity": 0.65,
        "feature_interest_score": 0.78,
        "device_type": "desktop",
        "previous_decisions": 5,
        "engagement_score": 0.82,
        "purchase_intent_score": 0.71
    }


@pytest.fixture
def sample_options():
    """Sample product options for comparison"""
    return [
        {
            "option_id": "A",
            "name": "Option A",
            "price": 500,
            "quality_score": 0.85,
            "feature_score": 0.75,
            "user_rating": 4.5,
            "review_count": 250
        },
        {
            "option_id": "B",
            "name": "Option B",
            "price": 650,
            "quality_score": 0.90,
            "feature_score": 0.85,
            "user_rating": 4.7,
            "review_count": 180
        },
        {
            "option_id": "C",
            "name": "Option C",
            "price": 450,
            "quality_score": 0.75,
            "feature_score": 0.70,
            "user_rating": 4.3,
            "review_count": 120
        }
    ]


@pytest.fixture
def sample_prediction_request(sample_user_behavior, sample_options):
    """Complete prediction request payload"""
    return {
        "user_behavior": sample_user_behavior,
        "options": sample_options,
        "context": {
            "timestamp": "2026-03-19T10:00:00Z",
            "session_id": "test_session_123"
        }
    }


@pytest.fixture
def sample_features():
    """Sample feature vector for SHAP/simulation"""
    return {
        "price_sensitivity": 0.65,
        "quality_score": 0.85,
        "feature_interest_score": 0.78,
        "engagement_ratio": 0.15,
        "decision_efficiency": 0.044,
        "interaction_score": 33.75,
        "category_diversity": 3,
        "behavior_intensity": 11,
        "device_type": "desktop"
    }


@pytest.fixture
def sample_explain_request(sample_features):
    """SHAP explanation request"""
    return {
        "features": sample_features,
        "model_type": "prediction",
        "top_n": 5
    }


@pytest.fixture
def sample_simulation_request(sample_features, sample_options):
    """Simulation request payload"""
    return {
        "base_features": sample_features,
        "options": sample_options,
        "scenarios": ["budget_increase", "quality_focus", "feature_emphasis"]
    }

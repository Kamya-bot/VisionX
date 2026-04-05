"""
VisionX Integration Tests - Error Handling & Edge Cases
Tests system behavior with invalid inputs, edge cases, and error scenarios
"""

import pytest


# ========== Input Validation Tests ==========

def test_prediction_with_missing_required_fields(client):
    """
    Test prediction endpoint rejects requests with missing required fields
    """
    incomplete_data = {
        "user_behavior": {"session_time": 100}  # Missing many required fields
    }
    
    response = client.post("/api/v1/ml/predict", json=incomplete_data)
    assert response.status_code == 422  # Validation error


def test_prediction_with_invalid_data_types(client):
    """
    Test prediction endpoint rejects invalid data types
    """
    invalid_data = {
        "user_behavior": {
            "session_time": "not_a_number",  # Should be number
            "clicks": "also_not_a_number"
        },
        "options": "not_a_list"  # Should be list
    }
    
    response = client.post("/api/v1/ml/predict", json=invalid_data)
    assert response.status_code == 422


def test_prediction_with_negative_values(client, sample_options):
    """
    Test prediction handles negative values gracefully
    """
    invalid_behavior = {
        "session_time": -100,  # Negative time
        "clicks": -50,  # Negative clicks
        "scroll_depth": -0.5,  # Invalid scroll depth
        "categories_viewed": [],
        "comparison_count": 0,
        "product_views": 0,
        "decision_time": 0,
        "price_sensitivity": 0.5,
        "feature_interest_score": 0.5,
        "device_type": "desktop",
        "previous_decisions": 0,
        "engagement_score": 0.5,
        "purchase_intent_score": 0.5
    }
    
    response = client.post("/api/v1/ml/predict", json={
        "user_behavior": invalid_behavior,
        "options": sample_options
    })
    
    # Should either reject (422) or handle gracefully (200 with warning)
    assert response.status_code in [200, 422]


def test_prediction_with_out_of_range_values(client, sample_options):
    """
    Test prediction handles out-of-range values
    """
    extreme_behavior = {
        "session_time": 999999,  # Extremely long session
        "clicks": 100000,  # Unrealistic clicks
        "scroll_depth": 5.0,  # > 1.0 (invalid)
        "categories_viewed": ["test"] * 1000,  # Huge list
        "comparison_count": 10000,
        "product_views": 10000,
        "decision_time": 999999,
        "price_sensitivity": 10.0,  # > 1.0 (invalid)
        "feature_interest_score": 10.0,  # > 1.0 (invalid)
        "device_type": "desktop",
        "previous_decisions": 10000,
        "engagement_score": 10.0,  # > 1.0 (invalid)
        "purchase_intent_score": 10.0  # > 1.0 (invalid)
    }
    
    response = client.post("/api/v1/ml/predict", json={
        "user_behavior": extreme_behavior,
        "options": sample_options
    })
    
    # Should handle gracefully (either validate or clip values)
    assert response.status_code in [200, 422]


def test_prediction_with_empty_options_list(client, sample_user_behavior):
    """
    Test prediction endpoint rejects empty options list
    """
    response = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": []  # Empty list
    })
    
    assert response.status_code == 422


def test_clustering_with_missing_features(client):
    """
    Test clustering handles missing features
    """
    incomplete_data = {
        "session_time": 300,
        "clicks": 45
        # Missing most features
    }
    
    response = client.get("/api/v1/ml/user-cluster", params=incomplete_data)
    
    # Should either reject or use defaults
    assert response.status_code in [200, 422]


# ========== Edge Case Tests ==========

def test_prediction_with_single_option(client, sample_user_behavior):
    """
    Test prediction with only one option to choose from
    """
    single_option = [{
        "option_id": "A",
        "name": "Only Option",
        "price": 500,
        "quality_score": 0.8
    }]
    
    response = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": single_option
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_option"] == "A"  # Only option available


def test_prediction_with_identical_options(client, sample_user_behavior):
    """
    Test prediction when all options are identical
    """
    identical_options = [
        {"option_id": f"Option_{i}", "price": 500, "quality_score": 0.8}
        for i in range(3)
    ]
    
    response = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": identical_options
    })
    
    assert response.status_code == 200
    # Should still return a prediction (may be random among identical options)


def test_recommendation_with_empty_item_pool(client, sample_user_behavior):
    """
    Test recommendation endpoint with empty item pool
    """
    response = client.post("/api/v1/ml/recommend", json={
        "user_behavior": sample_user_behavior,
        "item_pool": [],
        "top_n": 5
    })
    
    assert response.status_code in [200, 422]
    if response.status_code == 200:
        assert len(response.json()["recommendations"]) == 0


def test_simulation_with_extreme_feature_values(client, sample_options):
    """
    Test simulation handles extreme feature modifications
    """
    extreme_features = {
        "price_sensitivity": 0.0,  # Minimum
        "quality_score": 1.0,  # Maximum
        "feature_interest_score": 0.0,
        "engagement_ratio": 0.0,
        "decision_efficiency": 0.0,
        "interaction_score": 0.0,
        "category_diversity": 0,
        "behavior_intensity": 0,
        "device_type": "mobile"
    }
    
    response = client.post("/api/v1/ml/simulate", json={
        "base_features": extreme_features,
        "options": sample_options
    })
    
    # Should handle edge cases gracefully
    assert response.status_code in [200, 422]


# ========== System Error Tests ==========

def test_prediction_when_models_not_loaded(client, monkeypatch):
    """
    Test prediction behavior when models are not loaded
    
    Expected: 503 Service Unavailable or graceful error message
    """
    # Mock models_loaded = False
    from app.main import model_store
    original_loaded = model_store.models_loaded
    
    try:
        monkeypatch.setattr(model_store, "models_loaded", False)
        
        response = client.post("/api/v1/ml/predict", json={
            "user_behavior": {"session_time": 300},
            "options": [{"option_id": "A"}]
        })
        
        # Should return 503 (service unavailable) or 500 (internal error)
        assert response.status_code in [500, 503]
        
    finally:
        # Restore original state
        model_store.models_loaded = original_loaded


def test_concurrent_requests_dont_cause_deadlock(client, sample_prediction_request):
    """
    Test system handles concurrent requests without deadlock
    """
    import concurrent.futures
    import time
    
    def make_slow_request():
        """Make request and measure time"""
        start = time.time()
        response = client.post("/api/v1/ml/predict", json=sample_prediction_request)
        duration = time.time() - start
        return response.status_code, duration
    
    # Make 20 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(make_slow_request) for _ in range(20)]
        results = [f.result(timeout=10) for f in futures]  # 10s timeout
    
    # All should complete without timeout
    assert len(results) == 20
    
    # Most should be successful
    success_count = sum(1 for status, _ in results if status == 200)
    assert success_count >= 18, f"Only {success_count}/20 requests succeeded"


# ========== CORS & HTTP Tests ==========

def test_cors_preflight_request(client):
    """
    Test API handles CORS preflight (OPTIONS) requests
    """
    response = client.options("/api/v1/ml/predict")
    
    # Should return 200 for OPTIONS (preflight)
    # Note: TestClient may not fully simulate CORS behavior
    assert response.status_code in [200, 405]


def test_invalid_http_method(client):
    """
    Test API rejects unsupported HTTP methods
    """
    # Try GET on POST-only endpoint
    response = client.get("/api/v1/ml/predict")
    
    assert response.status_code == 405  # Method not allowed


def test_malformed_json_request(client):
    """
    Test API handles malformed JSON gracefully
    """
    import json
    
    # Send malformed JSON
    response = client.post(
        "/api/v1/ml/predict",
        data="{ invalid json ]",
        headers={"Content-Type": "application/json"}
    )
    
    # Should return 422 (validation error) or 400 (bad request)
    assert response.status_code in [400, 422]


def test_very_large_request_payload(client, sample_user_behavior):
    """
    Test API handles very large request payloads
    """
    # Create request with huge options list (1000 options)
    huge_options = [
        {
            "option_id": f"Option_{i}",
            "name": f"Product {i}",
            "price": i * 100,
            "quality_score": 0.8
        }
        for i in range(1000)
    ]
    
    response = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": huge_options
    })
    
    # Should either process (200) or reject as too large (413/422)
    assert response.status_code in [200, 413, 422]


# ========== Timeout & Performance Tests ==========

def test_prediction_completes_within_timeout(client, sample_prediction_request):
    """
    Test prediction completes within reasonable timeout
    """
    import time
    
    start = time.time()
    response = client.post("/api/v1/ml/predict", json=sample_prediction_request)
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 5.0, f"Prediction took {duration:.2f}s (expected <5s)"


def test_shap_explanation_completes_within_timeout(client, sample_features):
    """
    Test SHAP explanation completes within reasonable timeout
    """
    import time
    
    start = time.time()
    response = client.post("/api/v1/ml/explain", json={
        "features": sample_features
    })
    duration = time.time() - start
    
    assert response.status_code == 200
    assert duration < 5.0, f"SHAP explanation took {duration:.2f}s (expected <5s)"


# ========== Data Integrity Tests ==========

def test_prediction_confidence_always_valid_range(client, sample_prediction_request):
    """
    Test prediction confidence is always between 0 and 1
    """
    response = client.post("/api/v1/ml/predict", json=sample_prediction_request)
    
    assert response.status_code == 200
    confidence = response.json()["confidence"]
    
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0, 1]"


def test_cluster_id_always_valid(client, sample_user_behavior):
    """
    Test cluster ID is always one of valid clusters (0-3)
    """
    response = client.get("/api/v1/ml/user-cluster", params=sample_user_behavior)
    
    assert response.status_code == 200
    cluster_id = response.json()["cluster_id"]
    
    assert cluster_id in [0, 1, 2, 3], f"Invalid cluster_id: {cluster_id}"


def test_feature_importance_sums_to_reasonable_value(client, sample_prediction_request):
    """
    Test feature importance values are reasonable
    """
    response = client.post("/api/v1/ml/predict", json=sample_prediction_request)
    
    assert response.status_code == 200
    feature_importance = response.json().get("feature_importance")
    
    if isinstance(feature_importance, dict):
        # Check all importance values are reasonable
        for feature, importance in feature_importance.items():
            assert -1.0 <= importance <= 1.0, f"Feature {feature} importance out of range"


# ========== Security Tests ==========

def test_api_does_not_expose_sensitive_info_in_errors(client):
    """
    Test error messages don't expose sensitive system information
    """
    # Make invalid request
    response = client.post("/api/v1/ml/predict", json={"invalid": "data"})
    
    error_msg = str(response.json())
    
    # Should not expose file paths, credentials, internal details
    forbidden_strings = [
        "/home/", "/root/", "password", "secret", "key", "token",
        "stacktrace", "Traceback"
    ]
    
    for forbidden in forbidden_strings:
        assert forbidden.lower() not in error_msg.lower()


def test_api_rejects_sql_injection_attempts(client):
    """
    Test API rejects SQL injection patterns
    """
    injection_attempts = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--"
    ]
    
    for injection in injection_attempts:
        response = client.get("/api/v1/ml/user-cluster", params={
            "session_time": injection
        })
        
        # Should reject or handle safely (not crash)
        assert response.status_code in [200, 400, 422]


# ========== Monitoring Edge Cases ==========

def test_monitoring_handles_no_predictions_yet(client):
    """
    Test monitoring endpoints work when no predictions have been made yet
    """
    # Try to get metrics before any predictions
    response = client.get("/api/v1/ml/monitoring/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should return zero or empty data, not crash
    assert "prediction_count" in data


def test_monitoring_drift_with_insufficient_data(client):
    """
    Test drift detection handles case with too few predictions
    """
    response = client.get("/api/v1/ml/monitoring/drift")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should indicate insufficient data or no drift
    assert "drift_detected" in data
    assert isinstance(data["drift_detected"], bool)

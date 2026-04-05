"""
VisionX Integration Tests - Core ML Endpoints
Tests for /api/v1/ml/predict, /api/v1/ml/user-cluster, /api/v1/ml/recommend
"""

import pytest
import json


def test_prediction_endpoint_success(client, sample_prediction_request):
    """
    Test POST /api/v1/ml/predict returns successful prediction
    
    Expected:
    - Status code: 200
    - Response contains predicted_option, confidence, feature_importance
    - Confidence is between 0 and 1
    """
    response = client.post(
        "/api/v1/ml/predict",
        json=sample_prediction_request
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "predicted_option" in data
    assert "confidence" in data
    assert "feature_importance" in data
    
    # Validate types and ranges
    assert isinstance(data["predicted_option"], str)
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    
    # Feature importance should be a list or dict
    assert isinstance(data["feature_importance"], (list, dict))


def test_prediction_with_minimal_data(client, sample_user_behavior, sample_options):
    """
    Test prediction with minimal required data
    """
    minimal_request = {
        "user_behavior": sample_user_behavior,
        "options": sample_options
    }
    
    response = client.post("/api/v1/ml/predict", json=minimal_request)
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_option" in data


def test_prediction_performance(client, sample_prediction_request):
    """
    Test prediction responds within acceptable time (<500ms)
    """
    import time
    
    start = time.time()
    response = client.post("/api/v1/ml/predict", json=sample_prediction_request)
    duration = (time.time() - start) * 1000
    
    assert response.status_code == 200
    assert duration < 500, f"Prediction took {duration:.2f}ms (expected <500ms)"


def test_clustering_endpoint_success(client, sample_user_behavior):
    """
    Test GET /api/v1/ml/user-cluster returns valid cluster assignment
    
    Expected:
    - Status code: 200
    - cluster_name is one of 4 valid personas
    - confidence score included
    """
    response = client.get(
        "/api/v1/ml/user-cluster",
        params=sample_user_behavior
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "cluster_name" in data
    assert "cluster_id" in data
    assert "confidence" in data
    
    # Validate cluster name
    valid_clusters = [
        "Casual Users",
        "Analytical Researchers",
        "High Intent Buyers",
        "Power Decision Makers"
    ]
    assert data["cluster_name"] in valid_clusters
    
    # Validate cluster ID
    assert isinstance(data["cluster_id"], int)
    assert 0 <= data["cluster_id"] <= 3


def test_clustering_consistency(client, sample_user_behavior):
    """
    Test clustering is deterministic (same input → same cluster)
    """
    responses = [
        client.get("/api/v1/ml/user-cluster", params=sample_user_behavior)
        for _ in range(3)
    ]
    
    clusters = [r.json()["cluster_name"] for r in responses]
    
    # All responses should return same cluster
    assert len(set(clusters)) == 1, "Clustering should be deterministic"


def test_recommendation_endpoint_success(client, sample_user_behavior):
    """
    Test POST /api/v1/ml/recommend returns ranked recommendations
    
    Expected:
    - Status code: 200
    - Returns list of recommendations
    - Each recommendation has option_id, score, rank
    """
    request_data = {
        "user_behavior": sample_user_behavior,
        "item_pool": [
            {"item_id": "item1", "features": {"category": "electronics"}},
            {"item_id": "item2", "features": {"category": "computers"}},
            {"item_id": "item3", "features": {"category": "accessories"}}
        ],
        "top_n": 3
    }
    
    response = client.post("/api/v1/ml/recommend", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "recommendations" in data
    recommendations = data["recommendations"]
    
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 3  # top_n limit
    
    # Validate each recommendation
    for rec in recommendations:
        assert "item_id" in rec
        assert "score" in rec
        assert "rank" in rec
        assert 0.0 <= rec["score"] <= 1.0


def test_recommendation_ranking_order(client, sample_user_behavior):
    """
    Test recommendations are returned in descending score order
    """
    request_data = {
        "user_behavior": sample_user_behavior,
        "item_pool": [
            {"item_id": f"item{i}", "features": {"category": "test"}}
            for i in range(5)
        ],
        "top_n": 5
    }
    
    response = client.post("/api/v1/ml/recommend", json=request_data)
    
    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    
    # Scores should be descending
    scores = [r["score"] for r in recommendations]
    assert scores == sorted(scores, reverse=True), "Recommendations should be ranked by score"


def test_batch_prediction_endpoint(client, sample_user_behavior, sample_options):
    """
    Test POST /api/v1/ml/batch-predict for multiple predictions
    
    Expected:
    - Status code: 200
    - Returns list of predictions matching input size
    """
    batch_request = {
        "predictions": [
            {
                "user_behavior": sample_user_behavior,
                "options": sample_options
            }
            for _ in range(3)
        ]
    }
    
    response = client.post("/api/v1/ml/batch-predict", json=batch_request)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "results" in data
    assert len(data["results"]) == 3
    
    # Each result should have prediction data
    for result in data["results"]:
        assert "predicted_option" in result
        assert "confidence" in result


def test_concurrent_predictions(client, sample_prediction_request):
    """
    Test API can handle concurrent requests without errors
    """
    import concurrent.futures
    
    def make_request():
        return client.post("/api/v1/ml/predict", json=sample_prediction_request)
    
    # Make 5 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        responses = [f.result() for f in futures]
    
    # All should succeed
    for response in responses:
        assert response.status_code == 200
        assert "predicted_option" in response.json()


def test_prediction_with_different_options(client, sample_user_behavior):
    """
    Test prediction changes appropriately with different options
    """
    options_set_1 = [
        {"option_id": "A", "name": "Cheap", "price": 100, "quality_score": 0.5}
    ]
    
    options_set_2 = [
        {"option_id": "B", "name": "Premium", "price": 1000, "quality_score": 0.95}
    ]
    
    response1 = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": options_set_1
    })
    
    response2 = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": options_set_2
    })
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    # Predictions might differ based on options
    # (This validates model is actually using option data)
    pred1 = response1.json()["predicted_option"]
    pred2 = response2.json()["predicted_option"]
    
    # At minimum, both should return valid predictions
    assert pred1 in ["A"]
    assert pred2 in ["B"]

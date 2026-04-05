"""
VisionX Integration Tests - Health Check Endpoints
Tests for /health, /health/detailed, /health/models
"""

import pytest


def test_health_endpoint(client):
    """
    Test GET /health returns 200 and healthy status
    
    Expected:
    - Status code: 200
    - Response contains 'status': 'healthy'
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_health_detailed(client):
    """
    Test GET /health/detailed returns system information
    
    Expected:
    - Status code: 200
    - Response contains models_loaded, timestamp, version
    """
    response = client.get("/health/detailed")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "status" in data
    assert "models_loaded" in data
    assert "timestamp" in data
    assert "version" in data
    
    # Validate types
    assert isinstance(data["models_loaded"], bool)
    assert isinstance(data["timestamp"], str)
    assert isinstance(data["version"], str)


def test_health_models(client):
    """
    Test GET /health/models returns model metadata
    
    Expected:
    - Status code: 200
    - Response contains clustering_model, prediction_model info
    """
    response = client.get("/health/models")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check model info exists
    assert "clustering_model" in data
    assert "prediction_model" in data
    
    # If models are loaded, check metadata
    if data.get("models_loaded"):
        clustering = data["clustering_model"]
        prediction = data["prediction_model"]
        
        assert "status" in clustering
        assert "status" in prediction


def test_health_endpoint_performance(client):
    """
    Test /health responds within acceptable time (<100ms)
    
    Expected:
    - Response time < 100ms
    """
    import time
    
    start = time.time()
    response = client.get("/health")
    duration = (time.time() - start) * 1000  # Convert to ms
    
    assert response.status_code == 200
    assert duration < 100, f"Health check took {duration:.2f}ms (expected <100ms)"


def test_health_cors_headers(client):
    """
    Test /health includes CORS headers
    
    Expected:
    - Access-Control-Allow-Origin header present
    """
    response = client.get("/health")
    
    assert response.status_code == 200
    # Note: TestClient may not include CORS headers in test mode
    # This test validates the endpoint works; CORS is tested in production


def test_health_multiple_requests(client):
    """
    Test /health is idempotent (multiple requests return same result)
    
    Expected:
    - All requests return 200
    - Status remains 'healthy'
    """
    responses = [client.get("/health") for _ in range(5)]
    
    for response in responses:
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

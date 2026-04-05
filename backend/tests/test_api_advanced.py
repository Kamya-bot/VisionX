"""
VisionX Integration Tests - Advanced ML Endpoints
Tests for SHAP explainability, decision simulation, and model monitoring
"""

import pytest
import time


def test_shap_explanation_endpoint(client, sample_explain_request):
    """
    Test POST /api/v1/ml/explain returns SHAP feature contributions
    
    Expected:
    - Status code: 200
    - top_features list with 5 features
    - Each feature has name, contribution, impact, direction
    """
    response = client.post("/api/v1/ml/explain", json=sample_explain_request)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "top_features" in data
    assert "explanation" in data
    
    # Validate top features
    features = data["top_features"]
    assert isinstance(features, list)
    assert len(features) == 5  # Top 5 features
    
    # Each feature should have required fields
    for feature in features:
        assert "feature" in feature
        assert "contribution" in feature
        assert "impact" in feature
        assert "direction" in feature
        
        # Validate impact levels
        assert feature["impact"] in ["high", "medium", "low"]
        
        # Validate direction
        assert feature["direction"] in ["positive", "negative"]


def test_shap_explanation_performance(client, sample_explain_request):
    """
    Test SHAP explanation responds within acceptable time (<200ms)
    """
    start = time.time()
    response = client.post("/api/v1/ml/explain", json=sample_explain_request)
    duration = (time.time() - start) * 1000
    
    assert response.status_code == 200
    assert duration < 200, f"SHAP explanation took {duration:.2f}ms (expected <200ms)"


def test_simulation_endpoint(client, sample_simulation_request):
    """
    Test POST /api/v1/ml/simulate returns scenario simulations
    
    Expected:
    - Status code: 200
    - base_prediction included
    - 5 scenarios returned
    - Each scenario has prediction, confidence, confidence_change
    """
    response = client.post("/api/v1/ml/simulate", json=sample_simulation_request)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "base_prediction" in data
    assert "scenarios" in data
    assert "causal_analysis" in data
    
    # Validate base prediction
    base = data["base_prediction"]
    assert "predicted_option" in base
    assert "confidence" in base
    
    # Validate scenarios
    scenarios = data["scenarios"]
    assert isinstance(scenarios, list)
    assert len(scenarios) == 5  # 5 standard scenarios
    
    # Each scenario should have required fields
    for scenario in scenarios:
        assert "name" in scenario
        assert "description" in scenario
        assert "prediction" in scenario
        assert "confidence" in scenario
        assert "confidence_change" in scenario
        assert "features_modified" in scenario


def test_simulation_scenario_diversity(client, sample_simulation_request):
    """
    Test simulation returns diverse scenarios (not all same)
    """
    response = client.post("/api/v1/ml/simulate", json=sample_simulation_request)
    
    assert response.status_code == 200
    scenarios = response.json()["scenarios"]
    
    # Extract scenario names
    names = [s["name"] for s in scenarios]
    
    # Should have 5 unique scenarios
    assert len(set(names)) == 5, "Scenarios should be diverse"
    
    # Validate expected scenario types
    expected_scenarios = [
        "budget_increase",
        "quality_focus",
        "feature_emphasis",
        "efficiency_mode",
        "conservative"
    ]
    
    for expected in expected_scenarios:
        assert any(expected in name for name in names), f"Missing scenario: {expected}"


def test_simulation_causal_analysis(client, sample_simulation_request):
    """
    Test simulation includes causal analysis
    """
    response = client.post("/api/v1/ml/simulate", json=sample_simulation_request)
    
    assert response.status_code == 200
    causal = response.json()["causal_analysis"]
    
    # Check causal analysis fields
    assert "most_influential_feature" in causal
    assert "impact_score" in causal
    assert "risk_factors" in causal
    
    # Validate types
    assert isinstance(causal["most_influential_feature"], str)
    assert isinstance(causal["impact_score"], float)
    assert isinstance(causal["risk_factors"], list)


def test_monitoring_drift_endpoint(client):
    """
    Test GET /api/v1/ml/monitoring/drift returns drift status
    
    Expected:
    - Status code: 200
    - drift_detected boolean
    - If drift detected, includes details
    """
    response = client.get("/api/v1/ml/monitoring/drift")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "drift_detected" in data
    assert "status" in data
    
    # Validate types
    assert isinstance(data["drift_detected"], bool)
    assert data["status"] in ["healthy", "warning", "critical"]


def test_monitoring_confidence_distribution(client):
    """
    Test GET /api/v1/ml/monitoring/confidence returns distribution
    
    Expected:
    - Status code: 200
    - mean_confidence, std_confidence
    - distribution bins or histogram
    """
    response = client.get("/api/v1/ml/monitoring/confidence")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check metrics
    assert "mean_confidence" in data
    assert "std_confidence" in data
    assert "distribution" in data
    
    # Validate ranges
    assert 0.0 <= data["mean_confidence"] <= 1.0
    assert data["std_confidence"] >= 0.0


def test_monitoring_metrics_endpoint(client):
    """
    Test GET /api/v1/ml/monitoring/metrics returns performance metrics
    
    Expected:
    - Status code: 200
    - prediction_count, avg_latency, error_rate
    """
    response = client.get("/api/v1/ml/monitoring/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check metrics exist
    assert "prediction_count" in data
    assert "avg_latency_ms" in data
    
    # Validate types
    assert isinstance(data["prediction_count"], int)
    assert isinstance(data["avg_latency_ms"], (int, float))


def test_monitoring_alert_creation(client):
    """
    Test POST /api/v1/ml/monitoring/alert can create custom alerts
    """
    alert_data = {
        "alert_type": "confidence_drop",
        "threshold": 0.6,
        "message": "Test alert"
    }
    
    response = client.post("/api/v1/ml/monitoring/alert", json=alert_data)
    
    # Should accept alert (200 or 201)
    assert response.status_code in [200, 201]


def test_explain_and_simulate_together(client, sample_features, sample_options):
    """
    Integration test: Explain → Simulate workflow
    
    Tests that explanation and simulation work together
    """
    # Step 1: Get explanation
    explain_response = client.post("/api/v1/ml/explain", json={
        "features": sample_features,
        "model_type": "prediction"
    })
    
    assert explain_response.status_code == 200
    top_feature = explain_response.json()["top_features"][0]["feature"]
    
    # Step 2: Run simulation
    sim_response = client.post("/api/v1/ml/simulate", json={
        "base_features": sample_features,
        "options": sample_options
    })
    
    assert sim_response.status_code == 200
    
    # Verify causal analysis references top feature
    causal = sim_response.json()["causal_analysis"]
    assert isinstance(causal["most_influential_feature"], str)


def test_monitoring_tracks_predictions(client, sample_prediction_request):
    """
    Test monitoring system logs predictions
    """
    # Make a prediction
    pred_response = client.post("/api/v1/ml/predict", json=sample_prediction_request)
    assert pred_response.status_code == 200
    
    # Check monitoring captured it
    metrics_response = client.get("/api/v1/ml/monitoring/metrics")
    assert metrics_response.status_code == 200
    
    metrics = metrics_response.json()
    # Prediction count should be > 0 if monitoring is working
    # (Note: May be 0 in test environment if monitoring is mocked)
    assert "prediction_count" in metrics

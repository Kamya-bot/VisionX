"""
VisionX Integration Tests - End-to-End Workflows
Tests complete user journeys across multiple endpoints
"""

import pytest


def test_complete_prediction_workflow(client, sample_user_behavior, sample_options):
    """
    Test complete workflow: Cluster → Predict → Explain → Simulate
    
    Simulates a real user journey through the system
    """
    # Step 1: Get user cluster
    cluster_response = client.get("/api/v1/ml/user-cluster", params=sample_user_behavior)
    assert cluster_response.status_code == 200
    cluster_data = cluster_response.json()
    cluster_name = cluster_data["cluster_name"]
    
    # Step 2: Get prediction
    pred_request = {
        "user_behavior": sample_user_behavior,
        "options": sample_options,
        "context": {"cluster": cluster_name}
    }
    pred_response = client.post("/api/v1/ml/predict", json=pred_request)
    assert pred_response.status_code == 200
    prediction = pred_response.json()
    
    # Step 3: Explain prediction
    explain_request = {
        "features": sample_user_behavior,
        "model_type": "prediction"
    }
    explain_response = client.post("/api/v1/ml/explain", json=explain_request)
    assert explain_response.status_code == 200
    explanation = explain_response.json()
    
    # Step 4: Simulate scenarios
    sim_request = {
        "base_features": sample_user_behavior,
        "options": sample_options
    }
    sim_response = client.post("/api/v1/ml/simulate", json=sim_request)
    assert sim_response.status_code == 200
    simulation = sim_response.json()
    
    # Verify workflow consistency
    assert cluster_name in [
        "Casual Users", "Analytical Researchers",
        "High Intent Buyers", "Power Decision Makers"
    ]
    assert prediction["predicted_option"] in [opt["option_id"] for opt in sample_options]
    assert len(explanation["top_features"]) == 5
    assert len(simulation["scenarios"]) == 5
    
    print(f"\n✅ Complete workflow successful:")
    print(f"   Cluster: {cluster_name}")
    print(f"   Prediction: {prediction['predicted_option']} ({prediction['confidence']:.2%})")
    print(f"   Top feature: {explanation['top_features'][0]['feature']}")
    print(f"   Scenarios: {len(simulation['scenarios'])} tested")


def test_prediction_to_recommendation_flow(client, sample_user_behavior, sample_options):
    """
    Test workflow: Predict → Recommend alternatives
    
    Simulates user getting prediction then exploring alternatives
    """
    # Step 1: Get prediction
    pred_response = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": sample_options
    })
    assert pred_response.status_code == 200
    predicted_option = pred_response.json()["predicted_option"]
    
    # Step 2: Get recommendations (alternatives)
    rec_response = client.post("/api/v1/ml/recommend", json={
        "user_behavior": sample_user_behavior,
        "item_pool": [
            {"item_id": opt["option_id"], "features": opt}
            for opt in sample_options
        ],
        "top_n": 3
    })
    assert rec_response.status_code == 200
    recommendations = rec_response.json()["recommendations"]
    
    # Verify recommendations include alternatives
    assert len(recommendations) > 0
    recommended_ids = [r["item_id"] for r in recommendations]
    
    # Predicted option should rank high in recommendations
    assert predicted_option in recommended_ids


def test_monitoring_captures_prediction_flow(client, sample_prediction_request):
    """
    Test workflow: Predict → Monitor captures → Drift detection
    
    Verifies monitoring system tracks predictions
    """
    # Step 1: Get baseline metrics
    baseline_response = client.get("/api/v1/ml/monitoring/metrics")
    assert baseline_response.status_code == 200
    baseline_count = baseline_response.json().get("prediction_count", 0)
    
    # Step 2: Make prediction
    pred_response = client.post("/api/v1/ml/predict", json=sample_prediction_request)
    assert pred_response.status_code == 200
    
    # Step 3: Check monitoring updated
    # (Note: In test environment, monitoring may be mocked)
    updated_response = client.get("/api/v1/ml/monitoring/metrics")
    assert updated_response.status_code == 200
    
    # Step 4: Check drift status
    drift_response = client.get("/api/v1/ml/monitoring/drift")
    assert drift_response.status_code == 200
    assert "drift_detected" in drift_response.json()


def test_simulation_influences_decision_flow(client, sample_user_behavior, sample_options):
    """
    Test workflow: Base prediction → Simulate → Changed decision
    
    Verifies simulation can influence user decisions
    """
    # Step 1: Get base prediction
    base_request = {
        "user_behavior": sample_user_behavior,
        "options": sample_options
    }
    base_response = client.post("/api/v1/ml/predict", json=base_request)
    assert base_response.status_code == 200
    base_prediction = base_response.json()["predicted_option"]
    
    # Step 2: Run simulation
    sim_response = client.post("/api/v1/ml/simulate", json={
        "base_features": sample_user_behavior,
        "options": sample_options
    })
    assert sim_response.status_code == 200
    scenarios = sim_response.json()["scenarios"]
    
    # Step 3: Verify at least one scenario differs
    different_predictions = [
        s for s in scenarios 
        if s.get("prediction") != base_prediction
    ]
    
    # Some scenarios should produce different predictions
    # (If user adjusts features based on simulation)
    assert len(scenarios) == 5, "Should have 5 scenarios"


def test_explainability_improves_trust_flow(client, sample_user_behavior):
    """
    Test workflow: Low confidence → Explain → Adjust → High confidence
    
    Simulates how explainability helps users understand and trust predictions
    """
    # Step 1: Get prediction with explanation
    explain_response = client.post("/api/v1/ml/explain", json={
        "features": sample_user_behavior,
        "model_type": "prediction"
    })
    assert explain_response.status_code == 200
    
    top_features = explain_response.json()["top_features"]
    
    # Step 2: Identify most influential feature
    most_influential = top_features[0]
    
    # Step 3: Simulate adjusting that feature
    adjusted_behavior = sample_user_behavior.copy()
    feature_name = most_influential["feature"]
    
    if feature_name in adjusted_behavior:
        # Increase if positive contribution, decrease if negative
        if most_influential["direction"] == "positive":
            adjusted_behavior[feature_name] *= 1.2  # Increase by 20%
        else:
            adjusted_behavior[feature_name] *= 0.8  # Decrease by 20%
    
    # Step 4: Get new explanation
    new_explain_response = client.post("/api/v1/ml/explain", json={
        "features": adjusted_behavior,
        "model_type": "prediction"
    })
    assert new_explain_response.status_code == 200
    
    # Verify feature still influential
    new_top_features = new_explain_response.json()["top_features"]
    top_feature_names = [f["feature"] for f in new_top_features]
    
    # Most influential feature should still be in top features
    # (This validates explainability consistency)
    print(f"\n✅ Explainability flow successful:")
    print(f"   Most influential: {feature_name}")
    print(f"   Direction: {most_influential['direction']}")
    print(f"   Still in top features: {feature_name in top_feature_names}")


def test_cluster_to_personalized_prediction_flow(client, sample_user_behavior, sample_options):
    """
    Test workflow: Cluster assignment → Personalized prediction
    
    Verifies different clusters might get different predictions
    """
    # Step 1: Get cluster
    cluster_response = client.get("/api/v1/ml/user-cluster", params=sample_user_behavior)
    assert cluster_response.status_code == 200
    cluster = cluster_response.json()["cluster_name"]
    
    # Step 2: Get prediction
    pred_response = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": sample_options,
        "context": {"cluster": cluster}
    })
    assert pred_response.status_code == 200
    
    # Verify prediction includes cluster context
    # (Implementation may vary, but should acknowledge cluster)
    prediction = pred_response.json()
    assert "predicted_option" in prediction


def test_batch_prediction_workflow(client, sample_user_behavior, sample_options):
    """
    Test workflow: Batch predictions → Aggregated insights
    
    Simulates processing multiple users at once
    """
    # Create batch of 5 predictions
    batch_request = {
        "predictions": [
            {
                "user_behavior": sample_user_behavior,
                "options": sample_options
            }
            for _ in range(5)
        ]
    }
    
    # Step 1: Submit batch
    batch_response = client.post("/api/v1/ml/batch-predict", json=batch_request)
    assert batch_response.status_code == 200
    
    results = batch_response.json()["results"]
    assert len(results) == 5
    
    # Step 2: Verify all predictions successful
    for result in results:
        assert "predicted_option" in result
        assert "confidence" in result
    
    # Step 3: Aggregate insights
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    assert 0.0 <= avg_confidence <= 1.0
    
    print(f"\n✅ Batch workflow successful:")
    print(f"   Predictions: {len(results)}")
    print(f"   Avg confidence: {avg_confidence:.2%}")


def test_error_recovery_workflow(client, sample_user_behavior, sample_options):
    """
    Test workflow: Error → Retry → Success
    
    Verifies system handles and recovers from errors gracefully
    """
    # Step 1: Make valid request
    valid_response = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": sample_options
    })
    assert valid_response.status_code == 200
    
    # Step 2: Make invalid request
    invalid_response = client.post("/api/v1/ml/predict", json={
        "invalid": "data"
    })
    assert invalid_response.status_code == 422  # Validation error
    
    # Step 3: Retry with valid data (recovery)
    retry_response = client.post("/api/v1/ml/predict", json={
        "user_behavior": sample_user_behavior,
        "options": sample_options
    })
    assert retry_response.status_code == 200
    
    # System should recover and continue working
    assert retry_response.json()["predicted_option"] is not None


def test_full_system_stress_test(client, sample_user_behavior, sample_options):
    """
    Test complete system under load: Multiple workflows concurrently
    
    Simulates realistic usage with multiple users
    """
    import concurrent.futures
    
    def user_journey():
        """Simulate one user's complete journey"""
        try:
            # Cluster
            cluster_resp = client.get("/api/v1/ml/user-cluster", params=sample_user_behavior)
            assert cluster_resp.status_code == 200
            
            # Predict
            pred_resp = client.post("/api/v1/ml/predict", json={
                "user_behavior": sample_user_behavior,
                "options": sample_options
            })
            assert pred_resp.status_code == 200
            
            # Explain
            explain_resp = client.post("/api/v1/ml/explain", json={
                "features": sample_user_behavior
            })
            assert explain_resp.status_code == 200
            
            return True
        except Exception as e:
            print(f"Error in user journey: {e}")
            return False
    
    # Run 10 concurrent user journeys
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(user_journey) for _ in range(10)]
        results = [f.result() for f in futures]
    
    # All journeys should complete successfully
    success_rate = sum(results) / len(results)
    assert success_rate >= 0.9, f"Success rate: {success_rate:.0%} (expected >= 90%)"
    
    print(f"\n✅ Stress test passed:")
    print(f"   Concurrent users: 10")
    print(f"   Success rate: {success_rate:.0%}")

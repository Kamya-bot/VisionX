# VisionX Integration Tests

Testing framework for end-to-end validation of VisionX ML backend.

## 📋 **Test Coverage**

### **API Tests**
- `test_api_health.py` - Health check endpoints
- `test_api_ml_core.py` - Core ML endpoints (predict, cluster, recommend)
- `test_api_advanced.py` - Advanced ML (SHAP, simulation, monitoring)
- `test_integration_flow.py` - End-to-end workflows
- `test_error_handling.py` - Edge cases and error scenarios

### **Coverage Goals**
- **Target:** 80%+ code coverage
- **Focus:** Critical paths (prediction, explainability, simulation)
- **Tools:** pytest, pytest-asyncio, pytest-cov

---

## 🚀 **Quick Start**

```bash
cd backend

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api_ml_core.py -v
```

---

## 📊 **Test Structure**

```
backend/tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── test_api_health.py          # Health endpoints
├── test_api_ml_core.py         # Core ML
├── test_api_advanced.py        # SHAP + Simulation
├── test_integration_flow.py    # End-to-end
└── test_error_handling.py      # Edge cases
```

---

## 🧪 **Test Fixtures**

### **Client Fixture** (`conftest.py`)
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def sample_user_behavior():
    """Sample user behavior data"""
    return {
        "session_time": 300,
        "clicks": 45,
        "scroll_depth": 0.75,
        # ... more features
    }
```

---

## ✅ **Test Categories**

### **1. Health Check Tests**
```python
def test_health_endpoint(client):
    """Test /health returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_health_detailed(client):
    """Test /health/detailed returns system info"""
    response = client.get("/health/detailed")
    assert "models_loaded" in response.json()
```

### **2. Core ML Tests**
```python
def test_prediction_endpoint(client, sample_data):
    """Test POST /api/v1/ml/predict"""
    response = client.post("/api/v1/ml/predict", json=sample_data)
    assert response.status_code == 200
    assert "predicted_option" in response.json()
    assert "confidence" in response.json()

def test_clustering_endpoint(client, sample_behavior):
    """Test GET /api/v1/ml/user-cluster"""
    response = client.get("/api/v1/ml/user-cluster", params=sample_behavior)
    assert response.status_code == 200
    assert response.json()["cluster_name"] in [
        "Casual Users", 
        "Analytical Researchers",
        "High Intent Buyers",
        "Power Decision Makers"
    ]
```

### **3. Advanced ML Tests**
```python
def test_shap_explanation(client, sample_features):
    """Test POST /api/v1/ml/explain"""
    response = client.post("/api/v1/ml/explain", json=sample_features)
    assert response.status_code == 200
    assert "top_features" in response.json()
    assert len(response.json()["top_features"]) == 5

def test_simulation_engine(client, sample_features):
    """Test POST /api/v1/ml/simulate"""
    response = client.post("/api/v1/ml/simulate", json=sample_features)
    assert response.status_code == 200
    assert "scenarios" in response.json()
    assert len(response.json()["scenarios"]) == 5
```

### **4. Integration Flow Tests**
```python
def test_prediction_to_explanation_flow(client, sample_data):
    """Test: Predict → Explain workflow"""
    # Step 1: Get prediction
    pred_response = client.post("/api/v1/ml/predict", json=sample_data)
    assert pred_response.status_code == 200
    
    # Step 2: Explain prediction
    explain_data = {"features": sample_data["user_behavior"]}
    exp_response = client.post("/api/v1/ml/explain", json=explain_data)
    assert exp_response.status_code == 200
    assert "top_features" in exp_response.json()

def test_prediction_to_simulation_flow(client, sample_data):
    """Test: Predict → Simulate workflow"""
    # Step 1: Get base prediction
    pred_response = client.post("/api/v1/ml/predict", json=sample_data)
    base_prediction = pred_response.json()["predicted_option"]
    
    # Step 2: Run simulations
    sim_response = client.post("/api/v1/ml/simulate", json=sample_data)
    scenarios = sim_response.json()["scenarios"]
    
    # Verify scenarios differ from base
    assert any(s["prediction"] != base_prediction for s in scenarios)
```

### **5. Error Handling Tests**
```python
def test_invalid_prediction_input(client):
    """Test prediction with invalid data"""
    invalid_data = {"invalid_field": "test"}
    response = client.post("/api/v1/ml/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_missing_required_fields(client):
    """Test with missing required fields"""
    incomplete_data = {"session_time": 100}  # Missing other fields
    response = client.post("/api/v1/ml/predict", json=incomplete_data)
    assert response.status_code == 422

def test_models_not_loaded_scenario(client, monkeypatch):
    """Test behavior when models not loaded"""
    # Mock models_loaded = False
    from app.main import model_store
    monkeypatch.setattr(model_store, "models_loaded", False)
    
    response = client.post("/api/v1/ml/predict", json={})
    assert response.status_code == 503  # Service unavailable
```

---

## 📈 **Running Tests**

### **All Tests**
```bash
pytest tests/ -v
```

### **With Coverage Report**
```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

### **Specific Test File**
```bash
pytest tests/test_api_ml_core.py -v
```

### **Stop on First Failure**
```bash
pytest tests/ -x
```

### **Run Tests in Parallel**
```bash
pytest tests/ -n 4  # 4 parallel workers
```

---

## 🎯 **Expected Results**

### **Success Criteria**
- ✅ All tests pass (100%)
- ✅ Code coverage >80%
- ✅ No critical warnings
- ✅ API latency <500ms per test

### **Sample Output**
```
tests/test_api_health.py::test_health_endpoint PASSED                   [ 10%]
tests/test_api_health.py::test_health_detailed PASSED                   [ 20%]
tests/test_api_ml_core.py::test_prediction_endpoint PASSED              [ 30%]
tests/test_api_ml_core.py::test_clustering_endpoint PASSED              [ 40%]
tests/test_api_advanced.py::test_shap_explanation PASSED                [ 50%]
tests/test_api_advanced.py::test_simulation_engine PASSED               [ 60%]
tests/test_integration_flow.py::test_prediction_to_explanation PASSED   [ 70%]
tests/test_integration_flow.py::test_prediction_to_simulation PASSED    [ 80%]
tests/test_error_handling.py::test_invalid_prediction_input PASSED      [ 90%]
tests/test_error_handling.py::test_missing_required_fields PASSED       [100%]

========================= 10 passed in 2.54s ===========================

Coverage: 82%
```

---

## 🐛 **Debugging Failed Tests**

### **View Detailed Output**
```bash
pytest tests/ -vv --tb=long
```

### **Run Single Test**
```bash
pytest tests/test_api_ml_core.py::test_prediction_endpoint -vv
```

### **Print Debug Output**
```python
def test_prediction_endpoint(client, sample_data):
    response = client.post("/api/v1/ml/predict", json=sample_data)
    print(f"Response: {response.json()}")  # Debug print
    assert response.status_code == 200
```

---

## 📝 **Writing New Tests**

### **Template**
```python
def test_your_feature(client, fixture):
    """
    Test description
    
    Steps:
    1. Setup test data
    2. Call API endpoint
    3. Validate response
    """
    # Arrange
    test_data = {...}
    
    # Act
    response = client.post("/api/v1/your-endpoint", json=test_data)
    
    # Assert
    assert response.status_code == 200
    assert "expected_field" in response.json()
    assert response.json()["expected_field"] == expected_value
```

---

## 🎓 **Best Practices**

1. **Use fixtures** for shared test data
2. **Test one thing** per test function
3. **Name tests clearly** (`test_what_when_expected`)
4. **Use assertions** liberally
5. **Mock external dependencies** (API calls, file I/O)
6. **Test edge cases** (empty inputs, large inputs, invalid types)
7. **Document expected behavior** in docstrings

---

## 🚀 **CI/CD Integration**

### **GitHub Actions** (`.github/workflows/tests.yml`)
```yaml
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        cd backend
        pytest tests/ --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## 📊 **Coverage Badge**

Add to `README.md`:
```markdown
![Tests](https://github.com/your-username/visionx/workflows/Run%20Tests/badge.svg)
![Coverage](https://codecov.io/gh/your-username/visionx/branch/main/graph/badge.svg)
```

---

## 🎯 **Next Steps**

1. ✅ Run tests locally: `pytest tests/ -v`
2. ✅ Achieve 80%+ coverage
3. ✅ Fix any failing tests
4. ✅ Add tests for new features
5. ✅ Set up CI/CD pipeline
6. ✅ Add coverage badge to README

---

**Status:** Tests provide confidence that VisionX works end-to-end ✅  
**Impact:** Converts "project" → "reliable system" 🚀

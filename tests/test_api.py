from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_predict_minimal():
    # Use the default feature names provided by the fallback dummy model
    sample_features = {"feature_0": 0.1, "feature_1": 1.2, "feature_2": -0.3}
    resp = client.post("/predict", json={"features": sample_features})
    assert resp.status_code == 200, f"Bad response: {resp.status_code} {resp.json()}"
    data = resp.json()
    assert "probability" in data and "risk_label" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_label"] in (0, 1)

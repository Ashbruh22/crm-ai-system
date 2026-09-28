import pytest

# The service stack is not installed in the training environment
# (requirements-train.txt), and app.main still imports TensorFlow until phase 3
# swaps it for ONNX Runtime. Skip rather than error at collection so the
# training suite stays green; CI installs requirements.txt and runs these.
pytest.importorskip("fastapi", reason="service deps not installed")
pytest.importorskip("tensorflow", reason="app.main still imports TensorFlow (phase 3)")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_readiness_check():
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200

def test_predict_outcome_unauthenticated():
    with TestClient(app) as client:
        payload = {
            "opportunity_id": "OPP_10001",
            "sales_agent": "Sarah Chen",
            "product": "Enterprise Suite",
            "engage_date": "2024-01-15"
        }
        response = client.post("/api/v1/predict/outcome", json=payload)
        assert response.status_code == 401
        assert "Not authenticated" in response.text

def test_admin_retrain_forbidden_for_rep():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/admin/retrain", 
            headers={"Authorization": "Bearer test-rep-token"}
        )
        assert response.status_code == 403
        assert "Not enough permissions" in response.text

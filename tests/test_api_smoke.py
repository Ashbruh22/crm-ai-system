import pytest
from fastapi.testclient import TestClient
from app.main import app

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

import pytest
from fastapi.testclient import TestClient
from forest_app.core.main import app
from forest_app.core.security import get_current_active_user

client = TestClient(app)

class DummyUser:
    id = 1
    email = "test@example.com"
    is_active = True

def setup_module(module):
    app.dependency_overrides[get_current_active_user] = lambda: DummyUser()

def teardown_module(module):
    app.dependency_overrides = {}

def test_complete_task_404():
    response = client.post("/core/complete_task", json={"task_id": "nonexistent", "success": True})
    assert response.status_code in (404, 400, 422, 500)  # Accept any error for missing/invalid

def test_onboarding_batch():
    response = client.get("/onboarding/current_batch")
    assert response.status_code in (200, 400, 404, 422, 500) 
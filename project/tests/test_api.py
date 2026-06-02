import pytest
from fastapi.testclient import TestClient
from project.src.api.server import app, is_ready, startup_event

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_api():
    # Force API initialization inside tests
    startup_event()

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_cases():
    response = client.get("/api/v1/cases")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Check key columns
    first = data[0]
    assert "id" in first
    assert "account_id" in first
    assert "status" in first
    assert "customer_name" in first

def test_get_case_detail():
    # Fetch list first
    list_res = client.get("/api/v1/cases")
    case_id = list_res.json()[0]["id"]
    
    response = client.get(f"/api/v1/cases/{case_id}")
    assert response.status_code == 200
    data = response.json()
    assert "case" in data
    assert "risk_profile" in data
    assert "timeline" in data
    
    # Check calibrated risk and rules engine triggers
    rp = data["risk_profile"]
    assert "calibrated_score" in rp
    assert 300 <= rp["calibrated_score"] <= 900
    assert "risk_tier" in rp
    assert "triggers" in rp
    assert len(rp["triggers"]) > 0

def test_post_note():
    list_res = client.get("/api/v1/cases")
    case_id = list_res.json()[0]["id"]
    
    payload = {
        "analyst": "A. Sharma (Senior Forensic)",
        "note": "Unit test note: Verified baseline income records."
    }
    response = client.post(f"/api/v1/cases/{case_id}/notes", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_post_escalation():
    list_res = client.get("/api/v1/cases")
    case_id = list_res.json()[0]["id"]
    
    payload = {
        "status": "Escalated - Frozen",
        "escalation_level": "Hard Hold Applied",
        "analyst": "A. Sharma (Senior Forensic)",
        "log_msg": "Unit test escalation: applied hard debit hold."
    }
    response = client.post(f"/api/v1/cases/{case_id}/escalate", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_get_report():
    list_res = client.get("/api/v1/cases")
    case_id = list_res.json()[0]["id"]
    
    response = client.get(f"/api/v1/cases/{case_id}/report?analyst=M. Sen")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "report" in data
    assert "SECTION 1:" in data["report"]

def test_ask_assistant():
    list_res = client.get("/api/v1/cases")
    case_id = list_res.json()[0]["id"]
    
    payload = {
        "case_id": case_id,
        "question": "Why was this account flagged?"
    }
    response = client.post("/api/v1/assistant", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "answer" in data
    assert len(data["answer"]) > 0

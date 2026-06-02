"""
Live tests against the deployed Render API.
Run: pytest test_api.py -v
"""
import pytest
import requests

BASE = "https://customer-churn-prediction-02k2.onrender.com"

BANK_PAYLOAD = {
    "name": "Jane Smith",
    "credit_score": 650,
    "geography": "France",
    "gender": "Female",
    "age": 42,
    "tenure": 5,
    "balance": 75000.0,
    "num_products": 2,
    "has_cr_card": 1,
    "is_active_member": 1,
    "estimated_salary": 85000.0,
}

TELCO_PAYLOAD = {
    "name": "John Rivera",
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 89.85,
    "total_charges": 1078.20,
}


# ── Meta ──────────────────────────────────────────────────────────────────────

def test_health():
    r = requests.get(f"{BASE}/health", timeout=60)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["bank_models"] == 7
    assert body["telco_models"] == 5


def test_models():
    r = requests.get(f"{BASE}/models", timeout=60)
    assert r.status_code == 200
    body = r.json()
    assert len(body["bank"]) == 7
    assert len(body["telco"]) == 5


# ── Bank predictions ──────────────────────────────────────────────────────────

def test_predict_bank_returns_valid_probability():
    r = requests.post(f"{BASE}/predict/bank", json=BANK_PAYLOAD, timeout=60)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_bank_risk_level():
    r = requests.post(f"{BASE}/predict/bank", json=BANK_PAYLOAD, timeout=60)
    assert r.json()["risk_level"] in {"low", "medium", "high"}


def test_predict_bank_model_scores():
    r = requests.post(f"{BASE}/predict/bank", json=BANK_PAYLOAD, timeout=60)
    scores = r.json()["model_scores"]
    assert len(scores) == 7
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_predict_bank_dataset_label():
    r = requests.post(f"{BASE}/predict/bank", json=BANK_PAYLOAD, timeout=60)
    assert r.json()["dataset"] == "bank"


def test_predict_bank_high_risk_profile():
    payload = {**BANK_PAYLOAD, "age": 55, "balance": 0, "is_active_member": 0, "num_products": 4}
    r = requests.post(f"{BASE}/predict/bank", json=payload, timeout=60)
    assert r.status_code == 200
    assert r.json()["churn_probability"] > 0.4


def test_predict_bank_invalid_credit_score():
    r = requests.post(f"{BASE}/predict/bank", json={**BANK_PAYLOAD, "credit_score": 100}, timeout=60)
    assert r.status_code == 422


def test_predict_bank_invalid_gender():
    r = requests.post(f"{BASE}/predict/bank", json={**BANK_PAYLOAD, "gender": "Unknown"}, timeout=60)
    assert r.status_code == 422


# ── Telco predictions ─────────────────────────────────────────────────────────

def test_predict_telco_returns_valid_probability():
    r = requests.post(f"{BASE}/predict/telco", json=TELCO_PAYLOAD, timeout=60)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_telco_risk_level():
    r = requests.post(f"{BASE}/predict/telco", json=TELCO_PAYLOAD, timeout=60)
    assert r.json()["risk_level"] in {"low", "medium", "high"}


def test_predict_telco_model_scores():
    r = requests.post(f"{BASE}/predict/telco", json=TELCO_PAYLOAD, timeout=60)
    scores = r.json()["model_scores"]
    assert len(scores) == 5
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_predict_telco_dataset_label():
    r = requests.post(f"{BASE}/predict/telco", json=TELCO_PAYLOAD, timeout=60)
    assert r.json()["dataset"] == "telco"


def test_predict_telco_low_risk_profile():
    payload = {**TELCO_PAYLOAD, "contract": "Two year", "tenure": 60, "monthly_charges": 30.0}
    r = requests.post(f"{BASE}/predict/telco", json=payload, timeout=60)
    assert r.status_code == 200
    assert r.json()["churn_probability"] < 0.6


def test_predict_telco_invalid_contract():
    r = requests.post(f"{BASE}/predict/telco", json={**TELCO_PAYLOAD, "contract": "Weekly"}, timeout=60)
    assert r.status_code == 422

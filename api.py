"""
ChurnGuard REST API — FastAPI
Endpoints:
  GET  /health           health check
  GET  /models           list available models
  POST /predict/bank     bank churn prediction
  POST /predict/telco    telco churn prediction
  POST /train/run        execute an uploaded .ipynb notebook
  GET  /docs             auto-generated Swagger UI
"""
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("churnguard")

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Literal

_TRAIN_KEY = os.getenv("TRAIN_API_KEY", "")
_MAX_NB_BYTES = 5 * 1024 * 1024  # 5 MB

_api_key_header = APIKeyHeader(name="X-Train-Key", auto_error=False)

def _require_train_key(key: str | None = Depends(_api_key_header)) -> None:
    if not _TRAIN_KEY:
        raise HTTPException(status_code=503, detail="Notebook execution is disabled (TRAIN_API_KEY not configured)")
    if key != _TRAIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Train-Key header")

import numpy as np
from notebook_runner import run_notebook
from utils import load_models, load_scaler, preprocess, predict_all
from utils_telco import load_telco_models, load_telco_scaler, preprocess_telco, predict_telco

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ChurnGuard API",
    description=(
        "ML-powered customer churn prediction.\n\n"
        "Two datasets supported: **Bank Churn** (7 models) and **Telco Churn** (5 models).\n\n"
        "All endpoints return a churn probability, per-model scores, and a risk level."
    ),
    version="1.0.0",
    contact={"name": "ChurnGuard", "url": "https://github.com/Limeload/customer-churn-prediction"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ───────────────────────────────────────────────
def _load_or_exit(loader, label):
    try:
        artifact = loader()
        logger.info("Loaded %s", label)
        return artifact
    except Exception as exc:
        logger.critical("Failed to load %s: %s", label, exc)
        sys.exit(1)

bank_models  = _load_or_exit(load_models,       "bank models (7)")
bank_scaler  = _load_or_exit(load_scaler,       "bank scaler")
telco_models = _load_or_exit(load_telco_models, "telco models (5)")
telco_scaler = _load_or_exit(load_telco_scaler, "telco scaler")

# ── Schemas ───────────────────────────────────────────────────────────────────
class BankCustomer(BaseModel):
    name:             str   = Field("Jane Smith",  description="Customer name (used in LLM email)")
    credit_score:     int   = Field(650,  ge=300, le=850,  description="FICO-style credit score")
    geography:        str   = Field("France",              description="Country (mapped to France / Germany / Spain cluster)")
    gender:           Literal["Male", "Female"] = Field("Female")
    age:              int   = Field(42,   ge=18, le=100)
    tenure:           int   = Field(5,    ge=0,  le=10,    description="Years with the bank")
    balance:          float = Field(75000.0, ge=0,         description="Account balance in USD")
    num_products:     int   = Field(2,    ge=1,  le=4)
    has_cr_card:      Literal[0, 1] = Field(1,             description="1 = holds a credit card")
    is_active_member: Literal[0, 1] = Field(1,             description="1 = active member")
    estimated_salary: float = Field(85000.0, ge=0)

    model_config = {"json_schema_extra": {"example": {
        "name": "Jane Smith", "credit_score": 650, "geography": "France",
        "gender": "Female", "age": 42, "tenure": 5, "balance": 75000,
        "num_products": 2, "has_cr_card": 1, "is_active_member": 1, "estimated_salary": 85000,
    }}}


class TelcoCustomer(BaseModel):
    name:             str   = Field("John Rivera")
    gender:           Literal["Male", "Female"] = Field("Male")
    senior_citizen:   Literal[0, 1] = Field(0)
    partner:          Literal["Yes", "No"] = Field("No")
    dependents:       Literal["Yes", "No"] = Field("No")
    tenure:           int   = Field(12, ge=0, le=72, description="Months with the provider")
    phone_service:    Literal["Yes", "No"] = Field("Yes")
    multiple_lines:   Literal["Yes", "No"] = Field("No")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field("Fiber optic")
    online_security:  Literal["Yes", "No"] = Field("No")
    online_backup:    Literal["Yes", "No"] = Field("No")
    device_protection:Literal["Yes", "No"] = Field("No")
    tech_support:     Literal["Yes", "No"] = Field("No")
    streaming_tv:     Literal["Yes", "No"] = Field("Yes")
    streaming_movies: Literal["Yes", "No"] = Field("Yes")
    contract:         Literal["Month-to-month", "One year", "Two year"] = Field("Month-to-month")
    paperless_billing:Literal["Yes", "No"] = Field("Yes")
    payment_method:   Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field("Electronic check")
    monthly_charges:  float = Field(89.85, ge=0)
    total_charges:    float = Field(1078.20, ge=0)

    model_config = {"json_schema_extra": {"example": {
        "name": "John Rivera", "gender": "Male", "senior_citizen": 0,
        "partner": "No", "dependents": "No", "tenure": 12,
        "phone_service": "Yes", "multiple_lines": "No",
        "internet_service": "Fiber optic", "online_security": "No",
        "online_backup": "No", "device_protection": "No", "tech_support": "No",
        "streaming_tv": "Yes", "streaming_movies": "Yes",
        "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 89.85, "total_charges": 1078.20,
    }}}


class PredictionResponse(BaseModel):
    churn_probability: float = Field(description="Average churn probability across all models (0–1)")
    risk_level:        str   = Field(description="low / medium / high")
    model_scores:      dict  = Field(description="Per-model churn probabilities")
    dataset:           str   = Field(description="Which dataset's models were used")


def _risk(p: float) -> str:
    return "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health():
    """Returns API status and number of loaded models."""
    return {
        "status": "ok",
        "bank_models":  len(bank_models),
        "telco_models": len(telco_models),
    }


@app.get("/models", tags=["Meta"])
def list_models():
    """Lists all available ML models for each dataset."""
    return {
        "bank":  list(bank_models.keys()),
        "telco": list(telco_models.keys()),
    }


@app.post("/predict/bank", response_model=PredictionResponse, tags=["Prediction"])
def predict_bank(customer: BankCustomer):
    """
    Predict churn probability for a **bank customer**.

    Returns the ensemble average across 7 models:
    XGBoost, Random Forest, Gradient Boosting, Stacking,
    Decision Tree, SVM, KNN.
    """
    form = {
        "name":         customer.name,
        "credit_score": str(customer.credit_score),
        "geography":    customer.geography,
        "gender":       customer.gender,
        "age":          str(customer.age),
        "tenure":       str(customer.tenure),
        "balance":      str(customer.balance),
        "num_products": str(customer.num_products),
        "has_cr_card":       str(customer.has_cr_card),
        "is_active_member":  str(customer.is_active_member),
        "estimated_salary":  str(customer.estimated_salary),
    }
    try:
        features    = preprocess(form)
        predictions = predict_all(features, bank_models, bank_scaler)
        avg_prob    = predictions.pop("Average")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(
        churn_probability=avg_prob,
        risk_level=_risk(avg_prob),
        model_scores=predictions,
        dataset="bank",
    )


@app.post("/predict/telco", response_model=PredictionResponse, tags=["Prediction"])
def predict_telco_endpoint(customer: TelcoCustomer):
    """
    Predict churn probability for a **telco customer**.

    Returns the ensemble average across 5 models:
    XGBoost, Random Forest, Gradient Boosting, Stacking, Voting Ensemble.
    """
    form = {k: str(v) for k, v in customer.model_dump().items()}
    try:
        features    = preprocess_telco(form)
        predictions = predict_telco(features, telco_models, telco_scaler)
        avg_prob    = predictions.pop("Average")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(
        churn_probability=avg_prob,
        risk_level=_risk(avg_prob),
        model_scores=predictions,
        dataset="telco",
    )


@app.post("/train/run", tags=["Training"], dependencies=[Depends(_require_train_key)])
async def train_run(notebook: UploadFile = File(...)):
    """Execute an uploaded .ipynb notebook and return all cell outputs. Requires X-Train-Key header."""
    if not notebook.filename.lower().endswith(".ipynb"):
        raise HTTPException(status_code=400, detail="Only .ipynb files are accepted")

    content = await notebook.read()
    if len(content) > _MAX_NB_BYTES:
        raise HTTPException(status_code=413, detail="Notebook exceeds 5 MB limit")

    return run_notebook(content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

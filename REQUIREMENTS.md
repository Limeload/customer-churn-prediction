# ChurnGuard — Requirements Traceability

Maps each functional requirement to the code that implements it and the tests that verify it.

---

## Requirement Index

| ID | Requirement | Status |
|---|---|---|
| REQ-01 | Bank churn prediction | Implemented |
| REQ-02 | Telco churn prediction | Implemented |
| REQ-03 | Ensemble model scoring | Implemented |
| REQ-04 | Risk level classification | Implemented |
| REQ-05 | LLM churn explanation | Implemented |
| REQ-06 | Personalized retention email | Implemented |
| REQ-07 | LLM provider selection | Implemented |
| REQ-08 | Notebook training execution | Implemented |
| REQ-09 | API health check | Implemented |
| REQ-10 | Model listing | Implemented |
| REQ-11 | Input validation | Implemented |
| REQ-12 | Model retraining scripts | Implemented |

---

## REQ-01 — Bank Churn Prediction

Given structured bank customer data (11 fields), the system must return a churn probability, per-model scores, a risk level, and a dataset label.

| Layer | Location |
|---|---|
| REST API | `POST /predict/bank` → [api.py:147](api.py#L147) |
| Request schema | `BankCustomer` → [api.py:58](api.py#L58) |
| Feature preprocessing | `preprocess()` → [utils.py](utils.py) |
| Inference | `predict_all()` → [utils.py](utils.py) |
| Flask UI route | `POST /predict` → [app.py:90](app.py#L90) |
| UI form | Bank tab → [templates/index.html](templates/index.html) |
| Tests | `test_predict_bank_*` → [test_api.py:69](test_api.py#L69) |

**Input fields:** `credit_score`, `geography`, `gender`, `age`, `tenure`, `balance`, `num_products`, `has_cr_card`, `is_active_member`, `estimated_salary`, `name`

---

## REQ-02 — Telco Churn Prediction

Given structured telco customer data (20 fields), the system must return a churn probability, per-model scores, a risk level, and a dataset label.

| Layer | Location |
|---|---|
| REST API | `POST /predict/telco` → [api.py:184](api.py#L184) |
| Request schema | `TelcoCustomer` → [api.py:78](api.py#L78) |
| Feature preprocessing | `preprocess_telco()` → [utils_telco.py](utils_telco.py) |
| Inference | `predict_telco()` → [utils_telco.py](utils_telco.py) |
| Flask UI route | `POST /predict-telco` → [app.py:142](app.py#L142) |
| UI form | Telco tab → [templates/index.html](templates/index.html) |
| Tests | `test_predict_telco_*` → [test_api.py:112](test_api.py#L112) |

**Input fields:** `gender`, `senior_citizen`, `partner`, `dependents`, `tenure`, `phone_service`, `multiple_lines`, `internet_service`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`, `contract`, `paperless_billing`, `payment_method`, `monthly_charges`, `total_charges`, `name`

---

## REQ-03 — Ensemble Model Scoring

The prediction response must include individual probability scores from every loaded model, not just an aggregate.

| Layer | Location |
|---|---|
| Response schema | `PredictionResponse.model_scores` → [api.py:116](api.py#L116) |
| Bank models (7) | XGBoost, Random Forest, Gradient Boosting, Stacking, Decision Tree, SVM, KNN |
| Telco models (5) | XGBoost, Random Forest, Gradient Boosting, Stacking, Voting Ensemble |
| Tests | `test_predict_bank_model_scores` → [test_api.py:81](test_api.py#L81) |
| | `test_predict_telco_model_scores` → [test_api.py:124](test_api.py#L124) |

---

## REQ-04 — Risk Level Classification

The ensemble average probability must be bucketed into a human-readable risk label.

| Probability | Label |
|---|---|
| ≥ 0.70 | `high` |
| 0.40 – 0.69 | `medium` |
| < 0.40 | `low` |

| Layer | Location |
|---|---|
| Classification logic | `_risk()` → [api.py:123](api.py#L123) |
| Response field | `PredictionResponse.risk_level` → [api.py:118](api.py#L118) |
| Tests | `test_predict_bank_risk_level` → [test_api.py:76](test_api.py#L76) |
| | `test_predict_telco_risk_level` → [test_api.py:119](test_api.py#L119) |
| | `test_predict_bank_high_risk_profile` → [test_api.py:93](test_api.py#L93) |
| | `test_predict_telco_low_risk_profile` → [test_api.py:136](test_api.py#L136) |

---

## REQ-05 — LLM Churn Explanation

Given prediction results and customer data, the system must generate a 2–4 sentence plain-English explanation of why the customer is or is not likely to churn, referencing specific features.

| Layer | Location |
|---|---|
| System prompt | `SYSTEM_PROMPT` → [app.py:31](app.py#L31) |
| LLM call | `get_llm_response()` → [app.py:59](app.py#L59) |
| User message builder | `build_user_message()` → [app.py:73](app.py#L73) |
| Response field | `explanation` in `/predict` and `/predict-telco` responses |
| Tests | None (requires live LLM key; validate manually via Swagger UI) |

---

## REQ-06 — Personalized Retention Email

The system must generate a warm, personalized retention email addressed to the customer by name alongside the explanation.

| Layer | Location |
|---|---|
| Prompt instruction | `SYSTEM_PROMPT` → [app.py:31](app.py#L31) |
| LLM call | `get_llm_response()` → [app.py:59](app.py#L59) |
| JSON parsing | `parse_llm_json()` → [app.py:44](app.py#L44) |
| Response field | `email` in `/predict` and `/predict-telco` responses |
| Tests | None (requires live LLM key; validate manually via UI) |

---

## REQ-07 — LLM Provider Selection

Users must be able to choose between OpenAI GPT-4o and Groq Llama 3.3 70B from the UI; the selected provider must be reflected in the response label.

| Layer | Location |
|---|---|
| Provider config | `LLM_CONFIGS` → [app.py:22](app.py#L22) |
| Client factory | `_llm_client()` → [app.py:26](app.py#L26) |
| UI controls | LLM radio buttons → [templates/index.html](templates/index.html) |
| Response field | `llm_label` in prediction response |
| Tests | None |

---

## REQ-08 — Notebook Training Execution

Users must be able to upload a `.ipynb` notebook, have it executed server-side, and receive all cell outputs (text, images, HTML, errors).

| Layer | Location |
|---|---|
| REST API | `POST /train/run` → [api.py:208](api.py#L208) |
| Flask UI route | `POST /train/run` → [app.py:247](app.py#L247) |
| UI page | [templates/train.html](templates/train.html) |
| Notebook execution | `nbconvert.ExecutePreprocessor` |
| Output types handled | `stream`, `execute_result`, `display_data` (text/html/image/png), `error` |
| Tests | None (requires kernel; validate by uploading a notebook in the UI) |

---

## REQ-09 — API Health Check

The API must expose a health endpoint that returns its operational status and the count of loaded models per dataset.

| Layer | Location |
|---|---|
| REST API | `GET /health` → [api.py:128](api.py#L128) |
| Flask proxy | `GET /api/health` → [app.py:203](app.py#L203) |
| Tests | `test_health` → [test_api.py:50](test_api.py#L50) |

**Expected response:** `{ "status": "ok", "bank_models": 7, "telco_models": 5 }`

---

## REQ-10 — Model Listing

The API must expose an endpoint that returns the names of all available models for each dataset.

| Layer | Location |
|---|---|
| REST API | `GET /models` → [api.py:138](api.py#L138) |
| Flask proxy | `GET /api/models` → [app.py:211](app.py#L211) |
| Tests | `test_models` → [test_api.py:59](test_api.py#L59) |

---

## REQ-11 — Input Validation

Invalid inputs must be rejected before reaching the model layer; the API must return HTTP 422 with a descriptive error.

| Layer | Location |
|---|---|
| Validation mechanism | Pydantic schemas with `Field(ge=..., le=..., Literal[...])` constraints |
| Bank constraints | `credit_score` 300–850, `age` 18–100, `gender` Male/Female, `num_products` 1–4 |
| Telco constraints | `contract` one of three literals, `internet_service` one of three literals, etc. |
| Tests | `test_predict_bank_invalid_credit_score` → [test_api.py:100](test_api.py#L100) |
| | `test_predict_bank_invalid_gender` → [test_api.py:105](test_api.py#L105) |
| | `test_predict_telco_invalid_contract` → [test_api.py:143](test_api.py#L143) |

---

## REQ-12 — Model Retraining Scripts

The project must include scripts that train and serialize all models from the raw CSV datasets, so the model artifacts can be reproduced without manual notebook steps.

| Layer | Location |
|---|---|
| Bank training | [training/train_bank.py](training/train_bank.py) → outputs `models/bank/` |
| Telco training | [training/train_telco.py](training/train_telco.py) → outputs `models/telco/` |
| Hyperparameter tuning | [training/tune.py](training/tune.py) |
| Bank dataset | [data/churn.csv](data/churn.csv) |
| Telco dataset | [data/telco_churn.csv](data/telco_churn.csv) |
| Exploration notebooks | [notebooks/bank-churn-data-exploration-and-churn-prediction.ipynb](notebooks/bank-churn-data-exploration-and-churn-prediction.ipynb) |
| | [notebooks/churn.ipynb](notebooks/churn.ipynb) |
| Tests | None (artifacts validated at startup by model-count assertions in REQ-09) |

---

## Test Coverage Summary

| Requirement | Test count | Notes |
|---|---|---|
| REQ-01 Bank prediction | 5 | probability, risk, scores, label, high-risk profile |
| REQ-02 Telco prediction | 5 | probability, risk, scores, label, low-risk profile |
| REQ-03 Ensemble scoring | 2 | score count and 0–1 range |
| REQ-04 Risk classification | 4 | label enum + boundary profiles |
| REQ-05 LLM explanation | 0 | manual validation required |
| REQ-06 Retention email | 0 | manual validation required |
| REQ-07 LLM selection | 0 | manual validation required |
| REQ-08 Notebook execution | 0 | manual validation required |
| REQ-09 Health check | 1 | status + model counts |
| REQ-10 Model listing | 1 | counts per dataset |
| REQ-11 Input validation | 3 | invalid credit score, gender, contract |
| REQ-12 Retraining scripts | 0 | artifact presence checked by REQ-09 |

Run all integration tests against the live API:

```bash
pytest test_api.py -v
```

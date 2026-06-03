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
| Integration tests | `test_predict_bank_*` → [test_api.py:69](test_api.py#L69) |
| Unit tests | `TestBankPayloadFromForm::test_is_active_form_field_maps_to_is_active_member` → [test_unit.py:74](test_unit.py#L74) |
| | `TestBankPayloadFromForm::test_salary_form_field_maps_to_estimated_salary` → [test_unit.py:80](test_unit.py#L80) |
| | `TestBankPayloadFromForm::test_numeric_type_coercions` → [test_unit.py:86](test_unit.py#L86) |
| | `TestBankPayloadFromForm::test_defaults_applied_when_fields_absent` → [test_unit.py:93](test_unit.py#L93) |
| | `TestBankCustomerData::test_has_cr_card_1_renders_as_yes` → [test_unit.py:112](test_unit.py#L112) |
| | `TestBankCustomerData::test_has_cr_card_0_renders_as_no` → [test_unit.py:116](test_unit.py#L116) |
| | `TestBankCustomerData::test_is_active_member_rendered_as_yes_no` → [test_unit.py:120](test_unit.py#L120) |
| | `TestBankCustomerData::test_balance_formatted_with_dollar_and_commas` → [test_unit.py:124](test_unit.py#L124) |
| | `TestBankCustomerData::test_salary_formatted_with_dollar_and_commas` → [test_unit.py:128](test_unit.py#L128) |

**Input fields:** `credit_score`, `geography`, `gender`, `age`, `tenure`, `balance`, `num_products`, `has_cr_card`, `is_active_member`, `estimated_salary`, `name`

---

## REQ-02 — Telco Churn Prediction

Given structured telco customer data (20 fields), the system must return a churn probability, per-model scores, a risk level, and a dataset label. Unexpected categorical values must emit a warning log rather than silently producing a bad prediction.

| Layer | Location |
|---|---|
| REST API | `POST /predict/telco` → [api.py:184](api.py#L184) |
| Request schema | `TelcoCustomer` → [api.py:78](api.py#L78) |
| Feature preprocessing | `preprocess_telco()` → [utils_telco.py](utils_telco.py) |
| Categorical guard | `_warn_unexpected()` → [utils_telco.py](utils_telco.py) |
| Inference | `predict_telco()` → [utils_telco.py](utils_telco.py) |
| Flask UI route | `POST /predict-telco` → [app.py:142](app.py#L142) |
| UI form | Telco tab → [templates/index.html](templates/index.html) |
| Integration tests | `test_predict_telco_*` → [test_api.py:112](test_api.py#L112) |
| Unit tests | `TestTelcoPayloadFromForm::test_all_required_fields_present` → [test_unit.py:151](test_unit.py#L151) |
| | `TestTelcoPayloadFromForm::test_numeric_coercions` → [test_unit.py:163](test_unit.py#L163) |
| | `TestTelcoPayloadFromForm::test_defaults_applied_when_fields_absent` → [test_unit.py:170](test_unit.py#L170) |
| | `TestTelcoCustomerData::test_senior_citizen_1_renders_as_yes` → [test_unit.py:189](test_unit.py#L189) |
| | `TestTelcoCustomerData::test_senior_citizen_0_renders_as_no` → [test_unit.py:193](test_unit.py#L193) |
| | `TestTelcoCustomerData::test_monthly_charges_formatted` → [test_unit.py:198](test_unit.py#L198) |
| | `TestTelcoCustomerData::test_total_charges_formatted` → [test_unit.py:202](test_unit.py#L202) |
| | `TestWarnUnexpected::test_no_warning_for_all_valid_values` → [test_unit.py:213](test_unit.py#L213) |
| | `TestWarnUnexpected::test_warning_for_invalid_gender` → [test_unit.py:227](test_unit.py#L227) |
| | `TestWarnUnexpected::test_warning_for_invalid_contract` → [test_unit.py:231](test_unit.py#L231) |
| | `TestWarnUnexpected::test_warning_for_invalid_payment_method` → [test_unit.py:236](test_unit.py#L236) |

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
| System prompt | `SYSTEM_PROMPT` → [llm_service.py](llm_service.py) |
| LLM call | `get_llm_response()` → [llm_service.py](llm_service.py) |
| User message builder | `build_user_message()` → [llm_service.py](llm_service.py) |
| JSON parsing | `parse_llm_json()` → [llm_service.py](llm_service.py) |
| Response field | `explanation` in `/predict` and `/predict-telco` responses |
| Integration tests | None (requires live LLM key; validate manually via Swagger UI) |
| Unit tests | `TestParseLlmJson::test_valid_json_string` → [test_unit.py:18](test_unit.py#L18) |
| | `TestParseLlmJson::test_code_fenced_json` → [test_unit.py:24](test_unit.py#L24) |
| | `TestParseLlmJson::test_code_fenced_with_language_tag` → [test_unit.py:29](test_unit.py#L29) |
| | `TestParseLlmJson::test_invalid_json_returns_fallback` → [test_unit.py:34](test_unit.py#L34) |
| | `TestBuildUserMessage::test_probability_formatted_as_percentage` → [test_unit.py:45](test_unit.py#L45) |
| | `TestBuildUserMessage::test_contains_customer_data` → [test_unit.py:49](test_unit.py#L49) |
| | `TestBuildUserMessage::test_contains_model_scores` → [test_unit.py:54](test_unit.py#L54) |

---

## REQ-06 — Personalized Retention Email

The system must generate a warm, personalized retention email addressed to the customer by name alongside the explanation.

| Layer | Location |
|---|---|
| Prompt instruction | `SYSTEM_PROMPT` → [llm_service.py](llm_service.py) |
| LLM call | `get_llm_response()` → [llm_service.py](llm_service.py) |
| JSON parsing | `parse_llm_json()` → [llm_service.py](llm_service.py) |
| Response field | `email` in `/predict` and `/predict-telco` responses |
| Integration tests | None (requires live LLM key; validate manually via UI) |
| Unit tests | `TestParseLlmJson::test_valid_json_string` → [test_unit.py:18](test_unit.py#L18) _(shared with REQ-05)_ |
| | `TestParseLlmJson::test_invalid_json_returns_fallback` → [test_unit.py:34](test_unit.py#L34) _(shared with REQ-05)_ |

---

## REQ-07 — LLM Provider Selection

Users must be able to choose between OpenAI GPT-4o and Groq Llama 3.3 70B from the UI; the selected provider must be reflected in the response label.

| Layer | Location |
|---|---|
| Provider config | `LLM_CONFIGS` → [llm_service.py](llm_service.py) |
| Client factory | `_llm_client()` → [llm_service.py](llm_service.py) |
| UI controls | LLM radio buttons → [templates/index.html](templates/index.html) |
| Response field | `llm_label` in prediction response |
| Tests | None |

---

## REQ-08 — Notebook Training Execution

Users must be able to upload a `.ipynb` notebook, have it executed server-side, and receive all cell outputs (text, images, HTML, errors).

| Layer | Location |
|---|---|
| REST API | `POST /train/run` → [api.py:208](api.py#L208) |
| Flask UI route | `POST /train/run` → [app.py:251](app.py#L251) |
| Execution engine | `run_notebook()` → [notebook_runner.py](notebook_runner.py) |
| Output collector | `_collect_outputs()` → [notebook_runner.py](notebook_runner.py) |
| UI page | [templates/train.html](templates/train.html) |
| Output types handled | `stream`, `execute_result`, `display_data` (text/html/image/png), `error` |
| Authentication | `X-Train-Key` header must match `TRAIN_API_KEY` env var; endpoint returns 503 if var is unset |
| Size limit | Notebooks larger than 5 MB are rejected with HTTP 413 |
| Integration tests | None (requires kernel; validate by uploading a notebook in the UI) |
| Unit tests | `TestCollectOutputs::test_stream_stdout_collected` → [test_unit.py:253](test_unit.py#L253) |
| | `TestCollectOutputs::test_empty_stream_text_not_collected` → [test_unit.py:261](test_unit.py#L261) |
| | `TestCollectOutputs::test_error_output_collected` → [test_unit.py:268](test_unit.py#L268) |
| | `TestCollectOutputs::test_image_output_collected` → [test_unit.py:281](test_unit.py#L281) |
| | `TestCollectOutputs::test_non_code_cells_skipped` → [test_unit.py:293](test_unit.py#L293) |
| | `TestCollectOutputs::test_html_output_collected` → [test_unit.py:300](test_unit.py#L300) |
| | `TestCollectOutputs::test_ansi_codes_stripped_from_stream` → [test_unit.py:310](test_unit.py#L310) |

> **Security note:** Notebook execution runs arbitrary uploaded Python code. The API key guard and size cap are a minimum baseline. For production, also run `ExecutePreprocessor` in an isolated container with `--network none` and CPU/memory limits.

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

| Requirement | Integration | Unit | Unit test functions |
|---|---|---|---|
| REQ-01 Bank prediction | 5 | 9 | `TestBankPayloadFromForm` (4), `TestBankCustomerData` (5) |
| REQ-02 Telco prediction | 5 | 11 | `TestTelcoPayloadFromForm` (3), `TestTelcoCustomerData` (4), `TestWarnUnexpected` (4) |
| REQ-03 Ensemble scoring | 2 | 0 | — |
| REQ-04 Risk classification | 4 | 0 | — |
| REQ-05 LLM explanation | 0 | 7 | `TestParseLlmJson` (4), `TestBuildUserMessage` (3) |
| REQ-06 Retention email | 0 | 2 | `TestParseLlmJson::test_valid_json_string`, `test_invalid_json_returns_fallback` |
| REQ-07 LLM selection | 0 | 0 | manual validation required |
| REQ-08 Notebook execution | 0 | 7 | `TestCollectOutputs` (7) |
| REQ-09 Health check | 1 | 0 | — |
| REQ-10 Model listing | 1 | 0 | — |
| REQ-11 Input validation | 3 | 0 | — |
| REQ-12 Retraining scripts | 0 | 0 | artifact presence checked by REQ-09 |
| **Total** | **21** | **27** | |

Run unit tests (no live server or API keys required):

```bash
pytest test_unit.py -v
```

Run integration tests against the live API:

```bash
pytest test_api.py -v
```

Run the full suite:

```bash
pytest test_unit.py test_api.py -v
```

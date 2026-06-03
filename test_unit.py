"""
Unit tests for shared utility modules.
No live server, API keys, or loaded ML models required.
Run: pytest test_unit.py -v
"""
import json
import logging
import pytest

# ── llm_service: parse_llm_json (REQ-05, REQ-06) ─────────────────────────────

from llm_service import parse_llm_json, build_user_message
from llm_service import bank_payload_from_form, bank_customer_data
from llm_service import telco_payload_from_form, telco_customer_data


class TestParseLlmJson:
    def test_valid_json_string(self):
        raw = '{"explanation": "Low risk.", "email": "Dear Alice,"}'
        result = parse_llm_json(raw)
        assert result["explanation"] == "Low risk."
        assert result["email"] == "Dear Alice,"

    def test_code_fenced_json(self):
        raw = '```\n{"explanation": "High risk.", "email": "Dear Bob,"}\n```'
        result = parse_llm_json(raw)
        assert result["explanation"] == "High risk."

    def test_code_fenced_with_language_tag(self):
        raw = '```json\n{"explanation": "Medium risk.", "email": "Dear Carol,"}\n```'
        result = parse_llm_json(raw)
        assert result["explanation"] == "Medium risk."

    def test_invalid_json_returns_fallback(self):
        raw = "This is not JSON at all."
        result = parse_llm_json(raw)
        assert "explanation" in result
        assert "email" in result
        assert result["explanation"] == raw


# ── llm_service: build_user_message (REQ-05, REQ-06) ─────────────────────────

class TestBuildUserMessage:
    def test_probability_formatted_as_percentage(self):
        msg = build_user_message({"Name": "Alice"}, {"XGBoost": 0.8}, 0.75)
        assert "75.0%" in msg

    def test_contains_customer_data(self):
        msg = build_user_message({"Name": "Alice", "Age": 42}, {}, 0.5)
        assert "Alice" in msg
        assert "42" in msg

    def test_contains_model_scores(self):
        scores = {"XGBoost": 0.82, "Random Forest": 0.78}
        msg = build_user_message({}, scores, 0.8)
        assert "XGBoost" in msg
        assert "0.82" in msg


# ── llm_service: bank_payload_from_form (REQ-01, field-naming consistency) ───

class TestBankPayloadFromForm:
    def _form(self, **overrides):
        base = {
            "name": "Jane", "credit_score": "650", "geography": "France",
            "gender": "Female", "age": "42", "tenure": "5",
            "balance": "75000", "num_products": "2",
            "has_cr_card": "1", "is_active": "1", "salary": "85000",
        }
        base.update(overrides)
        return base

    def test_is_active_form_field_maps_to_is_active_member(self):
        payload = bank_payload_from_form(self._form(is_active="0"))
        assert "is_active_member" in payload
        assert "is_active" not in payload
        assert payload["is_active_member"] == 0

    def test_salary_form_field_maps_to_estimated_salary(self):
        payload = bank_payload_from_form(self._form(salary="90000"))
        assert "estimated_salary" in payload
        assert "salary" not in payload
        assert payload["estimated_salary"] == 90000.0

    def test_numeric_type_coercions(self):
        payload = bank_payload_from_form(self._form())
        assert isinstance(payload["credit_score"], int)
        assert isinstance(payload["age"], int)
        assert isinstance(payload["balance"], float)
        assert isinstance(payload["estimated_salary"], float)

    def test_defaults_applied_when_fields_absent(self):
        payload = bank_payload_from_form({})
        assert payload["name"] == "Valued Customer"
        assert payload["geography"] == "France"


# ── llm_service: bank_customer_data (REQ-01) ─────────────────────────────────

class TestBankCustomerData:
    def _payload(self, **overrides):
        base = {
            "name": "Jane", "age": 42, "geography": "France", "gender": "Female",
            "credit_score": 650, "tenure": 5, "balance": 75000.0,
            "num_products": 2, "has_cr_card": 1, "is_active_member": 1,
            "estimated_salary": 85000.0,
        }
        base.update(overrides)
        return base

    def test_has_cr_card_1_renders_as_yes(self):
        data = bank_customer_data(self._payload(has_cr_card=1))
        assert data["Has Credit Card"] == "Yes"

    def test_has_cr_card_0_renders_as_no(self):
        data = bank_customer_data(self._payload(has_cr_card=0))
        assert data["Has Credit Card"] == "No"

    def test_is_active_member_rendered_as_yes_no(self):
        assert bank_customer_data(self._payload(is_active_member=1))["Is Active Member"] == "Yes"
        assert bank_customer_data(self._payload(is_active_member=0))["Is Active Member"] == "No"

    def test_balance_formatted_with_dollar_and_commas(self):
        data = bank_customer_data(self._payload(balance=75000.0))
        assert data["Balance"] == "$75,000.00"

    def test_salary_formatted_with_dollar_and_commas(self):
        data = bank_customer_data(self._payload(estimated_salary=1234567.89))
        assert data["Estimated Salary"] == "$1,234,567.89"


# ── llm_service: telco_payload_from_form (REQ-02) ────────────────────────────

class TestTelcoPayloadFromForm:
    def _form(self, **overrides):
        base = {
            "name": "John", "gender": "Male", "senior_citizen": "0",
            "partner": "No", "dependents": "No", "tenure": "12",
            "phone_service": "Yes", "multiple_lines": "No",
            "internet_service": "Fiber optic", "online_security": "No",
            "online_backup": "No", "device_protection": "No", "tech_support": "No",
            "streaming_tv": "Yes", "streaming_movies": "Yes",
            "contract": "Month-to-month", "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": "89.85", "total_charges": "1078.20",
        }
        base.update(overrides)
        return base

    def test_all_required_fields_present(self):
        payload = telco_payload_from_form(self._form())
        required = [
            "name", "gender", "senior_citizen", "partner", "dependents", "tenure",
            "phone_service", "multiple_lines", "internet_service", "online_security",
            "online_backup", "device_protection", "tech_support", "streaming_tv",
            "streaming_movies", "contract", "paperless_billing", "payment_method",
            "monthly_charges", "total_charges",
        ]
        for field in required:
            assert field in payload, f"Missing field: {field}"

    def test_numeric_coercions(self):
        payload = telco_payload_from_form(self._form())
        assert isinstance(payload["tenure"], int)
        assert isinstance(payload["monthly_charges"], float)
        assert isinstance(payload["total_charges"], float)
        assert isinstance(payload["senior_citizen"], int)

    def test_defaults_applied_when_fields_absent(self):
        payload = telco_payload_from_form({})
        assert payload["contract"] == "Month-to-month"
        assert payload["payment_method"] == "Electronic check"


# ── llm_service: telco_customer_data (REQ-02) ────────────────────────────────

class TestTelcoCustomerData:
    def _payload(self):
        return {
            "name": "John", "gender": "Male", "senior_citizen": 1,
            "tenure": 12, "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "monthly_charges": 89.85, "total_charges": 1078.20,
            "payment_method": "Electronic check",
            "paperless_billing": "Yes", "tech_support": "No",
        }

    def test_senior_citizen_1_renders_as_yes(self):
        data = telco_customer_data(self._payload())
        assert data["Senior Citizen"] == "Yes"

    def test_senior_citizen_0_renders_as_no(self):
        p = self._payload()
        p["senior_citizen"] = 0
        assert telco_customer_data(p)["Senior Citizen"] == "No"

    def test_monthly_charges_formatted(self):
        data = telco_customer_data(self._payload())
        assert data["Monthly Charges"] == "$89.85"

    def test_total_charges_formatted(self):
        data = telco_customer_data(self._payload())
        assert data["Total Charges"] == "$1,078.20"


# ── utils_telco: _warn_unexpected (REQ-02, data integrity) ───────────────────

from utils_telco import _warn_unexpected


class TestWarnUnexpected:
    def test_no_warning_for_all_valid_values(self, caplog):
        valid = {
            "gender": "Male", "partner": "Yes", "dependents": "No",
            "phone_service": "Yes", "multiple_lines": "No",
            "internet_service": "Fiber optic", "online_security": "No",
            "online_backup": "No", "device_protection": "No", "tech_support": "No",
            "streaming_tv": "Yes", "streaming_movies": "Yes",
            "contract": "Month-to-month", "paperless_billing": "Yes",
            "payment_method": "Electronic check",
        }
        with caplog.at_level(logging.WARNING, logger="utils_telco"):
            _warn_unexpected(valid)
        assert caplog.records == []

    def test_warning_for_invalid_gender(self, caplog):
        with caplog.at_level(logging.WARNING, logger="utils_telco"):
            _warn_unexpected({"gender": "Unknown"})
        assert any("gender" in r.message for r in caplog.records)

    def test_warning_for_invalid_contract(self, caplog):
        with caplog.at_level(logging.WARNING, logger="utils_telco"):
            _warn_unexpected({"contract": "Weekly"})
        assert any("contract" in r.message for r in caplog.records)

    def test_warning_for_invalid_payment_method(self, caplog):
        with caplog.at_level(logging.WARNING, logger="utils_telco"):
            _warn_unexpected({"payment_method": "Crypto"})
        assert any("payment_method" in r.message for r in caplog.records)


# ── notebook_runner: _collect_outputs (REQ-08) ───────────────────────────────

from notebook_runner import _collect_outputs


def _make_cell(outputs, cell_type="code"):
    return {"cell_type": cell_type, "outputs": outputs}


class TestCollectOutputs:
    def test_stream_stdout_collected(self):
        nb = type("NB", (), {"cells": [
            _make_cell([{"output_type": "stream", "name": "stdout", "text": "hello\n"}])
        ]})()
        outputs = _collect_outputs(nb)
        assert len(outputs) == 1
        assert outputs[0]["kind"] == "stream"
        assert "hello" in outputs[0]["text"]

    def test_empty_stream_text_not_collected(self):
        nb = type("NB", (), {"cells": [
            _make_cell([{"output_type": "stream", "name": "stdout", "text": "   \n"}])
        ]})()
        assert _collect_outputs(nb) == []

    def test_error_output_collected(self):
        nb = type("NB", (), {"cells": [
            _make_cell([{
                "output_type": "error",
                "ename": "ValueError", "evalue": "bad input",
                "traceback": ["ValueError: bad input"],
            }])
        ]})()
        outputs = _collect_outputs(nb)
        assert len(outputs) == 1
        assert outputs[0]["kind"] == "error"
        assert "ValueError" in outputs[0]["text"]

    def test_image_output_collected(self):
        nb = type("NB", (), {"cells": [
            _make_cell([{
                "output_type": "display_data",
                "data": {"image/png": "abc123=="},
            }])
        ]})()
        outputs = _collect_outputs(nb)
        assert len(outputs) == 1
        assert outputs[0]["kind"] == "image"
        assert outputs[0]["b64"] == "abc123=="

    def test_non_code_cells_skipped(self):
        nb = type("NB", (), {"cells": [
            _make_cell([{"output_type": "stream", "name": "stdout", "text": "ignored"}],
                       cell_type="markdown"),
        ]})()
        assert _collect_outputs(nb) == []

    def test_html_output_collected(self):
        nb = type("NB", (), {"cells": [
            _make_cell([{
                "output_type": "execute_result",
                "data": {"text/html": "<table></table>"},
            }])
        ]})()
        outputs = _collect_outputs(nb)
        assert outputs[0]["kind"] == "html"

    def test_ansi_codes_stripped_from_stream(self):
        nb = type("NB", (), {"cells": [
            _make_cell([{
                "output_type": "stream", "name": "stdout",
                "text": "\x1b[32mgreen\x1b[0m",
            }])
        ]})()
        outputs = _collect_outputs(nb)
        assert outputs[0]["text"] == "green"

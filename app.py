import io
import os
import json
import requests as http
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

RENDER_API = os.getenv("RENDER_API_URL", "https://customer-churn-prediction-02k2.onrender.com")

LLM_CONFIGS = {
    "openai": {"model": "gpt-4o",                  "label": "OpenAI GPT-4o",       "temperature": 0.4, "max_tokens": 1024},
    "groq":   {"model": "llama-3.3-70b-versatile", "label": "Groq Llama 3.3 70B",  "temperature": 0.5, "max_tokens": 1024},
}

def _llm_client(llm: str) -> OpenAI:
    if llm == "groq":
        return OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a customer retention specialist at a bank. You receive structured data
about a customer and the churn probability predicted by multiple ML models.

Your job is to:
1. Explain in plain English why this customer is or isn't likely to churn, referencing the
   specific features (age, balance, products, activity, etc.).
2. Write a warm, personalized retention email addressed to the customer by name.

Be concise but insightful. Format your response as valid JSON with exactly two keys:
- "explanation": a 2-4 sentence analysis of the churn risk
- "email": the full text of the personalized retention email"""


def parse_llm_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return {
            "explanation": text,
            "email": "Could not parse email separately. See explanation above.",
        }


def get_llm_response(user_message: str, llm: str) -> dict:
    cfg = LLM_CONFIGS.get(llm, LLM_CONFIGS["openai"])
    completion = _llm_client(llm).chat.completions.create(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )
    return parse_llm_json(completion.choices[0].message.content)


def build_user_message(customer_data: dict, predictions: dict, churn_probability: float) -> str:
    return f"""Customer data:
{json.dumps(customer_data, indent=2)}

Churn probability scores from ML models:
{json.dumps(predictions, indent=2)}

Overall average churn probability: {churn_probability:.1%}

Provide your explanation and personalized retention email as JSON."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    llm = form_data.get("llm", "openai")

    payload = {
        "name":             form_data.get("name", "Valued Customer"),
        "credit_score":     int(form_data.get("credit_score", 650)),
        "geography":        form_data.get("geography", "France"),
        "gender":           form_data.get("gender", "Female"),
        "age":              int(form_data.get("age", 35)),
        "tenure":           int(form_data.get("tenure", 5)),
        "balance":          float(form_data.get("balance", 0)),
        "num_products":     int(form_data.get("num_products", 1)),
        "has_cr_card":      int(form_data.get("has_cr_card", 0)),
        "is_active_member": int(form_data.get("is_active", 0)),
        "estimated_salary": float(form_data.get("salary", 0)),
    }

    r = http.post(f"{RENDER_API}/predict/bank", json=payload, timeout=60)
    r.raise_for_status()
    result = r.json()

    predictions = result["model_scores"]
    avg_prob    = result["churn_probability"]

    customer_data = {
        "Name":               payload["name"],
        "Age":                payload["age"],
        "Geography":          payload["geography"],
        "Gender":             payload["gender"],
        "Credit Score":       payload["credit_score"],
        "Tenure (years)":     payload["tenure"],
        "Balance":            f"${payload['balance']:,.2f}",
        "Number of Products": payload["num_products"],
        "Has Credit Card":    "Yes" if payload["has_cr_card"] == 1 else "No",
        "Is Active Member":   "Yes" if payload["is_active_member"] == 1 else "No",
        "Estimated Salary":   f"${payload['estimated_salary']:,.2f}",
    }

    user_message = build_user_message(customer_data, predictions, avg_prob)
    llm_result   = get_llm_response(user_message, llm)

    return jsonify({
        "churn_probability": avg_prob,
        "model_scores":      predictions,
        "explanation":       llm_result.get("explanation", ""),
        "email":             llm_result.get("email", ""),
        "llm_label":         LLM_CONFIGS.get(llm, LLM_CONFIGS["openai"])["label"],
    })


@app.route("/predict-telco", methods=["POST"])
def predict_telco_route():
    form_data = request.form.to_dict()
    llm = form_data.get("llm", "openai")

    payload = {
        "name":             form_data.get("name", "Valued Customer"),
        "gender":           form_data.get("gender", "Male"),
        "senior_citizen":   int(form_data.get("senior_citizen", 0)),
        "partner":          form_data.get("partner", "No"),
        "dependents":       form_data.get("dependents", "No"),
        "tenure":           int(form_data.get("tenure", 0)),
        "phone_service":    form_data.get("phone_service", "Yes"),
        "multiple_lines":   form_data.get("multiple_lines", "No"),
        "internet_service": form_data.get("internet_service", "DSL"),
        "online_security":  form_data.get("online_security", "No"),
        "online_backup":    form_data.get("online_backup", "No"),
        "device_protection":form_data.get("device_protection", "No"),
        "tech_support":     form_data.get("tech_support", "No"),
        "streaming_tv":     form_data.get("streaming_tv", "No"),
        "streaming_movies": form_data.get("streaming_movies", "No"),
        "contract":         form_data.get("contract", "Month-to-month"),
        "paperless_billing":form_data.get("paperless_billing", "Yes"),
        "payment_method":   form_data.get("payment_method", "Electronic check"),
        "monthly_charges":  float(form_data.get("monthly_charges", 0)),
        "total_charges":    float(form_data.get("total_charges", 0)),
    }

    r = http.post(f"{RENDER_API}/predict/telco", json=payload, timeout=60)
    r.raise_for_status()
    result = r.json()

    predictions = result["model_scores"]
    avg_prob    = result["churn_probability"]

    customer_data = {
        "Name":              payload["name"],
        "Gender":            payload["gender"],
        "Senior Citizen":    "Yes" if payload["senior_citizen"] == 1 else "No",
        "Tenure (months)":   payload["tenure"],
        "Contract":          payload["contract"],
        "Internet Service":  payload["internet_service"],
        "Monthly Charges":   f"${payload['monthly_charges']:,.2f}",
        "Total Charges":     f"${payload['total_charges']:,.2f}",
        "Payment Method":    payload["payment_method"],
        "Paperless Billing": payload["paperless_billing"],
        "Tech Support":      payload["tech_support"],
    }

    user_message = build_user_message(customer_data, predictions, avg_prob)
    llm_result   = get_llm_response(user_message, llm)

    return jsonify({
        "churn_probability": avg_prob,
        "model_scores":      predictions,
        "explanation":       llm_result.get("explanation", ""),
        "email":             llm_result.get("email", ""),
        "llm_label":         LLM_CONFIGS.get(llm, LLM_CONFIGS["openai"])["label"],
    })


@app.route("/api/health")
def api_health():
    try:
        r = http.get(f"{RENDER_API}/health", timeout=30)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/api/models")
def api_models():
    try:
        r = http.get(f"{RENDER_API}/models", timeout=30)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/api/predict/bank", methods=["POST"])
def api_predict_bank():
    try:
        r = http.post(f"{RENDER_API}/predict/bank", json=request.json, timeout=60)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/api/predict/telco", methods=["POST"])
def api_predict_telco():
    try:
        r = http.post(f"{RENDER_API}/predict/telco", json=request.json, timeout=60)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/train")
def train_page():
    return render_template("train.html")


@app.route("/notebooks/<path:filename>")
def serve_notebook(filename):
    return send_from_directory(
        os.path.join(app.root_path, "notebooks"), filename, as_attachment=True
    )


@app.route("/train/run", methods=["POST"])
def train_run():
    try:
        if "notebook" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        nb_file = request.files["notebook"]
        nb = nbformat.read(io.TextIOWrapper(nb_file.stream, encoding="utf-8"), as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            ep.preprocess(nb, {"metadata": {"path": app.root_path}})
        except CellExecutionError:
            pass  # return cells even if a cell errored
        nb_json = json.loads(nbformat.writes(nb))
        return jsonify({"cells": nb_json["cells"]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

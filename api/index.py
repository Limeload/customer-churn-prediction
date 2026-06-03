import os
import sys
import requests as http
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv

load_dotenv()

_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _root)

from llm_service import (  # noqa: E402
    RENDER_API, LLM_CONFIGS,
    bank_payload_from_form, bank_customer_data,
    telco_payload_from_form, telco_customer_data,
    llm_predict_response,
)

app = Flask(
    __name__,
    template_folder=os.path.join(_root, "templates"),
)


@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": str(e)}), getattr(e, "code", 500)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()
        payload   = bank_payload_from_form(form_data)
        return jsonify(llm_predict_response(
            payload, bank_customer_data(payload),
            f"{RENDER_API}/predict/bank", form_data.get("llm", "openai"),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-telco", methods=["POST"])
def predict_telco_route():
    try:
        form_data = request.form.to_dict()
        payload   = telco_payload_from_form(form_data)
        return jsonify(llm_predict_response(
            payload, telco_customer_data(payload),
            f"{RENDER_API}/predict/telco", form_data.get("llm", "openai"),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        os.path.join(_root, "notebooks"), filename, as_attachment=True
    )


@app.route("/train/run", methods=["POST"])
def train_run():
    try:
        if "notebook" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        nb_file = request.files["notebook"]
        r = http.post(
            f"{RENDER_API}/train/run",
            files={"notebook": (nb_file.filename, nb_file.read(), "application/octet-stream")},
            timeout=660,
        )
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

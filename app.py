import os
import requests as http
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from notebook_runner import run_notebook
from llm_service import (
    RENDER_API, LLM_CONFIGS,
    bank_payload_from_form, bank_customer_data,
    telco_payload_from_form, telco_customer_data,
    llm_predict_response,
)

load_dotenv()

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=[])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    payload   = bank_payload_from_form(form_data)
    return jsonify(llm_predict_response(
        payload, bank_customer_data(payload),
        f"{RENDER_API}/predict/bank", form_data.get("llm", "openai"),
    ))


@app.route("/predict-telco", methods=["POST"])
def predict_telco_route():
    form_data = request.form.to_dict()
    payload   = telco_payload_from_form(form_data)
    return jsonify(llm_predict_response(
        payload, telco_customer_data(payload),
        f"{RENDER_API}/predict/telco", form_data.get("llm", "openai"),
    ))


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


_TRAIN_KEY = os.getenv("TRAIN_API_KEY", "")
_MAX_NB_BYTES = 5 * 1024 * 1024  # 5 MB


@app.route("/train/run", methods=["POST"])
@limiter.limit("10 per hour")
def train_run():
    if not _TRAIN_KEY:
        return jsonify({"error": "Notebook execution is disabled (TRAIN_API_KEY not configured)"}), 503
    if request.headers.get("X-Train-Key") != _TRAIN_KEY:
        return jsonify({"error": "Invalid or missing X-Train-Key header"}), 403
    try:
        if "notebook" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        nb_file = request.files["notebook"]
        filename = secure_filename(nb_file.filename or "")
        if not filename.lower().endswith(".ipynb"):
            return jsonify({"error": "Only .ipynb files are accepted"}), 400
        raw = nb_file.read()
        if len(raw) > _MAX_NB_BYTES:
            return jsonify({"error": "Notebook exceeds 5 MB limit"}), 413
        return jsonify(run_notebook(raw)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

import os
import json
import re
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from utils import load_models, load_scaler, preprocess, predict_all
from utils_telco import load_telco_models, load_telco_scaler, preprocess_telco, predict_telco

load_dotenv()

app = Flask(__name__)

models        = load_models()
scaler        = load_scaler()
telco_models  = load_telco_models()
telco_scaler  = load_telco_scaler()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

LLM_CONFIGS = {
    "openai": {
        "client": openai_client,
        "model": "gpt-4o",
        "label": "OpenAI GPT-4o",
        "temperature": 0.4,
        "max_tokens": 1024,
    },
    "groq": {
        "client": groq_client,
        "model": "llama-3.3-70b-versatile",
        "label": "Groq Llama 3.3 70B",
        "temperature": 0.5,
        "max_tokens": 1024,
    },
}

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
    completion = cfg["client"].chat.completions.create(
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

    features    = preprocess(form_data)
    predictions = predict_all(features, models, scaler)
    avg_prob    = predictions.pop("Average")

    customer_data = {
        "Name":             form_data.get("name", "Valued Customer"),
        "Age":              form_data.get("age"),
        "Geography":        form_data.get("geography"),
        "Gender":           form_data.get("gender"),
        "Credit Score":     form_data.get("credit_score"),
        "Tenure (years)":   form_data.get("tenure"),
        "Balance":          f"${float(form_data.get('balance', 0)):,.2f}",
        "Number of Products": form_data.get("num_products"),
        "Has Credit Card":  "Yes" if form_data.get("has_cr_card") == "1" else "No",
        "Is Active Member": "Yes" if form_data.get("is_active") == "1" else "No",
        "Estimated Salary": f"${float(form_data.get('salary', 0)):,.2f}",
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

    features    = preprocess_telco(form_data)
    predictions = predict_telco(features, telco_models, telco_scaler)
    avg_prob    = predictions.pop("Average")

    customer_data = {
        "Name":              form_data.get("name", "Valued Customer"),
        "Gender":            form_data.get("gender"),
        "Senior Citizen":    "Yes" if form_data.get("senior_citizen") == "1" else "No",
        "Tenure (months)":   form_data.get("tenure"),
        "Contract":          form_data.get("contract"),
        "Internet Service":  form_data.get("internet_service"),
        "Monthly Charges":   f"${float(form_data.get('monthly_charges', 0)):,.2f}",
        "Total Charges":     f"${float(form_data.get('total_charges', 0)):,.2f}",
        "Payment Method":    form_data.get("payment_method"),
        "Paperless Billing": "Yes" if form_data.get("paperless_billing") == "Yes" else "No",
        "Tech Support":      form_data.get("tech_support"),
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


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mK]')

def _clean(text: str) -> str:
    return _ANSI_RE.sub('', str(text))


@app.route("/train")
def train_page():
    return render_template("train.html")


@app.route("/notebooks/<path:filename>")
def serve_notebook(filename):
    return send_from_directory(os.path.join(app.root_path, "notebooks"), filename,
                               as_attachment=True)


@app.route("/train/run", methods=["POST"])
def train_run():
    if "notebook" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    nb_file = request.files["notebook"]
    if not nb_file.filename.lower().endswith(".ipynb"):
        return jsonify({"error": "Only .ipynb files are accepted"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nb.ipynb")
        nb_file.save(path)
        with open(path) as fh:
            nb = nbformat.read(fh, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        success, exec_error = True, None
        try:
            ep.preprocess(nb, {"metadata": {"path": tmpdir}})
        except Exception as exc:
            success = False
            exec_error = _clean(str(exc))

        outputs = []
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            for out in cell.get("outputs", []):
                ot = out.get("output_type", "")
                if ot == "stream":
                    txt = _clean(out.get("text", ""))
                    if txt.strip():
                        outputs.append({"kind": "stream", "name": out.get("name", "stdout"), "text": txt})
                elif ot in ("display_data", "execute_result"):
                    d = out.get("data", {})
                    if "image/png" in d:
                        outputs.append({"kind": "image", "b64": d["image/png"]})
                    elif "text/html" in d:
                        outputs.append({"kind": "html", "html": d["text/html"]})
                    elif "text/plain" in d:
                        txt = _clean(d["text/plain"])
                        if txt.strip():
                            outputs.append({"kind": "text", "text": txt})
                elif ot == "error":
                    tb = _clean("\n".join(out.get("traceback",
                        [f"{out.get('ename','Error')}: {out.get('evalue','')}"])))
                    outputs.append({"kind": "error", "text": tb})

    return jsonify({"success": success, "error": exec_error, "outputs": outputs})


if __name__ == "__main__":
    app.run(debug=True)

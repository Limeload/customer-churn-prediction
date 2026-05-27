import os
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI

from utils import load_models, load_scaler, preprocess, predict_all

load_dotenv()

app = Flask(__name__)

models = load_models()
scaler = load_scaler()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


def get_openai_response(user_message: str) -> dict:
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    text = completion.choices[0].message.content
    return parse_llm_json(text)


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

    features = preprocess(form_data)
    predictions = predict_all(features, models, scaler)
    avg_prob = predictions.pop("Average")

    customer_data = {
        "Name": form_data.get("name", "Valued Customer"),
        "Age": form_data.get("age"),
        "Geography": form_data.get("geography"),
        "Gender": form_data.get("gender"),
        "Credit Score": form_data.get("credit_score"),
        "Tenure (years)": form_data.get("tenure"),
        "Balance": f"${float(form_data.get('balance', 0)):,.2f}",
        "Number of Products": form_data.get("num_products"),
        "Has Credit Card": "Yes" if form_data.get("has_cr_card") == "1" else "No",
        "Is Active Member": "Yes" if form_data.get("is_active") == "1" else "No",
        "Estimated Salary": f"${float(form_data.get('salary', 0)):,.2f}",
    }

    user_message = build_user_message(customer_data, predictions, avg_prob)
    llm_result = get_openai_response(user_message)

    return jsonify(
        {
            "churn_probability": avg_prob,
            "model_scores": predictions,
            "explanation": llm_result.get("explanation", ""),
            "email": llm_result.get("email", ""),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

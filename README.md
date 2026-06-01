# ChurnGuard — Customer Churn Predictor

Predicts customer churn using an ensemble of ML models, explains the result with GPT-4o or Groq, and generates a personalized retention email.

Supports two datasets: **Bank** (7 models) and **Telco** (5 models).

## Setup

```bash
git clone https://github.com/shraddharaom/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

Add your API keys to `.env`:

```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

Run the app:

```bash
python app.py
# open http://127.0.0.1:5000
```

## Training

Datasets are included in `data/`. To retrain:

```bash
python training/train_bank.py   # → models/bank/
python training/train_telco.py  # → models/telco/
```

## REST API

A FastAPI service is also available:

```bash
uvicorn api:app --reload
# docs at http://127.0.0.1:8000/docs
```

## Deployment

```bash
gunicorn app:app          # Flask (Heroku/Railway via Procfile)
docker build -t churnguard . && docker run -p 8000:8000 churnguard  # FastAPI
```

`render.yaml` is included for one-click Render deployment.

## Stack

ML: scikit-learn, XGBoost · Web: Flask, FastAPI · UI: Tailwind CSS, Chart.js · AI: OpenAI GPT-4o, Groq Llama 3.3

## License

[MIT](LICENSE) © 2026 Shraddha Rao

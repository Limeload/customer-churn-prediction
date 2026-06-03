# ChurnGuard — Customer Churn Predictor

Predicts customer churn using an ensemble of ML models, explains the result with GPT-4o or Groq, and generates a personalized retention email.

Supports two datasets: **Bank** (7 models) and **Telco** (5 models).

See [REQUIREMENTS.md](REQUIREMENTS.md) for a full feature list and requirement-to-code traceability map.

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

A FastAPI service is also available locally:

```bash
uvicorn api:app --reload
# docs at http://127.0.0.1:8000/docs
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status + model count |
| GET | `/models` | List loaded models |
| POST | `/predict/bank` | Bank churn prediction |
| POST | `/predict/telco` | Telco churn prediction |

## Cloud Deployment

Models are hosted on [Hugging Face Hub](https://huggingface.co/shraddharaom/churnguard-models) and downloaded automatically on first startup. The API is deployed on [Render](https://render.com).

**1. Upload models to Hugging Face Hub** (one-time, after training)

- Go to [huggingface.co/new](https://huggingface.co/new) and create a model repo named `churnguard-models`
- In the repo, create two folders: `bank/` and `telco/`
- Upload all `.pkl` files from `models/bank/` into `bank/` and from `models/telco/` into `telco/`

**2. Deploy API to Render**

Connect the repo to Render — `render.yaml` configures everything automatically (Docker runtime, health check, `HF_REPO_ID` env var). On first boot the API pulls all `.pkl` files from HF Hub and caches them.

```bash
gunicorn app:app          # Flask UI (Heroku/Railway via Procfile)
docker build -t churnguard . && docker run -p 8000:8000 churnguard  # FastAPI locally
```

The deployed API has CORS open to all origins, so any web app can call it directly.

**3. Deploy Flask UI to Vercel**

```bash
vercel --prod
```

Add these environment variables in the Vercel dashboard → Settings → Environment Variables:

```
OPENAI_API_KEY   = sk-...
GROQ_API_KEY     = gsk_...
RENDER_API_URL   = https://customer-churn-prediction-02k2.onrender.com
```

The Flask UI has no ML dependencies — it proxies all predictions to the Render API and only calls OpenAI/Groq for the explanation and email.

## Stack

ML: scikit-learn, XGBoost · Web: Flask, FastAPI · UI: Tailwind CSS, Chart.js · AI: OpenAI GPT-4o, Groq Llama 3.3

## License

[MIT](LICENSE) © 2026 Shraddha Rao

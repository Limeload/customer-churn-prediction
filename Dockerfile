FROM python:3.11-slim

# Non-root service account — no login shell, no password
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Dependency layer is cached independently from source changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entrypoint before source so its layer is also cached separately
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY . .

# ── Environment defaults (override at runtime; never bake secrets into the image) ──
# PORT             — TCP port gunicorn binds to (match your cloud platform's expected port)
# WEB_CONCURRENCY  — gunicorn worker count; 2 is safe for large in-process ML models
# WORKER_TIMEOUT   — seconds before a worker is killed; must exceed notebook exec time (600 s)
# LOG_LEVEL        — gunicorn/uvicorn log verbosity: debug | info | warning | error | critical
# HF_HOME          — HuggingFace Hub cache; kept inside /app so the non-root user can write
#
# Secrets (never set defaults here):
#   OPENAI_API_KEY, GROQ_API_KEY, TRAIN_API_KEY, HF_REPO_ID
ENV PORT=8000 \
    WEB_CONCURRENCY=2 \
    WORKER_TIMEOUT=600 \
    LOG_LEVEL=info \
    HF_HOME=/app/.cache/huggingface

RUN chown -R app:app /app
USER app

EXPOSE 8000

# Liveness probe — passes once /health returns HTTP 200 (models loaded).
# start-period gives the process time to pull model artifacts from HuggingFace on cold start.
# For Kubernetes, wire an equivalent httpGet probe to GET /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/health')"

ENTRYPOINT ["/entrypoint.sh"]

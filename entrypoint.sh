#!/bin/sh
# exec replaces the shell with gunicorn so PID 1 receives SIGTERM/SIGINT directly.
# All tunables come from ENV vars; override any of them at `docker run` time or in
# your orchestrator's environment spec without rebuilding the image.
set -e
exec gunicorn api:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers       "${WEB_CONCURRENCY:-2}" \
    --bind          "0.0.0.0:${PORT:-8000}" \
    --timeout       "${WORKER_TIMEOUT:-600}" \
    --graceful-timeout 30 \
    --log-level     "${LOG_LEVEL:-info}" \
    --access-logfile -

# Runtime image for the ML service (spec section 12).
#
# Must fit a ~512 MB free-tier container. Two things dominate that budget:
# what goes in the image, and how many copies of the models are resident.

FROM python:3.12-slim AS builder
WORKDIR /app

# gcc is needed to build a few wheels; it does not follow into the runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Only requirements.txt. requirements-train.txt (torch, optuna, mlflow, shap)
# is deliberately absent: the LSTM arrives as artifacts/lstm.onnx and is served
# through onnxruntime, so no deep learning framework is installed here.
COPY requirements.txt .
# The second step drops pip and setuptools: ~15 MB of build tooling the
# running service never uses.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pip setuptools 2>/dev/null || true


FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy only what the service runs. .dockerignore excludes frontend/, training/,
# tests/ and the large synthetic CSVs; being explicit here as well means a new
# top-level directory does not silently end up in the image.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY artifacts ./artifacts
COPY data/synthetic/live_deals.csv data/synthetic/live_activities.csv \
     data/synthetic/manifest.json ./data/synthetic/
COPY docker-entrypoint.sh ./docker-entrypoint.sh

RUN chmod +x ./docker-entrypoint.sh \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# One worker by default, not four.
#
# Each gunicorn worker runs the app's lifespan, which loads XGBoost, the ONNX
# session and the rep statistics — so N workers means N resident copies of the
# models. At WEB_CONCURRENCY=4 that alone exceeds the free tier's memory. It
# would also start four stream consumers, which is four times the scoring work
# for one event.
ENV WEB_CONCURRENCY=1

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["gunicorn", "app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]

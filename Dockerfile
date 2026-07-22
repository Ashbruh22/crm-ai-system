# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app

# Install system dependencies required for compilation (psycopg2, etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install into /opt/venv so any user can access it
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy requirements and install into venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder (world-readable, accessible by appuser)
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user and set ownership only for /app
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Run Gunicorn with Uvicorn workers
# Configuration driven by environment variables (e.g., WEB_CONCURRENCY)
CMD ["gunicorn", "app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--preload", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]

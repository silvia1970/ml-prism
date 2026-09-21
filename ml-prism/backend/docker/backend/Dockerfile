# ============================================================
# PRISM API — Dockerfile
# Flask + PyTorch (CPU-only) backend
# ============================================================
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps needed by numpy/pandas/scipy build wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# ---------- dependencies layer (cached) ----------
COPY requirements/requirements-api.txt requirements/requirements-api.txt

# Install API deps first (smaller, faster cache layer)
RUN pip install --no-cache-dir -r requirements/requirements-api.txt

# Install PyTorch CPU-only (keeps image ~1.5 GB smaller than CUDA builds)
RUN pip install --no-cache-dir torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install MinIO SDK
RUN pip install --no-cache-dir minio==7.2.15

# ---------- application layer ----------
# Copy API code
COPY api/ api/
COPY start_api.py .

# Copy ML models & normalisation stats (needed at inference time)
COPY models/ models/
COPY datasets/ datasets/

# Copy .env.example as fallback
COPY .env.example .env.example

# Default env vars (overridable via docker-compose / env_file)
ENV FLASK_ENV=production \
    PORT=5000 \
    DATA_DIR=api_data \
    MINIO_ENABLED=false \
    MINIO_ENDPOINT=minio:9000 \
    MINIO_ACCESS_KEY=minioadmin \
    MINIO_SECRET_KEY=minioadmin \
    MINIO_BUCKET=prism-data \
    MINIO_SECURE=false

EXPOSE 5000

# Health-check — hits the public /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["python", "start_api.py"]

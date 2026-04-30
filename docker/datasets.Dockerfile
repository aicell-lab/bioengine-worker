# Lightweight image for the BioEngine Datasets server.
# Only zarr and FastAPI are needed — no Ray, no MinIO, no Redis.
FROM python:3.11.9-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-datasets.txt /app/

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements-datasets.txt

COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/

RUN pip install --no-cache-dir ".[datasets]"

CMD ["python", "-m", "bioengine.datasets", "--help"]

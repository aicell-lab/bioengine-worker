# Use an official slim Python 3.11 image
FROM python:3.11.9-slim

# Set environment variables for installation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install minio and minio-client
RUN curl -Lo /bin/minio.20240716T234641Z https://dl.min.io/server/minio/release/linux-amd64/archive/minio.RELEASE.2024-07-16T23-46-41Z && \
    chmod +x /bin/minio.20240716T234641Z

RUN curl -Lo /bin/mc.20250408T153949Z https://dl.min.io/client/mc/release/linux-amd64/archive/mc.RELEASE.2025-04-08T15-39-49Z && \
    chmod +x /bin/mc.20250408T153949Z

# Set environment variables for Hypha
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    MINIO_EXECUTABLE_PATH=/bin

# Set up working directory
WORKDIR /app

# Copy requirements first
COPY requirements-datasets.txt /app/

# Install Python packages from requirements (covers datasets extra)
RUN pip install -U pip && \
    pip install -r requirements-datasets.txt

# Copy the rest of the application code
COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/

# Install the bioengine package
RUN pip install .[datasets]

CMD [ "/bin/bash" ]
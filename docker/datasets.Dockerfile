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

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

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

# Install the bioengine package with datasets extra
RUN pip install .[datasets]

CMD [ "/bin/bash" ]

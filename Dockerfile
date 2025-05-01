# Use an official slim Python 3.11 image
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python packages from requirements
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Install the package
RUN pip install .

CMD [ "/bin/bash" ]
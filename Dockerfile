# Use an official slim Python 3.11 image
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
    hypha-rpc \
    "ray[data,train,serve]" \
    pyyaml \
    python-dotenv

# Set up working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/
COPY ./hpc_worker /app/hpc_worker/

# Install the package
RUN pip install .

# Create a non-root user
RUN useradd -m -u 1000 worker && \
    chown -R worker:worker /app

USER worker

# Use the start script as the entrypoint and forward arguments
CMD [ "/bin/bash" ]
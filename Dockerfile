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

# Create a non-root user
RUN useradd -m -u 1000 bioengine_worker && \
    chown -R bioengine_worker:bioengine_worker .

USER bioengine_worker

# Use the start script as the entrypoint and forward arguments
CMD [ "/bin/bash" ]
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
    gnupg2 \
    ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.opensuse.org/repositories/devel:kubic:libcontainers:stable/Debian_11/Release.key | gpg --dearmor > /etc/apt/keyrings/podman.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/podman.gpg] https://download.opensuse.org/repositories/devel:kubic:libcontainers:stable/Debian_11/ /" > /etc/apt/sources.list.d/podman.list \
    && apt-get update \
    && apt-get install -y podman \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# (fix ray version because of DeprecationWarning for `ray.state.available_resources_per_node`)
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
    hypha-rpc \
    "ray[client,data,train,serve]==2.42.1" \
    pyyaml \
    python-dotenv

# Set up working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/
COPY ./bioengine_worker /app/bioengine_worker/

# Install the package
RUN pip install .

# Create a non-root user
RUN useradd -m -u 1000 worker && \
    chown -R worker:worker /app

USER worker

# Use the start script as the entrypoint and forward arguments
CMD [ "/bin/bash" ]
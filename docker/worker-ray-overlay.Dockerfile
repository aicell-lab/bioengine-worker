# Lightweight overlay over a published BioEngine worker image: swaps
# only the Ray pin without rebuilding system packages, Python, or the
# rest of the dependency tree. Pulls one image layer set, runs one pip
# install, sets one env var — done.
#
# Build:
#   docker build \
#       --build-arg BIOENGINE_IMAGE=ghcr.io/aicell-lab/bioengine-worker:0.9.0 \
#       --build-arg RAY_VERSION=2.54.1 \
#       -f docker/worker-ray-overlay.Dockerfile \
#       -t bioengine-worker:0.9.0-ray2.54.1 .
#
# BIOENGINE_IMAGE: the published image to use as the base. Update this
#   when you want a newer BioEngine release as the floor.
# RAY_VERSION:     the exact Ray release to swap in. Must satisfy the
#   range BioEngine supports (>=2.33.0, <2.56.0) — see pyproject.toml.

ARG BIOENGINE_IMAGE=ghcr.io/aicell-lab/bioengine-worker:0.9.0
FROM ${BIOENGINE_IMAGE}

ARG RAY_VERSION=2.55.1
RUN pip install --no-cache-dir "ray[client,serve]==${RAY_VERSION}"

ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

# Base Ray image with uv preinstalled, used as the cluster image for
# KubeRay deployments that BioEngine drives. The upstream
# rayproject/ray-ml images were deprecated in Ray 2.31 (all current
# ray-ml tags carry a .deprecated suffix), so we build on the supported
# rayproject/ray base. uv is needed by BioEngine's runtime_env pip
# installs to take advantage of Ray's faster uv-based installer
# (preferred over pip from Ray 2.47+).
#
# Build both variants — cpu for the head pod, gpu for workers:
#
#   docker build --build-arg RAY_VARIANT=cpu \
#       -f docker/ray-cluster.Dockerfile \
#       -t ghcr.io/aicell-lab/bioengine-ray:2.55.1-py311-cpu .
#   docker build --build-arg RAY_VARIANT=gpu \
#       -f docker/ray-cluster.Dockerfile \
#       -t ghcr.io/aicell-lab/bioengine-ray:2.55.1-py311-gpu .

ARG RAY_VERSION=2.55.1
ARG RAY_PY_TAG=py311
ARG RAY_VARIANT=cpu

FROM rayproject/ray:${RAY_VERSION}-${RAY_PY_TAG}-${RAY_VARIANT}

USER root
# pin pydantic + pydantic-core to the versions that Ray runtime_env
# venvs typically resolve to (driven by transitive constraints from
# torch/keras/tensorflow that pull pydantic ~= 2.11.0). When the base
# image ships a newer pydantic than the venv installs, cloudpickle-ed
# FieldInfo objects fail to unpickle across the boundary
# ("AttributeError: 'FieldInfo' object has no attribute 'exclude_if'"
# on Ray Serve replicas). Keeping the base aligned with the venv avoids
# that mismatch.
RUN pip install --no-cache-dir uv==0.5.0 "pydantic==2.11.0" "pydantic-core==2.33.0"
USER ray

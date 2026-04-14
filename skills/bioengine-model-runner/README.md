# BioImage.IO ModelRunner – Deployment Overview

## Overview

**ModelRunner** is a BioEngine application built on **Ray Serve** that provides standardized loading, validation, and execution of BioImage.IO models.
It is implemented as a **two-deployment architecture**:

1. **Entry Deployment (CPU)** – ingress, metadata handling, validation, caching, and request routing
2. **Runtime Deployment (GPU)** – unified execution environment for model testing and inference

This separation enables clear responsibility boundaries, efficient resource utilization, and isolation of heavy ML dependencies to GPU-backed workers.

## Core BioEngine Base Packages

These packages are shared by all BioEngine applications. They are intentionally kept **minimal** to maximize compatibility across environments and deployments.

```
httpx==0.28.1
hypha-rpc==0.21.11
pydantic==2.11.9
```

### Package Roles

* **httpx**
  Used for web communication, including model downloads and external service calls.

* **hypha-rpc**
  Provides communication with **Hypha** services and infrastructure components.

* **pydantic**
  Used for schema definitions of services.

## ModelRunner Architecture

ModelRunner consists of two Ray Serve deployments:

```
[ Client / User ]
        |
        v
+-----------------------+
| Entry Deployment (CPU)|
+-----------------------+
        |
        v
+------------------------+
| Runtime Deployment (GPU)|
+------------------------+
```

## ModelRunner – Entry Deployment (CPU)

**Responsibilities**

The entry deployment acts as the ingress and control plane for ModelRunner. It:

* Receives and validates user requests
* Downloads models and manages the local model cache
* Extracts and serves model metadata (RDF)
* Validates model metadata via `bioimageio.core`
* Forwards model testing and inference requests to the runtime deployment

This deployment is designed to run **without GPU** and remain lightweight.

**Dependencies**

```text
bioimageio.core==0.9.5
numpy==1.26.4
tqdm>=4.64.0
aiofiles>=23.0.0
```

## ModelRunner – Runtime Deployment (GPU)

**Responsibilities**

The runtime deployment is responsible for **model execution**. It:

* Runs model testing
* Executes inference
* Selects the appropriate backend (PyTorch, TensorFlow, ONNX, etc.) based on the model format

This deployment is GPU-backed and contains all heavy ML dependencies.

**Design Note – Universal Runtime**

All supported frameworks are installed together to form a **single universal runtime image**.
This avoids the need for model-type-specific runtimes and simplifies scheduling, deployment, and maintenance at the cost of a larger runtime environment.

**Dependencies**

```text
bioimageio.core==0.9.5
careamics==0.0.16
cellpose==3.1.1.2
numpy==1.26.4
onnxruntime==1.20.1
tensorflow==2.16.1
torch==2.5.1
torchvision==0.20.1
xarray==2025.1.2
```

Notes:

* This environment supports:

  * PyTorch models
  * TensorFlow models
  * ONNX models
  * Cellpose-based workflows
  * CAREamics-based workflows
* All versions are **strictly pinned** to ensure reproducibility and avoid runtime incompatibilities.

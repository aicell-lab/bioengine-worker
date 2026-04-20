<div align="center">
  <img src="docs/assets/bioengine-icon.svg" alt="BioEngine Logo" width="200"/>

  # BioEngine

  **Execution and adaptation layer for bioimage AI — run, screen, fine-tune, and deploy BioImage Model Zoo models through AI agents**

  [![GitHub](https://img.shields.io/badge/github-aicell--lab%2Fbioengine--worker-black?logo=github)](https://github.com/aicell-lab/bioengine-worker)
  [![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/orgs/aicell-lab/packages/container/package/bioengine-worker)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## What is BioEngine?

Foundation models and curated repositories have transformed bioimage AI, yet most researchers cannot readily run, adapt, or extend them on available hardware. **BioEngine fills this gap as the execution and adaptation layer between curated AI and scalable compute**, deployable on a laptop, workstation, or cluster.

BioEngine exposes its capabilities through a **SKILL.md contract** — a plain-text file designed for general-purpose AI agents to acquire domain knowledge and invoke GPU services directly. A scientist describes their imaging goal in plain language to any AI agent. The agent parses the contract, selects the appropriate service, and dispatches the GPU workflow. Results return as segmented images, ranked comparison tables, or a live web application, with no command-line access, software installation, or IT ticket required.

## Capabilities

| Capability | Description |
|-----------|-------------|
| **Model screening** | Agent queries BioImage Model Zoo, filters by compatibility, runs inference, and ranks by mAP — 58 candidates screened to 4 ranked in a single session |
| **Real-time inference** | Sub-second latency for live microscopy feedback loops; per-frame statistics (cell count, masks, morphology) returned to the controlling agent |
| **Collaborative fine-tuning** | Browser-based annotation against Cellpose-SAM pre-segmentations; fine-tuning triggered with one click; F1 rose from 0.36 → 0.71 across 1,600 training epochs on PlantSeg data |
| **Agent-built applications** | Agent generates deployment manifest, GPU workflow, and web UI from a single plain-language prompt; mean F1 = 0.920 ± 0.037 on Lucchi++ FIB-SEM benchmark |

## Quick Start

### Public BioEngine (no setup)

Test AI models instantly via the community instance:

1. Visit **[BioImage.IO Model Zoo](https://bioimage.io/#/models)**
2. Select any model and click **"TEST RUN MODEL"**
3. Execution runs on the public BioEngine worker (`bioimage-io/bioengine-worker`)

### CLI

```bash
pip install "bioengine[cli] @ git+https://github.com/aicell-lab/bioengine-worker.git"

# Call any service method
bioengine call bioimage-io/bioengine-worker get_status

# List running applications
bioengine apps list --worker bioimage-io/bioengine-worker

# Run a model
bioengine call bioimage-io/my-app predict --arg input=image.tif
```

### Python SDK

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": token})
worker = await server.get_service("bioimage-io/bioengine-worker")

# Screen models
status = await worker.get_status()

# Deploy an application
app_id = await worker.deploy_app(
    artifact_id="bioimage-io/cellpose-finetuning",
    application_id="cellpose-finetuning",
)

# Get application status
app_status = await worker.get_app_status(application_ids=[app_id])
```

### Deploy Your Own Worker

```bash
# Docker (single machine)
git clone https://github.com/aicell-lab/bioengine-worker.git
cd bioengine-worker
mkdir -p .bioengine data
UID=$(id -u) GID=$(id -g) docker compose up
```

See [Deployment Guide](docs/deployment-guide.md) for SLURM/HPC and Kubernetes modes.

## Architecture

```
┌────────────────────────────────────────┐
│            Hypha Server                │
│   (RPC, service discovery, artifacts)  │
└────────────┬───────────────────────────┘
             │ WebSocket / RPC
┌────────────▼───────────────────────────┐
│         BioEngineWorker                │
│                                        │
│  ┌─────────────────────────────────┐   │
│  │  SKILL.md contract              │   │
│  │  (agent-readable interface)     │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Ray Cluster                    │   │
│  │  (SLURM / single / Kubernetes)  │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Applications Manager           │   │
│  │  (Ray Serve lifecycle +         │   │
│  │   artifact management)          │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Datasets Manager               │   │
│  │  (Zarr HTTP streaming)          │   │
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

**Stack:** [Ray](https://www.ray.io/) + [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) for distributed GPU inference, [Hypha](https://docs.amun.ai/#/) for RPC service discovery and artifact management.

## Deployment Modes

| Mode | Use Case | Guide |
|------|----------|-------|
| `single-machine` | Workstation, development, small-scale | [→](docs/deployment-guide.md#mode-1-single-machine) |
| `external-cluster` | Kubernetes, pre-configured Ray clusters | [→](docs/deployment-guide.md#mode-2-external-cluster-kubernetes--kuberay) |
| `slurm` | HPC clusters with SLURM scheduler | [→](docs/deployment-guide.md#mode-3-slurm--hpc) |

## Worker Service API

The worker registers as a Hypha service. Key methods:

| Method | Description |
|--------|-------------|
| `get_status()` | Worker and cluster status |
| `deploy_app(artifact_id, ...)` | Deploy an application |
| `stop_app(application_id)` | Stop a running application |
| `get_app_status(application_ids)` | Status of specific applications |
| `list_apps()` | All deployed applications |
| `upload_app(...)` | Create/update application artifact |
| `run_code(code, ...)` | Run Python in a Ray task |
| `list_datasets()` | Available datasets |

## Applications

BioEngine applications are self-contained deployable units: a `manifest.yaml` + Python deployment code + optional web frontend. They can compose multiple AI models, wrap models with custom pre/post-processing, and expose arbitrary web UIs.

An AI agent given a SKILL.md contract can generate and deploy a new application from a plain-language prompt — generating the manifest, GPU workflow, and web interface with no manual programming.

**Reference apps:**
- [`apps/demo-app/`](apps/demo-app/) — minimal single-deployment app
- [`apps/cellpose-finetuning/`](apps/cellpose-finetuning/) — browser-based collaborative fine-tuning
- [`apps/model-runner/`](apps/model-runner/) — production BioImage Model Zoo inference

See [Applications Guide](docs/apps-guide.md) for full documentation.

## Documentation

- [Applications Guide](docs/apps-guide.md) — build and deploy BioEngine applications
- [Datasets Guide](docs/datasets-guide.md) — share and stream large scientific datasets
- [Deployment Guide](docs/deployment-guide.md) — single-machine, Kubernetes, and SLURM setup
- [BioEngine Dashboard](https://bioimage.io/#/bioengine) — web-based configuration and management

## Development Setup

```bash
conda activate bioengine-worker
pip install -e ".[worker,cli,dev]"
source .env   # loads HYPHA_TOKEN

# Run locally
python -m bioengine.worker \
    --mode single-machine \
    --head-num-gpus 1 \
    --workspace-dir ~/.bioengine \
    --debug

# Run tests
pytest tests/end_to_end/ -v
```

## Citation

BioEngine is described in the following preprint (bioRxiv, submitted):

> Mechtel N, Dettner Källander H, Cheng S, Zhang H, AI4Life Consortium, Ouyang W.
> **BioEngine: scalable execution and adaptation of bioimage AI through agent-readable interfaces.**
> *bioRxiv* (2025).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

BioEngine is built on [Ray](https://www.ray.io/), [Hypha](https://docs.amun.ai/#/), and [Zarr](https://zarr.readthedocs.io/).
Supported by the SciLifeLab & Wallenberg Data Driven Life Science Program, AI4Life (EU Horizon Europe grant 101057970), and the Berzelius GPU resource.

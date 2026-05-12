<div align="center">
  <img src="docs/assets/bioengine-icon.svg" alt="BioEngine Logo" width="200"/>

  # BioEngine

  **Run, screen, fine-tune, and deploy bioimage AI models — on any GPU hardware, through AI agents or a browser**

  [![GitHub](https://img.shields.io/badge/github-aicell--lab%2Fbioengine-black?logo=github)](https://github.com/aicell-lab/bioengine)
  [![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/orgs/aicell-lab/packages/container/package/bioengine)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Use BioEngine now — no setup needed

The community BioEngine instance is publicly available. Use it through the web interface or an AI agent — no installation required.

### 🌐 Web interface

| Page | What you can do |
|------|----------------|
| [**BioEngine Dashboard**](https://bioimage.io/#/bioengine) | Browse all available BioEngine workers, check their status, and open their management panels |
| [**Worker Dashboard**](https://bioimage.io/#/bioengine/worker?service_id=bioimage-io/bioengine-worker) | Deploy and manage apps on the public worker — cluster resources, running deployments, available apps |
| [**Deployment Wizard**](https://bioimage.io/#/bioengine/worker?service_id=bioimage-io/bioengine-worker) | Interactive setup guide for deploying your own BioEngine worker (Docker, SLURM, Kubernetes) — built into the worker dashboard |

### 🤖 AI agent

Load the BioEngine skill in any AI agent (Claude, GPT-4, etc.) by providing this URL:

```
https://bioimage.io/skills/bioengine/SKILL.md
```

The skill gives the agent everything it needs to screen models, run inference, fine-tune, and deploy applications — all through natural language. No command-line access or software installation required for end users.

---

## What is BioEngine?

Foundation models and curated repositories have transformed bioimage AI, yet most researchers cannot run, adapt, or extend them on available hardware. **BioEngine fills this gap as the execution layer between curated AI and scalable compute**, deployable on a laptop, workstation, or institutional cluster.

| Capability | Description |
|-----------|-------------|
| **Model screening** | Query BioImage Model Zoo, filter by compatibility, run inference, rank by mAP |
| **Real-time inference** | Sub-second latency for live microscopy feedback loops |
| **Collaborative fine-tuning** | Browser-based annotation + one-click fine-tuning; F1 rose 0.36 → 0.71 on PlantSeg |
| **Agent-built applications** | Agent generates manifest, GPU workflow, and web UI from a plain-language prompt |

BioEngine exposes its capabilities through a **SKILL.md contract** — a plain-text file that any AI agent can read to discover and invoke GPU services directly.

---

## Deploy your own BioEngine worker

Facility managers and system administrators can deploy a private BioEngine worker on any hardware. The [interactive deployment wizard](https://bioimage.io/#/bioengine/worker?service_id=bioimage-io/bioengine-worker) on BioImage.IO walks through the full setup for all supported modes.

| Mode | Use case |
|------|----------|
| **Single machine** | Workstation, development, small-scale |
| **SLURM / HPC** | Auto-scaling on institutional HPC clusters |
| **Kubernetes** | Production deployment with KubeRay |

```bash
# Quick start: Docker (single machine)
git clone https://github.com/aicell-lab/bioengine.git
cd bioengine
mkdir -p .bioengine data
UID=$(id -u) GID=$(id -g) docker compose up
```

See [Deployment Guide](docs/deployment-guide.md) for full instructions.

---

## Developer documentation

The sections below are for developers building on top of BioEngine or contributing to it.

### Python SDK

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": token})
worker = await server.get_service("bioimage-io/bioengine-worker")

status = await worker.get_status()
app_id = await worker.deploy_app(
    artifact_id="bioimage-io/cellpose-finetuning",
    application_id="cellpose-finetuning",
)
```

### CLI

```bash
pip install "bioengine[cli] @ git+https://github.com/aicell-lab/bioengine.git"

bioengine call bioimage-io/bioengine-worker get_status
bioengine apps list --worker bioimage-io/bioengine-worker
```

### Worker service API

| Method | Admin | Description |
|--------|:-----:|-------------|
| `get_status()` | | Worker and cluster status |
| `deploy_app(artifact_id, ...)` | ✓ | Deploy an application |
| `stop_app(application_id)` | ✓ | Stop a running application |
| `get_app_status(application_ids)` | | Status of specific applications |
| `list_apps()` | ✓ | All deployed applications |
| `upload_app(files, ...)` | | Create/update application artifact |
| `list_app_directories()` | ✓ | List app working directories on disk |
| `clear_app_directory(application_id)` | ✓ | Delete a stopped app's working directory |
| `run_code(code, ...)` | ✓ | Run Python in a Ray task |
| `list_datasets()` | | Available datasets |

### Applications

BioEngine applications are self-contained deployable units: a `manifest.yaml` + Python deployment code + optional web frontend. An AI agent given a SKILL.md contract can generate and deploy a new application from a plain-language description.

**Reference apps:**
- [`apps/demo-app/`](apps/demo-app/) — minimal single-deployment app
- [`apps/cellpose-finetuning/`](apps/cellpose-finetuning/) — browser-based collaborative fine-tuning
- [`apps/model-runner/`](apps/model-runner/) — production BioImage Model Zoo inference

See [Applications Guide](docs/apps-guide.md).

### Architecture

```
┌────────────────────────────────────────┐
│            Hypha Server                │
│   (RPC, service discovery, artifacts)  │
└────────────┬───────────────────────────┘
             │ WebSocket / RPC
┌────────────▼───────────────────────────┐
│         BioEngineWorker                │
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

**Stack:** [Ray](https://www.ray.io/) + [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) for distributed GPU inference · [Hypha](https://docs.amun.ai/#/) for RPC and artifact management

### Development setup

```bash
git clone https://github.com/aicell-lab/bioengine.git
cd bioengine
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

### Documentation

- [Applications Guide](docs/apps-guide.md) — build and deploy BioEngine applications
- [Datasets Guide](docs/datasets-guide.md) — share and stream large scientific datasets
- [Deployment Guide](docs/deployment-guide.md) — single-machine, Kubernetes, and SLURM setup

---

## Citation

> Mechtel N, Dettner Källander H, Cheng S, Zhang H, AI4Life Consortium, Ouyang W.
> **BioEngine: scalable execution and adaptation of bioimage AI through agent-readable interfaces.**
> *bioRxiv* (2025).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

BioEngine is built on [Ray](https://www.ray.io/), [Hypha](https://docs.amun.ai/#/), and [Zarr](https://zarr.readthedocs.io/).
Supported by the SciLifeLab & Wallenberg Data Driven Life Science Program, AI4Life (EU Horizon Europe grant 101057970), and the Berzelius GPU resource.

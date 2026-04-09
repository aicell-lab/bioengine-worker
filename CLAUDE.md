# BioEngine Worker — CLAUDE.md

## Project Overview

BioEngine Worker is a **Kubernetes-based compute engine and AI application deployment platform** for life sciences. It enables users to manage and deploy AI models (e.g., image analysis, segmentation) and build full-stack applications on top of those models. The platform runs on [Ray](https://www.ray.io/) and [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) for distributed inference and integrates with [Hypha](https://hypha.aicell.io/) for RPC service discovery, artifact management, and authentication.

**Top-level goals:**
- Deploy and serve AI models and applications at scale (single-machine, SLURM HPC, Kubernetes/external Ray clusters)
- Provide a unified API for remote management of model deployments via Hypha RPC
- Stream large scientific datasets with privacy-preserving access control
- Support both compute backends (Ray Serve deployments) and static frontends (artifact-hosted web UIs)

---

## Architecture

```
┌────────────────────────────────────────┐
│            Hypha Server                │
│   (RPC, service discovery, artifacts)  │
└────────────┬───────────────────────────┘
             │ WebSocket / RPC
┌────────────▼───────────────────────────┐
│         BioEngineWorker                │  ← bioengine/worker/worker.py
│  ┌─────────────────────────────────┐   │
│  │  RayCluster                     │   │  ← bioengine/ray/ray_cluster.py
│  │  (SLURM / single / external)    │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  AppsManager                    │   │  ← bioengine/applications/apps_manager.py
│  │  (Ray Serve lifecycle +         │   │
│  │   artifact management)          │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  BioEngineDatasets              │   │  ← bioengine/datasets/
│  │  (Zarr HTTP streaming)          │   │
│  └─────────────────────────────────┘   │
└────────────────────────────────────────┘
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `BioEngineWorker` | `bioengine/worker/worker.py` | Main orchestrator; Hypha service registration |
| `AppsManager` | `bioengine/applications/apps_manager.py` | Application lifecycle (deploy/stop/status) |
| `AppBuilder` | `bioengine/applications/app_builder.py` | Build Ray Serve apps from artifacts |
| `RayCluster` | `bioengine/ray/ray_cluster.py` | Ray cluster lifecycle (SLURM/local/external) |
| `BioEngineDatasets` | `bioengine/datasets/datasets.py` | Zarr dataset streaming |
| Artifact utilities | `bioengine/utils/artifact_utils.py` | Hypha artifact CRUD helpers |

---

## Application Manifest

Every BioEngine application is described by a `manifest.yaml` file. This file is stored in the Hypha artifact manager alongside Python deployment code.

### Required Fields

```yaml
name: My Application        # Human-readable name
id: my-application          # Unique lowercase ID (hyphens only)
id_emoji: "🔬"              # Visual emoji identifier
description: "..."          # Short description
type: ray-serve             # Must be "ray-serve"
deployments:                # List of Python class entry points
  - module_file:ClassName
authorized_users:
  - "*"                     # Or specific user IDs
```

### Optional Fields

```yaml
# Static frontend hosting — set frontend_entry to enable automatically
frontend_entry: "frontend/index.html"  # Entry HTML file (relative to artifact root)

# Metadata
format_version: "0.5.0"
version: "1.0.0"
authors:
  - {name: "...", affiliation: "...", github_user: "..."}
license: MIT
documentation: README.md
tutorial: tutorial.ipynb
tags: [bioengine, image-analysis]
```

When `frontend_entry` is set, BioEngine configures a `view_config` on the Hypha artifact during `save_application` (while the artifact is staged). The `frontend_entry` determines `root_directory` and `index` (e.g., `frontend/index.html` → `root_directory: "frontend"`, `index: "index.html"`). The resulting URL is:
```
https://hypha.aicell.io/{workspace}/view/{artifact-id}/
```

---

## Deployment Modes

| Mode | Description |
|------|-------------|
| `single-machine` | Local Ray cluster (dev, small-scale) |
| `external-cluster` | Connect to existing Ray cluster (Kubernetes) |
| `slurm` | Auto-scaling via SLURM job scheduler (HPC) |

---

## Worker Service API

The BioEngine worker registers as a Hypha service. Key methods:

| Method | Admin | Description |
|--------|:-----:|-------------|
| `get_status` | | Overall worker status |
| `check_access` | | Check caller permissions |
| `list_applications` | ✓ | List deployed applications |
| `run_application` | ✓ | Deploy an application from artifact |
| `stop_application` | ✓ | Stop a running application |
| `get_application_status` | | Status of specific application |
| `save_application` | ✓ | Create/update application artifact |
| `get_application_manifest` | ✓ | Get manifest for an application |
| `delete_application` | ✓ | Delete an application artifact |
| `execute_python_code` | ✓ | Run Python code in Ray task |
| `list_datasets` | | Available datasets |

---

## Development Setup

Use the existing `bioengine-worker` conda environment:

```bash
conda activate bioengine-worker
pip install -e ".[worker,dev]"
source .env   # loads HYPHA_TOKEN
```

### Run Locally

```bash
python -m bioengine.worker \
    --mode single-machine \
    --head-num-gpus 1 \
    --head-num-cpus 4 \
    --workspace-dir ~/.bioengine \
    --debug
```

For local artifact development, set:
```bash
export BIOENGINE_LOCAL_ARTIFACT_PATH=/path/to/bioengine-worker/tests
```

### Run Tests

```bash
pytest tests/end_to_end/ -v
```

---

## Code Conventions

- **Permissions**: Use `check_permissions(context, authorized_users, resource_name)` from `bioengine.utils`
- **Schema methods**: Decorate public API methods with `@schema_method` and use `pydantic.Field` for parameter descriptions
- **Logging**: Use `create_logger("ComponentName", ...)` from `bioengine.utils`
- **Artifact IDs**: Always fully qualified as `workspace/alias`
- **Artifact config**: Use `{"permissions": {"*": "r"}}` for public read; `{"website_root": "<dir>"}` for static hosting

## Key File Locations

- `bioengine/utils/artifact_utils.py` — All Hypha artifact CRUD helpers
- `bioengine/applications/apps_manager.py` — `run_application`, `save_application`, lifecycle
- `bioengine/applications/app_builder.py` — `build()` constructs Ray Serve app from artifact
- `tests/demo_app/` — Minimal example BioEngine app
- `bioengine_apps/model-runner/` — Production model-runner app
- `pyproject.toml` — Package version and dependencies

## Agent Workflow Guidelines

- **Simplicity First**: Make every change as minimal as possible.
- **No Regressions**: Only change what's necessary; read before modifying.
- **Prove It Works**: Test and verify before marking done.
- Planning lives in model context — do NOT create planning files in the repo.
- Lessons go in `.github/copilot-instructions.md` (durable repo-level knowledge).

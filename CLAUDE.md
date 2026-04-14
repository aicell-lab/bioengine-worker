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
- `bioengine_apps/demo-app/` — Reference BioEngine app (single deployment + frontend; ping, ascii_art, list_datasets, reverse_text)
- `bioengine_apps/composition-demo/` — Multi-deployment composition app (entry + 3 runtimes, reference for composition pattern)
- `bioengine_apps/model-runner/` — Production model-runner app
- `bioengine_apps/cellpose-finetuning/` — Cellpose fine-tuning app
- `pyproject.toml` — Package version and dependencies

---

## BioEngine Skills

The `skills/` directory contains AI agent skills for working with BioEngine. Skills are Markdown documents that describe a capability to an AI agent so it can autonomously use the relevant APIs.

### Skill structure

```
skills/
├── bioengine/                          # Main entry-point skill — load this first
│   ├── SKILL.md                        # Core skill: app deployment + references to app skills
│   ├── bioengine_cli/                  # CLI source (installable with pip install -e .)
│   └── references/
│       ├── manifest_reference.md       # Full manifest.yaml field reference
│       └── cli_reference.md            # CLI command reference
├── bioengine-model-runner/             # App skill: BioImage.IO model inference
│   ├── SKILL.md
│   ├── bioengine_cli/                  # Same CLI source (shared)
│   └── references/
│       ├── api_reference.md
│       ├── cli_reference.md
│       └── rdf_format.md
└── bioengine-cellpose-finetuning/      # App skill: Cellpose fine-tuning
    └── SKILL.md
```

### How skills are used

- **`skills/bioengine/SKILL.md`** is the single skill users pass to an AI agent. It covers app deployment, the CLI, and all platform concepts. It also lists app-specific skills for the agent to load autonomously when needed.
- **App skills** (`bioengine-model-runner`, `bioengine-cellpose-finetuning`) are deeper references for specific deployed services. They are not loaded by default — the agent reads `bioengine/SKILL.md` first and picks up the relevant app skill based on the user's task.

### Working on skills

- **Main skill** (`skills/bioengine/SKILL.md`): Update when the worker API, CLI commands, manifest format, or deployment rules change.
- **Model runner skill** (`skills/bioengine-model-runner/SKILL.md`): Update when the `bioengine_apps/model-runner/` service API changes (new inference parameters, new CLI commands, new model formats).
- **Cellpose fine-tuning skill** (`skills/bioengine-cellpose-finetuning/SKILL.md`): Update when `bioengine_apps/cellpose-finetuning/main.py` service API changes (new training parameters, new status fields, new export options, known pitfalls discovered during testing).

### Adding a new app skill

1. Create `skills/<app-name>/SKILL.md` with frontmatter (`name`, `description`, `license`, `metadata.service-id`).
2. Add an entry to the `## BioEngine app skills` table in `skills/bioengine/SKILL.md`.
3. Add the path to the `metadata.app-skills` list in the `bioengine/SKILL.md` frontmatter.

---

## External Skills

| Skill | URL | Purpose |
|-------|-----|---------|
| Hypha | https://hypha.aicell.io/ws/agent-skills/SKILL.md | Connect to the Hypha distributed computing platform — obtain tokens, discover workspaces, call services via RPC or HTTP, manage artifacts, deploy apps |

---

## Agent Workflow Guidelines

- **Simplicity First**: Make every change as minimal as possible.
- **No Regressions**: Only change what's necessary; read before modifying.
- **Prove It Works**: Test and verify before marking done.
- Planning lives in model context — do NOT create planning files in the repo.
- Lessons go in `.github/copilot-instructions.md` (durable repo-level knowledge).
- **Commit after live deploy**: Once an app in `bioengine_apps/` is verified working on the live worker, commit the source to git so the deployed version is always reproducible:
  ```bash
  git add bioengine_apps/my-app/
  git commit -m "feat(my-app): describe change, bump version to X.Y.Z"
  git push
  ```
  The version in `manifest.yaml` must be bumped whenever app code changes.
- **Version bump rules**:
  - **`deploy-applications.yml`** is manual-dispatch only (push trigger disabled — agents deploy directly via the worker API). Always bump `version` in the affected app's `manifest.yaml` when app code changes.
  - **`docker-publish.yml`** triggers on changes to any of these paths: `bioengine/**`, `requirements*.txt`, `pyproject.toml`, `docker/**`, `.dockerignore`. It enforces that `version` in `pyproject.toml` is strictly greater than the latest published image tag — CI will fail if not bumped. **Always create a PR** (never push directly to `main`) and **bump `version` in `pyproject.toml`** before opening the PR whenever any of those paths are touched.
- **Clean up test deployments**: After testing is complete, stop and delete any temporary apps deployed to the live worker:
  ```python
  await worker.stop_application(application_id=app_id)   # stops the Ray Serve deployment
  await worker.delete_application(artifact_id=app_id)    # deletes the Hypha artifact
  ```
  Do not leave test/throwaway deployments running on `bioimage-io/bioengine-worker` — they consume shared cluster resources.

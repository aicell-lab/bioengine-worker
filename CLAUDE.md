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
│  │  RayCluster                     │   │  ← bioengine/cluster/ray_cluster.py
│  │  (SLURM / single / external)    │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  AppsManager                    │   │  ← bioengine/apps/manager.py
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
| `AppsManager` | `bioengine/apps/manager.py` | Application lifecycle (deploy/stop/status) |
| `AppBuilder` | `bioengine/apps/builder.py` | Build Ray Serve apps from artifacts |
| `RayCluster` | `bioengine/cluster/ray_cluster.py` | Ray cluster lifecycle (SLURM/local/external) |
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

When `frontend_entry` is set, BioEngine configures a `view_config` on the Hypha artifact during `upload_app` (while the artifact is staged). The `frontend_entry` determines `root_directory` and `index` (e.g., `frontend/index.html` → `root_directory: "frontend"`, `index: "index.html"`). The resulting URL is:
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
| `list_apps` | ✓ | List deployed applications |
| `deploy_app` | ✓ | Deploy an application from artifact |
| `stop_app` | ✓ | Stop a running application |
| `get_app_status` | | Status of specific application |
| `upload_app` | ✓ | Create/update application artifact |
| `get_app_manifest` | ✓ | Get manifest for an application |
| `delete_app` | ✓ | Delete an application artifact |
| `run_code` | ✓ | Run Python code in Ray task |
| `list_datasets` | | Available datasets |

---

## Development Setup

Use the existing `bioengine-worker` conda environment:

```bash
conda activate bioengine-worker
pip install -e ".[worker,cli,dev]"
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

### Test Organization

- `tests/end_to_end/` — Integration tests for the core worker (applications manager, datasets, code executor)
- `tests/apps/` — Tests for individual BioEngine apps; one subfolder per app (e.g. `tests/apps/cellpose/`)
- `tests/apps/<app-name>/` — **All tests specific to a BioEngine app must go here**, not in `tests/end_to_end/`

---

## Code Conventions

- **Git author**: Always commit as `nilsmechtel` (`nils.mech@gmail.com`) unless explicitly told otherwise.
- **App authorized_users**: When deploying an app, the worker's `admin_users` and the deploying user are always injected into every key of `authorized_users` (including `"*"`). This guarantees admins can always call any app method regardless of the app's access rules.
- **Permissions**: Use `check_permissions(context, authorized_users, resource_name)` from `bioengine.utils`
- **Schema methods**: Decorate public API methods with `@schema_method` and use `pydantic.Field` for parameter descriptions
- **Logging**: Use `create_logger("ComponentName", ...)` from `bioengine.utils`
- **Artifact IDs**: Always fully qualified as `workspace/alias`
- **Artifact config**: Use `{"permissions": {"*": "r"}}` for public read; `{"website_root": "<dir>"}` for static hosting

## Key File Locations

- `bioengine/cli/` — BioEngine CLI (`bioengine` command); entry point is `bioengine.cli.cli:main`
- `bioengine/utils/artifact_utils.py` — All Hypha artifact CRUD helpers
- `bioengine/apps/manager.py` — `deploy_app`, `upload_app`, lifecycle
- `bioengine/apps/builder.py` — `build()` constructs Ray Serve app from artifact
- `apps/demo-app/` — Reference BioEngine app (single deployment + frontend; ping, ascii_art, list_datasets, reverse_text); **always keep version at 1.0.0**
- `apps/composition-demo/` — Multi-deployment composition app (entry + 3 runtimes, reference for composition pattern); **always keep version at 1.0.0**
- `apps/model-runner/` — Production model-runner app
- `apps/cellpose-finetuning/` — Cellpose fine-tuning app
- `pyproject.toml` — Package version and dependencies; install with `pip install -e ".[cli]"` for CLI use
- `../bioimage.io/public/skills/bioengine/` — Agent skills for working with BioEngine (separate repo)

---

## BioEngine Skills

Skills live in the **`../bioimage.io/public/skills/bioengine/`** directory (separate repo). They are Markdown documents that describe BioEngine capabilities to an AI agent.

### Skill structure

```
bioimage.io/public/skills/bioengine/
├── SKILL.md                        # Main entry-point — load this first
├── references/
│   ├── manifest_reference.md       # Full manifest.yaml field reference
│   └── cli_reference.md            # CLI command reference
└── apps/                           # App-specific subskills
    ├── model-runner/               # BioImage.IO model inference
    ├── cellpose-finetuning.md      # Cellpose fine-tuning
    └── cell-image-search.md        # Cell image search
```

The CLI source lives in `bioengine/cli/` in this repo. Install with `pip install "bioengine[cli] @ git+https://github.com/aicell-lab/bioengine-worker.git"` (or `pip install -e ".[cli]"` for development).

### How skills are used

- **`SKILL.md`** is the single entry-point skill. It covers app deployment, CLI, and all platform concepts, and references app subskills for deeper detail.
- **App subskills** (`apps/model-runner/`, `apps/cellpose-finetuning.md`, etc.) are referenced from `SKILL.md` — agents load them on demand when the task requires a specific service.

### Working on skills

- **Main skill** (`SKILL.md`): Update when the worker API, CLI commands, manifest format, or deployment rules change.
- **Model runner skill** (`apps/model-runner/`): Update when `apps/model-runner/` service API changes.
- **Cellpose fine-tuning skill** (`apps/cellpose-finetuning.md`): Update when `apps/cellpose-finetuning/main.py` service API changes (new training parameters, new status fields, new export options, known pitfalls discovered during testing).

### Adding a new app skill

1. Create `apps/<app-name>.md` (or `apps/<app-name>/` for multi-file) in the bioimage.io skills directory.
2. Add an entry to the app skills table in `SKILL.md`.

---

## External Skills

| Skill | URL | Purpose |
|-------|-----|---------|
| **BioEngine** | `../bioimage.io/public/skills/bioengine/SKILL.md` (fallback: https://bioimage.io/skills/bioengine/SKILL.md) | Deploy apps, call services, use the CLI — load this first when working with BioEngine |
| Hypha | https://hypha.aicell.io/ws/agent-skills/SKILL.md | Connect to the Hypha distributed computing platform — obtain tokens, discover workspaces, call services via RPC or HTTP, manage artifacts, deploy apps |

---

## Agent Workflow Guidelines

- **Simplicity First**: Make every change as minimal as possible.
- **No Regressions**: Only change what's necessary; read before modifying.
- **Prove It Works**: Test and verify before marking done.
- Planning lives in model context — do NOT create planning files in the repo.
- Lessons go in `.github/copilot-instructions.md` (durable repo-level knowledge).
- **Test on the live worker**: When working on a BioEngine app, test and debug by deploying to the live `bioimage-io/bioengine-worker` and calling the service directly. Do not write standalone test scripts for app behaviour — use the live service. Deploy with a stable `application_id` matching the artifact alias so the service is consistently addressable:
  ```python
  app_id = await worker.deploy_app(
      artifact_id='bioimage-io/my-app',
      version='1.2.3',
      application_id='my-app',   # gives stable service ID, not a random name
  )
  svc = await client.get_service(f'bioimage-io/{app_id}')
  ```
- **CRITICAL — artifact ID ≠ app ID, omitting `application_id` always creates a NEW instance**: One artifact can be deployed multiple times with different `application_id`s. `deploy_app(artifact_id)` without `application_id` **always spawns a brand-new instance with a random ID** — it never updates an existing one. To update a running app, you MUST pass its `application_id` AND the new `version` explicitly:
  ```python
  # WRONG — creates a new random instance, does NOT update cellpose-finetuning:
  await worker.deploy_app('bioimage-io/cellpose-finetuning')
  
  # CORRECT — updates the running 'cellpose-finetuning' instance to the new version:
  await worker.deploy_app(
      'bioimage-io/cellpose-finetuning',
      application_id='cellpose-finetuning',
      version='0.0.28',
  )
  ```
  Before deploying, always check `list_apps()` or `get_app_status(None)` to find the correct running `application_id`.
- **Commit after live deploy**: Once an app in `apps/` is verified working on the live worker, commit the source to git so the deployed version is always reproducible:
  ```bash
  git add apps/my-app/
  git commit -m "feat(my-app): describe change, bump version to X.Y.Z"
  git push
  ```
  The version in `manifest.yaml` must be bumped whenever app code changes.
- **Version bump rules**:
  - **`deploy-applications.yml`** is manual-dispatch only (push trigger disabled — agents deploy directly via the worker API). Always bump `version` in the affected app's `manifest.yaml` when app code changes.
  - **`docker-publish.yml`** triggers on changes to any of these paths: `bioengine/**`, `requirements*.txt`, `pyproject.toml`, `docker/**`, `.dockerignore`. It enforces that `version` in `pyproject.toml` is strictly greater than the latest published image tag — CI will fail if not bumped. **Always create a PR** (never push directly to `main`) and **bump `version` in `pyproject.toml`** before opening the PR whenever any of those paths are touched.
- **PRs are only required for changes that trigger `docker-publish.yml`** (i.e. changes under `bioengine/**`, `requirements*.txt`, `pyproject.toml`, `docker/**`, `.dockerignore`). Changes to `apps/**` only — push directly to `main`, no PR needed.
- **NEVER push directly to `main` for worker/package code.** Always use a feature branch and open a PR for any change that touches the paths above. If the user asks you to push directly to main for those paths, refuse and create a PR instead.
- **Always open the PR immediately after pushing the branch** using the GitHub PAT from `.env` so the user can see and review it without having to navigate to GitHub manually. Never merge a PR — that is always left to the user.
- **Clean up test deployments**: After testing is complete, stop and delete any temporary apps deployed to the live worker:
  ```python
  await worker.stop_app(application_id=app_id)   # stops the Ray Serve deployment
  await worker.delete_app(artifact_id=app_id)    # deletes the Hypha artifact
  ```
  Do not leave test/throwaway deployments running on `bioimage-io/bioengine-worker` — they consume shared cluster resources.

# BioEngine Worker - AI Development Guide

## Architecture Overview

BioEngine Worker is an enterprise distributed AI platform with 3-tier architecture:

**Core Worker (`bioengine/worker/worker.py`)**
- Orchestrates 3 component managers: RayCluster, AppsManager, DatasetsManager  
- Registers as Hypha service with admin permission system
- Supports 3 modes: `slurm` (HPC), `single-machine` (local), `external-cluster`

**Component Managers Pattern**
- Each manager: async lifecycle (initialize → start → monitor → cleanup)
- All implement `get_status()`, admin permission checking, graceful shutdown
- Shared logger configuration with `create_logger()` from `utils/`

**Service Integration**
- Hypha RPC server for remote access (authenticate via `HYPHA_TOKEN`)
- Ray cluster for distributed computing + Ray Serve for model deployment
- HTTP/WebSocket services for dataset streaming and real-time communication

## Development Workflows

### Environment Setup
Use the existing `bioengine-worker` conda environment — do not create a new one.
```bash
conda activate bioengine-worker
pip install -r requirements.txt
pip install -r requirements-test.txt
```

The `HYPHA_TOKEN` required for tests and Hypha service access is stored in the `.env` file at the workspace root. Load it before running commands:
```bash
source .env
```

### Local Development
```bash
# Start worker locally (single-machine mode)
python -m bioengine.worker \
    --mode single-machine \
    --head-num-gpus 1 \
    --head-num-cpus 4 \
    --head-memory-in-gb 24 \
    --workspace-dir /home/nmechtel/.bioengine \
    --startup-applications '{"artifact_id": "ws-user-github|49943582/demo-app", "disable_gpu": true}' \
    --client-id "<debug-client-id>" \
    --debug
```

> **Note:** Do not reuse the same `--client-id` in a short period of time to avoid service registration conflicts.

Once running, the worker service is accessible at:
```
https://hypha.aicell.io/ws-user-github%7C49943582/services/<debug-client-id>:bioengine-worker
```

To deploy an application on a running worker, call `run_application` via the service URL, e.g.:
```
https://hypha.aicell.io/ws-user-github%7C49943582/services/<debug-client-id>:bioengine-worker/run_application?artifact_id=ws-user-github|49943582/model-runner
```

If `BIOENGINE_LOCAL_ARTIFACT_PATH` is set, the worker loads the app from the local path instead of the artifact registry. The path must contain the app directory — e.g.:
- `bioengine_apps/` for apps like `model-runner` and `cellpose-finetuning`
- `tests/` for the `composition-app`

```bash
# When working on BioEngine apps, set the local artifact path:
export BIOENGINE_LOCAL_ARTIFACT_PATH=/data/nmechtel/bioengine-worker/bioengine_apps

# With Docker Compose
UID=$(id -u) GID=$(id -g) docker compose up

# HPC mode with Apptainer
bash scripts/start_hpc_worker.sh --mode slurm
```

### Worker Service API

The BioEngine worker exposes the following Hypha service endpoints:

| Method | Description | Admin required |
|--------|-------------|:--------------:|
| `get_status` | Get overall worker status | |
| `stop_worker` | Stop the worker | ✓ |
| `check_access` | Check caller's access level | |
| `get_logs` | Retrieve worker logs | ✓ |
| `list_datasets` | List available datasets | |
| `execute_python_code` | Execute Python code in a Ray task | ✓ |
| `save_application` | Save/update an application artifact | ✓ |
| `list_applications` | List deployed applications | ✓ |
| `get_application_manifest` | Get manifest for an application | ✓ |
| `delete_application` | Delete an application | ✓ |
| `run_application` | Deploy and start an application | ✓ |
| `stop_application` | Stop a running application | ✓ |
| `stop_all_applications` | Stop all running applications | ✓ |
| `get_application_status` | Get status of a specific application | |

## Code Patterns & Conventions

### Component Manager Structure
```python
class ComponentManager:
    def __init__(self, log_file=None, debug=False):
        self.logger = create_logger("ComponentName", ...)
        
    async def initialize(self, server, admin_users):
        # Set server connection, admin permissions
        
    async def get_status(self) -> Dict:
        # Return comprehensive status dict
        
    async def cleanup(self, context):
        # Check admin permissions, cleanup resources
```

### Permission Checking Pattern
```python
from bioengine.utils import check_permissions

@schema_method  # For Hypha service methods
async def admin_operation(self, context=None):
    check_permissions(
        context=context,
        authorized_users=self.admin_users,
        resource_name="operation description"
    )
```

### Ray Cluster Modes
- **SLURM**: Auto-scaling workers via SLURM job submission, container-based with Apptainer
- **single-machine**: Local Ray cluster with resource limits  
- **external-cluster**: Connect to existing Ray cluster, data access only

### Dataset Management
- Manifest-driven: `dataset_dir/manifest.yaml` + `*.zarr` files
- HTTP streaming via `HttpZarrStore` 
- Load → Access → Close pattern through worker service API

### Application Deployment  
- Artifact-based: `manifest.yaml` defines deployment config + Python entry point
- Ray Serve deployment with resource allocation checking
- WebSocket + WebRTC support for real-time applications

## Key File Locations

### Core Components
- `bioengine/worker/worker.py` - Main orchestrator
- `bioengine/ray/ray_cluster.py` - Ray cluster lifecycle  
- `bioengine/applications/apps_manager.py` - Model deployment management
- `bioengine/datasets/datasets_manager.py` - Dataset streaming services

### Utilities & Shared Code
- `bioengine/utils/` - Logging, permissions, context creation
- `bioengine/ray/slurm_workers.py` - HPC autoscaling logic
- `bioengine/ray/proxy_actor.py` - Ray actor for cluster coordination

### Testing
- `tests/test_end_to_end/` - Integration tests via Hypha service API
- `tests/conftest.py` - Shared fixtures (requires `HYPHA_TOKEN`)
- Use existing `bioengine_worker_service_id` fixture for most tests

### Configuration & Deployment
- `pyproject.toml` - Package config, Ray 2.33.0 + hypha-rpc dependencies
- `docker-compose.yaml` - Local development setup
- `scripts/start_hpc_worker.sh` - HPC deployment helper

## Common Debugging

### Port Conflicts
Ray auto-allocates ports starting from defaults (6379, 8000, etc.). Check `ray_cluster.py/_set_cluster_ports()` for logic.

### SLURM Issues  
Worker containers mount `${HOME}/bioengine`. Check `slurm_workers.py` for job submission patterns.

### Hypha Connection
Authentication via `HYPHA_TOKEN` env var (loaded from `.env` file) or interactive login. Service discovery uses workspace-scoped IDs.

### Resource Allocation
Apps check available resources before deployment. SLURM mode can scale up workers if resources insufficient.

Always check component logs - each manager has detailed logging for troubleshooting distributed operations.

## Agent Workflow Guidelines

### Core Principles
- **Simplicity First:** Make every change as simple as possible. Touch minimal code.
- **No Laziness:** Find root causes. Avoid temporary fixes. Maintain senior-level standards.
- **Minimal Impact:** Only change what's necessary. Avoid introducing regressions.
- **Clarity Over Cleverness:** Prefer readable, obvious solutions over clever ones.
- **Consistency Over Preference:** Follow existing patterns unless there is a strong reason to improve them.
- **Prove It Works:** Evidence beats assumption — test and verify.
- **Long-Term Thinking:** Optimize for maintainability and team comprehension, not speed of completion.

### Workflow Orchestration

**Plan Mode Default**
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
- If something goes sideways, STOP and re-plan — don't push through a flawed approach.
- Use plan mode for verification, not just implementation.
- Write clear specs upfront to reduce ambiguity.
- Planning lives in model context — do NOT create planning files in the repo.

**Subagent Strategy**
- Use subagents to keep the main context window clean.
- Offload research, exploration, and parallel analysis.
- For complex problems, prefer structured reasoning over muddling through.
- One focused task per subagent for clarity and execution quality.

**Self-Improvement Loop (Durable Only)**
- After corrections, pause and ask: "Is there a reusable, repo-level lesson here?"
- Only create a lesson if it will matter for future work in this repo and is useful for other contributors (not just this session).
- Convert those into clear, durable rules in `tasks/lessons.md`.
- Do not log small mistakes, one-off decisions, or temporary context.

**Verification Before Done**
- Never mark a task complete without proving it works.
- Diff behavior between main and your changes when relevant.
- Ask: "Would a staff engineer approve this?"
- Run tests, check logs, and demonstrate correctness.
- If you cannot prove it works, it is not done.

**Demand Elegance (Balanced)**
- For non-trivial changes: pause and ask "Is there a more elegant way?"
- If a fix feels hacky, implement the clean solution.
- Skip over-engineering for simple, obvious fixes.
- Challenge your own work before presenting it.

**Autonomous Bug Fixing**
- When given a bug report: aim to fix it without hand-holding.
- Identify logs, errors, failing tests — then resolve them.
- Minimize context switching required from the user.
- Proactively fix failing CI tests.

### Task Management

1. **Plan First:** Create a clear plan with checkable steps (kept in model context).
2. **Verify Plan:** Confirm direction before heavy implementation when appropriate.
3. **Track Progress:** Mark steps complete as you go (in-context).
4. **Explain Changes:** Provide high-level summaries at meaningful milestones.
5. **Document Results:** Capture rationale and outcomes in PR descriptions or commits.
6. **Capture Lessons:** Update `tasks/lessons.md` only when a durable, repo-level rule emerges.

No persistent task tracking files.

### Lessons Management (`tasks/lessons.md`)

`tasks/lessons.md` is the single source of truth for durable lessons. Each lesson must capture a recurring pattern, be useful to future contributors, and stand alone without relying on session context.

**Promote a lesson when:**
- It would have prevented a real bug or painful rework.
- The same class of mistake is likely to recur.
- It changes how future work should be done.
- It qualifies as institutional knowledge.

**Do not log:**
- One-off debugging stories.
- Temporary design decisions.
- Minor implementation details.
- Planning artifacts.

**Writing rules:** Short and specific. Written as rules, constraints, or patterns. Focused on prevention and better defaults. No anecdotes.

**Review discipline:** Skim `tasks/lessons.md` before substantial work. Merge or prune redundant lessons. Delete lessons that stop being true or useful.

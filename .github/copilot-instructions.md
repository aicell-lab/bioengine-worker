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
- `tests/` for the `demo-app` and `composition-app`

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

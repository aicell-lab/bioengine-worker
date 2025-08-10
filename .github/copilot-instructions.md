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
```bash
conda create -n bioengine-worker python=3.11.9
conda activate bioengine-worker
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Testing Patterns
```bash
# Run end-to-end tests (requires HYPHA_TOKEN in .env)
pytest tests/test_end_to_end/ -v

# Test single component
pytest tests/test_end_to_end/test_app_manager.py -v

# Use bioengine-worker conda environment for tests
conda run -n bioengine-worker pytest tests/ -v
```

### Local Development
```bash
# Start worker locally
python -m bioengine.worker --mode single-machine --debug

# With Docker Compose
UID=$(id -u) GID=$(id -g) docker compose up

# HPC mode with Apptainer
bash scripts/start_hpc_worker.sh --mode slurm
```

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
- Manifest-driven: `dataset_dir/manifest.yml` + `*.zarr` files
- HTTP streaming via `HttpZarrStore` 
- Load → Access → Close pattern through worker service API

### Application Deployment  
- Artifact-based: `manifest.yml` defines deployment config + Python entry point
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
- `docker-compose.yml` - Local development setup
- `scripts/start_hpc_worker.sh` - HPC deployment helper

## Common Debugging

### Port Conflicts
Ray auto-allocates ports starting from defaults (6379, 8000, etc.). Check `ray_cluster.py/_set_cluster_ports()` for logic.

### SLURM Issues  
Worker containers mount `/tmp/bioengine` and `/data`. Check `slurm_workers.py` for job submission patterns.

### Hypha Connection
Authentication via `HYPHA_TOKEN` env var or interactive login. Service discovery uses workspace-scoped IDs.

### Resource Allocation
Apps check available resources before deployment. SLURM mode can scale up workers if resources insufficient.

Always check component logs - each manager has detailed logging for troubleshooting distributed operations.

---
name: bioengine
description: Complete BioEngine skill — builds, deploys, and manages BioEngine applications on Ray Serve/Hypha, and provides access to all pre-deployed BioEngine services (model runner, Cellpose fine-tuning, and more). Use this as the single entry point for any BioEngine task. For deep usage of specific services, load the referenced app skills below.
license: MIT
compatibility: Requires Python ≥3.9, Ray ≥2.33.0, hypha-rpc ≥0.21.11, pydantic ≥2.11.0, httpx ≥0.28.1. Network access to hypha.aicell.io required.
metadata:
  docs-source: https://raw.githubusercontent.com/aicell-lab/bioengine-worker/refs/heads/main/bioengine_apps/README.md
  cli-package: bioengine (pip install bioengine)
  app-skills:
    - ../bioengine-model-runner/SKILL.md      # BioImage.IO model search and inference
    - ../bioengine-cellpose-finetuning/SKILL.md  # Cellpose-SAM fine-tuning on custom data
---

# BioEngine Apps

BioEngine applications are Ray Serve classes packaged as Hypha artifacts and deployed across single machines to HPC clusters. They expose model inference, distributed training, data exploration, and custom processing pipelines as Hypha services.

## Application structure

Every BioEngine app is a directory with at minimum:

```
my-app/
├── manifest.yaml          # Required: app identity and deployment config
├── my_deployment.py       # Required: Ray Serve class implementation
├── README.md              # Optional
└── tutorial.ipynb         # Optional
```

## Quick start — minimal deployment

**manifest.yaml**:
```yaml
id: "my-unique-app"
id_emoji: "🔬"
name: "My App"
description: "What this app does"
type: ray-serve
authorized_users:
  - "*"
deployments:
  - "my_deployment:MyDeployment"
```

**my_deployment.py**:
```python
import logging
from ray import serve
from hypha_rpc.utils.schema import schema_method

logger = logging.getLogger("ray.serve")

@serve.deployment(
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
    max_ongoing_requests=10,
)
class MyDeployment:
    def __init__(self):
        pass  # sync init

    async def async_init(self) -> None:
        pass  # async setup (optional)

    @schema_method
    async def process(self, input_data: str) -> dict:
        return {"result": f"Processed: {input_data}"}
```

## Deploying with the CLI

The CLI is bundled in this skill at `bioengine_cli/`. Install its dependencies once:

```bash
pip install hypha-rpc httpx numpy tifffile Pillow click
```

Use the bundled CLI directly from the skill root:

```bash
python -m bioengine_cli apps deploy ./my-app/
python -m bioengine_cli apps list
python -m bioengine_cli apps status <app-id>
```

Or install as a package for the shorter `bioengine` command:

```bash
pip install -e bioengine_cli/   # installs from bundled source
# or: pip install bioengine     # install from PyPI
```

Set credentials (required for apps commands):

```bash
# Public BioEngine worker (shared, available without provisioning your own):
export BIOENGINE_WORKER_SERVICE_ID=bioimage-io/bioengine-worker

# For private/institutional workers, use your workspace:
# export BIOENGINE_WORKER_SERVICE_ID=my-workspace/bioengine-worker

export HYPHA_TOKEN=<your-token>   # required even with the public worker
```

Then deploy:

```bash
# Upload and deploy in one step
bioengine apps deploy ./my-app/

# Or separately: upload, then deploy
bioengine apps upload ./my-app/
bioengine apps run my-workspace/my-app

# Update an existing running app in-place (preserves env vars from previous deployment)
bioengine apps upload ./my-app/
bioengine apps run my-workspace/my-app --app-id existing-app-id

# Monitor
bioengine apps list
bioengine apps status <app-id>
bioengine apps logs <app-id>
bioengine apps stop <app-id>
```

## Updating a running application

To update a running application without losing its environment variables (e.g., secrets, tokens), use `--app-id` with the **same ID as the running app**. The worker reuses **all env vars and init args/kwargs** from the previous deployment — nothing needs to be re-supplied:

```bash
# 1. Upload new code (updates the artifact in-place)
bioengine apps upload ./my-app/

# 2. Redeploy with the SAME app-id → env vars + init args are preserved
bioengine apps run my-workspace/my-app --app-id my-running-app-id
```

This is the correct way to push code fixes to production apps. Creating a new app (without `--app-id`) starts fresh with no env vars or init args, which will break apps that depend on `HYPHA_TOKEN`, workspace-scoped tokens, or other secrets.

**Apps that require a `bioimage-io` workspace token** (e.g., `model-runner`, `cellpose-finetuning`) must be started with a `HYPHA_TOKEN` that has write access to the `bioimage-io` workspace. When updating such an app using `--app-id`, the token is automatically inherited — do **not** pass a new token unless intentionally rotating it.

## CLI reference

| Command | Purpose |
|---|---|
| `bioengine apps deploy <dir>` | Upload and deploy a local app directory |
| `bioengine apps upload <dir>` | Upload app files to Hypha artifact storage |
| `bioengine apps run <artifact-id>` | Deploy an uploaded artifact |
| `bioengine apps list` | List all artifacts in the workspace |
| `bioengine apps status [app-id...]` | Show status of running deployments |
| `bioengine apps logs <app-id>` | Show logs from a running deployment |
| `bioengine apps stop <app-id>` | Stop and remove a running deployment |

All commands accept `--json` for machine-parseable output. Use `--help` on any command for full options.

## Deployment rules

**Import pattern**: Only standard library + BioEngine runtime at module top. Import external packages inside methods — required by Ray's serialization constraints:

```python
# Top level: OK
import os, asyncio, logging
from ray import serve
from hypha_rpc.utils.schema import schema_method

# Inside method: required for external packages
async def predict(self, image):
    import numpy as np       # correct
    import torch             # correct
    ...
```

**Blocking operations**: Use thread pools for CPU-intensive or synchronous calls:

```python
from concurrent.futures import ThreadPoolExecutor

class MyDeployment:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process(self, data: str) -> dict:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self._blocking_op, data)
        return {"result": result}
```

**Logging**: Use Ray Serve logger — do not use `print`:

```python
logger = logging.getLogger("ray.serve")
logger.info("message")
```

**GPU resources**:

```python
@serve.deployment(ray_actor_options={"num_gpus": 1, "num_cpus": 4, "memory": 8 * 1024**3})
```

Omit `num_gpus` for CPU-only. Pass `--no-gpu` to `bioengine apps run` to override at deploy time.

**Local testing** (no Ray Serve needed):

The `@serve.deployment` decorator wraps the class and prevents direct instantiation. Mock the decorator before importing your module:

```python
import unittest.mock, sys

# Patch ray.serve before any import of your deployment file
ray_mock = unittest.mock.MagicMock()
ray_mock.serve.deployment = lambda **kwargs: (lambda cls: cls)
sys.modules["ray"] = ray_mock
sys.modules["ray.serve"] = ray_mock.serve

# Now import and instantiate normally
from my_deployment import MyDeployment

instance = MyDeployment()
await instance.async_init()   # if defined
result = await instance.process("test")
```

Note: `async_init()` is not called automatically outside Ray Serve — call it manually in tests if it sets up resources your methods need.

## Multi-deployment composition

Use `DeploymentHandle` for inter-deployment communication:

```python
from ray.serve.handle import DeploymentHandle

@serve.deployment(ray_actor_options={"num_cpus": 1})
class MainDeployment:
    def __init__(self, helper_deployment: DeploymentHandle):
        self.helper = helper_deployment

    @schema_method
    async def process(self, data: str) -> dict:
        preprocessed = await self.helper.preprocess.remote(data)
        return {"result": preprocessed}
```

List all deployments in manifest — first entry is the main service entry point.

## Lifecycle methods

```python
async def async_init(self) -> None:
    # async setup after __init__
    self.model = await self.load_model()

async def test_deployment(self) -> None:
    # background health check on startup
    result = await self.model.predict("test")
    assert "expected" in result
```

## Built-in resources available in every deployment

**Environment variables**:

| Variable | Value |
|---|---|
| `HOME` | App working directory |
| `TMPDIR` | Temporary directory |
| `HYPHA_SERVER_URL` | Hypha server endpoint |
| `HYPHA_WORKSPACE` | Current workspace |
| `HYPHA_ARTIFACT_ID` | Full artifact identifier |
| `BIOENGINE_WORKER_SERVICE_ID` | Worker service ID |
| `HYPHA_TOKEN` | User auth token (if provided) |

**Dataset access** via `self.bioengine_datasets`:

```python
datasets = await self.bioengine_datasets.list_datasets()
files = await self.bioengine_datasets.list_files(dataset_id=did, dir_path="sub/dir")
content = await self.bioengine_datasets.get_file(dataset_id=did, file_path="data.txt")
```

## Passing runtime configuration

Pass environment variables and kwargs at deploy time:

```bash
# Pass env vars (prefix with _ to mark as secret)
bioengine apps run my-workspace/my-app --env MODEL_PATH=/data/model.pt --env _API_KEY=secret

# Custom app ID for in-place updates
bioengine apps run my-workspace/my-app --app-id production-v1
```

## Secret management

Prefix env var names with `_` to mark as secret (hidden in status, stripped from key when accessed):

```bash
bioengine apps run my-workspace/my-app --env _API_KEY=secret
# Access inside deployment as: os.environ["API_KEY"]
```

## Monitor and manage

```bash
# Check all running apps
bioengine apps status

# Detailed status for one app (with 100 log lines)
bioengine apps status my-app-id --logs 100 --json

# View logs
bioengine apps logs my-app-id --tail 200

# Stop an app
bioengine apps stop my-app-id
```

## Connect to deployed service

After deployment, connect via Hypha:

**WebSocket** (server-routed):
```python
from hypha_rpc import connect_to_server
server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": token})
service = await server.get_service(websocket_service_id)
result = await service.process(input_data="Hello")
```

**WebRTC** (peer-to-peer, lower latency):
```python
peer = await get_rtc_service(server, webrtc_service_id)
svc = await peer.get_service(app_id)
result = await svc.process(input_data="Hello")
```

## References

- Full manifest fields and deployment config: [references/manifest_reference.md](references/manifest_reference.md)
- CLI source and advanced usage: [references/cli_reference.md](references/cli_reference.md)

## BioEngine app skills

The following app-specific skills provide deeper documentation for pre-deployed BioEngine services. Load them autonomously when the user's task involves one of these services:

| Skill file | When to load |
|---|---|
| [`../bioengine-model-runner/SKILL.md`](../bioengine-model-runner/SKILL.md) | User wants to search, run inference on, or validate BioImage.IO models (segmentation, denoising, detection, restoration) |
| [`../bioengine-cellpose-finetuning/SKILL.md`](../bioengine-cellpose-finetuning/SKILL.md) | User wants to fine-tune Cellpose on their own annotated microscopy images, monitor training metrics, or export a trained model |

More app skills will be added here as new BioEngine applications are developed.

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

BioEngine applications are Ray Serve classes packaged as Hypha artifacts and deployed on Ray clusters. They expose model inference, distributed training, data exploration, and custom processing pipelines as Hypha RPC services — accessible from browser, Python, or any HTTP client.

## Quick orientation

| Goal | Read |
|---|---|
| Build a single-deployment app | [Simple app template](#simple-app-template) |
| Build a multi-deployment composition app | [Composition app template](#composition-app-template) |
| Add a browser UI | [Frontend](#frontend-ui) |
| Save + deploy to BioEngine worker | [Deploy workflow](#deploy-workflow) |
| Update a running app | [Updating a running application](#updating-a-running-application) |

## Application structure

Every BioEngine app is a directory:

```
my-app/
├── manifest.yaml          # Required: identity, deployments, auth
├── my_deployment.py       # Required: Ray Serve class
├── frontend/              # Optional: static HTML/JS UI
│   └── index.html
└── README.md              # Optional
```

For a composition app:
```
my-app/
├── manifest.yaml
├── entry_deployment.py    # Entry point — orchestrates runtimes
├── runtime_a.py           # Runtime deployment A
├── runtime_b.py           # Runtime deployment B
└── frontend/index.html
```

---

## Manifest reference

```yaml
name: My Application        # Human-readable name
id: my-application          # Unique lowercase ID (hyphens only) — also the artifact alias
id_emoji: "🔬"              # Visual identifier
description: "..."
type: ray-serve             # Must be "ray-serve"
format_version: 0.5.0
version: 1.0.0
authors:
  - {name: "...", affiliation: "...", github_user: "..."}
license: MIT
tags: [bioengine, my-tag]

deployments:                # Ordered list: first is always the entry/main deployment
  - my_deployment:MyDeployment       # Single deployment
  # For composition:
  # - entry_deployment:EntryDeployment
  # - runtime_a:RuntimeA
  # - runtime_b:RuntimeB

authorized_users:
  - "*"                     # Public; or list specific user IDs

# Static frontend (optional)
frontend_entry: "frontend/index.html"
```

**Deployment format**: `python_filename_without_py:ClassName`  
**Naming rule for composition**: the filename part (e.g. `runtime_a`) **must exactly match** the parameter name in the entry deployment's `__init__` that accepts the `DeploymentHandle`.

---

## Pip requirements — set them early and freeze versions

**This is the most important performance rule.**

Ray caches the entire runtime environment (pip packages + Python interpreter) by content hash. If you change any package or its version — even a patch bump — Ray must rebuild the entire environment from scratch, which takes 5–15 minutes.

**Best practice:**
1. Decide all pip requirements **before** you start coding.
2. Pin every package to an exact version.
3. Only change requirements when absolutely necessary.

```python
@serve.deployment(
    ray_actor_options={
        "runtime_env": {
            "pip": [
                "numpy==1.26.4",       # pin exact versions
                "scipy==1.13.0",
                "pillow==10.4.0",
            ],
        },
    }
)
class MyDeployment:
    ...
```

If you need to iterate on code without changing packages, you can update the deployment files freely — only `runtime_env.pip` changes cause a rebuild.

---

## Import pattern

- **Standard library + BioEngine core** (`asyncio`, `os`, `logging`, `hypha_rpc`, `ray.serve`, `pydantic`): import at the **top** of the file.
- **Third-party packages** (anything in `runtime_env.pip`): import **inside each method** that uses them. Ray serializes deployment classes; top-level third-party imports break serialization.

```python
# Top level — OK
import asyncio, logging, os
from ray import serve
from hypha_rpc.utils.schema import schema_method

# Inside method — required for third-party packages
async def process(self, data):
    import numpy as np        # correct
    import torch              # correct
    result = np.array(data)
    ...
```

---

## Lifecycle methods

All lifecycle methods are optional but highly recommended.

```python
async def async_init(self) -> None:
    """Called once after __init__, before accepting requests.
    Use for: async setup, connecting to Hypha, loading models."""
    self.model = await self._load_model()

async def test_deployment(self) -> None:
    """Called once after async_init. Raise an exception to fail startup.
    Use for: smoke tests, import checks, env var checks."""
    import numpy as np        # verify package available
    assert os.environ.get("MY_TOKEN"), "MY_TOKEN not set"
    result = await self.predict(np.zeros((3, 64, 64)))
    assert result is not None

async def check_health(self) -> None:
    """Called periodically by Ray Serve.
    Raise to signal unhealthy — triggers restart."""
    if self._unhealthy:
        raise RuntimeError("Service unhealthy")
```

**Order**: `__init__` → `async_init` → `test_deployment` → `check_health` (periodic)

---

## Public API methods

Decorate public methods with `@schema_method`. These become the callable API when clients connect via Hypha. Use type hints and `pydantic.Field` for auto-generated docs.

```python
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

@schema_method
async def process(
    self,
    text: str = Field(..., description="Input text to process"),
    max_length: int = Field(100, description="Maximum output length"),
) -> dict:
    """Process text and return result."""
    return {"result": text[:max_length]}
```

Non-decorated methods are internal only.

**Constraint: no mutable defaults in `Field()`**. `list` and `dict` defaults fail at load time with a `ValueError`. Use `None` and check inside the method:

```python
# WRONG — crashes at startup
async def foo(self, values: list = Field([1, 2, 3], ...)) -> dict: ...

# CORRECT — use None, then assign default inside
async def foo(self, values: list = Field(None, description="Numbers")) -> dict:
    if values is None:
        values = [1, 2, 3]
    ...
```

---

## Logging

Use the Ray Serve logger — do not use `print`:

```python
logger = logging.getLogger("ray.serve")
logger.info("message")
logger.warning("something odd")
logger.error("something broke")
```

---

## GPU vs CPU resources

```python
@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,          # CPU-only
        "memory": 2 * 1024**3,  # 2 GB
    }
)
```

- Use `num_gpus: 0` for CPU-only deployments.
- **Use `num_gpus: 1` (never fractional) for GPU deployments.** Fractional values (e.g. `0.5`) allow multiple apps to share a single physical GPU, causing VRAM contention and OOM errors. With `num_gpus: 1`, Ray schedules the deployment on a dedicated GPU node.
- Entry/orchestrator deployments in composition apps should use `num_cpus: 0, num_gpus: 0` — they spend most of their time waiting for responses from runtime deployments.
- GPU allocation can be overridden at deploy time with `disable_gpu=True`.
- The `bioimage-io/bioengine-worker` cluster has **4 × NVIDIA A40 (16 GiB) worker nodes** — check current availability with `worker.get_status()` before deploying GPU-heavy apps (see "Checking cluster resources" below).

### Checking cluster resources

Call `get_status()` on the worker service to inspect total and used CPUs/GPUs across the cluster:

```python
status = await worker.get_status()
cluster = status["ray_cluster"]["cluster"]
print(f"GPUs: {cluster['used_gpu']} / {cluster['total_gpu']} used")
print(f"CPUs: {cluster['used_cpu']} / {cluster['total_cpu']} used")

# Per-node detail (GPU memory, accelerator type, etc.)
for node_id, node in status["ray_cluster"]["nodes"].items():
    if node["total_gpu"] > 0:
        vram_used = node["used_gpu_memory"] / 1024**3
        vram_total = node["total_gpu_memory"] / 1024**3
        print(f"  {node['node_ip']} ({node['accelerator_type']}): "
              f"{node['used_gpu']}/{node['total_gpu']} GPU, "
              f"{vram_used:.1f}/{vram_total:.1f} GiB VRAM")
```

---

## Available built-in resources

Every deployment automatically gets:

| Variable | Value |
|---|---|
| `HOME` | App working directory |
| `TMPDIR` | Temporary directory |
| `HYPHA_SERVER_URL` | `https://hypha.aicell.io` |
| `HYPHA_WORKSPACE` | Current workspace |
| `HYPHA_ARTIFACT_ID` | Full artifact identifier |
| `BIOENGINE_WORKER_SERVICE_ID` | Worker service ID |
| `HYPHA_TOKEN` | Auth token (if provided at deploy time) |

Dataset access via `self.bioengine_datasets`:
```python
datasets = await self.bioengine_datasets.list_datasets()
files = await self.bioengine_datasets.list_files(dataset_id=did, dir_path="sub/dir")
content = await self.bioengine_datasets.get_file(dataset_id=did, file_path="data.txt")
```

---

# Simple App Template

A single deployment app — one Python file, one Ray Serve class.

## `manifest.yaml`

```yaml
name: My Simple App
id: my-simple-app
id_emoji: "⚙️"
description: "A simple BioEngine application"
type: ray-serve
format_version: 0.5.0
version: 1.0.0
authors:
  - {name: "Your Name", affiliation: "Your Org"}
license: MIT
tags: [bioengine]
deployments:
  - my_deployment:MyDeployment
authorized_users:
  - "*"
frontend_entry: "frontend/index.html"
```

## `my_deployment.py`

```python
"""Single-deployment BioEngine app."""
import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Union

from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 1 * 1024**3,
        "runtime_env": {
            "pip": [
                # Freeze all versions here BEFORE writing business logic.
                # Changing these later requires a full environment rebuild (5-15 min).
                "numpy==1.26.4",
            ],
        },
    },
    max_ongoing_requests=10,
)
class MyDeployment:
    def __init__(self, greeting: str = "Hello") -> None:
        self.greeting = greeting
        self.start_time = time.time()

    async def async_init(self) -> None:
        """Async setup — connect to services, load models, etc."""
        logger.info("MyDeployment async_init complete")

    async def test_deployment(self) -> None:
        """Smoke test — runs once at startup. Raise to fail deployment."""
        import numpy as np  # verify pip package available
        arr = np.zeros((3, 3))
        assert arr.shape == (3, 3)
        result = await self.ping()
        assert result["status"] == "ok"

    async def check_health(self) -> None:
        """Periodic health check — raise to signal unhealthy."""
        pass

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float]]:
        """Ping the service.

        Returns:
            dict: status, message, timestamp, uptime
        """
        return {
            "status": "ok",
            "message": f"{self.greeting} from MyDeployment!",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
        }

    @schema_method
    async def process(
        self,
        values: list = Field(..., description="List of numbers to sum"),
    ) -> dict:
        """Sum a list of numbers using numpy.

        Returns:
            dict: result with the sum
        """
        import numpy as np
        arr = np.array(values, dtype=float)
        return {"result": float(np.sum(arr)), "count": len(values)}
```

## `frontend/index.html`

See [Frontend UI](#frontend-ui) for the template.

---

# Composition App Template

A multi-deployment app: one entry deployment that orchestrates multiple runtime deployments. The entry deployment has no CPUs/GPUs — it just routes calls to the runtimes.

**Architecture:**
```
Client → EntryDeployment (CPU=0) → RuntimeA (CPU=1, text)
                                 → RuntimeB (CPU=1, data)
                                 → RuntimeC (CPU=1, images)
```

## Naming convention — critical

The manifest lists deployments as `filename:ClassName`. The filename part (without `.py`) **must exactly match** the parameter name in `EntryDeployment.__init__` that holds the `DeploymentHandle`:

```yaml
# manifest.yaml
deployments:
  - entry_deployment:EntryDeployment   # entry — always first
  - runtime_a:RuntimeA                 # "runtime_a" must match param name below
  - runtime_b:RuntimeB                 # "runtime_b" must match param name below
  - runtime_c:RuntimeC                 # "runtime_c" must match param name below
```

```python
# entry_deployment.py
class EntryDeployment:
    def __init__(
        self,
        runtime_a: DeploymentHandle,   # matches "runtime_a" in manifest
        runtime_b: DeploymentHandle,   # matches "runtime_b"
        runtime_c: DeploymentHandle,   # matches "runtime_c"
    ) -> None:
```

## `manifest.yaml`

```yaml
name: My Composition App
id: my-composition-app
id_emoji: "🔬"
description: "Multi-deployment composition app with entry + 3 runtimes"
type: ray-serve
format_version: 0.5.0
version: 1.0.0
authors:
  - {name: "Your Name", affiliation: "Your Org"}
license: MIT
tags: [bioengine, composition]
deployments:
  - entry_deployment:EntryDeployment  # entry — always first
  - runtime_a:RuntimeA                # text processing
  - runtime_b:RuntimeB                # data analysis
  - runtime_c:RuntimeC                # image processing
authorized_users:
  - "*"
frontend_entry: "frontend/index.html"
```

## `entry_deployment.py`

```python
"""Entry deployment — orchestrates RuntimeA, RuntimeB, RuntimeC."""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Union

from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve
from ray.serve.handle import DeploymentHandle

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        # Entry deployment only routes requests — no CPU/GPU needed.
        "num_cpus": 0,
        "num_gpus": 0,
        "memory": 256 * 1024**2,   # 256 MB
        "runtime_env": {
            # Entry deployment pip: keep minimal — only what entry needs directly.
            # Each runtime has its own runtime_env with its own packages.
            "pip": [],
        },
    },
    max_ongoing_requests=20,
)
class EntryDeployment:
    def __init__(
        self,
        runtime_a: DeploymentHandle,   # must match filename "runtime_a" in manifest
        runtime_b: DeploymentHandle,   # must match filename "runtime_b"
        runtime_c: DeploymentHandle,   # must match filename "runtime_c"
    ) -> None:
        self.runtime_a = runtime_a
        self.runtime_b = runtime_b
        self.runtime_c = runtime_c
        self.start_time = time.time()

    async def async_init(self) -> None:
        logger.info("EntryDeployment async_init complete")

    async def test_deployment(self) -> None:
        """Test that all runtimes respond."""
        ping_a = await self.runtime_a.ping.remote()
        ping_b = await self.runtime_b.ping.remote()
        ping_c = await self.runtime_c.ping.remote()
        assert ping_a == "pong"
        assert ping_b == "pong"
        assert ping_c == "pong"

    @schema_method
    async def status(self) -> dict:
        """Get status of entry and all runtimes.

        Returns:
            dict: status from entry and each runtime
        """
        a, b, c = await asyncio.gather(
            self.runtime_a.get_status.remote(),
            self.runtime_b.get_status.remote(),
            self.runtime_c.get_status.remote(),
        )
        return {
            "entry_uptime": time.time() - self.start_time,
            "runtime_a": a,
            "runtime_b": b,
            "runtime_c": c,
        }

    @schema_method
    async def process_text(
        self,
        text: str = Field(..., description="Text to process"),
    ) -> dict:
        """Process text through RuntimeA.

        Returns:
            dict: word count and character count
        """
        return await self.runtime_a.process_text.remote(text)

    @schema_method
    async def analyze_data(
        self,
        values: list = Field(..., description="List of numbers"),
    ) -> dict:
        """Run statistical analysis through RuntimeB.

        Returns:
            dict: mean, std, min, max
        """
        return await self.runtime_b.analyze.remote(values)

    @schema_method
    async def process_image(
        self,
        width: int = Field(64, description="Image width"),
        height: int = Field(64, description="Image height"),
    ) -> dict:
        """Generate and process a test image through RuntimeC.

        Returns:
            dict: image stats
        """
        return await self.runtime_c.process_image.remote(width, height)

    @schema_method
    async def pipeline(
        self,
        text: str = Field(..., description="Text input"),
        values: list = Field(..., description="Numeric values"),
    ) -> dict:
        """Run all three runtimes in parallel and combine results.

        Returns:
            dict: combined results from all runtimes
        """
        text_result, data_result = await asyncio.gather(
            self.runtime_a.process_text.remote(text),
            self.runtime_b.analyze.remote(values),
        )
        return {"text": text_result, "data": data_result}
```

## `runtime_a.py`

```python
"""RuntimeA — text processing."""
import logging
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 512 * 1024**2,
        "runtime_env": {
            "pip": [
                # Freeze versions BEFORE coding.
                # Example: lightweight text processing only.
            ],
        },
    },
    max_ongoing_requests=5,
)
class RuntimeA:
    def __init__(self) -> None:
        pass

    async def async_init(self) -> None:
        logger.info("RuntimeA ready")

    async def test_deployment(self) -> None:
        result = await self.process_text("hello world")
        assert "word_count" in result

    async def ping(self) -> str:
        return "pong"

    async def get_status(self) -> dict:
        return {"name": "runtime_a", "status": "ok"}

    async def process_text(self, text: str) -> dict:
        words = text.split()
        return {"word_count": len(words), "char_count": len(text), "words": words}
```

## `runtime_b.py`

```python
"""RuntimeB — data analysis with numpy/scipy."""
import logging
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 512 * 1024**2,
        "runtime_env": {
            "pip": [
                "numpy==1.26.4",
                "scipy==1.13.0",
            ],
        },
    },
    max_ongoing_requests=5,
)
class RuntimeB:
    def __init__(self) -> None:
        pass

    async def async_init(self) -> None:
        import numpy as np
        logger.info(f"RuntimeB ready (numpy {np.__version__})")

    async def test_deployment(self) -> None:
        import numpy as np
        result = await self.analyze([1, 2, 3, 4, 5])
        assert "mean" in result

    async def ping(self) -> str:
        return "pong"

    async def get_status(self) -> dict:
        return {"name": "runtime_b", "status": "ok"}

    async def analyze(self, values: list) -> dict:
        import numpy as np
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }
```

## `runtime_c.py`

```python
"""RuntimeC — image processing with Pillow."""
import logging
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 1 * 1024**3,
        "runtime_env": {
            "pip": [
                "numpy==1.26.4",
                "pillow==10.4.0",
            ],
        },
    },
    max_ongoing_requests=5,
)
class RuntimeC:
    def __init__(self) -> None:
        pass

    async def async_init(self) -> None:
        from PIL import Image
        logger.info("RuntimeC ready")

    async def test_deployment(self) -> None:
        result = await self.process_image(64, 64)
        assert "mean_pixel" in result

    async def ping(self) -> str:
        return "pong"

    async def get_status(self) -> dict:
        return {"name": "runtime_c", "status": "ok"}

    async def process_image(self, width: int = 64, height: int = 64) -> dict:
        import numpy as np
        from PIL import Image
        # Generate a test gradient image
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, width)   # red gradient
        arr[:, :, 1] = np.linspace(0, 255, height).reshape(-1, 1)  # green gradient
        img = Image.fromarray(arr)
        pixel_arr = np.array(img)
        return {
            "width": width,
            "height": height,
            "channels": 3,
            "mean_pixel": float(pixel_arr.mean()),
            "format": "RGB",
        }
```

---

# Frontend UI

Add a `frontend/index.html` to your app directory and set `frontend_entry: "frontend/index.html"` in the manifest. BioEngine will serve this file at:

```
https://hypha.aicell.io/{workspace}/view/{artifact-id}/
```

The frontend connects to the BioEngine service using the `hypha-rpc` JavaScript library. The service URL and service ID are passed as URL query parameters (`server` and `ws_service_id`).

## Minimal frontend template

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>My BioEngine App</title>
  <style>
    body { font-family: system-ui; background: #0f172a; color: #e2e8f0;
           min-height: 100vh; display: flex; flex-direction: column;
           align-items: center; padding: 2rem 1rem; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: .75rem;
            padding: 1.5rem; width: 100%; max-width: 640px; margin-bottom: 1rem; }
    button { background: #0284c7; color: #fff; border: none; border-radius: .5rem;
             padding: .5rem 1.25rem; cursor: pointer; font-weight: 600; }
    button:disabled { background: #334155; color: #64748b; cursor: not-allowed; }
    pre { background: #0f172a; border-radius: .5rem; padding: 1rem;
          font-size: .8rem; white-space: pre-wrap; min-height: 3rem;
          color: #94a3b8; overflow-y: auto; max-height: 16rem; }
  </style>
</head>
<body>

<div class="card">
  <h2>My BioEngine App</h2>
  <button id="connectBtn" onclick="connect()">Connect</button>
  <p id="status">Not connected</p>
</div>

<div class="card">
  <button id="pingBtn" onclick="callPing()" disabled>Ping</button>
  <pre id="result">—</pre>
</div>

<script type="module">
import { connectToServer }
  from "https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.54/dist/hypha-rpc-websocket.mjs";

const p          = new URLSearchParams(window.location.search);
const SERVER_URL = p.get("server")        || "https://hypha.aicell.io";
const SERVICE_ID = p.get("ws_service_id") || "";

let svc = null;

window.connect = async () => {
  document.getElementById("status").textContent = "Connecting…";
  try {
    const server = await connectToServer({ server_url: SERVER_URL });
    svc = await server.getService(SERVICE_ID, { _rkwargs: true });
    document.getElementById("status").textContent = "Connected ✓";
    document.getElementById("pingBtn").disabled = false;
  } catch (e) {
    document.getElementById("status").textContent = "Error: " + e.message;
  }
};

window.callPing = async () => {
  document.getElementById("result").textContent = "Loading…";
  try {
    const r = await svc.ping({ _rkwargs: true });
    document.getElementById("result").textContent = JSON.stringify(r, null, 2);
  } catch (e) {
    document.getElementById("result").textContent = "Error: " + e.message;
  }
};
</script>
</body>
</html>
```

**Key points:**
- Import `connectToServer` from the CDN — no npm/bundler needed.
- `server_url` and `ws_service_id` come from URL query params injected by BioEngine into the static site URL.
- Always pass `{ _rkwargs: true }` to every service call in **JavaScript**. This is a JS-only convention required by `hypha-rpc`.
- In **Python**, call service methods directly without `_rkwargs`: `await service.ping()`.

---

# Deploy Workflow

## Prerequisites

```bash
# Install CLI (once)
pip install hypha-rpc httpx click bioengine

# Set credentials
export HYPHA_TOKEN=<your-token>
export BIOENGINE_WORKER_SERVICE_ID=bioimage-io/bioengine-worker  # public worker
# For private worker: export BIOENGINE_WORKER_SERVICE_ID=my-workspace/bioengine-worker
```

## Step 1 — Always save app files to the worker first

**Do not deploy directly without saving.** The worker stores the app as a Hypha artifact. All files (manifest, Python files, frontend) must be saved via the worker's `save_application` API:

```bash
# Upload + deploy in one step (recommended)
bioengine apps deploy ./my-app/

# Or separately:
bioengine apps upload ./my-app/   # saves to Hypha artifact
bioengine apps run my-workspace/my-app  # deploys from artifact
```

Or via Python (direct API):
```python
import asyncio, os
from pathlib import Path
from hypha_rpc import connect_to_server
from bioengine.utils import create_file_list_from_directory

async def deploy_app(app_dir: str, worker_service_id: str = "bioimage-io/bioengine-worker"):
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "token": os.environ["HYPHA_TOKEN"],
    })
    worker = await server.get_service(worker_service_id)

    # 1. Save app files to Hypha artifact (commits with the version in manifest.yaml)
    files = create_file_list_from_directory(Path(app_dir))
    artifact_id = await worker.save_application(files=files)
    print(f"Saved: {artifact_id}")

    # 2. Read the manifest version so we deploy the exact version we just saved
    import yaml
    manifest = yaml.safe_load(Path(app_dir, "manifest.yaml").read_text())
    app_version = manifest.get("version")  # e.g. "1.0.0"

    # 3. Deploy that specific version from the artifact
    app_id = await worker.run_application(
        artifact_id=artifact_id,
        version=app_version,              # pin to the version saved in step 1
        application_kwargs={
            "CompositionDeployment": {"some_param": "value"},  # optional init kwargs
            "RuntimeB": {"config": "value"},
        },
        hypha_token=os.environ["HYPHA_TOKEN"],  # optional: token for app internals
        disable_gpu=True,                        # CPU-only
    )
    print(f"Deployed app ID: {app_id}")
    return app_id

asyncio.run(deploy_app("./my-app/"))
```

## Step 2 — Monitor deployment + live debug

**Application states**: `NOT_STARTED` → `DEPLOYING` → `RUNNING` (success) or `DEPLOY_FAILED`.  
**Deployment states** (per deployment inside the app): `UPDATING` → `HEALTHY` (success) or `UNHEALTHY`.

The app is fully ready when `status == "RUNNING"` and all deployments show `HEALTHY`.

```bash
bioengine apps status
bioengine apps logs <app-id> --tail 100
```

Via Python — poll until ready, inspect on failure:
```python
import asyncio

async def wait_until_healthy(worker, app_id: str, timeout: int = 300):
    """Poll until app is RUNNING with all deployments HEALTHY."""
    for _ in range(timeout // 5):
        # Pass a single-element list → returns status dict directly (not nested)
        status = await worker.get_application_status(application_ids=[app_id])
        state = status.get("status", "")
        deployments = status.get("deployments", [])
        print(f"  [{state}] {status.get('message', '')}")
        for d in deployments:
            replica_states = [r.get("state") for r in d.get("replicas", [])]
            print(f"    {d['name']}: {d['status']} — replicas: {replica_states}")
        if state == "RUNNING" and all(d.get("status") == "HEALTHY" for d in deployments):
            return status
        if state == "DEPLOY_FAILED":
            # Print full logs for each unhealthy deployment
            for d in deployments:
                print(f"\n--- Logs for {d['name']} ---")
                for r in d.get("replicas", []):
                    print(r.get("log", ""))
            raise RuntimeError(f"Deployment failed: {status.get('message')}")
        await asyncio.sleep(5)
    raise TimeoutError(f"App {app_id} did not become healthy within {timeout}s")

status = await wait_until_healthy(worker, app_id)
print("App is healthy!")
```

The `status["deployments"]` list contains per-deployment details with `name`, `status`, `message`, and `replicas` (each replica has `state`, `actor_id`, `log`). Use this to diagnose startup failures.

## Step 3 — Call the deployed service

Extract the WebSocket service ID from the status and connect:

```python
from hypha_rpc import connect_to_server

# The ws_service_id is inside status["service_ids"][0]["websocket_service_id"]
ws_service_id = status["service_ids"][0]["websocket_service_id"]

server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": token})
service = await server.get_service(ws_service_id)

# Python: call methods directly — NO _rkwargs
result = await service.ping()
result2 = await service.process(values=[1, 2, 3])
```

The frontend URL (with service ID already embedded as query param) is at `status["static_site_url"]`.

## Step 4 — Bump version and commit after successful deploy

Once the live app is verified working:

1. **Bump the version** in `manifest.yaml` (e.g. `1.0.0` → `1.1.0` for a feature, `1.0.0` → `1.0.1` for a fix).
2. **Re-save** the updated app to keep the artifact in sync.
3. **Commit** the app source to git so the deployed version is always reproducible:

```bash
git add bioengine_apps/my-app/
git commit -m "feat(my-app): add X feature, bump version to 1.1.0"
git push
```

Always keep `bioengine_apps/` in sync with what is running live on the worker.

---

## CLI reference

```bash
# Install from bundled source (no PyPI needed)
pip install -e bioengine_cli/   # from this skill directory
# or: pip install bioengine

export BIOENGINE_WORKER_SERVICE_ID=bioimage-io/bioengine-worker
export HYPHA_TOKEN=<token>

bioengine apps deploy ./my-app/                              # upload + deploy
bioengine apps upload ./my-app/                              # upload only
bioengine apps run <workspace/app-id>                        # deploy from artifact
bioengine apps run <workspace/app-id> --app-id existing-id  # update existing
bioengine apps list                                          # list artifacts
bioengine apps status                                        # all running apps
bioengine apps status <app-id> --logs 50                     # one app + logs
bioengine apps logs <app-id> --tail 200                      # stream logs
bioengine apps stop <app-id>                                 # stop app
```

All commands accept `--json` for machine-readable output.

---

## Updating a running application

To update without losing env vars, init kwargs, or the HYPHA_TOKEN, use `--app-id` with the existing app's ID. The new version inherits **all env vars and all init args/kwargs** from the previous deployment:

```bash
bioengine apps upload ./my-app/
bioengine apps run my-workspace/my-app --app-id my-running-app-id
# HYPHA_TOKEN and all init kwargs are preserved automatically
```

Creating a new deployment (without `--app-id`) starts fresh — no env vars, no init kwargs.

**Apps that require a `bioimage-io` workspace token** (e.g., `model-runner`, `cellpose-finetuning`) must be started with a `HYPHA_TOKEN` scoped to that workspace. On updates with `--app-id`, the token is inherited automatically.

---

## Deployment rules summary

| Rule | Detail |
|---|---|
| Freeze pip versions early | Changing any package = full env rebuild (5–15 min) |
| Import third-party inside methods | Top-level imports break Ray serialization |
| Use `logger`, not `print` | `logger = logging.getLogger("ray.serve")` |
| Entry deployment CPU = 0 | Orchestrators just route; no compute needed |
| Save before deploy | Always call `save_application` (via CLI or API) first |
| `@schema_method` on public API | Non-decorated methods are internal only |
| Match manifest filename to param name | `runtime_a:RuntimeA` → `def __init__(self, runtime_a: DeploymentHandle)` |
| GPU deployments use `num_gpus: 1` | Never use fractional GPU values — they allow VRAM sharing across apps and cause OOM. Use `num_gpus: 0` for CPU-only. |

---

## CPU-blocking operations

Wrap CPU-intensive or synchronous code in a thread pool to avoid blocking the event loop:

```python
from concurrent.futures import ThreadPoolExecutor

class MyDeployment:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def process(self, data: str) -> dict:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._executor, self._blocking_op, data)
        return {"result": result}

    def _blocking_op(self, data: str) -> str:
        # CPU-intensive synchronous work here
        return data.upper()
```

---

## Secret management

Prefix env var names with `_` to mark as secret (hidden from status output, stripped from key inside the app):

```bash
bioengine apps run my-workspace/my-app --env _API_KEY=secret
# Inside deployment: os.environ["API_KEY"]  (underscore stripped)
```

---

## Local testing (no Ray Serve)

Test deployment logic without spinning up Ray:

```python
import unittest.mock, sys

# Patch ray before importing your deployment file
ray_mock = unittest.mock.MagicMock()
ray_mock.serve.deployment = lambda **kw: (lambda cls: cls)
ray_mock.serve.multiplexed = lambda **kw: (lambda fn: fn)
sys.modules["ray"] = ray_mock
sys.modules["ray.serve"] = ray_mock.serve

from my_deployment import MyDeployment

inst = MyDeployment(greeting="Hi")
await inst.async_init()
result = await inst.ping()
assert result["status"] == "ok"
```

---

## Multi-deployment composition — advanced

### Passing init kwargs at deploy time

```python
await worker.run_application(
    artifact_id="my-workspace/my-composition-app",
    version="1.0.0",                       # deploy a specific artifact version
    application_kwargs={
        # Keys are class names (not file names)
        "EntryDeployment": {},             # entry usually needs no kwargs
        "RuntimeB": {"threshold": 0.5},   # passed to RuntimeB.__init__
        "RuntimeC": {"model_path": "/data/model.pt"},
    },
)
```

### Calling runtime deployments in parallel

```python
@schema_method
async def batch_process(self, items: list) -> list:
    # Fan out to runtime in parallel
    results = await asyncio.gather(
        *[self.runtime_a.process_text.remote(item) for item in items]
    )
    return list(results)
```

### Streaming responses

For long-running operations, poll a status endpoint rather than holding the connection open:

```python
@schema_method
async def start_job(self, data: dict) -> str:
    """Returns job_id immediately."""
    job_id = str(uuid.uuid4())
    asyncio.create_task(self._run_job(job_id, data))
    return job_id

@schema_method
async def get_job_status(self, job_id: str) -> dict:
    """Poll for job status."""
    return self._jobs.get(job_id, {"status": "not_found"})
```

---

## References

- Full manifest fields and deployment config: [references/manifest_reference.md](references/manifest_reference.md)
- CLI source and advanced usage: [references/cli_reference.md](references/cli_reference.md)
- Example single-deployment app: [`bioengine_apps/demo-app/`](https://github.com/aicell-lab/bioengine-worker/tree/main/bioengine_apps/demo-app)
- Example model-runner (entry + runtime): [`bioengine_apps/model-runner/`](https://github.com/aicell-lab/bioengine-worker/tree/main/bioengine_apps/model-runner)

## BioEngine app skills

The following app-specific skills provide deeper documentation for pre-deployed BioEngine services. Load them autonomously when the user's task involves one of these services:

| Skill file | When to load |
|---|---|
| [`../bioengine-model-runner/SKILL.md`](../bioengine-model-runner/SKILL.md) | User wants to search, run inference on, or validate BioImage.IO models |
| [`../bioengine-cellpose-finetuning/SKILL.md`](../bioengine-cellpose-finetuning/SKILL.md) | User wants to fine-tune Cellpose on their own annotated microscopy images |

More app skills will be added here as new BioEngine applications are developed.

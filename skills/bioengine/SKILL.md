---
name: bioengine
description: Builds, deploys, and manages BioEngine applications on Ray Serve/Hypha, and calls any pre-deployed BioEngine service (model runner, Cellpose fine-tuning, etc.). Use this as the single entry point for any BioEngine task: building new apps, deploying to a worker, calling service methods, checking cluster resources. For deep usage of specific services, load the referenced app skills.
license: MIT
metadata:
  cli-package: bioengine (pip install -e skills/bioengine/bioengine_cli/)
  app-skills:
    - ../bioengine-model-runner/SKILL.md
    - ../bioengine-cellpose-finetuning/SKILL.md
---

# BioEngine Apps

BioEngine applications are Ray Serve classes packaged as Hypha artifacts. They expose model inference, distributed training, data exploration, and custom pipelines as Hypha RPC services.

## Quick orientation

| Goal | Read |
|---|---|
| Discover methods on any service | `bioengine call <service-id> --list-methods` |
| Call any deployed service | [CLI: bioengine call](#cli-quick-reference) |
| Build a single-deployment app | [Simple app template](#simple-app-template) |
| Build a multi-deployment composition app | [references/app_templates.md](references/app_templates.md) |
| Multiplexing / model integration / fine-tuning | [references/model_serving.md](references/model_serving.md) |
| Deploy + monitor | [Deploy workflow](#deploy-workflow) |
| Check cluster GPU availability | [Checking cluster resources](#checking-cluster-resources) |
| Full manifest fields | [references/manifest_reference.md](references/manifest_reference.md) |
| Full CLI reference | [references/cli_reference.md](references/cli_reference.md) |

---

## Server and workspace defaults

**Default Hypha server**: `https://hypha.aicell.io` — use this unless the user specifies another.

**Public BioEngine services** (live on `bioimage-io` workspace, no auth required to call):

| Service | Service ID | Description |
|---|---|---|
| Model Runner | `bioimage-io/model-runner` | Run/search/validate BioImage.IO models |
| Cellpose Fine-tuning | `bioimage-io/cellpose-finetuning` | Fine-tune Cellpose-SAM on custom data |
| BioEngine Worker | `bioimage-io/bioengine-worker` | Cluster management (admin ops) |

**User-deployed workers**: A user can start their own BioEngine worker in any Hypha workspace. In that case, derive service IDs from their workspace:

```
Workspace: ws-user-github|49943582
→ Worker:              ws-user-github|49943582/bioengine-worker
→ Model runner:        ws-user-github|49943582/model-runner
→ Cellpose finetuning: ws-user-github|49943582/cellpose-finetuning
```

Always use the public `bioimage-io` workspace unless the user explicitly provides their own workspace.

---

## Application structure

```
my-app/
├── manifest.yaml          # Required: identity, deployments, auth
├── my_deployment.py       # Required: Ray Serve class
└── frontend/index.html    # Optional: static UI
```

For a composition app (entry + multiple runtimes):
```
my-app/
├── manifest.yaml
├── entry_deployment.py    # Orchestrates runtimes — no CPU/GPU
├── runtime_a.py
└── runtime_b.py
```

---

## Manifest reference

```yaml
name: My Application
id: my-application          # Unique lowercase ID — also the artifact alias
id_emoji: "🔬"
description: "..."
type: ray-serve
format_version: 0.5.0
version: 1.0.0
authors:
  - {name: "...", affiliation: "...", github_user: "..."}
license: MIT
tags: [bioengine]

deployments:                # Ordered list: first is always the entry/main deployment
  - my_deployment:MyDeployment       # single deployment
  # For composition: - entry_deployment:EntryDeployment, - runtime_a:RuntimeA, ...

authorized_users:
  - "*"                     # Public; or list specific user IDs

frontend_entry: "frontend/index.html"   # Optional: static UI
```

**Deployment format**: `python_filename_without_py:ClassName`  
**Composition naming rule**: the filename (e.g. `runtime_a`) **must exactly match** the parameter name in `EntryDeployment.__init__` that holds the `DeploymentHandle`.

---

## Pip requirements — set them early and freeze versions

Ray caches runtime environments by content hash. Changing any package version causes a full rebuild (5–15 min). **Decide all pip requirements before writing business logic, and pin every version.**

```python
@serve.deployment(
    ray_actor_options={
        "runtime_env": {
            "pip": ["numpy==1.26.4", "scipy==1.13.0"],  # pin exact versions
        },
    }
)
```

---

## Import pattern

- **Standard library + BioEngine core** (`asyncio`, `os`, `logging`, `hypha_rpc`, `ray.serve`, `pydantic`): import at the **top** of the file.
- **Third-party packages** (`runtime_env.pip`): import **inside each method**. Ray serializes deployment classes; top-level third-party imports break serialization.

```python
# Top-level — OK
import asyncio, logging, os
from ray import serve
from hypha_rpc.utils.schema import schema_method

# Inside method — required for third-party packages
async def process(self, data):
    import numpy as np    # correct
    import torch          # correct
```

---

## Lifecycle methods

```python
async def async_init(self) -> None:
    """Called once after __init__, before accepting requests. Load models here."""
    self.model = await self._load_model()

async def test_deployment(self) -> None:
    """Called once after async_init. Raise to fail startup (smoke test)."""
    result = await self.ping()
    assert result["status"] == "ok"

async def check_health(self) -> None:
    """Called periodically by Ray Serve. Raise to signal unhealthy."""
    if self._unhealthy:
        raise RuntimeError("unhealthy")
```

**Order**: `__init__` → `async_init` → `test_deployment` → `check_health` (periodic)

---

## Public API methods

Decorate public methods with `@schema_method`. These become the callable API over Hypha RPC.

```python
from hypha_rpc.utils.schema import schema_method
from pydantic import Field

@schema_method
async def process(
    self,
    text: str = Field(..., description="Input text"),
    max_length: int = Field(100, description="Maximum output length"),
) -> dict:
    """Process text and return result."""
    return {"result": text[:max_length]}
```

**No mutable defaults in `Field()`**: use `None` and assign the default inside the method — `Field([1, 2, 3])` crashes at startup.

---

## Logging

```python
logger = logging.getLogger("ray.serve")
logger.info("message")    # never use print()
```

---

## GPU / CPU resources

```python
ray_actor_options={
    "num_cpus": 1,
    "num_gpus": 1,          # 1 = dedicated GPU node; 0 = CPU-only
    "memory": 4 * 1024**3,  # 4 GB RAM limit
}
```

- **Use `num_gpus: 1`** for GPU deployments — never fractional values. Fractional (e.g. `0.5`) allows VRAM sharing across apps, causing OOM errors.
- **Use `num_gpus: 0`** for CPU-only.
- Entry/orchestrator deployments in composition apps: `num_cpus: 0, num_gpus: 0`.
- Override at deploy time: `disable_gpu=True` in `run_application()`.

### Checking cluster resources

```python
status = await worker.get_status()
cluster = status["ray_cluster"]["cluster"]
print(f"GPUs: {cluster['used_gpu']} / {cluster['total_gpu']} used")

for nid, node in status["ray_cluster"]["nodes"].items():
    if node["total_gpu"] > 0:
        vram_gb = node["used_gpu_memory"] / 1024**3
        total_gb = node["total_gpu_memory"] / 1024**3
        print(f"  {node['node_ip']} ({node['accelerator_type']}): "
              f"{node['used_gpu']}/{node['total_gpu']} GPU, {vram_gb:.1f}/{total_gb:.1f} GiB VRAM")
```

Or via CLI: `bioengine cluster status`

---

## Image data conventions (RPC boundary)

Never pass raw numpy arrays across Hypha RPC — they are not JSON-serialisable.

| Direction | Client | Server |
|---|---|---|
| Send image | `image_list = img.tolist()` | `arr = np.array(image_list, dtype=np.float32)` |
| Receive labels | `labels = np.array(result["labels"])` | `"labels": masks.astype(np.int32).tolist()` |

For images > 100 MB, use Hypha artifact URLs instead of inline data.

---

## Simple app template

See [references/app_templates.md](references/app_templates.md) for the full code. Skeleton:

```python
@serve.deployment(
    ray_actor_options={
        "num_cpus": 1, "num_gpus": 0, "memory": 1 * 1024**3,
        "runtime_env": {"pip": ["numpy==1.26.4"]},
    },
    max_ongoing_requests=10,
)
class MyDeployment:
    def __init__(self) -> None: self.start_time = time.time()

    async def async_init(self) -> None: ...
    async def test_deployment(self) -> None: ...

    @schema_method
    async def ping(self) -> dict:
        """Ping the service."""
        return {"status": "ok", "uptime": time.time() - self.start_time}
```

For model multiplexing, model integration (HuggingFace, Zenodo, BioImage.IO), auto-scaling, and fine-tuning patterns, see [references/model_serving.md](references/model_serving.md).

---

## Deploy workflow

```bash
pip install -e skills/bioengine/bioengine_cli/

export HYPHA_TOKEN=<your-token>
export BIOENGINE_WORKER_SERVICE_ID=bioimage-io/bioengine-worker
```

**Step 1 — Upload and deploy:**
```bash
bioengine apps deploy ./my-app/                   # upload + deploy in one step
bioengine apps deploy ./my-app/ --app-id my-app   # stable ID (enables in-place updates)
```

**Step 2 — Monitor:**
```bash
bioengine apps status                             # all running apps
bioengine apps status my-app --logs 50            # one app + logs
bioengine apps logs my-app --tail 200
```

App states: `NOT_STARTED` → `DEPLOYING` → `RUNNING` or `DEPLOY_FAILED`.  
Deployment states: `UPDATING` → `HEALTHY` or `UNHEALTHY`. App is ready when all deployments are `HEALTHY`.

**Step 3 — Call the service:**
```bash
bioengine call bioimage-io/my-app ping --json
bioengine call bioimage-io/my-app process --args '{"values": [1, 2, 3]}' --json
```

Or via Python:
```python
from hypha_rpc import connect_to_server
server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": token})
svc = await server.get_service("bioimage-io/my-app")
result = await svc.ping()
```

**Step 4 — Bump version and commit:**
```bash
# After verifying the app works on the live worker:
git add bioengine_apps/my-app/
git commit -m "feat(my-app): add X, bump version to 1.1.0"
git push
```

Always bump `version` in `manifest.yaml` when changing app code.

---

## Updating a running application

Pass `--app-id` with the existing ID to update in-place (inherits all env vars and init kwargs):

```bash
bioengine apps upload ./my-app/
bioengine apps run my-workspace/my-app --app-id my-running-app-id
```

---

## CLI quick reference

```bash
# Install
pip install -e skills/bioengine/bioengine_cli/

# Call any service method
bioengine call <service-id> --list-methods           # discover available methods
bioengine call <service-id> <method> --json          # call a method, JSON output
bioengine call <service-id> <method> --args '{"k": "v"}' --json  # with arguments
bioengine call <service-id> <method> --arg key=val   # individual args (auto-typed)

# Deploy and manage apps
bioengine apps deploy ./my-app/ [--app-id ID]        # upload + deploy
bioengine apps upload ./my-app/                      # upload only
bioengine apps run <workspace/app-id> [--app-id ID]  # deploy from artifact
bioengine apps list                                  # list artifacts
bioengine apps status [APP_ID...] [--logs N]         # running app status
bioengine apps logs <app-id> [--tail N]              # view logs
bioengine apps stop <app-id> [-y]                    # stop app

# Cluster resources
bioengine cluster status [--json]                    # GPU/CPU availability
```

All commands accept `--json` for machine-readable output and respect `HYPHA_TOKEN`, `BIOENGINE_WORKER_SERVICE_ID`, `BIOENGINE_SERVER_URL` env vars.

---

## Deployment rules summary

| Rule | Detail |
|---|---|
| Freeze pip versions early | Changing any package = full env rebuild (5–15 min) |
| Import third-party inside methods | Top-level imports break Ray serialization |
| Use `logger`, not `print` | `logger = logging.getLogger("ray.serve")` |
| `num_gpus: 1` for GPU, never fractional | Fractional allows VRAM sharing → OOM |
| Entry deployment: `num_cpus: 0, num_gpus: 0` | Orchestrators just route; no compute needed |
| Save before deploy | Always call `save_application` (via CLI or API) first |
| `@schema_method` on public API | Non-decorated methods are internal only |
| Match manifest filename to param name | `runtime_a:RuntimeA` → `def __init__(self, runtime_a: DeploymentHandle)` |
| No mutable defaults in `Field()` | Use `None`, assign default inside method |

---

## Common pitfalls

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` at import | Add to `runtime_env.pip`, import inside method |
| numpy array returned over RPC | Call `.tolist()` on all arrays before returning |
| Long cold start on first request | Set `min_replicas: 1`, preload model in `async_init()` |
| Blocking inference stalls event loop | Wrap with `asyncio.get_event_loop().run_in_executor(None, fn)` |
| `AttributeError` deserializing exception | Re-raise third-party exceptions as plain `RuntimeError` at RPC boundary |
| `Multiple services found` error | Use `connect_service()` from bioengine_cli.utils (handles multi-replica) |

---

## References

- **App code templates**: [references/app_templates.md](references/app_templates.md) — simple app, composition app, frontend UI
- **Model serving patterns**: [references/model_serving.md](references/model_serving.md) — multiplexing, HuggingFace/Zenodo/BioImage.IO integration, auto-scaling, fine-tuning
- **Full manifest fields**: [references/manifest_reference.md](references/manifest_reference.md)
- **Full CLI reference**: [references/cli_reference.md](references/cli_reference.md)
- **Example apps**: [`bioengine_apps/demo-app/`](https://github.com/aicell-lab/bioengine-worker/tree/main/bioengine_apps/demo-app), [`bioengine_apps/model-runner/`](https://github.com/aicell-lab/bioengine-worker/tree/main/bioengine_apps/model-runner)

---

## BioEngine app skills

Load these autonomously when the user's task involves a specific deployed service:

| Skill file | When to load |
|---|---|
| [`../bioengine-model-runner/SKILL.md`](../bioengine-model-runner/SKILL.md) | Search, infer, validate, or compare BioImage.IO models |
| [`../bioengine-cellpose-finetuning/SKILL.md`](../bioengine-cellpose-finetuning/SKILL.md) | Fine-tune Cellpose on custom annotated microscopy images |

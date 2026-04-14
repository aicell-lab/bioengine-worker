# BioEngine CLI Reference — apps commands

CLI source: `/data/wei/workspace/bioengine-paper/bioengine_cli/apps.py`  
Install: `pip install bioengine`  
Entry point: `bioengine apps <command>`

## Prerequisites

Apps commands require a BioEngine worker service ID and an auth token:

```bash
export BIOENGINE_WORKER_SERVICE_ID=my-workspace/bioengine-worker
export HYPHA_TOKEN=<your-token>
```

Or pass per-command with `--worker` and `--token`.

---

## Command summary

```
bioengine apps deploy    Upload + deploy a local app directory (one step)
bioengine apps upload    Upload app files to Hypha artifact storage
bioengine apps run       Deploy an uploaded artifact
bioengine apps list      List all artifacts in the workspace
bioengine apps status    Show status of running deployments
bioengine apps logs      Show logs from a running deployment
bioengine apps stop      Stop and remove a running deployment
```

---

## `bioengine apps deploy`

```
Usage: bioengine apps deploy [OPTIONS] APP_DIR

Upload and immediately deploy a local BioEngine app directory.

Arguments:
  APP_DIR   Directory with manifest.yaml and deployment .py file(s)

Options:
  --app-id ID        Custom application instance ID (default: auto-generated)
  --no-gpu           Disable GPU even if worker has GPUs
  --env KEY=VALUE    Environment variable (repeat for multiple)
  --worker SERVICE   Worker service ID (or BIOENGINE_WORKER_SERVICE_ID)
  --token TOKEN      Auth token (or HYPHA_TOKEN)
```

**Examples:**
```bash
bioengine apps deploy ./my-app/
bioengine apps deploy ./my-pipeline/ --app-id pipeline-v1 --no-gpu
bioengine apps deploy ./my-app/ --env MODEL_PATH=/data/model.pt --env _API_KEY=secret
```

**Internally:** reads all files in APP_DIR, encodes binary as base64, calls `worker.save_application(files=[...])` then `worker.run_application(artifact_id=...)`.

---

## `bioengine apps upload`

```
Usage: bioengine apps upload [OPTIONS] APP_DIR

Upload a local BioEngine app directory to Hypha artifact storage.

Arguments:
  APP_DIR   Directory with manifest.yaml and deployment .py file(s)

Options:
  --public   Make artifact publicly readable
  --worker / --token / --server-url   (see above)
```

**Output:** Prints the artifact ID on success. Pass this ID to `bioengine apps run`.

---

## `bioengine apps run`

```
Usage: bioengine apps run [OPTIONS] ARTIFACT_ID

Deploy a BioEngine application from artifact storage.

Arguments:
  ARTIFACT_ID   Artifact ID returned by `bioengine apps upload`

Options:
  --app-id ID        Custom instance ID (pass same ID to update in-place)
  --version VER      Specific artifact version (default: latest)
  --no-gpu           Disable GPU
  --env KEY=VALUE    Environment variable (repeat for multiple)
  --worker / --token / --server-url
```

**Examples:**
```bash
bioengine apps run my-workspace/my-app
bioengine apps run my-workspace/my-app --app-id production-v1
bioengine apps run my-workspace/my-app --no-gpu --env DEBUG=true
```

**Note:** Calling `run` with the same `--app-id` but a different artifact updates the app in-place.

---

## `bioengine apps list`

```
Usage: bioengine apps list [OPTIONS]

List all available BioEngine application artifacts in the current workspace.

Options:
  --json   Output as JSON
  --worker / --token / --server-url
```

Shows artifacts (uploaded but not necessarily running). For running deployments use `status`.

---

## `bioengine apps status`

```
Usage: bioengine apps status [OPTIONS] [APP_ID...]

Show status of deployed BioEngine applications.

Arguments:
  APP_ID...   One or more app IDs (default: all running apps)

Options:
  --logs N   Number of log lines per replica [default: 30]
  --json     Output full status as JSON
  --worker / --token / --server-url
```

**Examples:**
```bash
bioengine apps status
bioengine apps status my-app-id --logs 100
bioengine apps status app-a app-b --json
```

---

## `bioengine apps logs`

```
Usage: bioengine apps logs [OPTIONS] APP_ID

Show logs for a deployed BioEngine application.

Arguments:
  APP_ID   Application instance ID

Options:
  -n, --tail N   Number of log lines [default: 100]
  --json         Output as JSON
  --worker / --token / --server-url
```

---

## `bioengine apps stop`

```
Usage: bioengine apps stop [OPTIONS] APP_ID

Stop and remove a deployed BioEngine application.

Arguments:
  APP_ID   Application instance ID

Options:
  -y, --yes   Skip confirmation prompt
  --worker / --token / --server-url
```

**Note:** Stops the running deployment; does NOT delete the artifact from storage.

---

## Environment variables

| Variable | Description |
|---|---|
| `BIOENGINE_WORKER_SERVICE_ID` | BioEngine worker service ID (required for all apps commands) |
| `HYPHA_TOKEN` or `BIOENGINE_TOKEN` | Auth token |
| `BIOENGINE_SERVER_URL` | Hypha server URL (default: `https://hypha.aicell.io`) |

---

## File encoding for upload

- Text files (`.py`, `.yaml`, `.md`, `.ipynb`): uploaded as UTF-8 text
- Binary files (images, model weights, etc.): uploaded as base64
- `__pycache__` directories are automatically excluded

---

## Implementation notes

- API verified against `bioengine-worker/bioengine/worker/worker.py` (`run_application`, `stop_application`, `get_application_status`, `list_applications`, `save_application`).
- The `save_application` call is proxied through the worker service (not directly to artifact manager) to ensure proper collection membership validation.
- `application_env_vars` format expected by the server: `{"DeploymentName": {"KEY": "value"}}`. The CLI maps `--env KEY=VALUE` to `{"*": {KEY: value}}` as a shorthand for all deployments.

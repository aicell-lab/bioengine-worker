# BioEngine Datasets Server

A lightweight server for streaming scientific datasets with per-user access control. Datasets are served in-place from a local directory — no data copying is required. Token authentication is delegated to the central Hypha server on each request.

## Overview

The datasets server exposes a simple HTTP API for:
- Listing available datasets and their metadata
- Listing files within a dataset
- Streaming file bytes directly, including zarr chunks with HTTP Range request support

Clients connect directly to the datasets server using the `BioEngineDatasets` Python client. Authentication tokens are validated against `https://hypha.aicell.io` on demand and cached locally.

---

## Dataset Directory Structure

Your data directory contains one subdirectory per dataset. Each subdirectory must contain a `manifest.yaml` file. All other files in the directory (zarr stores, text files, etc.) are served as-is.

```
/path/to/datasets/
├── blood_atlas/
│   ├── manifest.yaml        ← required
│   ├── data.zarr/           ← zarr store (directory)
│   │   ├── zarr.json
│   │   └── cells/
│   │       ├── zarr.json
│   │       └── c/
│   │           └── 0/
│   │               └── 0    ← binary chunk file
│   └── README.md            ← optional documentation
│
└── spatial_txn/
    ├── manifest.yaml
    └── data.zarr/
```

> The directory name does not determine the dataset ID — the `id` field in `manifest.yaml` is used.

---

## manifest.yaml

Every dataset directory must contain a `manifest.yaml`. The following fields are recognised:

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier used in API calls and client code |
| `name` | `str` | Human-readable display name |
| `description` | `str` | Short description of the dataset |
| `authorized_users` | `list` | Who can access files — see [Access Control](#access-control) |

### Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | `str` | Dataset version string |
| `license` | `str` | License identifier (e.g. `CC-BY-4.0`) |
| `authors` | `list` | List of `{name, affiliation}` entries |
| `tags` | `list` | Keywords for discovery |
| `documentation` | `str` | URL to external documentation |
| `git_repo` | `str` | URL to associated repository |

### Minimal example

```yaml
id: blood-atlas
name: Blood Cell Atlas
description: Single-cell RNA-seq data from 50,000 human blood cells.
authorized_users:
  - "*"
```

### Full example

```yaml
id: blood-atlas
name: Blood Cell Atlas
description: Single-cell RNA-seq data from 50,000 human blood cells collected
  across 10 healthy donors. Includes raw counts and normalised expression layers.
version: "1.2"
license: CC-BY-4.0
authors:
  - name: Jane Smith
    affiliation: KTH Royal Institute of Technology
  - name: Erik Andersson
    affiliation: Karolinska Institutet
tags:
  - single-cell
  - RNA-seq
  - blood
documentation: https://example.com/blood-atlas-docs
git_repo: https://github.com/aicell-lab/blood-atlas
authorized_users:
  - researcher@university.edu
  - collaborator@institute.org
```

---

## Access Control

The `authorized_users` field in `manifest.yaml` controls who can list files and download data.

| Value | Effect |
|-------|--------|
| `["*"]` | Any authenticated user can access the dataset |
| `["user@example.com"]` | Only the user whose Hypha email matches |
| `["user:abc123"]` | Only the user whose Hypha user ID matches |
| `[]` or absent | No access granted to anyone |

Listing available datasets (`GET /datasets`) never requires authentication — anyone can see what datasets exist and read their manifests. Access control applies only to file listing and file downloads.

---

## Starting the Server

### Command line

```bash
python -m bioengine.datasets --data-dir /path/to/datasets
```

The server scans `--data-dir` at startup, registers the found datasets, and begins serving requests. No credentials are required at startup.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir PATH` | *(required)* | Directory containing dataset subdirectories |
| `--server-ip IP` | auto-detected | IP address written into the auto-discovery file and used in URLs |
| `--server-port PORT` | auto (39527+) | Port. Scans upward from 39527 if not set |
| `--authentication-server-url URL` | `https://hypha.aicell.io` | Hypha server used for token validation |
| `--log-file PATH` | auto-timestamped | Log file. Pass `off` for console-only logging |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIOENGINE_DATASETS_ZARR_STORE_CACHE_SIZE` | `1` | Default in-memory chunk cache size in GB for zarr access. Applied to both the Python client and standalone `HttpZarrStore` instances |

### Docker

```bash
docker run --rm \
  -v /path/to/datasets:/data \
  -v ~/.bioengine:/home/.bioengine \
  -e HOME=/home \
  -p 39527:39527 \
  ghcr.io/aicell-lab/bioengine-datasets:latest \
  python -m bioengine.datasets --data-dir /data --server-port 39527
```

### Docker Compose

Set environment variables and start the `data-server` service from the repository root:

```bash
export DATA_DIR=/path/to/datasets
export UID=$(id -u)
export GID=$(id -g)
docker compose up data-server
```

The compose file does not publish any ports. On Linux, the Docker bridge network is reachable from the host, so local clients (same machine) can connect via the URL written to `~/.bioengine/datasets/bioengine_current_server`. For access from other machines, either add a `ports:` mapping to the service or pass `--server-ip <public-ip>` so the auto-discovery file contains the correct external address.

---

## HTTP API

All endpoints are at `http://<server-ip>:<port>`.

### `GET /health/liveness`

Returns `{"status": "ok"}`. Used by Docker health checks.

### `GET /ping`

Returns `"pong"`. Simple connectivity check.

### `GET /datasets`

Returns metadata for all datasets. No authentication required.

```bash
curl http://localhost:39527/datasets
```

```json
{
  "blood-atlas": {
    "id": "blood-atlas",
    "name": "Blood Cell Atlas",
    "description": "...",
    "authorized_users": ["*"]
  }
}
```

### `GET /datasets/{id}/files`

Lists all files in a dataset. Requires a valid token for non-public datasets.

| Parameter | In | Description |
|-----------|----|-------------|
| `token` | query | Hypha authentication token (required if not public) |
| `dir_path` | query | Subdirectory to list (e.g. `data.zarr`) |

```bash
curl "http://localhost:39527/datasets/blood-atlas/files?token=your_token"
```

Returns paths relative to the dataset root:

```json
["README.md", "manifest.yaml", "data.zarr/zarr.json", "data.zarr/cells/c/0/0"]
```

### `GET /data/{dataset_id}/{path}`

Serves raw file bytes. Supports HTTP Range requests for partial content, which zarr clients use to fetch individual chunks efficiently.

```bash
# Full file
curl "http://localhost:39527/data/blood-atlas/data.zarr/cells/c/0/0?token=your_token"

# Partial content
curl -H "Range: bytes=0-1023" \
  "http://localhost:39527/data/blood-atlas/data.zarr/cells/c/0/0?token=your_token"
# → HTTP 206 Partial Content
```

---

## Python Client

### Installation

```bash
pip install "bioengine[datasets]"
```

### Initialisation

```python
from bioengine.datasets import BioEngineDatasets
import os

# Auto-discovers server URL from ~/.bioengine/datasets/bioengine_current_server
client = BioEngineDatasets(
    data_server_url="auto",          # default
    hypha_token=os.getenv("HYPHA_TOKEN"),
    chunk_cache_size_gb=1,           # optional, see Chunk Cache below
)

# Or connect to an explicit URL
client = BioEngineDatasets(
    data_server_url="http://192.168.1.10:39527",
    hypha_token=os.getenv("HYPHA_TOKEN"),
)
```

> In BioEngine applications, `self.bioengine_datasets` is pre-configured automatically.

### List datasets

```python
datasets = await client.list_datasets()
# {'blood-atlas': {'id': 'blood-atlas', 'name': 'Blood Cell Atlas', ...}}
```

### List files

```python
# All files in a dataset
files = await client.list_files("blood-atlas")

# Only files inside data.zarr/
files = await client.list_files("blood-atlas", dir_path="data.zarr")
```

### Get a zarr store

Returns an `HttpZarrStore` compatible with `zarr` and `anndata`. Only the chunks you access are downloaded.

```python
import zarr

store = await client.get_file("blood-atlas", file_name="data.zarr")
group = zarr.open_group(store=store, mode="r")
print(list(group.array_keys()))
```

### Get a non-zarr file

Returns raw bytes.

```python
readme = await client.get_file("blood-atlas", file_name="README.md")
print(readme.decode("utf-8"))
```

### Read with AnnData

```python
import asyncio
import anndata

store = await client.get_file("blood-atlas", file_name="data.zarr")

# Lazy load — metadata only, no data transferred yet
adata = await asyncio.to_thread(
    anndata.experimental.read_lazy, store, load_annotation_index=True
)
print(adata)           # AnnData object summary

# Access a slice — fetches only the required zarr chunks
counts = adata.layers["X_binned"][0:10, :].compute()
```

### Chunk cache

The client keeps a per-client LRU cache of zarr chunks in memory. All zarr stores opened by the same `BioEngineDatasets` instance share one cache budget.

```python
# Set cache size at construction (default: 1 GB, or BIOENGINE_DATASETS_ZARR_STORE_CACHE_SIZE env var)
client = BioEngineDatasets(chunk_cache_size_gb=4)

# Resize at runtime — evicts LRU entries immediately if needed
await client.set_chunk_cache_size_gb(2)

# Disable caching entirely
await client.set_chunk_cache_size_gb(0)
```

The default can also be set process-wide via the environment variable:

```bash
export BIOENGINE_DATASETS_ZARR_STORE_CACHE_SIZE=4
```

### Client auto-discovery

When the server starts it writes its URL to `~/.bioengine/datasets/bioengine_current_server`. Passing `data_server_url="auto"` (the default) makes the client read this file automatically, so no URL needs to be configured when server and client run on the same machine.

---

## Architecture

```
Client (BioEngineDatasets)
       │
       │  HTTP (direct connection)
       ▼
BioEngine Datasets Server (FastAPI)
  ├─ /datasets            manifest metadata from manifest.yaml
  ├─ /datasets/{id}/files filesystem scan of dataset directory
  └─ /data/{id}/{path}    serves file bytes (Range-aware)
       │
       │  per-request token validation (cached)
       ▼
https://hypha.aicell.io  (central auth server)
```

The server reads dataset directories and `manifest.yaml` files at startup. No database, no object store, and no separate services are required.

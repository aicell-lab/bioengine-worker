# BioEngine Datasets

Privacy-preserved scientific data management system for streaming large datasets with fine-grained access control.

## Overview

BioEngine Datasets provides a secure, efficient way to serve and access large scientific datasets. It implements a manifest-driven architecture with HTTP-based streaming, enabling partial data access to Zarr-formatted datasets while maintaining per-user access control.

## Data Preparation

### Directory Structure

The data directory can contain multiple datasets, each in a separate folder. A dataset is recognized by the presence of a `manifest.yaml` file. Each dataset folder can contain an arbitrary number of files.

```
/path/to/data/
└── my_dataset/
    ├── data.zarr/               # Optional: Zarr dataset
    ├── example.txt              # Optional: Text file
    ├── manifest.yaml            # Required: Dataset configuration
    └── subdirectory/            # Optional: Subdirectory with more files
        └── file.txt
```

> **Note**: The folder name does not determine the dataset ID. The `id` field in `manifest.yaml` specifies the dataset identifier.

### Manifest Configuration

Each dataset requires a `manifest.yaml` file with the following required fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier for accessing the dataset |
| `name` | `str` | Human-readable name for the dataset |
| `description` | `str` | Brief description of the dataset |
| `authorized_users` | `List[str]` | Hypha user IDs or email addresses with access |

#### Minimal Example

```yaml
id: my-dataset
name: My Dataset
description: Description of the dataset contents.
authorized_users:
  - user@example.com
  - another.user@domain.org
```

#### Full Example

```yaml
id: blood_perturb_rna_001
name: Blood-Perturb-RNA
description: CRISPR perturbation screen of 19 transcription factors in human HSPCs with scRNA-seq.

# Access control
authorized_users:
  - user@example.com

# Optional metadata
authors:
  - name: Author Name
    affiliation: Institution Name
license: CC-BY-4.0
documentation: https://example.com/docs
git_repo: https://github.com/org/repo
tags:
  - perturbation
  - scRNA-seq
```

### Access Control

The `authorized_users` field controls who can list and retrieve files from the dataset:

- **Specific users**: List email addresses or Hypha user IDs
- **Public access**: Use `["*"]` to allow unrestricted access

```yaml
# Restricted access
authorized_users:
  - user1@example.com
  - user2@example.com

# Public access (no authentication required)
authorized_users:
  - "*"
```

> **Note**: Listing available datasets does not require authentication. Access control only applies to listing files within a dataset and retrieving file contents.

## Running the Data Server

### From Python

```bash
python -m bioengine.datasets --data-dir /path/to/data --workspace-dir $HOME/.bioengine
```

### From Docker

```bash
docker run -it --rm \
    --user $(id -u):$(id -g) \
    -v $HOME/.bioengine:/.bioengine \
    -v /path/to/data:/data \
    ghcr.io/aicell-lab/bioengine-worker:latest \
    python -m bioengine.datasets --data-dir /data --workspace-dir /.bioengine --server-port 9527
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--data-dir PATH` | **Required.** Root directory containing datasets |
| `--workspace-dir PATH` | Directory for workspace and temporary files |
| `--server-ip IP` | IP address for the proxy server (default: localhost) |
| `--server-port PORT` | Port for the proxy server |
| `--minio-port PORT` | Port for the MinIO S3 backend |
| `--log-file PATH` | Log file path (use `off` for console-only logging) |

## Accessing Datasets

### Initializing BioEngineDatasets

```python
import os
from bioengine.datasets import BioEngineDatasets

bioengine_datasets = BioEngineDatasets(
    data_server_url="auto",
    hypha_token=os.getenv("HYPHA_TOKEN"),
)
```

> **Note**: In BioEngine applications, `self.bioengine_datasets` is automatically available and pre-configured. See the [BioEngine Applications User Guide](../../bioengine_apps/README.md#selfbioengine_datasets) for details.

### List Available Datasets

```python
datasets = await bioengine_datasets.list_datasets()
```

### List Files in a Dataset

```python
files = await bioengine_datasets.list_files(
    dataset_id="my-dataset",
    dir_path=None,  # Root directory
    token=None,     # Use default token from initialization
)
# Output: ["data.zarr", "example.txt", "subdirectory"]
```

> **Note**: The manifest file (`manifest.yaml`) is not included in the file listing.


### Get File Contents

```python
content = await bioengine_datasets.get_file(
    dataset_id="my-dataset",
    file_path="example.txt",
    token=None,
)
file_text = content.decode("utf-8")
```

### Stream Zarr Data

```python
import zarr

# Get Zarr store with HTTP streaming
zarr_store = await bioengine_datasets.get_file(
    dataset_id="my-dataset",
    file_path="data.zarr",
    token=None,
)

# Open as Zarr group
zarr_group = zarr.open_group(store=zarr_store, mode="r")
arrays = list(zarr_group.array_keys())
```

### Access Subdirectories

```python
# List files in a subdirectory
files = await bioengine_datasets.list_files(
    dataset_id="my-dataset",
    dir_path="subdirectory",
    token=None,
)

# Get a file from a subdirectory
content = await bioengine_datasets.get_file(
    dataset_id="my-dataset",
    file_path="subdirectory/file.txt",
    token=None,
)
```

### Authentication

For datasets with restricted access, provide a user token:

```python
content = await bioengine_datasets.get_file(
    dataset_id="restricted-dataset",
    file_path="data.zarr",
    token="user-authentication-token",
)
```

If no token is provided, `BioEngineDatasets` uses the token passed during initialization.

## Architecture

The BioEngine Datasets system consists of:

- **Proxy Server**: HTTP server with Hypha integration and MinIO S3 backend
- **BioEngineDatasets Client**: Async interface for dataset access from applications
- **HttpZarrStore**: Efficient streaming store for partial Zarr data access

This architecture enables efficient streaming of large datasets without requiring full downloads, with privacy-preserved access control managed through Hypha authentication.

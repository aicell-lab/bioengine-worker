# BioEngine Worker

Manages Ray cluster lifecycle BioEngine Apps and Datasets on HPC systems, single machine Ray instances or pre-existing Ray environments.

Provides a Hypha service interface for streaming datasets, executing python code remotely or deploying models through Ray.

The BioEngine worker comes in three modes:
- `slurm`: Start a latent BioEngine worker service with very low resource consumption that activates upon usage and scales a Ray cluster by submitting SLURM jobs to recruit additional workers. An autoscaling system will handle up- and down-scaling.
- `single-machine`: Start a local BioEngine worker with Ray running on a single machine.
- `external-cluster`: Start a BioEngine worker with data access, but execute all computations on a remote Ray cluster.

## Start your own BioEngine worker

The BioEngine worker comes in containerized form as Docker or Apptainer image.

### Docker (for workstations or K8s)

A prebuilt Docker image is available under `ghcr.io/aicell-lab/bioengine-worker:latest` (see [all available versions](https://github.com/orgs/aicell-lab/packages/container/package/bioengine-worker) to use a specific version).

To make use of the predefined settings, clone this Github repository and run `docker compose up`.

This assumes the following directories in the current working directory for mounting into the docker container:
- `.bioengine`: A temporary directory for Ray (read and write)
- `data`: A directory with datasets (available only as read-only)

A token for the Hypha workspace can either be added using the tag `--token` or provided in the `.env` file as `HYPHA_TOKEN`. Otherwise, you will be prompted to login when starting the worker.
The default workspace can be changed using the tag `--workspace`.

By default, the BioEngine worker will start a local Ray cluster with the provided resources `--head_num_gpus` and `--head_num_cpus`. To connect to a running Ray cluster, change the tag `--mode` to `"external-cluster"`.

An overview of all tags for the BioEngine worker can be accessed via:
```bash
docker run --rm ghcr.io/aicell-lab/bioengine-worker:latest python -m bioengine.worker --help
```

To run as your own user, the variables `UID` and `GID` are required. If not set, `export` them before running docker compose with `export UID=$(id -u)` and `export GID=$(id -g)` or add them to your `.env` file.

As a shortcut, you can also run:

```bash
UID=$(id -u) GID=$(id -g) docker compose up
```

### Apptainer (for HPC)

The bash script [`start_hpc_worker.sh`](scripts/start_hpc_worker.sh) helps to start a BioEngine worker in Apptainer. Either clone this Github repository to run the script:

```bash
bash scripts/start_hpc_worker.sh
```

or access the script directly from Github like this:
```bash
bash <(curl -s https://raw.githubusercontent.com/aicell-lab/bioengine-worker/main/scripts/start_hpc_worker.sh)
```

An overview of all tags for the BioEngine worker can be accessed via:
```bash
bash scripts/start_hpc_worker.sh --help
```

The script will pull the latest BioEngine worker docker image and convert it into Singularity Image Format (SIF) using Apptainer. These Apptainer images will be saved to the directory `./images/`. 

To avoid interactive login to Hypha, pass the token with the tag `--token` or save it to `HYPHA_TOKEN` in the `.env` file in the root directory of the project. The script will automatically load the token from the `.env` file if it exists.

The directory `.bioengine` will be automatically created in the current working directory if the respective tag `--cache_dir` is not specified.

#### BioEngine worker with different base images

The default image `ghcr.io/aicell-lab/bioengine-worker` only has a minimal list of Python packages installed. All additional Python packages need to be installed via a separate pip runtime environment when executing python code on or deploying a model to the BioEngine worker.

It is possible to start a BioEngine worker with a different base image, provided that `bioengine` is installed. All installations will be available when executing python code on or deploying a model.

Here are two examples of how this can be done, from a remote docker image:
```bash
bash scripts/start_hpc_worker.sh --image <remote_docker_image>
```

or from a local Apptainer image file:
```bash
bash scripts/start_hpc_worker.sh --image <path_to_apptainer_image>.sif
```

Note: When a Ray runtime environment is provided, it is not possible to access installations from the base image anymore.

## How to use the BioEngine worker

The BioEngine worker is based on Hypha and registers different services once started.

### Services of the BioEngine worker

Once a BioEngine worker is started, you can access it remotely through Hypha:

```python
from hypha_rpc import connect_to_server

server = await connect_to_server(
    {
        "server_url": server_url,
        "workspace": workspace,
        "token": token,
    }
)
worker_service = await server.get_service("<workspace>/<service_id>")
```

The default `service_id` is `bioengine_worker`.

The worker service provides the following functions:
- `get_status()`
- `stop_worker()`
- `test_access()`
- `list_datasets()`
- `update_datasets()`
- `execute_python_code(...)`
- `save_application(files)`
- `list_applications()`
- `get_application_manifest(application_id)`
- `delete_application(application_id)`
- `run_application(artifact_id, ...)`
- `stop_application(application_id)`
- `get_application_status()`

As an example, the worker status can be called like this:
```python
status = await worker_service.get_status()
```

The status contains information about the:
- Hypha service (`service`)
- Ray cluster (`ray_cluster`)

### Deploying models to the BioEngine worker

The BioEngine worker deploys models from the Hypha artifact manager. To create a deployment artifact, run the following script:

```bash
python scripts/manage_artifact.py --deployment_dir <path_to_your_deployment_dir>
```

A deployment requires a `manifest.yml` file and a python script defining the deployment model.

The `manifest.yml` requires at minimum the following fields:
- `name`
- `description`
- `type`
- `class_name`

With the field `deployment_config`, you can define required resources (e.g., `num_cpus` and `num_gpus`), a pip runtime environment, and more.

The field `python_file` defines the python script name, by default `main.py`. This python script must define the model class. The name of the model class needs to be specified in the `class_name` field. The model class requires the `__call__` method. Also note that all imports must be made within the class!

An example deployment can be found in [`bioengine_apps/example_deployment`](bioengine_apps/example_deployment).

### Streaming datasets with the BioEngine worker

A dataset consists of a folder in the specified data directory containing a manifest file and the corresponding zarr files.

#### Dataset Structure
```
example_dataset/
├── data_file_1.zarr/
│   └── ...
├── data_file_2.zarr/
│   └── ...
└── manifest.yml
```

#### Manifest File Format
The `manifest.yml` file defines the dataset metadata and files:
```yaml
description: "This is an example dataset"
files:
  data_file_1.zarr:  # Note: filename must match actual file
    description: "Example file 1"
    version: "1.0.0"
  data_file_2.zarr:
    description: "Example file 2"
    version: "1.0.0"
```

#### Working with Datasets

1. **Load the dataset** by calling `load_dataset` from the worker service:
   ```python
   dataset_id = "example_dataset"
   await worker_service.load_dataset(dataset_id)
   ```
   
   This registers an ASGI app that enables streaming the dataset using the `HttpZarrStore`.

2. **Access the dataset** in your code using the provided HTTP interface.

3. **Release resources** when finished:
   ```python
   await worker_service.close_dataset(dataset_id)
   ```

### Accessing other datasets

All other files that are not zarr and have no manifest.yml file will be accessible under `/data` from `execute_python_code` or from deployed models.


### Build a multi-arch docker image

To build a multi-arch docker image, you can use the `buildx` command. This requires Docker Buildx to be installed and set up on your system.
```bash
docker buildx create --use --name multiarch-builder
docker buildx inspect multiarch-builder --bootstrap
```
This ensures you're using a builder that supports multiple platforms.

To build the image, run the following command:
```bash
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t ghcr.io/aicell-lab/bioengine-worker:latest \
    --push .
```

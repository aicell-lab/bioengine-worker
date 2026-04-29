"""
BioEngine Datasets Proxy Server - Privacy-preserved dataset management service.

Serves zarr datasets in-place from a local directory. Registers a service to
the remote central Hypha server for RPC-based dataset discovery and access
control, while a local FastAPI app handles the actual file byte serving.

Architecture:
- Dataset directories are scanned for manifest.yaml at startup
- Access control is defined per-dataset in manifest.yaml (authorized_users field)
- Token authentication is delegated to the remote Hypha server (cached locally)
- RPC service (list_datasets, list_files, get_presigned_url) registered to remote Hypha
- Local FastAPI app serves zarr chunks directly with Range request support
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import RemoteService

from bioengine import __version__
from bioengine.utils import (
    acquire_free_port,
    create_logger,
    get_internal_ip,
)
from bioengine.utils.permissions import check_permissions

AUTHENTICATION_SERVER_URL: str = "https://hypha.aicell.io"
_datasets: Dict[str, dict] = {}  # dataset_id -> {manifest, path, authorized_users}
_cached_user_info: Dict[str, dict] = {}
_local_http_url: str = ""

logger = logging.getLogger("ProxyServer")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_log_config(log_file: Optional[Path] = None) -> Dict[str, Any]:
    # Set logging level and config (based on uvicorn default config with asctime added)
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s %(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    if log_file:
        for handler in log_config["handlers"].values():
            handler["class"] = "logging.FileHandler"
            handler["filename"] = str(log_file)
            handler.pop("stream", None)
    return log_config


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_datasets(data_dir: Path) -> Dict[str, dict]:
    """Scan data_dir for subdirectories containing manifest.yaml.

    Returns a dict keyed by dataset_id (manifest["id"] or directory name).
    Each value has: manifest (dict), path (Path), authorized_users (list).
    """
    datasets = {}
    if not data_dir.is_dir():
        raise ValueError(f"data_dir does not exist or is not a directory: {data_dir}")

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        manifest_file = subdir / "manifest.yaml"
        if not manifest_file.exists():
            continue
        try:
            manifest = yaml.safe_load(manifest_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to parse {manifest_file}: {e}")
            continue

        dataset_id = manifest.get("id") or subdir.name
        authorized_users = manifest.get("authorized_users", [])
        if isinstance(authorized_users, str):
            authorized_users = [authorized_users]

        datasets[dataset_id] = {
            "manifest": manifest,
            "path": subdir,
            "authorized_users": authorized_users,
        }
        logger.info(f"Loaded dataset '{dataset_id}' from {subdir}")

    return datasets


# ---------------------------------------------------------------------------
# Token authentication
# ---------------------------------------------------------------------------


async def parse_token(
    token: Union[str, None],
    cached_user_info: Dict[str, dict],
) -> Dict[str, str]:
    global AUTHENTICATION_SERVER_URL

    if token is not None:
        if (
            token in cached_user_info
            and cached_user_info[token].get("expires_at", 0) > time.time()
        ):
            return cached_user_info[token]

        async with connect_to_server(
            {"server_url": AUTHENTICATION_SERVER_URL, "token": token}
        ) as user_client:
            user_info = user_client.config.user

        cached_user_info[token] = user_info
        if len(cached_user_info) > 1000:
            cached_user_info.pop(next(iter(cached_user_info)))
    else:
        user_info = {"id": "anonymous-user", "email": "no-email"}

    return user_info


# ---------------------------------------------------------------------------
# RPC service functions (registered to remote Hypha)
# ---------------------------------------------------------------------------


async def list_datasets(datasets: Dict[str, dict]) -> Dict[str, dict]:
    """Return manifest metadata for all datasets. No auth required for listing."""
    return {dataset_id: info["manifest"] for dataset_id, info in datasets.items()}


async def list_files(
    dataset_id: str,
    datasets: Dict[str, dict],
    cached_user_info: Dict[str, dict],
    dir_path: Optional[str] = None,
    token: Optional[str] = None,
) -> List[str]:
    """List files in a dataset directory after checking access permissions."""
    if dataset_id not in datasets:
        raise ValueError(f"Dataset '{dataset_id}' does not exist")

    user_info = await parse_token(token=token, cached_user_info=cached_user_info)
    check_permissions(
        context={"user": user_info},
        authorized_users=datasets[dataset_id]["authorized_users"],
        resource_name=f"list files in dataset '{dataset_id}'",
    )

    base_path = datasets[dataset_id]["path"]
    scan_path = base_path / dir_path if dir_path else base_path

    if not scan_path.exists():
        raise ValueError(f"Path '{dir_path}' does not exist in dataset '{dataset_id}'")

    files = []
    for root, dirs, filenames in os.walk(scan_path):
        dirs.sort()
        for fname in sorted(filenames):
            full = Path(root) / fname
            files.append(str(full.relative_to(base_path)))

    return files


async def get_presigned_url(
    dataset_id: str,
    file_path: str,
    datasets: Dict[str, dict],
    cached_user_info: Dict[str, dict],
    local_http_url: str,
    token: Optional[str] = None,
) -> str:
    """Return a URL to fetch the file from the local HTTP server."""
    if dataset_id not in datasets:
        raise ValueError(f"Dataset '{dataset_id}' does not exist")

    user_info = await parse_token(token=token, cached_user_info=cached_user_info)
    check_permissions(
        context={"user": user_info},
        authorized_users=datasets[dataset_id]["authorized_users"],
        resource_name=f"access file '{file_path}' in dataset '{dataset_id}'",
    )

    # Verify the file exists before returning a URL
    full_path = datasets[dataset_id]["path"] / file_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"File '{file_path}' not found in dataset '{dataset_id}'"
        )

    url = f"{local_http_url}/data/{dataset_id}/{file_path}"
    if token:
        url += f"?token={token}"
    return url


# ---------------------------------------------------------------------------
# Service registration (runs as background task)
# ---------------------------------------------------------------------------


async def _run_service_forever(
    remote_url: str,
    workspace: str,
    service_token: str,
    datasets: Dict[str, dict],
    cached_user_info: Dict[str, dict],
    local_http_url: str,
) -> None:
    """Connect to remote Hypha and keep the service registered until cancelled."""
    while True:
        try:
            logger.info(
                f"Registering bioengine-datasets service to {remote_url} "
                f"(workspace: {workspace})"
            )
            async with connect_to_server(
                {
                    "server_url": remote_url,
                    "token": service_token,
                    "workspace": workspace,
                }
            ) as server:
                await server.register_service(
                    {
                        "id": "bioengine-datasets",
                        "name": "BioEngine Datasets",
                        "description": f"BioEngine Datasets proxy server v{__version__}",
                        "config": {"visibility": "public"},
                        "ping": lambda: "pong",
                        "list_datasets": partial(list_datasets, datasets=datasets),
                        "list_files": partial(
                            list_files,
                            datasets=datasets,
                            cached_user_info=cached_user_info,
                        ),
                        "get_presigned_url": partial(
                            get_presigned_url,
                            datasets=datasets,
                            cached_user_info=cached_user_info,
                            local_http_url=local_http_url,
                        ),
                    }
                )
                logger.info("bioengine-datasets service registered successfully")
                # keep the connection open indefinitely
                await asyncio.get_event_loop().create_future()
        except asyncio.CancelledError:
            logger.info("Service registration task cancelled")
            return
        except Exception as e:
            logger.warning(f"Service connection lost ({e}), reconnecting in 5 s...")
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# FastAPI app (file byte serving)
# ---------------------------------------------------------------------------


def _build_app(
    datasets: Dict[str, dict],
    cached_user_info: Dict[str, dict],
    remote_url: str,
    workspace: str,
    service_token: str,
    local_http_url: str,
) -> FastAPI:
    service_task: Optional[asyncio.Task] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal service_task
        service_task = asyncio.create_task(
            _run_service_forever(
                remote_url=remote_url,
                workspace=workspace,
                service_token=service_token,
                datasets=datasets,
                cached_user_info=cached_user_info,
                local_http_url=local_http_url,
            )
        )
        yield
        if service_task and not service_task.done():
            service_task.cancel()
            try:
                await service_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="BioEngine Datasets",
        description="Local file server for BioEngine dataset zarr chunks",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/data/{dataset_id}/{path:path}")
    async def serve_file(
        dataset_id: str,
        path: str,
        token: Optional[str] = None,
        request: Request = None,
    ):
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

        try:
            user_info = await parse_token(
                token=token, cached_user_info=cached_user_info
            )
            check_permissions(
                context={"user": user_info},
                authorized_users=datasets[dataset_id]["authorized_users"],
                resource_name=f"access file '{path}' in dataset '{dataset_id}'",
            )
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))

        full_path = datasets[dataset_id]["path"] / path
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail=f"File '{path}' not found")

        # FileResponse handles Range requests (206 Partial Content) natively
        return FileResponse(full_path)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def start_proxy_server(
    data_dir: Union[str, Path],
    server_ip: Optional[str] = None,
    server_port: int = 9527,
    workspace: str = "bioimage-io",
    service_token: Optional[str] = None,
    authentication_server_url: str = "https://hypha.aicell.io",
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    Start the BioEngine Datasets proxy server.

    Scans data_dir for dataset subdirectories (each must contain a manifest.yaml),
    registers a service on the remote Hypha server for dataset discovery and access
    control, and starts a local FastAPI server that streams zarr chunk files directly
    to clients.

    No data is copied — datasets are served in-place from data_dir.

    Args:
        data_dir: Directory containing dataset subdirectories. Each subdirectory
                  must have a manifest.yaml with at minimum an "id" and
                  "authorized_users" field.
        server_ip: IP address for the local file-serving HTTP server. Defaults to
                   the machine's internal IP.
        server_port: Port for the local HTTP server (default: 9527).
        workspace: Hypha workspace to register the service in (default: "bioimage-io").
        service_token: Hypha token used to register the service. Falls back to the
                       HYPHA_TOKEN environment variable if not provided.
        authentication_server_url: URL of the central Hypha server used for token
                                   validation (default: "https://hypha.aicell.io").
        log_file: Path to log file. Pass "off" to disable file logging. Defaults to
                  a timestamped file in <data_dir>/../logs/.
    """
    global AUTHENTICATION_SERVER_URL

    data_dir = Path(data_dir).resolve()
    AUTHENTICATION_SERVER_URL = authentication_server_url

    # Resolve service token
    if service_token is None:
        service_token = os.environ.get("HYPHA_TOKEN")
    if not service_token:
        raise ValueError(
            "A service token is required to register with the remote Hypha server. "
            "Pass --service-token or set the HYPHA_TOKEN environment variable."
        )

    # Initialize logging
    if log_file != "off":
        if log_file is None:
            log_dir = data_dir.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"bioengine_datasets_{time.strftime('%Y%m%d_%H%M%S')}.log"
        else:
            log_file = Path(log_file).resolve()
            log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = None

    global logger
    logger = create_logger("ProxyServer", log_file=log_file)

    # File used by BioEngineDatasets("auto") client discovery
    current_server_file = data_dir.parent / "bioengine_current_server"

    try:
        logger.info(f"Starting BioEngine Datasets proxy server v{__version__}")
        logger.info(f"Serving datasets from: {data_dir}")

        # Load dataset manifests
        datasets = load_datasets(data_dir)
        logger.info(f"Found {len(datasets)} dataset(s): {list(datasets.keys())}")

        # Determine local HTTP server address
        server_ip = server_ip or get_internal_ip()
        free_server_port, server_s = acquire_free_port(
            port=server_port, step=1, ip=server_ip, keep_open=True
        )
        server_s.close()
        if free_server_port != server_port:
            logger.warning(
                f"Port {server_port} is not available. Using {free_server_port} instead."
            )

        local_http_url = f"http://{server_ip}:{free_server_port}"

        # Write remote service base URL for BioEngineDatasets("auto") discovery
        remote_base_url = f"{authentication_server_url}/{workspace}"
        current_server_file.write_text(remote_base_url)
        logger.info(
            f"Service will be available at "
            f"{remote_base_url}/public/services/bioengine-datasets"
        )
        logger.info(f"Local file server: {local_http_url}")

        cached_user_info: Dict[str, dict] = {}

        app = _build_app(
            datasets=datasets,
            cached_user_info=cached_user_info,
            remote_url=authentication_server_url,
            workspace=workspace,
            service_token=service_token,
            local_http_url=local_http_url,
        )

        log_config = get_log_config(log_file=log_file)

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=free_server_port,
            log_config=log_config,
            log_level="info",
        )

    except Exception as e:
        logger.error(f"Failed to start BioEngine Datasets proxy server: {e}")
        raise

    finally:
        if current_server_file.exists():
            try:
                current_server_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up current server file: {e}")

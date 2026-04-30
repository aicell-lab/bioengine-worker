"""
BioEngine Datasets Proxy Server - Privacy-preserved dataset management service.

Serves zarr datasets in-place from a local directory. A FastAPI app handles
all client-facing endpoints (dataset listing, file listing, and zarr chunk
serving) with no Hypha service registration needed.

Token authentication is delegated to the remote Hypha server on demand and
cached locally — the server itself requires no credentials at startup.

Architecture:
- Dataset directories are scanned for manifest.yaml at startup
- A background asyncio task polls data_dir every 30 s and hot-reloads the
  in-memory registry when datasets are added, removed, or their manifest
  changes — no server restart needed
- Access control is defined per-dataset in manifest.yaml (authorized_users field)
- Token authentication is delegated to the remote Hypha server (cached locally)
- FastAPI exposes all endpoints under /public/services/bioengine-datasets/
- File bytes are served directly at /data/{dataset_id}/{path} with Range support
"""

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from hypha_rpc import connect_to_server

from bioengine import __version__
from bioengine.utils import (
    acquire_free_port,
    create_logger,
    get_internal_ip,
)
from bioengine.utils.permissions import check_permissions

AUTHENTICATION_SERVER_URL: str = "https://hypha.aicell.io"

logger = logging.getLogger("ProxyServer")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_log_config(log_file: Optional[Path] = None) -> Dict[str, Any]:
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
        logger.debug(f"Loaded dataset '{dataset_id}' from {subdir}")

    return datasets


async def _watch_data_dir(
    data_dir: Path,
    datasets: Dict[str, dict],
    poll_interval: int = 30,
) -> None:
    """Poll data_dir every poll_interval seconds and hot-reload the datasets registry.

    Updates the shared ``datasets`` dict in-place so all running route handlers
    see changes immediately without a server restart.
    """
    while True:
        await asyncio.sleep(poll_interval)
        try:
            fresh = load_datasets(data_dir)

            added = set(fresh) - set(datasets)
            removed = set(datasets) - set(fresh)
            changed = {
                ds_id
                for ds_id in set(fresh) & set(datasets)
                if fresh[ds_id]["manifest"] != datasets[ds_id]["manifest"]
            }

            for ds_id in added:
                datasets[ds_id] = fresh[ds_id]
                logger.info(f"Dataset added: '{ds_id}' ({fresh[ds_id]['path']})")

            for ds_id in removed:
                del datasets[ds_id]
                logger.info(f"Dataset removed: '{ds_id}'")

            for ds_id in changed:
                datasets[ds_id] = fresh[ds_id]
                logger.info(f"Dataset reloaded (manifest changed): '{ds_id}'")

        except Exception as e:
            logger.warning(f"Data directory watch error: {e}")


# ---------------------------------------------------------------------------
# Token authentication
# ---------------------------------------------------------------------------


async def parse_token(
    token: Optional[str],
    cached_user_info: Dict[str, dict],
) -> Dict[str, str]:
    """Validate a user token against the remote Hypha server (cached)."""
    global AUTHENTICATION_SERVER_URL

    if token is not None:
        cached = cached_user_info.get(token)
        if cached is not None and cached.get("expires_at", 0) > time.time():
            return cached

        async with connect_to_server(
            {"server_url": AUTHENTICATION_SERVER_URL, "token": token}
        ) as user_client:
            user_info = user_client.config.user

        cached_user_info[token] = user_info
        # Evict oldest entry when cache exceeds 1000 tokens
        if len(cached_user_info) > 1000:
            cached_user_info.pop(next(iter(cached_user_info)))
    else:
        user_info = {"id": "anonymous-user", "email": "no-email"}

    return user_info


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def _build_app(
    data_dir: Path,
    datasets: Dict[str, dict],
    cached_user_info: Dict[str, dict],
    watch_interval: int = 30,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(
            _watch_data_dir(data_dir, datasets, poll_interval=watch_interval)
        )
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(
        title="BioEngine Datasets",
        description="Zarr dataset server with manifest-based access control",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health/liveness")
    async def liveness():
        return {"status": "ok"}

    @app.get("/ping")
    async def ping():
        return "pong"

    @app.get("/datasets")
    async def list_datasets_route():
        return {
            dataset_id: info["manifest"]
            for dataset_id, info in datasets.items()
        }

    @app.get("/datasets/{dataset_id}/files")
    async def list_files_route(
        dataset_id: str,
        dir_path: Optional[str] = None,
        token: Optional[str] = None,
    ):
        if dataset_id not in datasets:
            raise HTTPException(
                status_code=400,
                detail=f"ValueError: Dataset '{dataset_id}' does not exist",
            )

        try:
            user_info = await parse_token(token, cached_user_info)
            check_permissions(
                context={"user": user_info},
                authorized_users=datasets[dataset_id]["authorized_users"],
                resource_name=f"list files in dataset '{dataset_id}'",
            )
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))

        base_path = datasets[dataset_id]["path"]
        scan_path = base_path / dir_path if dir_path else base_path

        if not scan_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"ValueError: Path '{dir_path}' does not exist in dataset '{dataset_id}'",
            )

        files = []
        for root, dirs, filenames in os.walk(scan_path):
            dirs.sort()
            for fname in sorted(filenames):
                full = Path(root) / fname
                files.append(str(full.relative_to(base_path)))
        return files

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
            user_info = await parse_token(token, cached_user_info)
            check_permissions(
                context={"user": user_info},
                authorized_users=datasets[dataset_id]["authorized_users"],
                resource_name=f"access '{path}' in dataset '{dataset_id}'",
            )
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))

        full_path = datasets[dataset_id]["path"] / path
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail=f"'{path}' not found")

        file_size = full_path.stat().st_size
        range_header = request.headers.get("Range") if request else None

        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m:
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                def read_range():
                    with open(full_path, "rb") as f:
                        f.seek(start)
                        return f.read(length)

                data = await asyncio.to_thread(read_range)
                return Response(
                    content=data,
                    status_code=206,
                    media_type="application/octet-stream",
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(length),
                    },
                )

        return FileResponse(full_path, headers={"Accept-Ranges": "bytes"})

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def start_proxy_server(
    data_dir: Union[str, Path],
    server_ip: Optional[str] = None,
    server_port: Optional[int] = None,
    authentication_server_url: str = "https://hypha.aicell.io",
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    Start the BioEngine Datasets proxy server.

    Scans data_dir for dataset subdirectories (each must contain a manifest.yaml),
    then starts a FastAPI server that:
    - Lists datasets and their manifests
    - Lists files within a dataset (access-controlled)
    - Streams zarr chunk files with HTTP Range request support

    No data is copied — datasets are served in-place from data_dir.
    No Hypha service registration is performed — clients connect directly
    to this server. The local server URL is written to
    ~/.bioengine/datasets/bioengine_current_server for client auto-discovery.

    Args:
        data_dir: Directory containing dataset subdirectories. Each must have a
                  manifest.yaml with at minimum an "id" and "authorized_users" field.
        server_ip: IP address for the HTTP server. Defaults to the machine's
                   internal IP.
        server_port: Port for the HTTP server. If None (default), scans for a
                     free port starting from 39527.
        authentication_server_url: URL of the central Hypha server used for
                                   per-request token validation
                                   (default: https://hypha.aicell.io).
        log_file: Path to log file. Pass "off" for console-only logging.
                  Defaults to a timestamped file in ~/.bioengine/logs/.
    """
    global AUTHENTICATION_SERVER_URL

    data_dir = Path(data_dir).resolve()
    AUTHENTICATION_SERVER_URL = authentication_server_url

    # Logging setup
    if log_file != "off":
        if log_file is None:
            log_dir = Path.home() / ".bioengine" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"bioengine_datasets_{time.strftime('%Y%m%d_%H%M%S')}.log"
        else:
            log_file = Path(log_file).resolve()
            log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = None

    global logger
    logger = create_logger("ProxyServer", log_file=log_file)

    # Client auto-discovery file — fixed location read by BioEngineDatasets("auto")
    current_server_file = Path.home() / ".bioengine" / "datasets" / "bioengine_current_server"
    try:
        current_server_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Could not create auto-discovery directory: {e}")
        current_server_file = None

    try:
        logger.info(f"Starting BioEngine Datasets proxy server v{__version__}")
        logger.info(f"Serving datasets from: {data_dir}")

        datasets = load_datasets(data_dir)
        logger.info(
            f"Found {len(datasets)} dataset(s): {list(datasets.keys())} "
            f"(watching for changes every 30 s)"
        )

        server_ip = server_ip or get_internal_ip()
        # Default port range: 39527–39999 (avoids Hypha's 9527 and Ray's ports)
        start_port = server_port if server_port is not None else 39527
        free_server_port, server_s = acquire_free_port(
            port=start_port, step=1, ip=server_ip, keep_open=True
        )
        server_s.close()
        if server_port is not None and free_server_port != server_port:
            logger.warning(
                f"Port {server_port} unavailable, using {free_server_port} instead."
            )

        local_http_url = f"http://{server_ip}:{free_server_port}"

        # Write local URL for BioEngineDatasets("auto") discovery
        if current_server_file is not None:
            current_server_file.write_text(local_http_url)
        logger.info(f"Server URL: {local_http_url}")

        cached_user_info: Dict[str, dict] = {}

        app = _build_app(
            data_dir=data_dir,
            datasets=datasets,
            cached_user_info=cached_user_info,
        )

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=free_server_port,
            log_config=get_log_config(log_file=log_file),
            log_level="info",
        )

    except Exception as e:
        logger.error(f"Failed to start BioEngine Datasets proxy server: {e}")
        raise

    finally:
        if current_server_file is not None and current_server_file.exists():
            try:
                current_server_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up current server file: {e}")

"""
BioEngine Datasets Proxy Server - Privacy-preserved dataset management service.

This module implements a comprehensive dataset management service with privacy
preservation, access control, and efficient data streaming capabilities. It provides
both the server-side implementation of the datasets API and the infrastructure for
deploying standalone dataset services.

The proxy server integrates with Hypha for authentication and service registration
and MinIO for secure object storage.

Key Components:
- Hypha server integration for authentication and service discovery
- MinIO S3 backend for secure, scalable data storage
- Manifest-driven dataset configuration with access controls
- Presigned URL generation for secure, efficient data access
- Comprehensive logging and monitoring for production deployments

Service Architecture:
The proxy server implements a multi-tier architecture with Hypha for front-end
authentication, MinIO for backend storage, and custom middleware for access control
and presigned URL generation. It supports both local filesystem access and remote
S3-compatible storage backends.

Usage:
The start_proxy_server function is the main entry point for deploying the service,
either programmatically or through the command-line interface in __main__.py.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import time
from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
import yaml
from bioengine import __version__
from bioengine.utils import (
    acquire_free_port,
    create_logger,
    date_format,
    get_internal_ip,
)
from bioengine.utils.permissions import check_permissions
from hypha.server import create_application, get_argparser
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import ObjectProxy, RemoteService

BIOENGINE_DATA_DIR: Path
MINIO_CONFIG = {
    "mc_exe": str,
    "minio_port": int,
    "minio_user": str,
    "minio_password": str,
}
AUTHENTICATION_SERVER_URL: str


def get_log_config(log_file: Optional[Path] = None) -> Dict[str, Any]:
    # Set logging level and config (based on uvicorn default config with asctime added)
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s - %(levelprefix)s %(message)s",
                "datefmt": date_format,
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": date_format,
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
            "uvicorn.error": {"level": "ERROR"},
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    # Add file handler if log_file is set
    if log_file is not None:
        # Add file formatter and handler
        log_config["formatters"]["file"] = {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)s - %(levelprefix)s %(message)s",
            "datefmt": date_format,
            "use_colors": False,
        }
        log_config["formatters"]["file_access"] = {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": date_format,
            "use_colors": False,
        }
        log_config["handlers"]["file"] = {
            "formatter": "file",
            "class": "logging.FileHandler",
            "filename": str(log_file),
        }
        log_config["handlers"]["file_access"] = {
            "formatter": "file_access",
            "class": "logging.FileHandler",
            "filename": str(log_file),
        }
        # Add file handler to uvicorn loggers
        log_config["loggers"]["uvicorn"]["handlers"].append("file")
        log_config["loggers"]["uvicorn.access"]["handlers"].append("file_access")

    return log_config


async def create_datasets_collection(
    artifact_manager: ObjectProxy, logger: logging.Logger
) -> None:
    try:
        collection_alias = "bioengine-datasets"
        collection_id = f"public/{collection_alias}"

        logger.info(f"Creating collection '{collection_id}'")
        collection_manifest = {
            "name": "BioEngine Datasets",
            "description": "A collection of privacy-preserved datasets",
        }
        await artifact_manager.create(
            type="collection",
            alias=collection_alias,
            manifest=collection_manifest,
        )
    except Exception as e:
        logger.error(f"Failed to create applications collection: {e}")
        raise e

    return collection_id


async def mirror_dataset_to_artifact(
    artifact_s3_id: str,
    artifact_alias: str,
    dataset_dir: Path,
    minio_config: Dict[str, Union[str, int]],
    logger: logging.Logger,
) -> None:
    try:
        mc_exe = minio_config["mc_exe"]
        port = minio_config["minio_port"]
        root_user = minio_config["minio_user"]
        root_password = minio_config["minio_password"]

        artifact_id = f"public/{artifact_alias}"

        env = {
            **dict(**os.environ),  # inherit current env
            "MC_HOST_local": f"http://{root_user}:{root_password}@localhost:{port}",
        }

        # Filesystem path to all files except manifest.yaml (already handled separately)
        all_files = [
            f
            for f in await asyncio.to_thread(lambda: list(dataset_dir.iterdir()))
            if (f.is_file() and f.name != "manifest.yaml") or f.is_dir()
        ]
        for data_file in all_files:

            # Path to the artifact directory in MinIO
            s3_dest = f"local/hypha-workspaces/public/artifacts/{artifact_s3_id}/v0/{data_file.name}"

            logger.info(
                f"Copying {dataset_dir}/{data_file.name} to artifact '{artifact_id}'"
            )

            # Use 'cp' for files and 'mirror' for directories
            if data_file.is_file():
                args = [str(mc_exe), "cp", str(data_file), s3_dest]
            else:  # directory
                args = [str(mc_exe), "mirror", str(data_file), s3_dest]
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise subprocess.CalledProcessError(
                    proc.returncode, " ".join(args), stderr=error_msg
                )

    except Exception as e:
        logger.error(
            f"Error mirroring dataset '{dataset_dir.name}' to artifact '{artifact_id}': {e}"
        )
        raise e


async def create_dataset_artifact(
    artifact_manager: ObjectProxy,
    parent_id: str,
    dataset_dir: Path,
    minio_config: Dict[str, Union[str, int]],
    logger: logging.Logger,
) -> None:
    try:
        artifact = None
        logger.info(f"Creating dataset from folder '{dataset_dir}/'")

        dataset_manifest_content = (dataset_dir / "manifest.yaml").read_text()
        dataset_manifest = await asyncio.to_thread(
            yaml.safe_load, dataset_manifest_content
        )
        authorized_users = dataset_manifest.get("authorized_users")
        if not authorized_users or not any(len(user) > 0 for user in authorized_users):
            raise ValueError("Manifest does not have any authorized users specified.")

        artifact = await artifact_manager.create(
            type="dataset",
            parent_id=parent_id,
            alias=dataset_manifest["id"],
            manifest=dataset_manifest_content,
            stage=True,
        )

        await mirror_dataset_to_artifact(
            artifact_s3_id=artifact._id,
            artifact_alias=artifact.alias,
            dataset_dir=dataset_dir,
            minio_config=minio_config,
            logger=logger,
        )

        await artifact_manager.commit(artifact.id)
        logger.info(f"Created artifact with id '{artifact.id}'")

        return dataset_manifest["id"], authorized_users

    except Exception as e:
        # Delete the artifact if creation failed; don't raise exception to not affect other datasets
        logger.error(f"Failed to create dataset from folder '{dataset_dir}/': {e}")
        if artifact:
            await artifact_manager.delete(artifact.id)
        return None, None


async def parse_token(
    token: Union[str, None], cached_user_info: Dict[str, dict]
) -> Dict[str, str]:
    global AUTHENTICATION_SERVER_URL

    if token is not None:
        if (
            token in cached_user_info
            and cached_user_info[token]["expires_at"] > time.time()
        ):
            user_info = cached_user_info[token]
        else:
            async with connect_to_server(
                {"server_url": AUTHENTICATION_SERVER_URL, "token": token}
            ) as user_client:
                user_info = user_client.config.user

            cached_user_info[token] = user_info
            if len(cached_user_info) > 1000:
                # Limit cache size to 1000 entries
                cached_user_info.pop(next(iter(cached_user_info)))
    else:
        user_info = {"id": "anonymous-user", "email": "no-email"}

    return user_info


async def list_datasets(
    artifact_manager: ObjectProxy,
    collection_id: str,
) -> Dict[str, dict]:
    """List all datasets in the artifact manager."""
    # No checks for user authentication for listing datasets

    dataset_artifacts = await artifact_manager.list(parent_id=collection_id)
    return {
        artifact.alias: await asyncio.to_thread(yaml.safe_load, artifact.manifest)
        for artifact in dataset_artifacts
    }


async def list_files(
    dataset_id: str,
    cached_user_info: Dict[str, dict],
    authorized_users_collection: Dict[str, List[str]],
    artifact_manager: ObjectProxy,
    dir_path: Optional[str] = None,
    token: Optional[str] = None,
) -> List[str]:
    """List all files in a dataset."""
    if dataset_id not in authorized_users_collection:
        raise ValueError(f"Dataset '{dataset_id}' does not exist")

    user_info = await parse_token(
        token=token,
        cached_user_info=cached_user_info,
    )

    authorized_users = authorized_users_collection[dataset_id]
    check_permissions(
        context={"user": user_info},
        authorized_users=authorized_users,
        resource_name=f"list files in the dataset '{dataset_id}'",
    )

    files = await artifact_manager.list_files(
        artifact_id=f"public/{dataset_id}",
        dir_path=dir_path,
    )
    return [file.name for file in files]


async def get_presigned_url(
    dataset_id: str,
    file_path: str,
    cached_user_info: Dict[str, dict],
    authorized_users_collection: Dict[str, List[str]],
    artifact_manager: ObjectProxy,
    logger: logging.Logger,
    token: Optional[str] = None,
) -> Union[str, None]:
    """Get a pre-signed URL for a dataset artifact."""
    if dataset_id not in authorized_users_collection:
        raise ValueError(f"Dataset '{dataset_id}' does not exist")

    user_info = await parse_token(
        token=token,
        cached_user_info=cached_user_info,
    )
    user_id = user_info["id"]
    user_email = user_info["email"]

    authorized_users = authorized_users_collection[dataset_id]
    check_permissions(
        context={"user": user_info},
        authorized_users=authorized_users,
        resource_name=f"access '{file_path}' in the dataset '{dataset_id}'",
    )

    start_time = asyncio.get_event_loop().time()
    try:
        url = await artifact_manager.get_file(
            artifact_id=f"public/{dataset_id}",
            file_path=file_path,
        )
        time_taken = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"Generated pre-signed URL for user '{user_id}' ({user_email}) in {time_taken:.2f} seconds"
        )
        return url
    except Exception as e:
        time_taken = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"Failed to generate pre-signed URL for user '{user_id}' ({user_email}) after {time_taken:.2f} seconds"
        )
        raise e


async def create_bioengine_datasets(server: RemoteService):
    """Register BioEngine datasets service to Hypha.

    Args:
        server (RemoteService): The server instance to register the tools with.
    """
    global BIOENGINE_DATA_DIR
    global MINIO_CONFIG

    logger = logging.getLogger("ProxyServer")

    try:
        server_url = server.config.public_base_url
        logger.info(
            f"Creating BioEngine datasets artifacts at '{server_url}' in workspace 'public'"
        )

        artifact_manager = await server.get_service("public/artifact-manager")

        # Create bioengine-datasets collection
        collection_id = await create_datasets_collection(
            artifact_manager=artifact_manager,
            logger=logger,
        )

        # Create an artifact for each dataset in the data directory
        datasets = [
            d
            for d in BIOENGINE_DATA_DIR.iterdir()
            if d.is_dir() and (d / "manifest.yaml").exists()
        ]
        authorized_users_collection = {}
        for dataset_dir in datasets:
            dataset_id, authorized_users = await create_dataset_artifact(
                artifact_manager=artifact_manager,
                parent_id=collection_id,
                dataset_dir=dataset_dir,
                minio_config=MINIO_CONFIG,
                logger=logger,
            )
            if dataset_id is not None:
                authorized_users_collection[dataset_id] = authorized_users

        logger.info(f"Available datasets: {list(authorized_users_collection.keys())}")

        # Register service to hand out pre-signed urls
        cached_user_info = {}
        service_info = await server.register_service(
            {
                "id": "bioengine-datasets",
                "name": "BioEngine Datasets",
                "description": (
                    "Service for verifying users access and handing out "
                    "pre-signed URLs for BioEngine datasets"
                ),
                "type": "bioengine-datasets",
                # Authentication happens via user tokens at the central hypha server
                "config": {"visibility": "public", "require_context": False},
                "ping": lambda: "pong",
                "list_datasets": partial(
                    list_datasets,
                    artifact_manager=artifact_manager,
                    collection_id=collection_id,
                ),
                "list_files": partial(
                    list_files,
                    cached_user_info=cached_user_info,
                    authorized_users_collection=authorized_users_collection,
                    artifact_manager=artifact_manager,
                ),
                "get_presigned_url": partial(
                    get_presigned_url,
                    cached_user_info=cached_user_info,
                    authorized_users_collection=authorized_users_collection,
                    artifact_manager=artifact_manager,
                    logger=logger,
                ),
            }
        )
        logger.info(f"Registered BioEngine datasets service: {service_info['id']}")

    except Exception as e:
        logger.error(f"Failed to create BioEngine datasets: {e}")
        raise e


def start_proxy_server(
    data_dir: Union[str, Path],
    bioengine_workspace_dir: Union[str, Path] = f"{os.environ['HOME']}/.bioengine",
    server_ip: Optional[str] = None,
    server_port: int = 9527,
    minio_port: int = 10000,
    authentication_server_url: str = "https://hypha.aicell.io",
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    Start the BioEngine Datasets proxy server with comprehensive data management.

    Deploys a complete dataset management service including Hypha server for authentication,
    MinIO for secure object storage, and custom middleware for access control and data
    streaming. The service provides a secure API for accessing scientific datasets with
    privacy preservation and fine-grained access control.

    Service Components:
    - Hypha server with authentication and service registration
    - MinIO S3-compatible object storage for data files
    - Dataset discovery and manifest-based configuration
    - Access control with user-specific permissions
    - Presigned URL generation for secure data access
    - HTTP API for efficient data streaming and metadata access

    Deployment Process:
    1. Sets up cache directories and logging infrastructure
    2. Initializes MinIO server for secure object storage
    3. Registers datasets as artifacts in Hypha for discovery
    4. Starts Hypha server with dataset service and authentication
    5. Monitors service status and handles graceful shutdown

    Args:
        data_dir: Root directory containing dataset folders with manifest.yaml files
                Each dataset folder should have a manifest.yaml with metadata and
                access control configuration.
        bioengine_workspace_dir: Directory for cache files, logs, and temporary storage
                           Must be writable by the service process.
        server_ip: IP address for the service to listen on. If None, automatically
                 determined based on network configuration.
        server_port: Port for the Hypha server (default: 9527)
        minio_port: Port for the MinIO S3 service (default: 10000)
        log_file: Path to log file for service logging. If None, logs are created
                in the cache directory with timestamp-based filenames.
    """
    # Set paths
    global BIOENGINE_DATA_DIR
    global MINIO_CONFIG
    global AUTHENTICATION_SERVER_URL

    BIOENGINE_DATA_DIR = Path(data_dir).resolve()
    bioengine_workspace_dir = Path(bioengine_workspace_dir).resolve()
    datasets_dir = bioengine_workspace_dir / "datasets"
    current_server_file = datasets_dir / "bioengine_current_server"
    executable_path = Path(os.getenv("MINIO_EXECUTABLE_PATH") or datasets_dir / "bin")
    minio_workdir = datasets_dir / "s3"
    minio_config_dir = datasets_dir / "config"

    AUTHENTICATION_SERVER_URL = authentication_server_url

    # Initialize logging
    if log_file != "off":
        if log_file is None:
            # Create a timestamped log file in the workspace directory
            log_file = (
                bioengine_workspace_dir
                / "logs"
                / f"bioengine_datasets_{time.strftime('%Y%m%d_%H%M%S')}.log"
            )
        else:
            log_file = Path(log_file).resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = None

    logger = create_logger("ProxyServer", log_file=log_file)

    try:
        logger.info(f"Starting BioEngine Datasets proxy server v{__version__}")

        # Create the datasets workspace directory
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Set internal IP and ports
        server_ip = server_ip or get_internal_ip()
        free_server_port, server_s = acquire_free_port(
            port=server_port, step=1, ip=server_ip, keep_open=True
        )
        if free_server_port != server_port:
            logger.warning(
                f"Hypha Server port {server_port} is not available. Using {free_server_port} instead."
            )

        free_minio_port = copy(minio_port)
        while True:
            free_minio_port, minio_s = acquire_free_port(
                port=free_minio_port, step=1, ip=server_ip, keep_open=True
            )
            free_console_port, console_s = acquire_free_port(
                port=free_minio_port + 1, step=1, ip=server_ip, keep_open=True
            )
            if free_console_port == free_minio_port + 1:
                break

        if free_minio_port != minio_port:
            logger.warning(
                f"MinIO server port {minio_port} and console port {minio_port+1} are not available. "
                f"Using ports {free_minio_port} and {free_console_port} instead."
            )

        # Close the sockets after acquiring ports
        for s in (server_s, minio_s, console_s):
            s.close()

        # Save the server URL to the bioengine cache
        server_url = f"http://{server_ip}:{free_server_port}"
        current_server_file.write_text(server_url)

        # Set Hypha server configuration
        hypha_parser = get_argparser()
        hypha_args = hypha_parser.parse_args([])  # use default values
        hypha_args.public_base_url = server_url

        # Minio configuration
        hypha_args.start_minio_server = True
        hypha_args.enable_s3_proxy = True
        hypha_args.executable_path = executable_path
        hypha_args.minio_workdir = minio_workdir
        hypha_args.minio_port = free_minio_port
        hypha_args.minio_root_user = "bioengine_admin"
        hypha_args.minio_root_password = str(os.urandom(16).hex())
        hypha_args.minio_version = "RELEASE.2024-07-16T23-46-41Z"
        hypha_args.mc_version = "RELEASE.2025-04-08T15-39-49Z"
        hypha_args.minio_file_system_mode = False

        # Set global minio config
        mc_version_short = hypha_args.mc_version.replace("RELEASE", "").replace("-", "")
        mc_exe_path = executable_path / ("mc" + mc_version_short)
        MINIO_CONFIG = {
            "mc_exe": mc_exe_path,
            "minio_port": hypha_args.minio_port,
            "minio_user": hypha_args.minio_root_user,
            "minio_password": hypha_args.minio_root_password,
        }

        os.environ["MINIO_CONFIG_DIR"] = str(minio_config_dir / "minio")
        os.environ["MC_CONFIG_DIR"] = str(minio_config_dir / "mc")

        # Pass startup function (available at http://<ip>:<port>/public/services/bioengine-datasets)
        hypha_args.startup_functions = [
            "bioengine.datasets.proxy_server:create_bioengine_datasets"
        ]

        # Create the Hypha application
        hypha_app = create_application(hypha_args)

        # Create a log config for uvicorn
        log_config = get_log_config(log_file=log_file)

        # Start the server
        uvicorn.run(
            hypha_app,
            host="0.0.0.0",
            port=free_server_port,
            log_config=log_config,
            log_level="info",
        )
    except Exception as e:
        logger.error(f"Failed to start BioEngine Datasets proxy server: {e}")
        raise

    finally:
        if minio_workdir.exists():
            try:
                shutil.rmtree(minio_workdir)
            except Exception as e:
                logger.warning(f"Failed to clean up MinIO workdir: {e}")

        if current_server_file.exists():
            try:
                current_server_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up current server file: {e}")

        if minio_config_dir.exists():
            try:
                shutil.rmtree(minio_config_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up MinIO config dir: {e}")

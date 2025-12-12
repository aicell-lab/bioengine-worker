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
from hypha.server import create_application, get_argparser
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import ObjectProxy, RemoteService

from bioengine import __version__
from bioengine.utils import (
    acquire_free_port,
    create_logger,
    date_format,
    get_internal_ip,
)
from bioengine.utils.permissions import check_permissions

DATA_IMPORT_DIR: Path
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


async def sync_dataset_to_artifact(
    artifact_s3_id: str,
    artifact_alias: str,
    dataset_dir: Path,
    minio_config: Dict[str, Union[str, int]],
    logger: logging.Logger,
    artifact_manager: ObjectProxy,
) -> None:
    """Sync dataset to artifact by adding new files and removing deleted files."""
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

        # Get current files in the dataset directory (except manifest.yaml)
        current_files = set()
        all_files = []
        for f in await asyncio.to_thread(lambda: list(dataset_dir.iterdir())):
            if f.name != "manifest.yaml":
                current_files.add(f.name)
                if f.is_file() or f.is_dir():
                    all_files.append(f)

        # Get existing files in the artifact
        try:
            existing_files_objs = await artifact_manager.list_files(
                artifact_id=artifact_id,
                dir_path="",
            )
            existing_files = {f.name for f in existing_files_objs}
        except Exception:
            # If artifact doesn't exist or is empty, existing_files is empty
            existing_files = set()

        # Files to remove (exist in artifact but not in dataset directory)
        files_to_remove = existing_files - current_files

        # Remove deleted files from artifact
        for file_name in files_to_remove:
            s3_path = f"local/hypha-workspaces/public/artifacts/{artifact_s3_id}/v0/{file_name}"
            logger.info(f"Removing {file_name} from artifact '{artifact_id}'")
            args = [str(mc_exe), "rm", "-r", "--force", s3_path]
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            await proc.communicate()
            # Don't fail if removal fails - file might not exist

        # Sync all current files (mirror handles updates automatically)
        for data_file in all_files:
            s3_dest = f"local/hypha-workspaces/public/artifacts/{artifact_s3_id}/v0/{data_file.name}"

            logger.info(
                f"Syncing {dataset_dir}/{data_file.name} to artifact '{artifact_id}'"
            )

            # Use 'cp' for files (overwrites by default) and 'mirror' for directories
            if data_file.is_file():
                # For files, remove first then copy to ensure clean overwrite
                rm_args = [str(mc_exe), "rm", "--force", s3_dest]
                proc = await asyncio.create_subprocess_exec(
                    *rm_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                await proc.communicate()
                # Ignore errors from rm - file might not exist

                # Now copy the file
                args = [str(mc_exe), "cp", str(data_file), s3_dest]
            else:  # directory
                args = [
                    str(mc_exe),
                    "mirror",
                    "--overwrite",
                    "--remove",
                    str(data_file),
                    s3_dest,
                ]

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
            f"Error syncing dataset '{dataset_dir.name}' to artifact '{artifact_id}': {e}"
        )
        raise e


async def create_or_update_dataset_artifact(
    artifact_manager: ObjectProxy,
    parent_id: str,
    dataset_dir: Path,
    minio_config: Dict[str, Union[str, int]],
    logger: logging.Logger,
) -> bool:
    """Create a new dataset artifact or update an existing one. Returns True if successful."""
    try:
        dataset_manifest_content = (dataset_dir / "manifest.yaml").read_text()
        dataset_manifest = await asyncio.to_thread(
            yaml.safe_load, dataset_manifest_content
        )
        authorized_users = dataset_manifest.get("authorized_users")
        if not authorized_users or not any(len(user) > 0 for user in authorized_users):
            raise ValueError("Manifest does not have any authorized users specified.")

        dataset_id = dataset_manifest["id"]
        artifact_id = f"public/{dataset_id}"

        # Check if artifact already exists
        try:
            existing_artifact = await artifact_manager.read(artifact_id)
            logger.info(
                f"Artifact '{artifact_id}' exists. Updating dataset from folder '{dataset_dir}/'"
            )

            # Update manifest if changed
            await artifact_manager.edit(
                artifact_id=artifact_id,
                manifest=dataset_manifest_content,
                stage=True,
            )

            # Sync data files (add new, update existing, remove deleted)
            await sync_dataset_to_artifact(
                artifact_s3_id=existing_artifact._id,
                artifact_alias=dataset_id,
                dataset_dir=dataset_dir,
                minio_config=minio_config,
                logger=logger,
                artifact_manager=artifact_manager,
            )

            await artifact_manager.commit(artifact_id)
            logger.info(f"Updated artifact '{artifact_id}'")

            return True

        except Exception:
            # Artifact doesn't exist, create new one
            logger.info(f"Creating new dataset artifact from folder '{dataset_dir}/'")
            artifact = None

            try:
                artifact = await artifact_manager.create(
                    type="dataset",
                    parent_id=parent_id,
                    alias=dataset_id,
                    manifest=dataset_manifest_content,
                    stage=True,
                )

                # Initial sync of all files
                await sync_dataset_to_artifact(
                    artifact_s3_id=artifact._id,
                    artifact_alias=artifact.alias,
                    dataset_dir=dataset_dir,
                    minio_config=minio_config,
                    logger=logger,
                    artifact_manager=artifact_manager,
                )

                await artifact_manager.commit(artifact.id)
                logger.info(f"Created artifact with id '{artifact.id}'")

                return True

            except Exception as create_error:
                # Delete the artifact if creation failed
                logger.error(
                    f"Failed to create dataset from folder '{dataset_dir}/': {create_error}"
                )
                if artifact:
                    try:
                        await artifact_manager.delete(artifact.id)
                    except Exception:
                        pass
                return False

    except Exception as e:
        # Don't raise exception to not affect other datasets
        logger.error(f"Failed to process dataset from folder '{dataset_dir}/': {e}")
        return False


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
    global DATA_IMPORT_DIR
    global MINIO_CONFIG

    logger = logging.getLogger("ProxyServer")

    try:
        server_url = server["config"]["public_base_url"]
        workspace = server["config"]["workspace"]
        if workspace != "public":
            raise ValueError(
                f"Expected workspace to be 'public', but got '{workspace}'"
            )

        logger.info(
            f"Creating BioEngine datasets artifacts at '{server_url}' in workspace 'public'"
        )

        artifact_manager = await server.get_service("public/artifact-manager")

        # Create bioengine-datasets collection
        collection_id = await create_datasets_collection(
            artifact_manager=artifact_manager,
            logger=logger,
        )

        # Create or update artifacts for each dataset in the data import directory
        if DATA_IMPORT_DIR is not None:
            logger.info("Processing datasets from data import directory.")
            datasets = [
                d
                for d in DATA_IMPORT_DIR.iterdir()
                if d.is_dir() and (d / "manifest.yaml").exists()
            ]
            for dataset_dir in datasets:
                await create_or_update_dataset_artifact(
                    artifact_manager=artifact_manager,
                    parent_id=collection_id,
                    dataset_dir=dataset_dir,
                    minio_config=MINIO_CONFIG,
                    logger=logger,
                )
        else:
            logger.info(
                "No data import directory provided. Using existing datasets from S3 storage."
            )

        # Build authorized_users_collection from all artifacts in the collection
        authorized_users_collection = {}
        try:
            all_artifacts = await artifact_manager.list(parent_id=collection_id)
            for artifact in all_artifacts:
                try:
                    manifest = await asyncio.to_thread(
                        yaml.safe_load, artifact.manifest
                    )
                    authorized_users = manifest.get("authorized_users", [])
                    if authorized_users and any(
                        len(user) > 0 for user in authorized_users
                    ):
                        authorized_users_collection[artifact.alias] = authorized_users
                except Exception as e:
                    logger.warning(
                        f"Failed to load manifest for artifact '{artifact.alias}': {e}"
                    )
        except Exception as e:
            logger.warning(f"Failed to load datasets from artifact manager: {e}")

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
    data_import_dir: Optional[Union[str, Path]] = None,
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
    - MinIO S3-compatible object storage for data files (persisted across restarts)
    - Dataset discovery and manifest-based configuration
    - Access control with user-specific permissions
    - Presigned URL generation for secure data access
    - HTTP API for efficient data streaming and metadata access
    - Incremental sync for updating existing datasets

    Deployment Process:
    1. Sets up cache directories and logging infrastructure
    2. Initializes MinIO server (reuses existing S3 data if available)
    3. If data_import_dir is provided:
       - Creates new dataset artifacts or updates existing ones
       - Syncs files: adds new files, updates modified files, removes deleted files
    4. If data_import_dir is None:
       - Loads existing datasets from S3 storage
    5. Starts Hypha server with dataset service and authentication

    Data Persistence:
    - S3 data (minio_workdir) is preserved across server restarts
    - MinIO configuration is preserved to maintain data consistency
    - Only current_server_file is cleaned up on shutdown

    Args:
        data_import_dir: Optional directory containing dataset folders to sync with S3 storage.
                       Each dataset folder should have a manifest.yaml with metadata and
                       access control configuration. When provided:
                       - New datasets are created as artifacts
                       - Existing datasets are updated (files added/updated/removed)
                       If None, the proxy server loads existing datasets from S3 storage.
        bioengine_workspace_dir: Directory for cache files, logs, and persistent S3 storage.
                           Must be writable by the service process. S3 data persists across
                           restarts in <workspace_dir>/datasets/s3/.
        server_ip: IP address for the service to listen on. If None, automatically
                 determined based on network configuration.
        server_port: Port for the Hypha server (default: 9527)
        minio_port: Port for the MinIO S3 service (default: 10000)
        log_file: Path to log file for service logging. If None, logs are created
                in the cache directory with timestamp-based filenames.
    """
    # Set paths
    global DATA_IMPORT_DIR
    global MINIO_CONFIG
    global AUTHENTICATION_SERVER_URL

    DATA_IMPORT_DIR = Path(data_import_dir).resolve() if data_import_dir else None
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
        bioengine_datasets_dir.mkdir(parents=True, exist_ok=True)

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
        # Keep minio_workdir and minio_config_dir to persist S3 data across restarts
        if current_server_file.exists():
            try:
                current_server_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up current server file: {e}")

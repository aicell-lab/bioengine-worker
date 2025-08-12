import asyncio
import logging
import os
import shutil
import subprocess
import time
from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import uvicorn
import yaml
from hypha.server import create_application, get_argparser
from hypha_rpc.rpc import ObjectProxy, RemoteService

from bioengine import __version__
from bioengine.utils import (
    acquire_free_port,
    create_logger,
    date_format,
    get_internal_ip,
)

BIOENGINE_DATA_DIR: Path
ACCESS_TOKEN_FILE: Path
MINIO_CONFIG = {
    "mc_exe": str,
    "minio_port": int,
    "minio_user": str,
    "minio_password": str,
}


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
            "description": f"A collection of Zarr-file datasets for workspace 'public'",
        }
        # Allow all users access to the collection
        collection_config = {"permissions": {"*": "*"}}

        collection = await artifact_manager.create(
            type="collection",
            alias=collection_alias,
            manifest=collection_manifest,
            config=collection_config,
        )
        logger.info(f"Applications collection created with ID: {collection.id}")
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

        # Filesystem path to Zarr files
        zarr_files = await asyncio.to_thread(dataset_dir.glob, "*.zarr")
        for zarr_file in zarr_files:
            # Path to the artifact directory in MinIO
            s3_dest = f"local/hypha-workspaces/public/artifacts/{artifact_s3_id}/v0/{zarr_file.name}"

            args = [str(mc_exe), "mirror", str(zarr_file), s3_dest]
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

            logger.info(
                f"Mirrored {dataset_dir.name}/{zarr_file.name} to artifact '{artifact_id}'"
            )

    except Exception as e:
        logger.info(
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
        logger.info(f"Creating artifact for dataset '{dataset_dir.name}'")

        dataset_manifest_content = (dataset_dir / "manifest.yml").read_text()
        dataset_manifest = await asyncio.to_thread(
            yaml.safe_load, dataset_manifest_content
        )
        authorized_users = dataset_manifest["authorized_users"]
        logger.info(
            f"Setting permissions for authorized users: {', '.join(authorized_users)}"
        )
        dataset_config = {
            "permissions": {
                user_id: "r" for user_id in authorized_users
            },  # Only allow authorized users to access the dataset
        }

        artifact = await artifact_manager.create(
            type="dataset",
            parent_id=parent_id,
            alias=dataset_manifest["id"],
            manifest=dataset_manifest_content,
            config=dataset_config,
            stage=True,
        )
        logger.info(f"Creating new artifact with id '{artifact.id}'")

        await mirror_dataset_to_artifact(
            artifact_s3_id=artifact._id,
            artifact_alias=artifact.alias,
            dataset_dir=dataset_dir,
            minio_config=minio_config,
            logger=logger,
        )

        await artifact_manager.commit(artifact.id)
        logger.info(f"Committed artifact with id '{artifact.id}'")

    except Exception as e:
        # Delete the artifact if creation failed; don't raise exception to not affect other datasets
        logger.error(f"Failed to create artifact for dataset {dataset_dir.name}: {e}")
        if artifact:
            await artifact_manager.delete(artifact.id)


async def create_bioengine_datasets(
    server: RemoteService,
    service_id: str = "bioengine-datasets",
):
    """Register BioEngine datasets service to Hypha.

    Args:
        server (RemoteService): The server instance to register the tools with.
    """
    global BIOENGINE_DATA_DIR
    global ACCESS_TOKEN_FILE
    global MINIO_CONFIG

    logger = logging.getLogger("BioEngineDatasets")

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

        # Generate access token
        access_token = await server.generate_token()
        ACCESS_TOKEN_FILE.write_text(access_token)

        # Restrict access to the token file to only the user
        ACCESS_TOKEN_FILE.chmod(0o600)
        # TODO: This is not enough if the Ray cluster in BioEngine worker runs as the same user

        # Create bioengine-datasets collection
        collection_id = await create_datasets_collection(
            artifact_manager=artifact_manager,
            logger=logger,
        )

        # Create an artifact for each dataset in the data directory
        datasets = [
            d
            for d in BIOENGINE_DATA_DIR.iterdir()
            if d.is_dir() and (d / "manifest.yml").exists()
        ]
        for dataset_dir in datasets:
            await create_dataset_artifact(
                artifact_manager=artifact_manager,
                parent_id=collection_id,
                dataset_dir=dataset_dir,
                minio_config=MINIO_CONFIG,
                logger=logger,
            )

    except Exception as e:
        logger.error(f"Failed to create BioEngine datasets: {e}")
        raise e


def start_proxy_server(
    data_dir: Union[str, Path],
    bioengine_cache_dir: Union[str, Path] = f"{os.environ['HOME']}/.bioengine",
    server_ip: Optional[str] = None,
    server_port: int = 9527,
    minio_port: int = 10000,
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    # Set paths
    global BIOENGINE_DATA_DIR
    global ACCESS_TOKEN_FILE
    global MINIO_CONFIG

    BIOENGINE_DATA_DIR = Path(data_dir).resolve()
    bioengine_cache_dir = Path(bioengine_cache_dir).resolve()
    datasets_cache_dir = bioengine_cache_dir / "datasets"
    current_server_file = datasets_cache_dir / "bioengine_current_server"
    ACCESS_TOKEN_FILE = datasets_cache_dir / ".access_token"
    executable_path = Path(
        os.getenv("MINIO_EXECUTABLE_PATH") or datasets_cache_dir / "bin"
    )
    minio_workdir = datasets_cache_dir / "s3"
    minio_config_dir = datasets_cache_dir / "config" / "minio"
    mc_config_dir = datasets_cache_dir / "config" / "mc"

    # Initialize logging
    if log_file != "off":
        if log_file is None:
            # Create a timestamped log file in the cache directory
            log_file = (
                bioengine_cache_dir
                / "logs"
                / f"bioengine_datasets_{time.strftime('%Y%m%d_%H%M%S')}.log"
            )
        else:
            log_file = Path(log_file).resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = None

    logger = create_logger("BioEngineDatasets", log_file=log_file)

    try:
        logger.info(f"Starting BioEngineDatasets v{__version__}")

        # Create the datasets cache directory
        datasets_cache_dir.mkdir(parents=True, exist_ok=True)

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

        hypha_args.start_minio_server = True
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

        minio_config_dir.mkdir(parents=True, exist_ok=True)
        mc_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MINIO_CONFIG_DIR"] = str(minio_config_dir)
        os.environ["MC_CONFIG_DIR"] = str(mc_config_dir)

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
            host=server_ip,
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

        if ACCESS_TOKEN_FILE.exists():
            try:
                ACCESS_TOKEN_FILE.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up access token file: {e}")


if __name__ == "__main__":
    start_proxy_server(
        data_dir="/data/nmechtel/bioengine-worker/data",
        bioengine_cache_dir="/data/nmechtel/bioengine-worker/.bioengine",
        # log_file="off",
    )

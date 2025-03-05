"""
Register HPC worker to Chiron platform

This script registers a HPC worker service with the Hypha server, enabling:
- Remote monitoring of worker status
- Ray cluster management (start/stop)
- Submitting worker jobs to the HPC system
- Managing trusted models and datasets

Usage:
    python -m hpc_worker [options]

Options:
    --config_file       Path to configuration file
    --server_url        Hypha server URL (default: https://hypha.aicell.io)
    --num_gpu           Number of available GPUs (default: 3)
    --dataset_paths     Space-separated list of dataset directories 
    --trusted_models    Space-separated list of trusted docker images

Container execution:
    Run in image chiron_worker_0.1.0.sif

    Pull the image with:
    `apptainer pull chiron_worker_0.1.0.sif docker://ghcr.io/aicell-lab/chiron-worker:0.1.0`

    Run with:
    `apptainer run --contain --nv chiron_worker_0.1.0.sif python -m hpc_worker [options]`
"""

import logging
import os
import time
from functools import partial

from dotenv import find_dotenv, load_dotenv
from hypha_rpc import connect_to_server, login

from hpc_worker.config_manager import create_worker_config
from hpc_worker.worker_status import worker_status
from hpc_worker.ray_cluster import (
    start_ray_cluster,
    shutdown_ray_cluster,
    submit_ray_worker_job,
    get_ray_worker_jobs,
    cancel_ray_worker_jobs,
)


# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Configure logger
logger = logging.getLogger("hpc_worker")
logger.setLevel(logging.INFO)
# Disable propagation to avoid duplication of logs
logger.propagate = False
# Create a new console handler
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Set the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def register_hpc_worker(args):
    config_path = args.config_file or os.path.join(
        os.path.dirname(__file__), "worker_config.yaml"
    )

    if not args.config_file:
        create_worker_config(
            dataset_paths=args.dataset_paths,
            max_gpus=args.num_gpu,
            trusted_models=args.trusted_models,
            config_path=config_path,
        )

    registered_at = int(time.time())

    # Create bound functions with the configured logger
    status_func = partial(
        worker_status, config_path=config_path, registered_at=registered_at
    )
    start_ray_func = partial(start_ray_cluster, logger=logger)
    shutdown_ray_func = partial(shutdown_ray_cluster, logger=logger)
    submit_job_func = partial(submit_ray_worker_job, logger=logger)
    get_jobs_func = partial(get_ray_worker_jobs, logger=logger)
    cancel_jobs_func = partial(cancel_ray_worker_jobs, logger=logger)

    logger.info("Connecting to Hypha server...")
    hypha_token = os.environ.get("HYPHA_TOKEN") or await login(
        {"server_url": args.server_url}
    )
    server = await connect_to_server(
        {"server_url": args.server_url, "token": hypha_token}
    )

    # Register service with both status and ray cluster control
    service_info = await server.register_service(
        {
            "name": "HPC Worker",
            "id": "hpc-worker",
            "config": {
                "visibility": "public",
                "require_context": True,
                "run_in_executor": False,
            },
            "ping": lambda context: "pong",
            "get_worker_status": status_func,
            "start_ray_cluster": start_ray_func,
            "shutdown_ray_cluster": shutdown_ray_func,
            "submit_ray_worker_job": submit_job_func,
            "get_ray_worker_jobs": get_jobs_func,
            "cancel_ray_worker_jobs": cancel_jobs_func,
        }
    )

    logger.info(f"Service registered with ID: {service_info.id}")
    sid = service_info.id.split("/")[1]
    service_url = f"{args.server_url}/{server.config.workspace}/services/{sid}"
    logger.info(f"Test the HPC worker service here: {service_url}/get_worker_status")
    logger.info(f"Start Ray cluster with: {service_url}/start_ray_cluster")
    logger.info(f"Shutdown Ray cluster with: {service_url}/shutdown_ray_cluster")
    logger.info(f"Submit a Ray worker job with: {service_url}/submit_ray_worker_job")
    logger.info(f"Get Ray worker jobs status with: {service_url}/get_ray_worker_jobs")
    logger.info(f"Cancel Ray worker jobs with: {service_url}/cancel_ray_worker_jobs")


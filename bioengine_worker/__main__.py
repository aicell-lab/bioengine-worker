import argparse
import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Literal

from bioengine_worker import __version__
from bioengine_worker.utils import create_logger
from bioengine_worker.worker import BioEngineWorker


async def main(group_configs):
    """Main function to initialize and register BioEngine worker"""

    # Set up logging
    log_dir = Path(group_configs["options"]["cache_dir"]) / "logs"
    log_file = log_dir / f"bioengine_worker_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger = create_logger("__main__", log_file=log_file)

    # Pass log file to group configs
    group_configs["options"]["log_file"] = log_file

    # Setup signal-aware shutdown
    is_shutting_down = False
    bioengine_worker = None

    def _handle_shutdown_signal(sig_name: Literal["SIGINT", "SIGTERM"]):
        """
        Handle shutdown signals for graceful termination.

        Args:
            sig_name: The name of the signal received, either "SIGINT" for user interrupt (Ctrl+C) or "SIGTERM" for termination request.
        """
        nonlocal is_shutting_down, bioengine_worker
        if is_shutting_down and sig_name == "SIGINT":
            logger.info("Received second SIGINT, stopping immediately...")
            sys.exit(1)

        logger.info(
            f"Received {sig_name}, starting graceful shutdown. Press Ctrl+C again to force exit."
        )
        is_shutting_down = True

        if bioengine_worker:
            # TODO: when running in Apptainer, the container’s overlay filesystem is torn down before the graceful shutdown completes -> results in OSError [Errno 107] because executables like Ray or scancel are not found
            admin_context = {
                "user": {
                    "id": bioengine_worker.admin_users[0],
                    "email": bioengine_worker.admin_users[1],
                }
            }
            asyncio.create_task(
                bioengine_worker.cleanup(context=admin_context),
                name="BioEngineWorker.cleanup",
            )

    # Register signal handlers
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _handle_shutdown_signal, "SIGINT")
    loop.add_signal_handler(signal.SIGTERM, _handle_shutdown_signal, "SIGTERM")

    try:
        # Get Bioengine worker configuration
        worker_config = group_configs["options"]

        # Get Hypha configuration
        hypha_config = group_configs["Hypha Options"]

        # Get Ray Cluster Manager configuration
        ray_cluster_config = group_configs["Ray Cluster Manager Options"]

        # Get SLURM Job configuration
        slurm_job_config = group_configs["SLURM Job Options"]

        # Get Ray Autoscaler configuration
        ray_autoscaling_config = group_configs["Ray Autoscaler Options"]

        # Create BioEngine worker instance
        bioengine_worker = BioEngineWorker(
            **worker_config,
            **hypha_config,
            ray_cluster_config={
                **ray_cluster_config,
                **slurm_job_config,
                **ray_autoscaling_config,
            },
        )

        # Initialize worker
        await bioengine_worker.start()

        # Wait until shutdown is triggered
        await bioengine_worker.serve()

    except Exception as e:
        logger.error(f"Exception in main: {str(e)}")
        raise


def create_parser():
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        description="BioEngine Worker Registration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        default="slurm",
        type=str,
        choices=["slurm", "single-machine", "connect"],
        help="Mode of operation: 'slurm' for managing a Ray cluster with SLURM jobs, 'single-machine' for local Ray cluster, 'connect' for connecting to an existing Ray cluster.",
    )
    # TODO: use --head_node_address and --client_server_port to connect to existing Ray cluster
    parser.add_argument(
        "--admin_users",
        type=str,
        nargs="+",
        help="List of admin users for BioEngine apps and datasets. If not set, defaults to the logged-in user.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/bioengine",
        help="BioEngine cache directory. This should be a mounted directory if running in container.",
    )
    parser.add_argument(
        "--data_dir",
        default="/data",
        type=str,
        help="Data directory served by the dataset manager. This should be a mounted directory if running in container.",
    )
    parser.add_argument(
        "--startup_deployments",
        type=str,
        nargs="+",
        help="List of artifact IDs to deploy on worker startup",
    )

    # Hypha related options
    hypha_group = parser.add_argument_group("Hypha Options")
    hypha_group.add_argument(
        "--server_url",
        default="https://hypha.aicell.io",
        type=str,
        help="URL of the Hypha server",
    )
    hypha_group.add_argument(
        "--workspace",
        type=str,
        help="Hypha workspace to connect to. If not set, the workspace associated with the token will be used.",
    )
    hypha_group.add_argument(
        "--token",
        type=str,
        help="Authentication token for Hypha server. If not set, the environment variable 'HYPHA_TOKEN' will be used, otherwise the user will be prompted to log in.",
    )
    hypha_group.add_argument(
        "--client_id",
        type=str,
        help="Client ID for the worker. If not set, a client ID will be generated automatically.",
    )

    # Ray Cluster Configuration parameters
    ray_cluster_group = parser.add_argument_group("Ray Cluster Manager Options")
    ray_cluster_group.add_argument(
        "--head_node_address",
        type=str,
        help="Address of head node. If not set, the first system IP will be used.",
    )
    ray_cluster_group.add_argument(
        "--head_node_port",
        type=int,
        default=6379,
        help="Port for Ray head node and GCS server",
    )
    ray_cluster_group.add_argument(
        "--node_manager_port",
        type=int,
        default=6700,
        help="Port for Ray node manager services",
    )
    ray_cluster_group.add_argument(
        "--object_manager_port",
        type=int,
        default=6701,
        help="Port for object manager service",
    )
    ray_cluster_group.add_argument(
        "--redis_shard_port",
        type=int,
        default=6702,
        help="Port for Redis sharding",
    )
    ray_cluster_group.add_argument(
        "--serve_port",
        type=int,
        default=8100,
        help="Port for Ray Serve",
    )
    ray_cluster_group.add_argument(
        "--dashboard_port",
        type=int,
        default=8269,
        help="Port for Ray dashboard",
    )
    ray_cluster_group.add_argument(
        "--client_server_port",
        type=int,
        default=10001,
        help="Port for Ray client server",
    )
    ray_cluster_group.add_argument(
        "--redis_password",
        type=str,
        help="Redis password for Ray cluster. If not set, a random password will be generated.",
    )
    ray_cluster_group.add_argument(
        "--head_num_cpus",
        type=int,
        default=0,
        help="Number of CPUs for head node if starting locally",
    )
    ray_cluster_group.add_argument(
        "--head_num_gpus",
        type=int,
        default=0,
        help="Number of GPUs for head node if starting locally",
    )
    ray_cluster_group.add_argument(
        "--runtime_env_pip_cache_size_gb",
        type=int,
        default=30,
        help="Size of the pip cache in GB for Ray runtime environment",
    )
    ray_cluster_group.add_argument(
        "--connection_address",
        type=str,
        default="auto",
        help="Address of existing Ray cluster to connect to (format: 'auto' for auto-discovery, 'ip:port' for specific address).",
    )
    ray_cluster_group.add_argument(
        "--skip_cleanup",
        action="store_true",
        default=False,
        help="Skip cleanup of previous Ray cluster",
    )
    ray_cluster_group.add_argument(
        "--status_interval_seconds",
        default=10,
        type=int,
        help="Interval in seconds to check the status of the Ray cluster",
    )
    ray_cluster_group.add_argument(
        "--max_status_history_length",
        default=100,
        type=int,
        help="Maximum length of the status history for the Ray cluster",
    )

    # SLURM Job Configuration parameters
    slurm_job_group = parser.add_argument_group("SLURM Job Options")
    slurm_job_group.add_argument(
        "--image",
        default=f"ghcr.io/aicell-lab/bioengine-worker:{__version__}",
        type=str,
        help="Worker image for SLURM job",
    )
    slurm_job_group.add_argument(
        "--worker_cache_dir",
        type=str,
        help="Cache directory mounted to the container when starting a worker. Required in SLURM mode.",
    )
    slurm_job_group.add_argument(
        "--worker_data_dir",
        type=str,
        help="Data directory mounted to the container when starting a worker. Required in SLURM mode.",
    )
    slurm_job_group.add_argument(
        "--default_num_gpus",
        default=1,
        type=int,
        help="Default number of GPUs per worker",
    )
    slurm_job_group.add_argument(
        "--default_num_cpus",
        default=8,
        type=int,
        help="Default number of CPUs per worker",
    )
    slurm_job_group.add_argument(
        "--default_mem_per_cpu",
        default=16,
        type=int,
        help="Default memory per CPU in GB",
    )
    slurm_job_group.add_argument(
        "--default_time_limit",
        default="4:00:00",
        type=str,
        help="Default time limit for workers",
    )
    slurm_job_group.add_argument(
        "--further_slurm_args",
        type=str,
        nargs="+",
        help="Additional arguments for SLURM job script",
    )

    # Autoscaling configuration parameters
    ray_autoscaling_group = parser.add_argument_group("Ray Autoscaler Options")
    ray_autoscaling_group.add_argument(
        "--min_workers",
        default=0,
        type=int,
        help="Minimum number of worker nodes",
    )
    ray_autoscaling_group.add_argument(
        "--max_workers",
        default=4,
        type=int,
        help="Maximum number of worker nodes",
    )
    ray_autoscaling_group.add_argument(
        "--check_interval_seconds",
        default=60,
        type=int,
        help="Interval in seconds to check scale up/down",
    )
    ray_autoscaling_group.add_argument(
        "--scale_down_threshold_seconds",
        default=300,
        type=int,
        help="Time threshold before scaling down idle nodes",
    )
    ray_autoscaling_group.add_argument(
        "--scale_up_cooldown_seconds",
        default=180,
        type=int,
        help="Cooldown period before scaling up",
    )
    ray_autoscaling_group.add_argument(
        "--scale_down_cooldown_seconds",
        default=60,
        type=int,
        help="Cooldown period before scaling down",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Set logger to debug level",
    )

    return parser


def get_args_by_group(parser):
    args = parser.parse_args()
    args_dict = vars(args)

    group_configs = {}
    for group in parser._action_groups:
        group_keys = [a.dest for a in group._group_actions]
        group_configs[group.title] = {
            k: args_dict[k] for k in group_keys if k in args_dict
        }

    return group_configs


if __name__ == "__main__":
    description = "Register BioEngine worker to Hypha server"

    parser = create_parser()
    group_configs = get_args_by_group(parser)
    asyncio.run(main(group_configs))

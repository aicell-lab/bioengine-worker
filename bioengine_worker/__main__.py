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
            asyncio.create_task(bioengine_worker.cleanup())

    # Register signal handlers
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _handle_shutdown_signal, "SIGINT")
    loop.add_signal_handler(signal.SIGTERM, _handle_shutdown_signal, "SIGTERM")

    try:
        # Get Hypha configuration
        hypha_config = group_configs["Hypha Options"]

        # Get Dataset Manager configuration
        dataset_config = group_configs["Dataset Manager Options"]
        dataset_config["service_id"] = dataset_config.pop("dataset_service_id")

        # Get Ray Cluster Manager configuration
        ray_cluster_config = group_configs["Ray Cluster Manager Options"]
        clean_up_previous_cluster = not ray_cluster_config.pop("skip_cleanup")

        # Get Ray Autoscaler configuration
        ray_autoscaler_config = group_configs["Ray Autoscaler Options"]

        # Get Ray Deployment Manager configuration
        ray_deployment_config = group_configs["Ray Deployment Manager Options"]
        ray_deployment_config["service_id"] = ray_deployment_config.pop(
            "deployment_service_id"
        )

        # Get Ray Connection options
        ray_connection_kwargs = group_configs["Ray Connection Options"]
        ray_connection_kwargs = {
            k.replace("ray_", ""): v
            for k, v in ray_connection_kwargs.items()
            if v is not None
        }

        # Create BioEngine worker instance
        bioengine_worker = BioEngineWorker(
            workspace=hypha_config["workspace"],
            server_url=hypha_config["server_url"],
            token=hypha_config["token"],
            service_id=hypha_config["worker_service_id"],
            client_id=hypha_config["client_id"],
            mode=group_configs["options"]["mode"],
            dataset_config=dataset_config,
            ray_cluster_config=ray_cluster_config,
            clean_up_previous_cluster=clean_up_previous_cluster,
            ray_autoscaler_config=ray_autoscaler_config,
            ray_deployment_config=ray_deployment_config,
            ray_connection_kwargs=ray_connection_kwargs,
            cache_dir=group_configs["options"]["cache_dir"],
            log_file=group_configs["options"]["log_file"],
            _debug=group_configs["options"]["debug"],
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

    parser.add_argument(
        "--log_dir",
        type=str,
        help="Directory for logs. This should be a mounted directory if running in container.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp",
        help="Directory for caching data. This should be a mounted directory if running in container.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Set logger to debug level",
    )

    # Hypha related options
    hypha_group = parser.add_argument_group("Hypha Options")
    hypha_group.add_argument(
        "--workspace",
        type=str,
        help="Hypha workspace to connect to",
    )
    hypha_group.add_argument(
        "--server_url",
        default="https://hypha.aicell.io",
        type=str,
        help="URL of the Hypha server",
    )
    hypha_group.add_argument(
        "--token",
        type=str,
        help="Authentication token for Hypha server",
    )
    hypha_group.add_argument(
        "--worker_service_id",
        default="bioengine-worker",
        type=str,
        help="Service ID for the worker",
    )
    hypha_group.add_argument(
        "--client_id",
        type=str,
        help="Client ID for the worker. If not set, a client ID will be generated automatically.",
    )

    # Dataset Manager options
    dataset_group = parser.add_argument_group("Dataset Manager Options")
    dataset_group.add_argument(
        "--data_dir",
        type=str,
        help="Data directory served by the dataset manager. This should be a mounted directory if running in container.",
    )
    dataset_group.add_argument(
        "--dataset_service_id",
        type=str,
        default="bioengine-datasets",
        help="Service ID for the dataset manager",
    )

    # Ray Cluster Manager options
    cluster_group = parser.add_argument_group("Ray Cluster Manager Options")
    cluster_group.add_argument(
        "--head_node_ip",
        type=str,
        help="IP address for head node. Uses first system IP if None",
    )
    cluster_group.add_argument(
        "--head_node_port",
        type=int,
        default=6379,
        help="Port for Ray head node and GCS server",
    )
    cluster_group.add_argument(
        "--node_manager_port",
        type=int,
        default=6700,
        help="Port for Ray node manager services",
    )
    cluster_group.add_argument(
        "--object_manager_port",
        type=int,
        default=6701,
        help="Port for object manager service",
    )
    cluster_group.add_argument(
        "--redis_shard_port",
        type=int,
        default=6702,
        help="Port for Redis sharding",
    )
    cluster_group.add_argument(
        "--serve_port",
        type=int,
        default=8100,
        help="Port for Ray Serve",
    )
    cluster_group.add_argument(
        "--dashboard_port",
        type=int,
        default=8269,
        help="Port for Ray dashboard",
    )
    cluster_group.add_argument(
        "--ray_client_server_port",
        type=int,
        default=10001,
        help="Port for Ray client server",
    )
    cluster_group.add_argument(
        "--redis_password",
        type=str,
        help="Redis password for Ray cluster",
    )
    cluster_group.add_argument(
        "--ray_temp_dir",
        type=str,
        help="Temporary directory for Ray. If not set, defaults to '<cache_dir>/ray'. This should be a mounted directory if running in container.",
    )
    cluster_group.add_argument(
        "--head_num_cpus",
        type=int,
        default=0,
        help="Number of CPUs for head node if starting locally",
    )
    cluster_group.add_argument(
        "--head_num_gpus",
        type=int,
        default=0,
        help="Number of GPUs for head node if starting locally",
    )
    cluster_group.add_argument(
        "--skip_cleanup",
        action="store_true",
        default=False,
        help="Skip cleanup of previous Ray cluster",
    )
    cluster_group.add_argument(
        "--image",
        default=f"./apptainer_images/bioengine-worker_{__version__}.sif",
        type=str,
        help="Worker image for SLURM job",
    )
    cluster_group.add_argument(
        "--worker_data_dir",
        type=str,
        help="Data directory mounted to the container when starting a worker. If not set, the data_dir will be used.",
    )
    cluster_group.add_argument(
        "--slurm_log_dir",
        type=str,
        help="Directory for SLURM job logs. If not set, the log_dir will be used.",
    )
    cluster_group.add_argument(
        "--further_slurm_args",
        type=str,
        nargs="+",
        help="Additional arguments for SLURM job script",
    )

    # Ray Autoscaler options
    autoscaler_group = parser.add_argument_group("Ray Autoscaler Options")
    autoscaler_group.add_argument(
        "--default_num_gpus",
        default=1,
        type=int,
        help="Default number of GPUs per worker",
    )
    autoscaler_group.add_argument(
        "--default_num_cpus",
        default=8,
        type=int,
        help="Default number of CPUs per worker",
    )
    autoscaler_group.add_argument(
        "--default_mem_per_cpu",
        default=16,
        type=int,
        help="Default memory per CPU in GB",
    )
    autoscaler_group.add_argument(
        "--default_time_limit",
        default="4:00:00",
        type=str,
        help="Default time limit for workers",
    )
    autoscaler_group.add_argument(
        "--min_workers",
        default=0,
        type=int,
        help="Minimum number of worker nodes",
    )
    autoscaler_group.add_argument(
        "--max_workers",
        default=4,
        type=int,
        help="Maximum number of worker nodes",
    )
    autoscaler_group.add_argument(
        "--metrics_interval_seconds",
        default=60,
        type=int,
        help="Interval for collecting metrics",
    )
    autoscaler_group.add_argument(
        "--gpu_idle_threshold",
        default=0.05,
        type=float,
        help="GPU utilization threshold for idle nodes",
    )
    autoscaler_group.add_argument(
        "--cpu_idle_threshold",
        default=0.1,
        type=float,
        help="CPU utilization threshold for idle nodes",
    )
    autoscaler_group.add_argument(
        "--scale_down_threshold_seconds",
        default=300,
        type=int,
        help="Time threshold before scaling down idle nodes",
    )
    autoscaler_group.add_argument(
        "--scale_up_cooldown_seconds",
        default=120,
        type=int,
        help="Cooldown period before scaling up",
    )
    autoscaler_group.add_argument(
        "--scale_down_cooldown_seconds",
        default=600,
        type=int,
        help="Cooldown period before scaling down",
    )
    autoscaler_group.add_argument(
        "--node_grace_period_seconds",
        default=600,
        type=int,
        help="Grace period before considering a node for scaling down",
    )

    # Ray Deployment Manager options
    deployment_group = parser.add_argument_group("Ray Deployment Manager Options")
    deployment_group.add_argument(
        "--deployment_service_id",
        default="bioengine-apps",
        type=str,
        help="Service ID for deployed models",
    )
    deployment_group.add_argument(
        "--admin_users",
        type=str,
        nargs="+",
        help="List of admin users for the deployment",
    )
    deployment_group.add_argument(
        "--startup_deployments",
        type=str,
        nargs="+",
        help="List of artifact IDs to deploy on worker startup",
    )
    deployment_group.add_argument(
        "--deployment_cache_dir",
        type=str,
        help="Working directory for Ray Serve deployments. If not set, defaults to cache_dir. This should be a mounted directory if running in container.",
    )

    # Ray Connection options
    connection_group = parser.add_argument_group("Ray Connection Options")
    connection_group.add_argument(
        "--ray_address",
        type=str,
        help="Address of existing Ray cluster to connect to",
    )
    connection_group.add_argument(
        "--ray_namespace",
        type=str,
        help="Ray namespace to use",
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

    # Set up logging
    if group_configs["options"].get("log_dir"):
        log_dir = Path(group_configs["options"].get("log_dir")).resolve()
        log_file = log_dir / f"bioengine_worker_{time.strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_file = None
    group_configs["options"]["log_file"] = log_file
    logger = create_logger("__main__", log_file=log_file)

    try:
        asyncio.run(main(group_configs))
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)

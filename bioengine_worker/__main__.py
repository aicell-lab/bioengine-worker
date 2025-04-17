import argparse
import asyncio
import contextlib
import os
import signal
import sys

from hypha_rpc import login

from bioengine_worker.utils.logger import create_logger
from bioengine_worker.worker import BioEngineWorker

logger = create_logger("__main__")


async def main(group_configs):
    """Main function to initialize and register BioEngine worker"""
    # Setup signal-aware shutdown
    stop_event = asyncio.Event()
    is_shutting_down = False

    def _handle_shutdown_signal(sig_name):
        nonlocal is_shutting_down
        if is_shutting_down and sig_name == "SIGINT":
            logger.info("Received second SIGINT, stopping immediately...")
            sys.exit(1)
        
        logger.info(f"Received {sig_name}, starting graceful shutdown...")
        is_shutting_down = True
        stop_event.set()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _handle_shutdown_signal, "SIGINT")
    loop.add_signal_handler(signal.SIGTERM, _handle_shutdown_signal, "SIGTERM")

    bioengine_worker = None

    try:
        # Get Hypha configuration
        hypha_config = group_configs["Hypha Options"]
        token = hypha_config["token"] or os.environ["HYPHA_TOKEN"]
        if not token:
            token = await login({"server_url": hypha_config["server_url"]})

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
            k: v for k, v in ray_connection_kwargs.items() if v is not None
        }

        # Create BioEngine worker instance
        bioengine_worker = BioEngineWorker(
            workspace=hypha_config["workspace"],
            server_url=hypha_config["server_url"],
            token=token,
            service_id=hypha_config["worker_service_id"],
            ray_cluster_config=ray_cluster_config,
            clean_up_previous_cluster=clean_up_previous_cluster,
            ray_autoscaler_config=ray_autoscaler_config,
            ray_deployment_config=ray_deployment_config,
            ray_connection_kwargs=ray_connection_kwargs,
            _debug=group_configs["options"]["debug"],
        )

        # Initialize worker
        await bioengine_worker.start()

        # Start the server in the background
        serve_task = asyncio.create_task(bioengine_worker.serve())

        # Wait until shutdown is triggered
        await stop_event.wait()

        # Wait for the server task to complete (it should run forever, so we just cancel it when done)
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task

    except Exception as e:
        logger.error(f"Exception in main: {str(e)}")
        raise
    finally:
        if bioengine_worker:
            try:
                await bioengine_worker.cleanup()
            except Exception as cleanup_err:
                logger.error(f"Error during cleanup: {str(cleanup_err)}")


def create_parser():
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        description="BioEngine Worker Registration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Hypha related options
    hypha_group = parser.add_argument_group("Hypha Options")
    hypha_group.add_argument(
        "--workspace",
        default="chiron-platform",
        help="Hypha workspace to connect to",
    )
    hypha_group.add_argument(
        "--server_url",
        default="https://hypha.aicell.io",
        help="URL of the Hypha server",
    )
    hypha_group.add_argument(
        "--token",
        help="Authentication token for Hypha server",
    )
    hypha_group.add_argument(
        "--worker_service_id",
        default="bioengine-worker",
        help="Service ID for the worker",
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
        default="/tmp/ray",
        help="Temporary directory for Ray",
    )
    cluster_group.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory mounted to worker containers",
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
        "--image_path",
        type=str,
        default="./apptainer_images/bioengine-worker_0.1.5.sif",
        help="Worker container image path",
    )
    cluster_group.add_argument(
        "--slurm_logs_dir",
        type=str,
        default="./slurm_logs",
        help="Directory for SLURM logs",
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
        type=int,
        default=1,
        help="Default number of GPUs per worker",
    )
    autoscaler_group.add_argument(
        "--default_num_cpus",
        type=int,
        default=8,
        help="Default number of CPUs per worker",
    )
    autoscaler_group.add_argument(
        "--default_mem_per_cpu",
        type=int,
        default=16,
        help="Default memory per CPU in GB",
    )
    autoscaler_group.add_argument(
        "--default_time_limit",
        type=str,
        default="4:00:00",
        help="Default time limit for workers",
    )
    autoscaler_group.add_argument(
        "--min_workers",
        type=int,
        default=0,
        help="Minimum number of worker nodes",
    )
    autoscaler_group.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker nodes",
    )
    autoscaler_group.add_argument(
        "--metrics_interval_seconds",
        type=int,
        default=60,
        help="Interval for collecting metrics",
    )
    autoscaler_group.add_argument(
        "--gpu_idle_threshold",
        type=float,
        default=0.05,
        help="GPU utilization threshold for idle nodes",
    )
    autoscaler_group.add_argument(
        "--cpu_idle_threshold",
        type=float,
        default=0.1,
        help="CPU utilization threshold for idle nodes",
    )
    autoscaler_group.add_argument(
        "--scale_down_threshold_seconds",
        type=int,
        default=300,
        help="Time threshold before scaling down idle nodes",
    )
    autoscaler_group.add_argument(
        "--scale_up_cooldown_seconds",
        type=int,
        default=120,
        help="Cooldown period before scaling up",
    )
    autoscaler_group.add_argument(
        "--scale_down_cooldown_seconds",
        type=int,
        default=600,
        help="Cooldown period before scaling down",
    )
    autoscaler_group.add_argument(
        "--node_grace_period_seconds",
        type=int,
        default=600,
        help="Grace period before considering a node for scaling down",
    )

    # Ray Deployment Manager options
    deployment_group = parser.add_argument_group("Ray Deployment Manager Options")
    deployment_group.add_argument(
        "--deployment_service_id",
        type=str,
        default="ray-model-services",
        help="Service ID for deployed models",
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

    # Others
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

    try:
        asyncio.run(main(group_configs))
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)

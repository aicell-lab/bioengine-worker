#!/usr/bin/env python3
"""
BioEngine Worker Command-Line Interface

Enterprise-grade command-line interface for deploying and managing BioEngine workers
across diverse computational environments. Supports HPC clusters with SLURM scheduling,
single-machine deployments, and external Ray cluster connections.

This module provides comprehensive configuration management, signal handling for graceful
shutdown, and structured logging for production deployments. It serves as the primary
entry point for BioEngine worker services in both development and production environments.

Key Features:
- Multi-environment deployment support (SLURM, single-machine, external clusters)
- Comprehensive command-line argument parsing with validation
- Graceful shutdown handling with signal management
- Structured logging with file output and debug modes
- Production-ready error handling and cleanup procedures
- Hypha server integration with authentication management

Usage:
    python -m bioengine_worker --mode slurm --admin_users admin@institution.edu
    python -m bioengine_worker --mode single-machine --debug
    python -m bioengine_worker --mode external-cluster --server_url https://custom.hypha.io

Example Deployment:
    # SLURM HPC environment with custom configuration
    python -m bioengine_worker \\
        --mode slurm \\
        --admin_users admin@institution.edu researcher@institution.edu \\
        --cache_dir /shared/bioengine/cache \\
        --data_dir /shared/datasets \\
        --max_workers 20 \\
        --default_num_gpus 2 \\
        --server_url https://hypha.aicell.io

Author: BioEngine Development Team
License: MIT
"""

import argparse
import asyncio
import json
import sys
from typing import Dict

from bioengine_worker import __version__
from bioengine_worker.worker import BioEngineWorker


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the comprehensive argument parser for BioEngine worker.

    Sets up command-line argument parsing with detailed help text, validation,
    and organized argument groups for different configuration categories.
    Provides extensive customization options for all deployment environments.

    Returns:
        Configured ArgumentParser instance with all BioEngine worker options

    Argument Groups:
        - Core Options: Basic worker configuration (mode, users, directories)
        - Hypha Options: Server connection and authentication settings
        - Ray Cluster Options: Cluster management and networking configuration
        - SLURM Job Options: HPC-specific deployment parameters
        - Ray Autoscaler Options: Autoscaling behavior and resource limits
    """
    parser = argparse.ArgumentParser(
        description="BioEngine Worker - Enterprise AI Model Deployment Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SLURM HPC deployment with autoscaling
  %(prog)s --mode slurm --max_workers 10 --admin_users admin@institution.edu

  # Single-machine development deployment  
  %(prog)s --mode single-machine --debug --cache_dir ./cache

  # Connect to existing Ray cluster
  %(prog)s --mode external-cluster --head_node_address 10.0.0.100

For detailed documentation, visit: https://github.com/aicell-lab/bioengine-worker
""",
    )

    # Core configuration options
    core_group = parser.add_argument_group("Core Options", "Basic worker configuration")
    core_group.add_argument(
        "--mode",
        default="single-machine",
        type=str,
        choices=["single-machine", "slurm", "external-cluster"],
        help="Deployment mode: 'single-machine' for local Ray cluster, "
        "'slurm' for HPC clusters with SLURM job scheduling, 'external-cluster' for connecting "
        "to an existing Ray cluster",
    )
    core_group.add_argument(
        "--admin_users",
        type=str,
        nargs="+",
        metavar="EMAIL",
        help="List of user emails/IDs with administrative privileges for worker management. "
        "If not specified, defaults to the authenticated user from Hypha login.",
    )
    core_group.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/bioengine",
        metavar="PATH",
        help="Directory for worker cache, temporary files, and Ray data storage. "
        "Should be accessible across worker nodes in distributed deployments. "
        "Default: /tmp/bioengine",
    )
    core_group.add_argument(
        "--data_dir",
        default="/data",
        type=str,
        metavar="PATH",
        help="Root directory for dataset storage and access by the dataset manager. "
        "Should be mounted shared storage in distributed environments. Default: /data",
    )
    core_group.add_argument(
        "--startup_applications",
        type=str,
        nargs="+",
        metavar="JSON",
        help="List of applications to deploy automatically during worker startup. "
        "Each element should be a JSON string with deployment configuration. "
        'Example: \'{"artifact_id": "my_model", "application_id": "my_app"}\'',
    )
    core_group.add_argument(
        "--monitoring_interval_seconds",
        default=10,
        type=int,
        metavar="SECONDS",
        help="Interval in seconds for worker status monitoring and health checks. "
        "Lower values provide faster response but increase overhead. Default: 10",
    )
    core_group.add_argument(
        "--dashboard_url",
        type=str,
        default="https://bioimage.io/#/bioengine",
        metavar="URL",
        help="Base URL of the BioEngine dashboard for worker management interfaces. "
        "Default: https://bioimage.io/#/bioengine",
    )
    core_group.add_argument(
        "--log_file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the log file. If set to 'off', logging will only go to console. "
        "If not specified (None), a log file will be created in '<cache_dir>/logs'. ",
    )
    core_group.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug-level logging for detailed troubleshooting and development. "
        "Increases log verbosity significantly.",
    )
    core_group.add_argument(
        "--graceful_shutdown_timeout",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Timeout in seconds for graceful shutdown operations. " "Default: 60",
    )

    # Hypha server connection options
    hypha_group = parser.add_argument_group(
        "Hypha Options", "Server connection and authentication"
    )
    hypha_group.add_argument(
        "--server_url",
        default="https://hypha.aicell.io",
        type=str,
        metavar="URL",
        help="URL of the Hypha server for service registration and remote access. "
        "Must be accessible from the deployment environment. Default: https://hypha.aicell.io",
    )
    hypha_group.add_argument(
        "--workspace",
        type=str,
        metavar="NAME",
        help="Hypha workspace name for service isolation and organization. "
        "If not specified, uses the workspace associated with the authentication token.",
    )
    hypha_group.add_argument(
        "--token",
        type=str,
        metavar="TOKEN",
        help="Authentication token for Hypha server access. If not provided, will use "
        "the HYPHA_TOKEN environment variable or prompt for interactive login. "
        "Recommend using a long-lived token for production deployments.",
    )
    hypha_group.add_argument(
        "--client_id",
        type=str,
        metavar="ID",
        help="Unique client identifier for Hypha connection. If not specified, "
        "an identifier will be generated automatically to ensure unique registration.",
    )

    # Ray cluster management options
    ray_cluster_group = parser.add_argument_group(
        "Ray Cluster Options", "Cluster networking and resource configuration"
    )
    ray_cluster_group.add_argument(
        "--head_node_address",
        type=str,
        metavar="ADDRESS",
        help="IP address of the Ray head node. For external-cluster mode, this specifies "
        "the cluster to connect to. If not set in other modes, uses the first available "
        "system IP address.",
    )
    ray_cluster_group.add_argument(
        "--head_node_port",
        type=int,
        default=6379,
        metavar="PORT",
        help="Port for Ray head node and GCS (Global Control Service) server. "
        "Must be accessible from all worker nodes. Default: 6379",
    )
    ray_cluster_group.add_argument(
        "--node_manager_port",
        type=int,
        default=6700,
        metavar="PORT",
        help="Port for Ray node manager services. Used for inter-node communication "
        "and coordination. Default: 6700",
    )
    ray_cluster_group.add_argument(
        "--object_manager_port",
        type=int,
        default=6701,
        metavar="PORT",
        help="Port for Ray object manager service. Handles distributed object storage "
        "and transfer between nodes. Default: 6701",
    )
    ray_cluster_group.add_argument(
        "--redis_shard_port",
        type=int,
        default=6702,
        metavar="PORT",
        help="Port for Redis sharding in Ray's internal metadata storage. "
        "Used for cluster state management. Default: 6702",
    )
    ray_cluster_group.add_argument(
        "--serve_port",
        type=int,
        default=8000,
        metavar="PORT",
        help="Port for Ray Serve HTTP endpoint serving deployed models and applications. "
        "This is where model inference requests are handled. Default: 8000",
    )
    ray_cluster_group.add_argument(
        "--dashboard_port",
        type=int,
        default=8265,
        metavar="PORT",
        help="Port for Ray dashboard web interface. Provides cluster monitoring "
        "and debugging capabilities. Default: 8265",
    )
    ray_cluster_group.add_argument(
        "--client_server_port",
        type=int,
        default=10001,
        metavar="PORT",
        help="Port for Ray client server connections. Used by external Ray clients "
        "to connect to the cluster. Default: 10001",
    )
    ray_cluster_group.add_argument(
        "--redis_password",
        type=str,
        metavar="PASSWORD",
        help="Password for Ray cluster Redis authentication. If not specified, "
        "a secure random password will be generated automatically.",
    )
    ray_cluster_group.add_argument(
        "--head_num_cpus",
        type=int,
        default=0,
        metavar="COUNT",
        help="Number of CPU cores allocated to the head node for task execution. "
        "Set to 0 to reserve head node for coordination only. Default: 0",
    )
    ray_cluster_group.add_argument(
        "--head_num_gpus",
        type=int,
        default=0,
        metavar="COUNT",
        help="Number of GPU devices allocated to the head node for task execution. "
        "Typically 0 to reserve GPUs for worker nodes. Default: 0",
    )
    ray_cluster_group.add_argument(
        "--head_memory_in_gb",
        type=int,
        metavar="GB",
        help="Memory allocation in GB for head node task execution. "
        "If not specified, Ray will auto-detect available memory.",
    )
    ray_cluster_group.add_argument(
        "--runtime_env_pip_cache_size_gb",
        type=int,
        default=30,
        metavar="GB",
        help="Size limit in GB for Ray runtime environment pip package cache. "
        "Larger cache improves environment setup time. Default: 30",
    )
    ray_cluster_group.add_argument(
        "--skip_ray_cleanup",
        action="store_true",
        default=False,
        help="Skip cleanup of previous Ray cluster processes and data. "
        "Use with caution as it may cause port conflicts or resource issues.",
    )

    # SLURM job configuration options (for HPC environments)
    slurm_job_group = parser.add_argument_group(
        "SLURM Job Options", "HPC job scheduling and worker deployment"
    )
    slurm_job_group.add_argument(
        "--image",
        default=f"ghcr.io/aicell-lab/bioengine-worker:{__version__}",
        type=str,
        metavar="IMAGE",
        help=f"Container image for SLURM worker jobs. Should include all required "
        f"dependencies and be accessible on compute nodes. Default: ghcr.io/aicell-lab/bioengine-worker:{__version__}",
    )
    slurm_job_group.add_argument(
        "--worker_cache_dir",
        type=str,
        metavar="PATH",
        help="Cache directory path mounted to worker containers in SLURM jobs. "
        "Must be accessible from compute nodes. Required for SLURM mode.",
    )
    slurm_job_group.add_argument(
        "--default_num_gpus",
        default=1,
        type=int,
        metavar="COUNT",
        help="Default number of GPU devices to request per SLURM worker job. "
        "Can be overridden per deployment. Default: 1",
    )
    slurm_job_group.add_argument(
        "--default_num_cpus",
        default=8,
        type=int,
        metavar="COUNT",
        help="Default number of CPU cores to request per SLURM worker job. "
        "Should match typical model inference requirements. Default: 8",
    )
    slurm_job_group.add_argument(
        "--default_mem_in_gb_per_cpu",
        default=16,
        type=int,
        metavar="GB",
        help="Default memory allocation in GB per CPU core for SLURM workers. "
        "Total memory = num_cpus * mem_per_cpu. Default: 16",
    )
    slurm_job_group.add_argument(
        "--default_time_limit",
        default="4:00:00",
        type=str,
        metavar="TIME",
        help="Default time limit for SLURM worker jobs in HH:MM:SS format. "
        "Jobs will be terminated after this duration. Default: 4:00:00",
    )
    slurm_job_group.add_argument(
        "--further_slurm_args",
        type=str,
        nargs="+",
        metavar="ARG",
        help="Additional SLURM sbatch arguments for specialized cluster configurations. "
        'Example: "--partition=gpu" "--qos=high-priority"',
    )

    # Ray autoscaling configuration options
    ray_autoscaling_group = parser.add_argument_group(
        "Ray Autoscaler Options", "Automatic worker scaling behavior"
    )
    ray_autoscaling_group.add_argument(
        "--min_workers",
        default=0,
        type=int,
        metavar="COUNT",
        help="Minimum number of worker nodes to maintain in the cluster. "
        "Workers below this threshold will be started immediately. Default: 0",
    )
    ray_autoscaling_group.add_argument(
        "--max_workers",
        default=4,
        type=int,
        metavar="COUNT",
        help="Maximum number of worker nodes allowed in the cluster. "
        "Prevents unlimited scaling and controls costs. Default: 4",
    )
    ray_autoscaling_group.add_argument(
        "--scale_up_cooldown_seconds",
        default=60,
        type=int,
        metavar="SECONDS",
        help="Cooldown period in seconds between scaling up operations. "
        "Prevents rapid scaling oscillations. Default: 60",
    )
    ray_autoscaling_group.add_argument(
        "--scale_down_check_interval_seconds",
        default=60,
        type=int,
        metavar="SECONDS",
        help="Interval in seconds between checks for scaling down idle workers. "
        "More frequent checks enable faster response to load changes. Default: 60",
    )
    ray_autoscaling_group.add_argument(
        "--scale_down_threshold_seconds",
        default=300,
        type=int,
        metavar="SECONDS",
        help="Time threshold in seconds before scaling down idle worker nodes. "
        "Longer thresholds reduce churn but may waste resources. Default: 300",
    )

    return parser


def get_args_by_group(parser: argparse.ArgumentParser) -> Dict[str, Dict[str, any]]:
    """
    Parse command-line arguments and organize them by argument group.

    Parses all command-line arguments and organizes them into dictionaries
    based on their argument group membership. This organization facilitates
    passing configuration to different components of the BioEngine worker.

    Args:
        parser: Configured ArgumentParser instance

    Returns:
        Dictionary mapping argument group names to their respective arguments:
            - "Core Options": Basic worker configuration
            - "Hypha Options": Server connection settings
            - "Ray Cluster Options": Cluster networking configuration
            - "SLURM Job Options": HPC job parameters
            - "Ray Autoscaler Options": Autoscaling configuration

    Example:
        ```python
        parser = create_parser()
        groups = get_args_by_group(parser)
        worker_config = groups["Core Options"]
        hypha_config = groups["Hypha Options"]
        ```
    """
    args = parser.parse_args()
    args_dict = vars(args)

    group_configs = {}
    for group in parser._action_groups:
        # Skip built-in argument groups (positional arguments, options)
        if group.title in ["positional arguments", "options"]:
            continue

        group_keys = [a.dest for a in group._group_actions]
        group_configs[group.title] = {
            k: args_dict[k] for k in group_keys if k in args_dict
        }

    return group_configs


def read_startup_applications(
    group_configs: Dict[str, Dict[str, any]],
) -> Dict[str, Dict[str, any]]:
    """
    Parse and validate startup application configurations from JSON strings.

    Processes startup application specifications from command-line arguments,
    parsing JSON configuration strings and validating their format. Each
    application configuration should specify deployment parameters. The parsed
    dictionaries are passed to BioEngineWorker, replacing the original JSON strings.

    Args:
        group_configs: Dictionary of grouped command-line arguments

    Returns:
        Updated group_configs with parsed startup_applications as list of dictionaries

    Raises:
        ValueError: If JSON configuration is malformed or invalid

    Example:
        Valid startup application JSON:
        ```json
        {"artifact_id": "my_artifact", "num_gpus": 1}
        ```
    """
    # Use startup_applications from Core Options group
    core_options = group_configs.get("Core Options", {})
    startup_apps_raw = core_options.get("startup_applications")

    if not startup_apps_raw:
        return group_configs

    startup_applications = []
    for json_str in startup_apps_raw:
        try:
            application_config = json.loads(json_str.strip())
            if not isinstance(application_config, dict):
                raise ValueError("Application configuration must be a JSON object")

            # Validate required fields
            if "artifact_id" not in application_config:
                raise ValueError(
                    "Application configuration must include 'artifact_id' field"
                )

            startup_applications.append(application_config)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON format for startup application: {e}. "
                "A valid format would be: "
                '\'{"artifact_id": "my_artifact", "num_gpus": 1}\'. '
                f"Received: '{json_str}'"
            )

    group_configs["Core Options"]["startup_applications"] = startup_applications
    return group_configs


async def main(group_configs):
    """Main function to initialize and register BioEngine worker"""
    # Create BioEngine worker instance
    bioengine_worker = BioEngineWorker(
        **group_configs["Core Options"],
        **group_configs["Hypha Options"],
        ray_cluster_config={
            **group_configs["Ray Cluster Options"],
            **group_configs["SLURM Job Options"],
            **group_configs["Ray Autoscaler Options"],
        },
    )

    # Start the worker and wait until shutdown is triggered
    await bioengine_worker.start(blocking=True)


if __name__ == "__main__":
    """
    Entry point for BioEngine worker command-line interface.

    Parses command-line arguments, validates configuration, and starts the worker.
    """
    try:
        parser = create_parser()
        group_configs = get_args_by_group(parser)

        # Process startup applications if provided
        group_configs = read_startup_applications(group_configs)

        # Start the main worker process
        asyncio.run(main(group_configs))

    except Exception as e:
        print(f"Failed to start BioEngine worker: {e}")
        sys.exit(1)

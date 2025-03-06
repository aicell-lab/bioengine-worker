import logging
import os
import time
import socket
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import yaml
import numpy as np
from dotenv import find_dotenv, load_dotenv
from hypha_rpc import connect_to_server, login

from hpc_worker.ray_cluster_manager import RayClusterManager
from hpc_worker.ray_deployment_manager import RayDeploymentManager

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


class HpcWorker:
    """HPC Worker service for Chiron platform integration"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dir: Optional[str] = None,
        server_url: str = "https://hypha.aicell.io",
        num_gpu: Optional[int] = None,
        dataset_paths: Optional[List[str]] = None,
        trusted_models: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize HPC worker

        Args:
            config_path: Path to existing configuration file
            config_dir: Directory to save new configuration file (if config_path not provided)
            server_url: Hypha server URL
            num_gpu: Number of available GPUs
            dataset_paths: List of dataset directories
            trusted_models: List of trusted Docker images
            logger: Optional logger instance

        Raises:
            ValueError: If config_path is not provided and any of num_gpu, dataset_paths,
                       or trusted_models is missing
        """
        # Setup logging
        self.logger = logger or logging.getLogger("hpc_worker")
        if not logger:
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Set default server URL (may be overwritten by config)
        self.server_url = server_url

        # Handle configuration
        if config_path:
            # Use existing config file
            self.config_path = config_path

            # Check if config file exists and has content
            if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
                # Create a new config with provided parameters
                if (
                    num_gpu is not None
                    and dataset_paths is not None
                    and trusted_models is not None
                ):
                    self.create_worker_config(
                        dataset_paths=dataset_paths,
                        max_gpus=num_gpu,
                        trusted_models=trusted_models,
                        server_url=server_url,
                        config_path=self.config_path,
                    )
                else:
                    # Create a minimal config with just server_url
                    self.create_worker_config(
                        dataset_paths=[],
                        max_gpus=0,
                        trusted_models=[],
                        server_url=server_url,
                        config_path=self.config_path,
                    )
                self.logger.info(
                    f"Created worker configuration file at {self.config_path}"
                )
            else:
                # Load config to get server_url (if present)
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f) or {}  # Default to empty dict if None
                    if "server_url" in config:
                        self.server_url = config["server_url"]
        else:
            # Check that all required parameters are provided
            if num_gpu is None or dataset_paths is None or trusted_models is None:
                raise ValueError(
                    "If config_path is not provided, all of num_gpu, dataset_paths, "
                    "and trusted_models must be specified."
                )

            # Determine config directory and path
            if config_dir is None:
                config_dir = os.path.join(os.path.dirname(__file__), "config")
            os.makedirs(config_dir, exist_ok=True)

            self.config_path = os.path.join(config_dir, "worker_config.yaml")

            # Create new config file
            self.create_worker_config(
                dataset_paths=dataset_paths,
                max_gpus=num_gpu,
                trusted_models=trusted_models,
                server_url=server_url,
                config_path=self.config_path,
            )
            self.logger.info(f"Created worker configuration file at {self.config_path}")

        # Initialize registration timestamp
        self.registered_at = int(time.time())

        # Initialize Ray cluster manager
        self.ray_manager = RayClusterManager(logger=self.logger)

        # Initialize Ray deployment manager
        self.deployment_manager = RayDeploymentManager(
            hpc_worker=self,
            logger=self.logger,
            service_id="ray-model-services",
        )

        # Server connection - initialized during register()
        self.server = None
        self.service_info = None

    # Configuration management functions
    def create_worker_config(
        self,
        dataset_paths: List[str],
        max_gpus: int,
        trusted_models: List[str],
        server_url: str,
        config_path: str,
    ) -> None:
        """Create worker configuration file

        Args:
            dataset_paths: List of dataset directories to include
            max_gpus: Maximum number of GPUs available
            trusted_models: List of trusted Docker/SIF images
            server_url: Hypha server URL
            config_path: Path to save the config file
        """
        config = {
            "machine_name": socket.gethostname(),
            "max_gpus": max_gpus,
            "dataset_paths": dataset_paths,
            "trusted_models": trusted_models,
            "server_url": server_url,
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

    # Worker status functions
    def format_time(
        self, last_deployed_time_s: int, tz: timezone = timezone.utc
    ) -> Dict:
        """Format a timestamp into human-readable format with duration

        Args:
            last_deployed_time_s: Unix timestamp
            tz: Timezone to use for formatting

        Returns:
            Dictionary with formatted time information
        """
        current_time = datetime.now(tz)
        last_deployed_time = datetime.fromtimestamp(last_deployed_time_s, tz)

        duration = current_time - last_deployed_time
        days = duration.days
        seconds = duration.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        duration_parts = []
        if days > 0:
            duration_parts.append(f"{days}d")
        if hours > 0:
            duration_parts.append(f"{hours}h")
        if minutes > 0:
            duration_parts.append(f"{minutes}m")
        if remaining_seconds > 0:
            duration_parts.append(f"{remaining_seconds}s")

        return {
            "timestamp": last_deployed_time.strftime("%Y/%m/%d %H:%M:%S"),
            "timezone": str(tz),
            "duration_since": " ".join(duration_parts) if duration_parts else "0s",
        }

    def load_dataset_info(self, dataset_path: str) -> Optional[Dict]:
        """Load dataset information from info.npz file

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary with dataset information or None if info file not found
        """
        info_path = os.path.join(dataset_path, "info.npz")
        if not os.path.exists(info_path):
            return None

        info = np.load(info_path)
        return {"name": os.path.basename(dataset_path), "samples": int(info["length"])}

    def process_model_info(self, image: str) -> Dict:
        """Process docker image string into model info

        Args:
            image: Docker image string (repo:tag)

        Returns:
            Dictionary with parsed model information
        """
        repo, version = image.split(":")
        name = repo.split("/")[-1]
        return {"name": name, "image": image, "version": version}

    def get_worker_status(self, context: Dict = None) -> Dict:
        """Get complete worker status

        Args:
            context: RPC context

        Returns:
            Dictionary with complete worker status information
        """
        # Load config file
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Process datasets
        datasets = []
        for path in config["dataset_paths"]:
            if os.path.exists(path):
                info = self.load_dataset_info(path)
                if info:
                    datasets.append(info)

        # Process models
        models = [self.process_model_info(image) for image in config["trusted_models"]]

        # Get ray cluster status - fixed to use the instance's check_cluster method
        ray_status = self.ray_manager.check_cluster()

        # Format time information
        time_info = self.format_time(self.registered_at)

        status = {
            "machine": {"name": config["machine_name"], "max_gpus": config["max_gpus"]},
            "ray_cluster": {
                "head_node_running": ray_status["head_running"],
                "active_workers": ray_status["worker_count"],
            },
            "registration": {
                "registered_at": time_info["timestamp"],
                "timezone": time_info["timezone"],
                "uptime": time_info["duration_since"],
            },
            "datasets": datasets,
            "trusted_models": models,
            "server_url": config.get("server_url", self.server_url),
        }

        return status

    # Service registration
    async def register(self) -> Dict:
        """Register worker service with Hypha server

        Returns:
            Dict containing registration status and service information
        """
        self.logger.info("Connecting to Hypha server...")

        try:
            # Login to Hypha server
            hypha_token = os.environ.get("HYPHA_TOKEN")
            if not hypha_token:
                hypha_token = await login({"server_url": self.server_url})

            # Connect to server
            self.server = await connect_to_server(
                {"server_url": self.server_url, "token": hypha_token}
            )

            # Register service with Hypha
            self.service_info = await self.server.register_service(
                {
                    "name": "HPC Worker",
                    "id": "hpc-worker",
                    "config": {
                        "visibility": "public",
                        "require_context": True,
                        "run_in_executor": False,
                    },
                    # Service methods
                    "ping": lambda context: "pong",
                    "get_worker_status": self.get_worker_status,
                    # Ray cluster methods
                    "start_ray_cluster": self.ray_manager.start_cluster,
                    "shutdown_ray_cluster": self.ray_manager.shutdown_cluster,
                    "submit_ray_worker_job": self.ray_manager.submit_worker_job,
                    "get_ray_worker_jobs": self.ray_manager.get_worker_jobs,
                    "cancel_ray_worker_jobs": self.ray_manager.cancel_worker_jobs,
                    # Ray deployment methods
                    "deploy_artifact": self.deployment_manager.deploy_artifact,
                    "undeploy_artifact": self.deployment_manager.undeploy_artifact,
                    "list_deployments": self.deployment_manager.list_deployments,
                    "deploy_all_artifacts": self.deployment_manager.deploy_all_artifacts,
                }
            )

            # Initialize the deployment manager with the server connection
            await self.deployment_manager.initialize(self.server)

            self.logger.info(f"Service registered with ID: {self.service_info.id}")
            sid = self.service_info.id.split("/")[1]
            service_url = (
                f"{self.server_url}/{self.server.config.workspace}/services/{sid}"
            )

            # Log service endpoints
            self.logger.info(
                f"Test the HPC worker service here: {service_url}/get_worker_status"
            )
            self.logger.info(f"Start Ray cluster with: {service_url}/start_ray_cluster")
            self.logger.info(
                f"Shutdown Ray cluster with: {service_url}/shutdown_ray_cluster"
            )
            self.logger.info(
                f"Submit a Ray worker job with: {service_url}/submit_ray_worker_job"
            )
            self.logger.info(
                f"Get Ray worker jobs status with: {service_url}/get_ray_worker_jobs"
            )
            self.logger.info(
                f"Cancel Ray worker jobs with: {service_url}/cancel_ray_worker_jobs"
            )

            return {
                "success": True,
                "service_id": self.service_info.id,
                "service_url": service_url,
            }

        except Exception as e:
            self.logger.error(f"Error registering service: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}

    # Cleanup methods
    async def cleanup(self) -> Dict:
        """Clean up resources on shutdown

        Returns:
            Dict containing cleanup status information
        """
        self.logger.info("Cleaning up before shutdown...")
        results = {}

        # Clean up Ray Serve deployments
        try:
            self.logger.info("Cleaning up Ray Serve deployments")
            results["deployments_cleanup"] = await self.deployment_manager.cleanup()
        except Exception as e:
            self.logger.error(f"Error cleaning up deployments: {str(e)}")
            results["deployments_cleanup"] = {"success": False, "message": str(e)}

        # First try to shutdown Ray cluster
        try:
            self.logger.info("Attempting to shut down Ray cluster")
            results["ray_shutdown"] = self.ray_manager.shutdown_cluster()
            if results["ray_shutdown"]["success"]:
                self.logger.info("Ray cluster shut down successfully")
            else:
                self.logger.warning(
                    f"Ray cluster shutdown issue: {results['ray_shutdown'].get('message', 'Unknown error')}"
                )
        except Exception as e:
            self.logger.error(f"Error shutting down Ray cluster: {str(e)}")
            results["ray_shutdown"] = {"success": False, "message": str(e)}

        # Then cancel all worker jobs
        try:
            self.logger.info("Cancelling all Ray worker jobs")
            results["jobs_cancel"] = self.ray_manager.cancel_worker_jobs()
            if results["jobs_cancel"]["success"]:
                cancelled = results["jobs_cancel"].get("cancelled_jobs", 0)
                self.logger.info(f"Successfully cancelled {cancelled} worker jobs")
            else:
                self.logger.warning(
                    f"Job cancellation issue: {results['jobs_cancel'].get('message', 'Unknown error')}"
                )
        except Exception as e:
            self.logger.error(f"Error cancelling Ray worker jobs: {str(e)}")
            results["jobs_cancel"] = {"success": False, "message": str(e)}

        self.logger.info("Cleanup completed")
        return results


if __name__ == "__main__":
    """Test the HpcWorker class functionality."""
    import asyncio

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hpc_worker")

    print("===== Testing HPC Worker class =====", end="\n\n")

    # Define config path in hpc_worker/config/
    config_dir = os.path.join(os.path.dirname(__file__), "config")
    os.makedirs(config_dir, exist_ok=True)
    config_file_path = os.path.join(config_dir, "worker_config.yaml")

    # Create HPC worker instance
    print("Creating HPC worker instance...")
    worker = HpcWorker(
        config_path=config_file_path,
        num_gpu=2,
        dataset_paths=["/proj/aicell/users/x_nilme/tabula/tabula/resource/demo/blood"],
        trusted_models=["ghcr.io/aicell-lab/tabula:0.1.1"],
    )
    print(f"Worker created with config at {config_file_path}", end="\n\n")

    # Print worker config
    print("Worker configuration:")
    with open(config_file_path, "r") as f:
        print(f.read(), end="\n\n")

    # Test get_worker_status
    print("Getting worker status...")
    status = worker.get_worker_status()
    print(f"Worker status: {status}", end="\n\n")

    # Test ray manager functionality
    print("Testing Ray cluster functionality...")

    # Check cluster status
    print("Checking Ray cluster status...")
    ray_status = worker.ray_manager.check_cluster()
    print(f"Ray cluster status: {ray_status}", end="\n\n")

    # Test starting Ray cluster (only if not already running)
    if not ray_status["head_running"]:
        print("Starting Ray cluster...")
        start_result = worker.ray_manager.start_cluster()
        print(f"Start result: {start_result}", end="\n\n")
    else:
        print("Ray cluster already running, skipping start", end="\n\n")

    # Test submit job functionality
    print("Submitting Ray worker job...")
    job_result = worker.ray_manager.submit_worker_job()
    print(f"Job submission result: {job_result}", end="\n\n")

    # Get worker jobs
    print("Checking Ray worker jobs...")
    jobs_result = worker.ray_manager.get_worker_jobs()
    print(f"Worker jobs: {jobs_result}", end="\n\n")

    # Cancel worker jobs
    print("Cancelling Ray worker jobs...")
    cancel_result = worker.ray_manager.cancel_worker_jobs()
    print(f"Cancel result: {cancel_result}", end="\n\n")

    # Test shutting down Ray cluster
    print("Shutting down Ray cluster...")
    shutdown_result = worker.ray_manager.shutdown_cluster()
    print(f"Shutdown result: {shutdown_result}", end="\n\n")

    # Test cleanup method
    print("Testing cleanup...")
    loop = asyncio.get_event_loop()
    cleanup_result = loop.run_until_complete(worker.cleanup())
    print(f"Cleanup result: {cleanup_result}", end="\n\n")

    print("===== HPC Worker class tests completed =====")

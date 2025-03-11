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
from hpc_worker.ray_autoscaler import RayAutoscaler

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
        autoscale: bool = False,
        gpus_per_worker: Optional[int] = None,
        worker_config: Optional[Dict] = None,
        autoscaler_config: Optional[Dict] = None,
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
            autoscale: Enable autoscaling
            gpus_per_worker: Number of GPUs per worker
            worker_config: Configuration for worker jobs
            autoscaler_config: Configuration for autoscaler

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

        # Default configurations
        default_worker_config = {
            "num_cpus": 8,
            "mem_per_cpu": 16,
            "time_limit": "4:00:00"
        }
        
        default_autoscaler_config = {
            "min_workers": 0,
            "target_gpu_utilization": 0.7,
            "scale_up_cooldown": 30,
            "scale_down_cooldown": 300
        }

        # Handle configuration from file or parameters
        if config_path and os.path.exists(config_path):
            # Load existing config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Use config file values but allow override from parameters
            self.server_url = server_url or config.get('server_url', "https://hypha.aicell.io")
            self.gpus_per_worker = gpus_per_worker or config.get('gpus_per_worker', 1)
            self.worker_config = worker_config or config.get('worker_config', default_worker_config)
            self.autoscaler_config = autoscaler_config or config.get('autoscaler', default_autoscaler_config)
            
            # Other values from config
            self.max_gpus = num_gpu or config.get('max_gpus', 1)
            self.dataset_paths = dataset_paths or config.get('dataset_paths', [])
            self.trusted_models = trusted_models or config.get('trusted_models', [])
            
            self.config_path = config_path
        else:
            # Create new config from parameters
            if not all([num_gpu is not None, dataset_paths, trusted_models]):
                raise ValueError(
                    "If config_path is not provided or invalid, all of num_gpu, "
                    "dataset_paths, and trusted_models must be specified."
                )

            # Use provided parameters or defaults
            self.server_url = server_url
            self.max_gpus = num_gpu
            self.dataset_paths = dataset_paths
            self.trusted_models = trusted_models
            self.gpus_per_worker = gpus_per_worker or 1
            self.worker_config = worker_config or default_worker_config
            self.autoscaler_config = autoscaler_config or default_autoscaler_config

            # Create config directory and file if needed
            if not config_path:
                if not config_dir:
                    config_dir = os.path.join(os.path.dirname(__file__), "config")
                os.makedirs(config_dir, exist_ok=True)
                self.config_path = os.path.join(config_dir, "worker_config.yaml")
            else:
                self.config_path = config_path

            # Create new config file
            self.create_worker_config(
                dataset_paths=self.dataset_paths,
                max_gpus=self.max_gpus,
                trusted_models=self.trusted_models,
                server_url=self.server_url,
                config_path=self.config_path,
                gpus_per_worker=self.gpus_per_worker,
                worker_config=self.worker_config,
                autoscaler_config=self.autoscaler_config,
            )
            self.logger.info(f"Created worker configuration file at {self.config_path}")

        # Calculate max workers based on available GPUs and GPUs per worker
        self.max_workers = self.max_gpus // self.gpus_per_worker

        # Initialize registration timestamp
        self.registered_at = int(time.time())

        # Initialize Ray cluster manager
        self.ray_manager = RayClusterManager(logger=self.logger)

        # Start Ray cluster immediately
        self.logger.info("Starting Ray cluster for deployment manager...")
        ray_result = self.ray_manager.start_cluster()
        if not ray_result["success"]:
            self.logger.warning(f"Failed to start Ray cluster: {ray_result['message']}")
            self.logger.warning("Will retry Ray initialization during registration")

        # Initialize Ray deployment manager only after Ray is started
        try:
            self.deployment_manager = RayDeploymentManager(
                logger=self.logger,
                deployment_collection_id="ray-deployments",
                service_id="ray-model-services",
            )
        except RuntimeError as e:
            self.logger.warning(f"Could not initialize deployment manager yet: {e}")
            self.deployment_manager = None  # Will initialize during register()

        # Initialize Ray autoscaler with calculated values
        self.autoscaler = RayAutoscaler(
            ray_manager=self.ray_manager,
            logger=self.logger,
            min_workers=self.autoscaler_config.get('min_workers', 0),
            max_workers=self.max_workers,
            target_gpu_utilization=self.autoscaler_config.get('target_gpu_utilization', 0.7),
            scale_up_cooldown_seconds=self.autoscaler_config.get('scale_up_cooldown', 60),
            scale_down_cooldown_seconds=self.autoscaler_config.get('scale_down_cooldown', 300),
            gpus_per_worker=self.gpus_per_worker
        )
        self.autoscale_enabled = autoscale

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
        gpus_per_worker: int = 1,
        worker_config: Optional[Dict] = None,
        autoscaler_config: Optional[Dict] = None,
    ) -> None:
        """Create worker configuration file

        Args:
            dataset_paths: List of dataset directories to include
            max_gpus: Maximum number of GPUs available
            trusted_models: List of trusted Docker/SIF images
            server_url: Hypha server URL
            config_path: Path to save the config file
            gpus_per_worker: Number of GPUs per worker
            worker_config: Configuration for worker jobs
            autoscaler_config: Configuration for autoscaler
        """
        config = {
            "machine_name": socket.gethostname(),
            "max_gpus": max_gpus,
            "gpus_per_worker": gpus_per_worker,
            "worker_config": worker_config or {
                "num_cpus": 8,
                "mem_per_cpu": 16,
                "time_limit": "8:00:00"
            },
            "autoscaler": autoscaler_config or {
                "min_workers": 0,
                "target_gpu_utilization": 0.7,
                "scale_up_cooldown": 60,
                "scale_down_cooldown": 300
            },
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
            # Ensure Ray is running
            ray_status = self.ray_manager.check_cluster()
            if not ray_status["head_running"]:
                self.logger.info("Starting Ray cluster...")
                ray_result = self.ray_manager.start_cluster()
                if not ray_result["success"]:
                    raise RuntimeError(f"Failed to start Ray cluster: {ray_result['message']}")
                self.logger.info("Ray cluster started successfully")

            # Initialize deployment manager if not already initialized
            if self.deployment_manager is None:
                self.logger.info("Initializing Ray deployment manager...")
                self.deployment_manager = RayDeploymentManager(
                    logger=self.logger,
                    deployment_collection_id="ray-deployments",
                    service_id="ray-model-services",
                )

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
                    # Autoscaling methods
                    "start_autoscaling": self.start_autoscaling,
                    "stop_autoscaling": self.stop_autoscaling,
                    "get_autoscaler_status": self.get_autoscaler_status,
                    "configure_autoscaler": self.configure_autoscaler,
                }
            )

            # Initialize the deployment manager with the server connection
            await self.deployment_manager.initialize(self.server)

            # Start autoscaler if enabled in config
            if self.autoscale_enabled:
                await self.start_autoscaling()

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

    # Add these autoscaler-related methods
    async def start_autoscaling(self, min_workers: int = 0, max_workers: int = 4, context=None) -> Dict:
        """Start the Ray cluster autoscaler.
        
        Args:
            min_workers: Minimum number of worker nodes to maintain
            max_workers: Maximum number of worker nodes to allow
            context: RPC context
            
        Returns:
            Dict with operation status
        """
        try:
            # Update configuration if provided
            self.autoscaler.min_workers = min_workers
            self.autoscaler.max_workers = max_workers
            
            # Start the autoscaler
            await self.autoscaler.start()
            self.autoscale_enabled = True
            
            return {
                "success": True, 
                "message": f"Autoscaler started with min_workers={min_workers}, max_workers={max_workers}"
            }
        except Exception as e:
            self.logger.error(f"Error starting autoscaler: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def stop_autoscaling(self, context=None) -> Dict:
        """Stop the Ray cluster autoscaler.
        
        Args:
            context: RPC context
            
        Returns:
            Dict with operation status
        """
        try:
            # Stop the autoscaler
            await self.autoscaler.stop()
            self.autoscale_enabled = False
            
            return {"success": True, "message": "Autoscaler stopped"}
        except Exception as e:
            self.logger.error(f"Error stopping autoscaler: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def get_autoscaler_status(self, context=None) -> Dict:
        """Get the current status of the autoscaler.
        
        Args:
            context: RPC context
            
        Returns:
            Dict with autoscaler status
        """
        try:
            status = self.autoscaler.get_autoscaler_status()
            status["success"] = True
            return status
        except Exception as e:
            self.logger.error(f"Error getting autoscaler status: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def configure_autoscaler(
        self, 
        min_workers: Optional[int] = None,
        max_workers: Optional[int] = None, 
        target_gpu_utilization: Optional[float] = None, 
        target_cpu_utilization: Optional[float] = None,
        scale_up_cooldown_seconds: Optional[int] = None,
        scale_down_cooldown_seconds: Optional[int] = None,
        context=None
    ) -> Dict:
        """Configure the Ray cluster autoscaler.
        
        Args:
            min_workers: Minimum number of worker nodes
            max_workers: Maximum number of worker nodes
            target_gpu_utilization: Target GPU utilization (0-1)
            target_cpu_utilization: Target CPU utilization (0-1)
            scale_up_cooldown_seconds: Cooldown period between scale up operations
            scale_down_cooldown_seconds: Cooldown period between scale down operations
            context: RPC context
            
        Returns:
            Dict with operation status
        """
        try:
            # Update configuration for provided parameters
            if min_workers is not None:
                self.autoscaler.min_workers = min_workers
            if max_workers is not None:
                self.autoscaler.max_workers = max_workers
            if target_gpu_utilization is not None:
                self.autoscaler.target_gpu_utilization = max(0.0, min(1.0, target_gpu_utilization))
            if target_cpu_utilization is not None:
                self.autoscaler.target_cpu_utilization = max(0.0, min(1.0, target_cpu_utilization))
            if scale_up_cooldown_seconds is not None:
                self.autoscaler.scale_up_cooldown = scale_up_cooldown_seconds
            if scale_down_cooldown_seconds is not None:
                self.autoscaler.scale_down_cooldown = scale_down_cooldown_seconds
            
            # Get updated status
            status = self.autoscaler.get_autoscaler_status()
            status["success"] = True
            status["message"] = "Autoscaler configuration updated"
            
            return status
        except Exception as e:
            self.logger.error(f"Error configuring autoscaler: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}

    # Cleanup methods
    async def cleanup(self) -> Dict:
        """Clean up resources on shutdown

        Returns:
            Dict containing cleanup status information
        """
        self.logger.info("Cleaning up before shutdown...")
        results = {}

        # Stop the autoscaler if running
        if self.autoscale_enabled:
            try:
                self.logger.info("Stopping autoscaler")
                await self.autoscaler.stop()
                results["autoscaler"] = {"success": True, "message": "Autoscaler stopped"}
            except Exception as e:
                self.logger.error(f"Error stopping autoscaler: {str(e)}")
                results["autoscaler"] = {"success": False, "message": str(e)}

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
    """Test the HpcWorker class functionality including autoscaling."""
    import asyncio

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hpc_worker")

    print("===== Testing HPC Worker class =====", end="\n\n")

    # Create HPC worker instance with direct parameters
    print("Creating HPC worker instance with autoscaling...")
    worker = HpcWorker(
        server_url="https://hypha.aicell.io",
        num_gpu=4,
        dataset_paths=["/proj/aicell/users/x_nilme/tabula/tabula/resource/demo/blood"],
        trusted_models=["ghcr.io/aicell-lab/tabula:0.1.1"],
        autoscale=True,
        gpus_per_worker=1,
        worker_config={
            "num_cpus": 8,
            "mem_per_cpu": 16,
            "time_limit": "1:00:00"
        },
        autoscaler_config={
            "min_workers": 0,
            "target_gpu_utilization": 0.7,
            "scale_up_cooldown": 5,
            "scale_down_cooldown": 30
        }
    )
    print("Worker created with direct parameters", end="\n\n")

    async def run_tests():
        try:
            # Test get_worker_status
            print("Getting worker status...")
            status = worker.get_worker_status()
            print(f"Worker status: {status}", end="\n\n")

            # Start Ray cluster
            print("Starting Ray cluster...")
            ray_status = worker.ray_manager.check_cluster()
            if not ray_status["head_running"]:
                start_result = worker.ray_manager.start_cluster()
                print(f"Start result: {start_result}", end="\n\n")
            else:
                print("Ray cluster already running", end="\n\n")

            # Start autoscaler
            print("Starting autoscaler...")
            autoscale_result = await worker.start_autoscaling()
            print(f"Autoscaler start result: {autoscale_result}", end="\n\n")

            # Test autoscaler status
            print("Getting autoscaler status...")
            status = worker.get_autoscaler_status()
            print(f"Initial autoscaler status: {status}", end="\n\n")

            # Submit some test jobs to trigger autoscaling
            print("Submitting test jobs to trigger autoscaling...")
            for i in range(3):
                job_result = worker.ray_manager.submit_worker_job()
                print(f"Job {i+1} submission result: {job_result}")

            # Monitor autoscaling for a while
            print("\nMonitoring autoscaler for 60 seconds...")
            for i in range(6):
                await asyncio.sleep(10)
                status = worker.get_autoscaler_status()
                print(f"\nAutoscaler status at {i*10}s:")
                print(f"Workers: {status['current_metrics']['worker_count']}")
                print(f"GPU Utilization: {status['current_metrics'].get('gpu', 0):.2f}")
                print(f"Pending workers: {status['scaling_status']['pending_workers']}")

            # Stop autoscaler
            print("\nStopping autoscaler...")
            stop_result = await worker.stop_autoscaling()
            print(f"Autoscaler stop result: {stop_result}", end="\n\n")

            # Cancel any remaining jobs
            print("Cancelling worker jobs...")
            cancel_result = worker.ray_manager.cancel_worker_jobs()
            print(f"Cancel result: {cancel_result}", end="\n\n")

            # Final cleanup
            print("Running final cleanup...")
            cleanup_result = await worker.cleanup()
            print(f"Cleanup result: {cleanup_result}", end="\n\n")

        except Exception as e:
            print(f"Error during tests: {str(e)}")
            # Ensure cleanup runs even if tests fail
            await worker.cleanup()

    # Run the tests
    print("Starting tests...")
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_tests())
    finally:
        loop.close()
    print("===== HPC Worker class tests completed =====")

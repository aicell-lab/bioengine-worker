import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import ray
from hypha_rpc import connect_to_server

from bioengine_worker.ray_autoscaler import RayAutoscaler
from bioengine_worker.ray_cluster_manager import RayClusterManager
from bioengine_worker.ray_deployment_manager import RayDeploymentManager
from bioengine_worker.utils.format_time import format_time
from bioengine_worker.utils.logger import create_logger, logging_format


class BioEngineWorker:
    """Manages Ray cluster lifecycle and model deployments on HPC systems or pre-existing Ray environments.

    Provides a Hypha service interface for controlling Ray cluster operations,
    autoscaling, and model deployments through Ray Serve.
    """

    def __init__(
        self,
        workspace: str,
        server_url: str = "https://hypha.aicell.io",
        token: Optional[str] = None,
        service_id: str = "bioengine-worker",
        logger: Optional[logging.Logger] = None,
        ray_cluster_config: Optional[Dict] = None,
        clean_up_previous_cluster: bool = True,
        ray_autoscaler_config: Optional[Dict] = None,
        ray_deployment_config: Optional[Dict] = None,
        ray_connection_kwargs: Optional[Dict] = None,
    ):
        """Initialize BioEngine worker with component managers.

        Args:
            workspace: Hypha workspace to connect to.
            server_url: URL of the Hypha server to register the worker with.
            token: Optional authentication token for the Hypha server.
            service_id: Service ID used when registering with the Hypha server.
            logger: Optional custom logger. If not provided, a default logger will be created.
            ray_cluster_config: Configuration for RayClusterManager component.
            clean_up_previous_cluster: Flag to indicate whether to cleanup of previous Ray cluster.
            ray_autoscaler_config: Configuration for the RayAutoscaler component.
            ray_deployment_config: Configuration for the RayDeploymentManager component.
            ray_connection_kwargs: Optional arguments passed to `ray.init()` to connect to an existing ray cluster. If provided, disables cluster management.
        """
        self.workspace = workspace
        self.server_url = server_url
        self._token = token
        self.service_id = service_id
        self.logger = logger or create_logger("BioEngineWorker")
        self.start_time = time.time()

        ray_cluster_config = ray_cluster_config or {}
        self._clean_up = clean_up_previous_cluster
        self._ray_autoscaler_config = ray_autoscaler_config or {}
        ray_deployment_config = ray_deployment_config or {}
        self.ray_connection_kwargs = ray_connection_kwargs or {}

        # Inject default logging format if not already set
        if "logging_format" not in self.ray_connection_kwargs:
            self.ray_connection_kwargs["logging_format"] = logging_format
        else:
            self.logger.warning(
                f"Overriding default Ray logging_format. Provided format: {self.ray_connection_kwargs['logging_format']!r}"
            )

        # Initialize component managers
        if "cluster_manager" in self._ray_autoscaler_config:
            raise ValueError(
                "RayAutoscaler config should not contain 'cluster_manager' key."
            )
        if "autoscaler" in ray_deployment_config:
            raise ValueError(
                "RayDeploymentManager config should not contain 'autoscaler' key."
            )

        manage_cluster = not bool(ray_connection_kwargs)
        self.cluster_manager = (
            RayClusterManager(**ray_cluster_config) if manage_cluster else None
        )
        self.autoscaler = None
        self.deployment_manager = RayDeploymentManager(**ray_deployment_config)

        # Internal state
        self.server = None

    async def _connect_to_server(self) -> None:
        """Connect to Hypha server using provided URL.

        Args:
            server_url: URL of the Hypha server
            token: Token for authentication
            workspace: Workspace to connect to
        """
        self.logger.info(f"Connecting to Hypha server at {self.server_url}")
        self.server = await connect_to_server(
            {
                "server_url": self.server_url,
                "token": self._token,
                "workspace": self.workspace,
            }
        )
        if not self.server:
            raise ValueError("Failed to connect to Hypha server")
        self.logger.info(
            f"Connected to workspace '{self.workspace}' with client ID '{self.server.config.client_id}'"
        )

    async def _register(self) -> None:
        """Initialize connection to Hypha and register service interface.

        Args:
            server: Hypha server connection

        Returns:
            bool: True if initialization successful
        """
        # Register service interface
        service_info = await self.server.register_service(
            {
                "id": self.service_id,
                "name": "BioEngine worker",
                "description": "Controls Ray cluster on HPC system",
                "config": {"visibility": "public", "require_context": True},
                "deploy_artifact": self.deployment_manager.deploy_artifact,
                "undeploy_artifact": self.deployment_manager.undeploy_artifact,
                "get_deployment_name": self.deployment_manager.get_deployment_name,
                "get_status": self.get_status,
                "get_deployment_service_id": lambda context: self.deployment_manager.deployment_service_id,
            },
            {"overwrite": True},
        )

        self.logger.info(
            f"Successfully registered BioEngine worker service: {service_info.id}"
        )

        server_url = self.server.config.public_base_url
        workspace, sid = service_info.id.split("/")
        service_url = f"{server_url}/{workspace}/services/{sid}"
        self.logger.info(f"Service available at: {service_url}")
        self.logger.info(f"Get worker status at: {service_url}/get_status")
        self.logger.info(
            f"Get deployment service ID at: {service_url}/get_deployment_service_id"
        )

        return service_info.id

    async def start(self) -> str:
        """Start the BioEngine worker by initializing the Ray cluster or attaching to an existing one,
        connecting to the Hypha server, and initializing the deployment manager.

        Returns:
            The service ID assigned after successful registration with Hypha.
        """
        if self.cluster_manager:
            # Start the Ray cluster
            self.cluster_manager.start_cluster(force_clean_up=self._clean_up)

            # If running on a HPC system, use the RayAutoscaler to manage the Ray cluster
            if self.cluster_manager.slurm_available:
                self.autoscaler = RayAutoscaler(
                    cluster_manager=self.cluster_manager,
                    **self._ray_autoscaler_config,
                )
                await self.autoscaler.start()
                self.deployment_manager.autoscaler = self.autoscaler
        else:
            # Connect to an existing Ray cluster
            if ray.is_initialized():
                raise RuntimeError(
                    "Ray is already initialized. Please stop the existing Ray instance before starting the worker."
                )
            try:
                ray.init(**self.ray_connection_kwargs)
            except Exception as e:
                self.logger.error(f"Failed to connect to existing Ray cluster: {e}")
                raise
            self.logger.info("Connected to existing Ray cluster.")

        # Connect to the Hypha server and register the service
        await self._connect_to_server()
        await self.deployment_manager.initialize(self.server)
        sid = await self._register()
        return sid

    async def serve(self) -> None:
        """Keep the BioEngine worker running and serving requests."""
        if not self.server:
            raise RuntimeError("Server not initialized. Call start() first.")
        await self.server.serve()

    async def get_status(self, context: Optional[Dict[str, Any]] = None) -> Dict:
        """Get comprehensive status of the Ray cluster or connected Ray instance.

        Returns:
            Dict containing service, cluster, autoscaler and deployment status.
        """
        formatted_service_time = format_time(self.start_time)
        status = {
            "service": {
                "start_time": formatted_service_time["start_time"],
                "uptime": formatted_service_time["duration_since"],
            }
        }
        if ray.is_initialized():
            if self.cluster_manager:
                # Ray started via autoscaler
                formatted_ray_time = format_time(self.cluster_manager.ray_start_time)
                status["cluster"] = {
                    "head_address": self.cluster_manager.head_node_address,
                    "start_time": formatted_ray_time["start_time"],
                    "uptime": formatted_ray_time["duration_since"],
                }
                if self.autoscaler:
                    status["cluster"]["autoscaler"] = self.autoscaler.get_status()
                else:
                    status["cluster"]["autoscaler"] = "N/A"
                    status["cluster"][
                        "note"
                    ] = "Autoscaler is only available on HPC systems."
            else:
                # Ray connected externally
                ray_info = ray._private.services.get_node_ip_address()
                status["cluster"] = {
                    "address": ray_info,
                    "start_time": "N/A",
                    "uptime": "N/A",
                    "autoscaler": "N/A",
                    "note": "Connected to existing Ray cluster; no autoscaler info available.",
                }
            status["deployments"] = self.deployment_manager.get_deployments()
        else:
            status["cluster"] = "Not running"

        return status

    async def cleanup(self) -> None:
        """Clean up resources and stop the Ray cluster if managed by this worker."""
        self.logger.info("Cleaning up resources...")

        await self.deployment_manager.cleanup_deployments()

        if self.autoscaler:
            await self.autoscaler.shutdown_cluster()
        elif self.cluster_manager:
            self.cluster_manager.shutdown_cluster()

        self.logger.info("Worker cleanup complete.")


if __name__ == "__main__":
    """Test the BioEngineWorker class functionality"""
    import os
    from pathlib import Path

    from hypha_rpc import login

    from bioengine_worker.ray_deployment_manager import \
        create_example_deployment

    async def test_bioengine_worker(keep_running=True):
        try:
            # Create BioEngine worker instance
            server_url = "https://hypha.aicell.io"
            token = os.environ["HYPHA_TOKEN"] or await login({"server_url": server_url})
            bioengine_worker = BioEngineWorker(
                workspace="chiron-platform",
                server_url=server_url,
                token=token,
                service_id="bioengine-worker-test",
                ray_cluster_config={
                    "head_num_cpus": 4,
                    "temp_dir": str(Path(__file__).parent.parent / "ray_sessions"),
                },
                ray_autoscaler_config={
                    "metrics_interval_seconds": 10,
                    "temp_dir": "/tmp/ray",
                    "data_dir": str(Path(__file__).parent.parent / "data"),
                    "container_image": "/proj/aicell/users/x_nilme/autoscaler/tabula_0.1.1.sif",
                },
            )
            bioengine_worker.logger.setLevel(logging.DEBUG)

            # Initialize worker
            sid = await bioengine_worker.start()

            # Test registered service
            server = await connect_to_server(
                {
                    "server_url": server_url,
                    "token": token,
                    "workspace": bioengine_worker.workspace,
                }
            )
            worker_service = await server.get_service(sid)

            # Get initial status
            status = await worker_service.get_status()
            print("\nInitial status:", status)

            artifact_id = await create_example_deployment(
                bioengine_worker.deployment_manager.artifact_manager
            )
            # Test deploying an artifact
            deployment_name = await worker_service.deploy_artifact(
                artifact_id=artifact_id,
            )

            # Test registered deployment service
            deployment_service_id = await worker_service.get_deployment_service_id()
            deployment_service = await server.get_service(deployment_service_id)

            # Get the list of deployments
            deployments = await deployment_service.list_deployments()
            assert deployment_name in deployments

            # Keep server running if requested
            if keep_running:
                print("Server running. Press Ctrl+C to stop.")
                while True:
                    await asyncio.sleep(1)

        except Exception as e:
            print(f"Test error: {e}")
            raise e
        finally:
            # Cleanup
            await bioengine_worker.cleanup()

    # Run the test
    asyncio.run(test_bioengine_worker())

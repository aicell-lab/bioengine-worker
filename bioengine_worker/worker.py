import asyncio
import logging
import os
import time
from typing import Optional, Dict

from hypha_rpc import connect_to_server
import ray

from bioengine_worker.ray_autoscaler import RayAutoscaler
from bioengine_worker.ray_deployment_manager import RayDeploymentManager
from bioengine_worker.utils.logger import create_logger, logging_format
from bioengine_worker.utils.format_time import format_time


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
        ray_autoscaler_config: Optional[Dict] = None,
        ray_deployment_manager_config: Optional[Dict] = None,
        manage_cluster: bool = True,
        ray_init_kwargs: Optional[Dict] = None,
    ):
        """Initialize BioEngine worker with component managers.

        Args:
            workspace: Path to the workspace directory used for cluster logs and state.
            server_url: URL of the Hypha server to register the worker with.
            token: Optional authentication token for the Hypha server.
            service_id: Service ID used when registering with the Hypha server.
            logger: Optional custom logger. If not provided, a default logger will be created.
            ray_cluster_config: Configuration dictionary for initializing the Ray cluster.
            ray_autoscaler_config: Configuration for the RayAutoscaler component.
            ray_deployment_manager_config: Configuration for model deployment manager.
            manage_cluster: If True, manages the cluster lifecycle via RayAutoscaler. 
                            If False, assumes Ray is already running and connects to it.
            ray_init_kwargs: Optional arguments passed to `ray.init()` if `manage_cluster` is False.
        """
        self.workspace = workspace
        self.server_url = server_url
        self.token = token
        self.service_id = service_id
        self.logger = logger or create_logger("BioEngineWorker")
        self.start_time = time.time()

        ray_cluster_config = ray_cluster_config or {}
        ray_autoscaler_config = ray_autoscaler_config or {}
        ray_deployment_manager_config = ray_deployment_manager_config or {}
        ray_init_kwargs = ray_init_kwargs or {}

        # Inject default logging format if not already set
        if "logging_format" not in ray_init_kwargs:
            ray_init_kwargs["logging_format"] = logging_format
        else:
            self.logger.warning(
                f"Overriding default Ray logging_format. Provided format: {ray_init_kwargs['logging_format']!r}"
            )

        self.ray_init_kwargs = ray_init_kwargs
        self.manage_cluster = manage_cluster

        # Initialize component managers
        self.autoscaler = (
            RayAutoscaler(**ray_autoscaler_config, **ray_cluster_config)
            if self.manage_cluster else None
        )
        self.deployment_manager = RayDeploymentManager(
            **ray_deployment_manager_config,
            autoscaler=self.autoscaler
        )

        # Internal state
        self.server = None
        self.ray_start_time = None

    async def start(self):
        """Start the BioEngine worker by initializing the Ray cluster or attaching to an existing one,
        connecting to the Hypha server, and initializing the deployment manager.
        
        Returns:
            The service ID assigned after successful registration with Hypha.
        """
        if self.manage_cluster:
            await self.autoscaler.start()
        else:
            if not ray.is_initialized():
                self.logger.info("Connecting to existing Ray cluster with provided init kwargs...")
                ray.init(**self.ray_init_kwargs)
                self.logger.info("Connected to Ray cluster.")

        await self._connect_to_server()
        await self.deployment_manager.initialize(self.server)
        sid = await self._register()
        self.logger.info(f"BioEngine worker started and registered with Hypha service: {sid}")
        return sid

    async def _connect_to_server(self) -> None:
        """Connect to Hypha server using provided URL.
        
        Args:
            server_url: URL of the Hypha server
            token: Token for authentication
            workspace: Workspace to connect to
        """
        self.server = await connect_to_server({"server_url": self.server_url, "token": self.token, "workspace": self.workspace})
        if not self.server:
            raise ValueError("Failed to connect to Hypha server")
        self.logger.info(f"Connected to Hypha server at {self.server_url}")
        self.logger.info(f"Using workspace: {self.workspace}")

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
                "get_status": self.get_status,
            },
            {"overwrite": True},
        )

        self.logger.info(
            f"Successfully registered BioEngine worker service: {service_info.id}"
        )
        return service_info.id

    async def get_status(self, context=None) -> Dict:
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
            if self.autoscaler:
                # Ray started via autoscaler
                formatted_ray_time = format_time(self.autoscaler.cluster_manager.ray_start_time)
                status["cluster"] = {
                    "address": self.autoscaler.cluster_manager.head_node_address,
                    "start_time": formatted_ray_time["start_time"],
                    "uptime": formatted_ray_time["duration_since"],
                }
                status["autoscaler"] = self.autoscaler.get_status()
            else:
                # Ray connected externally
                ray_info = ray._private.services.get_node_ip_address()
                status["cluster"] = {
                    "address": ray_info,
                    "start_time": "N/A",
                    "uptime": "N/A",
                    "note": "Connected to existing Ray cluster; no autoscaler info available.",
                }
            status["deployments"] = self.deployment_manager.get_status()
        else:
            status["cluster"] = "Not running"

        return status




if __name__ == "__main__":
    """Test the BioEngineWorker class functionality"""
    import os
    from hypha_rpc import connect_to_server, login
    from bioengine_worker.ray_deployment_manager import create_example_deployment
    async def test_bioengine_worker(manage_cluster=False, keep_running=True):
        try:
            # Create BioEngine worker instance
            server_url="https://hypha.aicell.io"
            token = os.environ["HYPHA_TOKEN"] or await login({"server_url": server_url})
            bioengine_worker = BioEngineWorker(
                workspace="chiron-platform",
                server_url=server_url,
                token=token,
                service_id="bioengine-worker-test",
                manage_cluster=manage_cluster,
                ray_autoscaler_config={
                    "metrics_interval_seconds": 10,
                    "temp_dir": "/tmp/ray",
                    "data_dir": os.path.dirname(__file__),
                    "container_image": "/proj/aicell/users/x_nilme/autoscaler/tabula_0.1.1.sif",
                }
            )
            bioengine_worker.logger.setLevel(logging.DEBUG)

            # Initialize worker
            sid = await bioengine_worker.start()

            # Test service registration
            server = await connect_to_server(
                {"server_url": server_url, "token": token, "workspace": bioengine_worker.workspace}
            )
            service = await server.get_service(sid)

            # Get initial status
            status = await service.get_status()
            print("\nInitial status:", status)

            artifact_id = await create_example_deployment(
                bioengine_worker.deployment_manager.artifact_manager
            )
            # Test deployment
            await service.deploy_artifact(
                artifact_id=artifact_id,
            )
            
            # Test registered Hypha service
            service_info = await server.get_service(bioengine_worker.deployment_manager.service_id)
            deployment_name = bioengine_worker.deployment_manager._get_deployment_name(artifact_id)

            # Get the list of deployments
            deployments = await service_info.list_deployments()
            assert deployment_name in deployments

            # Keep server running if requested
            if keep_running:
                print("Server running. Press Ctrl+C to stop.")
                while True:
                    await asyncio.sleep(1)

        except Exception as e:
            print(f"Test error: {e}")
            raise e
        
    # Run the test
    asyncio.run(test_bioengine_worker())

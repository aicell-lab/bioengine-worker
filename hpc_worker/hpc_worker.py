import asyncio
import logging
import time
from typing import Optional, Dict

from hpc_worker.ray_autoscaler import RayAutoscaler
from hpc_worker.ray_deployment_manager import RayDeploymentManager
from hpc_worker.utils.logger import create_logger
from hpc_worker.utils.format_time import format_time


class HpcWorker:
    """Manages Ray cluster lifecycle and model deployments on HPC systems.
    
    Provides a Hypha service interface for controlling Ray cluster operations,
    autoscaling, and model deployments through Ray Serve.
    """

    def __init__(
        self,
        workspace: str,
        server_url: str = "https://hypha.aicell.io",
        token: Optional[str] = None,
        service_id: str = "hpc-worker",
        logger: Optional[logging.Logger] = None,
        ray_cluster_config: Optional[Dict] = None,
        ray_autoscaler_config: Optional[Dict] = None,
        ray_deployment_manager_config: Optional[Dict] = None,
    ):
        """Initialize HPC worker with component managers.
        
        Args:
            workspace: Workspace for the HPC worker
            server_url: URL for the Hypha server
            service_id: ID for the Hypha service
            logger: Optional logger instance
        """
        self.workspace = workspace
        self.server_url = server_url
        self.token = token
        self.service_id = service_id
        self.logger = logger or create_logger("HpcWorker")
        self.start_time = time.time()

        ray_cluster_config = ray_cluster_config or {}
        ray_autoscaler_config = ray_autoscaler_config or {}
        ray_deployment_manager_config = ray_deployment_manager_config or {}

        # Initialize component managers
        self.autoscaler = RayAutoscaler(**ray_autoscaler_config, **ray_cluster_config)
        self.deployment_manager = RayDeploymentManager(**ray_deployment_manager_config, autoscaler=self.autoscaler)

        # Initialize state
        self.server = None
        self.ray_start_time = None

    async def start(self):
        await self.autoscaler.start()
        await self._connect_to_server()
        await self.deployment_manager.initialize(self.server)
        sid = await self._register()
        self.logger.info(f"HPC Worker started and registered with Hypha service: {sid}")

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
                "name": "HPC Worker",
                "description": "Controls Ray cluster on HPC system",
                "config": {"visibility": "public", "require_context": True},
                "deploy_artifact": self.deployment_manager.deploy_artifact,
                "undeploy_artifact": self.deployment_manager.undeploy_artifact,
                "get_status": self.get_status,
            },
            {"overwrite": True},
        )

        self.logger.info(
            f"Successfully registered HPC Worker service: {service_info.id}"
        )
        return service_info.id

    async def get_status(self, context=None) -> Dict:
        """Get comprehensive status of the Ray cluster.
        
        Returns:
            Dict containing service, cluster, autoscaler and deployment status
        """
        # Get status from all components
        formatted_service_time = format_time(self.start_time)
        formatted_ray_time = format_time(self.autoscaler.cluster_manager.ray_start_time)
        status = {
            "service": {
                "start_time": formatted_service_time["start_time"],
                "uptime": formatted_service_time["duration_since"],
            }
        }
        if self.autoscaler.cluster_manager.head_node_address:
            status["cluster"] = {
                "address": self.autoscaler.cluster_manager.head_node_address,
                "start_time": formatted_ray_time["start_time"],
                "uptime": formatted_ray_time["duration_since"],
            }
            status["autoscaler"] = self.autoscaler.get_status()
            status["deployments"] = self.deployment_manager.get_status()
        else:
            status["cluster"] = "Not running"

        return status



if __name__ == "__main__":
    """Test the HpcWorker class functionality"""
    import os
    from hypha_rpc import connect_to_server, login
    from hpc_worker.ray_deployment_manager import create_example_deployment
    async def test_hpc_worker():
        try:
            # Create HPC worker instance
            server_url="https://hypha.aicell.io"
            token = os.environ["HYPHA_TOKEN"] or await login({"server_url": server_url})
            hpc_worker = HpcWorker(
                workspace="ws-user-github|49943582",
                server_url=server_url,
                token=token,
                service_id="hpc-worker-test",
                ray_autoscaler_config={
                    "metrics_interval_seconds": 10,
                    "temp_dir": "/proj/aicell/ray_tmp",
                    "data_dir": os.path.dirname(__file__),
                    "container_image": "/proj/aicell/users/x_nilme/autoscaler/tabula_0.1.1.sif",
                }
            )
            hpc_worker.logger.setLevel(logging.DEBUG)

            # Initialize worker
            sid = await hpc_worker.start()

            # Test service registration
            server = await connect_to_server(
                {"server_url": server_url, "token": token, "workspace": hpc_worker.workspace}
            )
            service = await server.get_service(sid)

            # Get initial status
            status = await service.get_status()
            print("\nInitial status:", status)

            artifact_id = await create_example_deployment(
                hpc_worker.deployment_manager.artifact_manager
            )
            # Test deployment
            await service.deploy_artifact(
                artifact_id=artifact_id,
            )
            
            # Test registered Hypha service
            service_info = await server.get_service(hpc_worker.deployment_manager.service_id)
            deployment_name = hpc_worker.deployment_manager._get_deployment_name(artifact_id)

            # Get the list of deployments
            deployments = await service_info.list_deployments()
            assert deployment_name in deployments

        except Exception as e:
            print(f"Test error: {e}")
            raise e
        
    # Run the test
    asyncio.run(test_hpc_worker())

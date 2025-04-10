import logging
import os
import ray
from ray import serve
import httpx
from typing import Dict, Optional, Any, List
from functools import partial
import asyncio
import yaml

from bioengine_worker.utils.logger import create_logger
from bioengine_worker.ray_autoscaler import RayAutoscaler
from bioengine_worker.utils.format_time import format_time


class RayDeploymentManager:
    """Manages Ray Serve deployments using Hypha artifact manager

    This class integrates with Hypha artifact manager to deploy
    artifacts from Hypha as Ray Serve deployments.
    """

    def __init__(
        self,
        deployment_collection_id: str = "ray-deployments",
        service_id: str = "ray-model-services",
        autoscaler: Optional[RayAutoscaler] = None,
        # Logger
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the Ray Deployment Manager

        Args:
            deployment_collection_id: Artifact collection ID for deployments
            service_id: ID to use for the Hypha service exposing deployed models
            autoscaler: Optional RayAutoscaler instance
            logger: Optional logger instance
        """
        # Store parameters
        self.deployment_collection_id = deployment_collection_id
        self.service_id = service_id
        self.autoscaler = autoscaler

        # Initialize state variables
        self.server = None
        self.artifact_manager = None
        self.service_info = None

        # Set up logging
        self.logger = logger or create_logger("RayDeploymentManager")

    @property
    def deployments(self) -> Dict:
        """Get a dictionary of currently deployed models"""
        output = {}
        if not ray.is_initialized():
            self.logger.error("Can not get deployments - Ray cluster is not running")
            raise RuntimeError("Ray cluster is not running")

        serve_status = ray.serve.status()
        for application_name, application in serve_status.applications.items():
            formatted_time = format_time(application.last_deployed_time_s)
            output[application_name] = {
                "status": application.status.value,
                "last_deployed_at": formatted_time["start_time"],
                "duration_since": formatted_time["duration_since"],
            }
            deployments = application.deployments
            for name, deployment in deployments.items():
                output[application_name][name] = {
                    "status": deployment.status.value,
                    "replica_states": deployment.replica_states,
                }
        self.logger.debug(
            f"Current deployments: {list(serve_status.applications.keys())}"
        )
        return output

    async def initialize(self, server) -> None:
        """Initialize the deployment manager with a Hypha server connection

        Args:
            server: Hypha server connection
        """
        try:
            # Store server connection
            self.server = server

            # Get artifact manager service
            self.artifact_manager = await self.server.get_service(
                "public/artifact-manager"
            )
            self.logger.info("Successfully connected to artifact manager")

        except Exception as e:
            self.logger.error(f"Error initializing Ray Deployment Manager: {e}")
            self.server = None
            self.artifact_manager = None

            raise e

    def _get_deployment_name(self, artifact_id: str) -> str:
        """Convert artifact ID to a deployment name

        Args:
            artifact_id: The artifact ID to convert

        Returns:
            str: The converted deployment name
        """
        try:
            return artifact_id.split("/")[1].replace("-", "_")
        except IndexError:
            return artifact_id.replace("-", "_")

    async def _load_deployment_code(
        self,
        class_name: str,
        artifact_id: str,
        version=None,
        file_path: str = "main.py",
        timeout: int = 30,
    ) -> Any:
        """Load and execute deployment code from an artifact directly in memory

        Args:
            class_name: Name of the class to load from the deployment code
            artifact_id: ID of the artifact
            version: Optional version of the artifact
            file_path: Path to the file within the artifact (default: main.py)
            timeout: Timeout in seconds for network requests (default: 30)

        Returns:
            Any: The loaded class or None if not found
        """
        try:
            # Ensure artifact manager is available
            if not self.artifact_manager:
                raise RuntimeError(
                    "Artifact manager not initialized. Call initialize() first."
                )

            # Get download URL for the file
            download_url = await self.artifact_manager.get_file(
                artifact_id=artifact_id, version=version, file_path=file_path
            )

            # Download the file content with timeout
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(download_url)
                response.raise_for_status()
                code_content = response.text

            # Create a restricted globals dictionary for sandboxed execution
            safe_globals = {}
            # Execute the code in a sandboxed environment
            exec(code_content, safe_globals)
            if class_name not in safe_globals:
                self.logger.error(f"{class_name} not found in {artifact_id}")
                return None

            model = safe_globals.get(class_name)
            if model:
                self.logger.info(f"Class '{class_name}' loaded from {artifact_id}")
                return model

        except Exception as e:
            self.logger.error(f"Error loading deployment code for {artifact_id}: {e}")
            raise e

    async def deploy_artifact(
        self,
        artifact_id: str,
        version=None,
        skip_update=False,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Deploy a single artifact to Ray Serve

        Args:
            artifact_id: ID of the artifact to deploy
            version: Optional version of the artifact
            skip_update: Skip updating services after deployment
            context: Context for Hypha service
        """
        try:
            # Verify client is connected to Hypha server
            if not self.server:
                raise RuntimeError("Hypha server connection not available")

            # Verify Ray is initialized
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            # Read the manifest to get deployment configuration
            artifact = await self.artifact_manager.read(artifact_id, version=version)
            manifest = artifact.get("manifest")
            if not manifest:
                raise ValueError(f"Manifest not found for artifact '{artifact_id}'")

            deployment_config = manifest.get("deployment_config")
            if not deployment_config:
                raise ValueError(
                    f"Deployment configuration not found for artifact '{artifact_id}'"
                )
            deployment_name = self._get_deployment_name(artifact_id)

            # Load the deployment code
            class_name = manifest.get("class_name")
            if not class_name:
                raise ValueError(f"Class name not found for artifact '{artifact_id}'")
            deployment_config["name"] = class_name

            model = await self._load_deployment_code(
                class_name,
                artifact_id,
                version=version,
                file_path=manifest.get("entry_point", "main.py"),
            )
            if not model:
                raise ValueError(
                    f"Failed to load model class '{class_name}' from artifact {artifact_id}"
                )

            # Create the Ray Serve deployment
            model_deployment = serve.deployment(**deployment_config)(model)

            # Bind the arguments to the deployment and return an Application
            app = model_deployment.bind()

            # Deploy the application in a separate thread to avoid blocking
            task = asyncio.to_thread(
                serve.run, app, name=deployment_name, route_prefix=None
            )
            if self.autoscaler:
                # Wait for the deployment to be ready
                await asyncio.sleep(1)
                # Notify the autoscaler of the new deployment
                await self.autoscaler.notify()
                # TODO: notifying works, but the application will only be deployed one awaited

            # Wait for the deployment to complete
            await task

            self.logger.info(f"Successfully deployed {artifact_id}")

            if not skip_update:
                await self._update_services()

        except Exception as e:
            self.logger.error(f"Error deploying '{artifact_id}': {e}")
            raise e

    async def undeploy_artifact(
        self,
        artifact_id: str,
        skip_update=False,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Remove a deployment from Ray Serve

        Args:
            artifact_id: ID of the artifact to undeploy
            skip_update: Skip updating services after undeployment
            context: Context for Hypha service
        """
        try:
            # Verify client is connected to Hypha server
            if not self.server:
                raise RuntimeError("Hypha server connection not available")

            # Verify Ray is initialized
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            deployment_name = self._get_deployment_name(artifact_id)

            if deployment_name not in self.deployments.keys():
                raise ValueError(f"Deployment '{deployment_name}' not found")

            # Delete the deployment in a separate thread
            await asyncio.to_thread(serve.delete, deployment_name)

            self.logger.info(f"Successfully undeployed {artifact_id}")

            if not skip_update:
                await self._update_services()

        except Exception as e:
            self.logger.error(f"Error undeploying {artifact_id}: {e}")
            raise e

    async def _update_services(self) -> None:
        """Update Hypha services based on currently deployed models"""
        try:
            # Ensure server connection
            if not self.server:
                raise RuntimeError("Hypha server connection not available")

            # Create service functions for each deployment
            service_functions = {}

            # Add a list_deployments endpoint
            async def list_deployments(context=None):
                return list(self.deployments.keys())

            service_functions["list_deployments"] = list_deployments

            # Get deployment handles for each tracked deployment
            async def create_model_function(application_name, data=None, context=None):
                # TODO: support other functions than __call__
                try:
                    app_handle = serve.get_app_handle(name=application_name)
                    return await app_handle.remote(data=data)
                except Exception as e:
                    self.logger.error(
                        f"Failed to get handle for '{application_name}': {e}"
                    )
                    raise e

            for deployment_name in self.deployments.keys():
                try:
                    service_functions[deployment_name] = partial(
                        create_model_function, deployment_name
                    )
                    self.logger.info(
                        f"Prepared model function for deployment '{deployment_name}'"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to get handle for {deployment_name}: {e}"
                    )
                    raise e

            # Register all model functions as a single service
            if len(service_functions) > 1:
                service_info = await self.server.register_service(
                    {
                        "id": self.service_id,
                        "name": "Ray Model Services",
                        "description": "Deployed Ray Serve models",
                        "config": {"visibility": "public", "require_context": True},
                        **service_functions,
                    },
                    {"overwrite": True},
                )
                self.logger.info("Successfully registered  model services")
                server_url = self.server.config.public_base_url
                workspace = service_info.config.workspace
                sid = service_info.id.split("/")[-1]
                service_url = (
                    f"{server_url}/{workspace}/services/{sid}"
                )
                self.logger.debug(f"Service available at: {service_url}")
                self.logger.debug(f"List all deployments at: {service_url}/list_deployments")
                self.logger.debug(f"Test the deployment service at: {service_url}/<deployment_name>")
                self.service_info = service_info
            else:
                self.logger.info("No deployments to register as services")
                self.service_info = None

        except Exception as e:
            self.logger.error(f"Error updating services: {e}")
            raise e

    async def deploy_all_artifacts(self) -> List[str]:
        """Deploy all artifacts in the deployment collection to Ray Serve

        Returns:
            list: List of artifact IDs that were successfully deployed
        """
        deployed_artifact_ids = []
        try:
            # Ensure artifact manager is available
            if not self.artifact_manager:
                raise ValueError("Artifact manager not initialized")

            # Get all artifacts in the collection
            artifacts = await self.artifact_manager.list(
                parent_id=self.deployment_collection_id
            )

            # Deploy each artifact
            for artifact in artifacts:
                try:
                    artifact_id = artifact["id"]
                    await self.deploy_artifact(artifact_id, skip_update=True)
                    deployed_artifact_ids.append(artifact_id)
                except Exception as e:
                    self.logger.error(f"Failed to deploy {artifact_id}: {e}")
                    raise e

            # Update services after all deployments
            await self._update_services()

            return deployed_artifact_ids

        except Exception as e:
            self.logger.error(f"Error deploying all artifacts: {e}")
            raise e

    async def cleanup_deployments(self) -> None:
        """Cleanup Ray Serve deployments"""
        try:
            # Ensure Ray is initialized
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            for deployment_name in self.deployments.keys():
                serve.delete(deployment_name)
                self.logger.info(f"Deleted application '{deployment_name}'")

            if len(self.deployments) == 0:
                self.logger.info("Successfully cleaned up all deployments")
            else:
                self.logger.warning(
                    f"Failed to clean up all deployments, {len(self.deployments)} remaining."
                )

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise e

    def get_status(self):
        """Get the current status of the Ray Deployment Manager."""
        return {
            "deployments": self.deployments,
            "service_info": self.service_info,
        }


async def create_example_deployment(artifact_manager, logger=None) -> str:
    """Create an example deployment for testing

    Args:
        artifact_manager: Hypha artifact manager instance
        logger: Optional logger instance

    Returns:
        str: ID of the created artifact
    """
    if not logger:
        logger = create_logger("ArtifactManager")

    # Get example deployment directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_deployment_dir = os.path.join(base_dir, "example_deployment")
    with open(os.path.join(example_deployment_dir, "manifest.yaml"), "r") as f:
        deployment_manifest = yaml.safe_load(f)

    # Check existing deployments
    artifacts = await artifact_manager.list()
    deployment_name = deployment_manifest.get("name")
    exists = False
    for artifact in artifacts:
        if artifact.manifest.name == deployment_name:
            logger.info(f"Deployment '{deployment_name}' already exists.")
            exists = True
            break

    if exists:
        test_artifact = await artifact_manager.edit(
            artifact_id=artifact.id,
            manifest=deployment_manifest,
            version="stage",
        )
    else:
        # Add the deployment to the gallery and stage it for review
        test_artifact = await artifact_manager.create(
            manifest=deployment_manifest,
            version="stage",
        )
    logger.info(f"Artifact created with ID: {test_artifact.id}")

    # Upload main.py
    upload_url = await artifact_manager.put_file(test_artifact.id, file_path="main.py")
    with open(os.path.join(example_deployment_dir, "main.py"), "r") as f:
        file_content = f.read()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(upload_url, data=file_content)
        response.raise_for_status()
        logger.info(f"Uploaded main.py to artifact")

    # Upload manifest.yaml
    upload_url = await artifact_manager.put_file(
        test_artifact.id, file_path="manifest.yaml"
    )
    with open(os.path.join(example_deployment_dir, "manifest.yaml"), "r") as f:
        file_content = f.read()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(upload_url, data=file_content)
        response.raise_for_status()
        logger.info(f"Uploaded manifest.yaml to artifact")

    # Commit the artifact
    await artifact_manager.commit(
        artifact_id=test_artifact.id,
        version="new",
    )
    logger.info(f"Committed artifact")

    return test_artifact.id


if __name__ == "__main__":
    """Test the RayDeploymentManager functionality with a real Ray cluster and model deployment."""

    from hypha_rpc import connect_to_server, login

    print("\n===== Testing RayDeploymentManager =====\n")

    # Create and start the autoscaler with shorter thresholds for quicker testing
    autoscaler = RayAutoscaler(
        # Use shorter times for faster testing
        default_time_limit="00:10:00",
        max_workers=1,
        metrics_interval_seconds=10,
        # Cluster Manager parameters
        temp_dir="/proj/aicell/ray_tmp",
        data_dir=os.path.dirname(__file__),
        container_image="/proj/aicell/users/x_nilme/autoscaler/tabula_0.1.1.sif",
    )
    autoscaler.cluster_manager.logger.setLevel(logging.DEBUG)
    autoscaler.logger.setLevel(logging.DEBUG)

    async def test_deployment_manager(server_url="https://hypha.aicell.io", keep_running=False):
        try:
            # Start autoscaler
            await autoscaler.start()

            # Create deployment manager
            deployment_manager = RayDeploymentManager(autoscaler=autoscaler)
            deployment_manager.logger.setLevel(logging.DEBUG)

            # Connect to Hypha server using token from environment
            token = os.environ["HYPHA_TOKEN"] or await login({"server_url": server_url})
            server = await connect_to_server({"server_url": server_url, "token": token})

            # Initialize deployment manager
            await deployment_manager.initialize(server)

            # Upload the example deployment code to Hypha as a new artifact
            artifact_id = await create_example_deployment(
                deployment_manager.artifact_manager
            )

            # Deploy the artifact
            await deployment_manager.deploy_artifact(artifact_id)

            # Test registered Hypha service
            deployment_service = await server.get_service(deployment_manager.service_id)
            deployment_name = deployment_manager._get_deployment_name(artifact_id)

            # Get the list of deployments
            deployments = await deployment_service.list_deployments()
            assert deployment_name in deployments

            # Call the deployed model
            response = await deployment_service[deployment_name]()
            deployment_manager.logger.info(f"Response from deployed model: {response}")

            response = await deployment_service[deployment_name]()
            deployment_manager.logger.info(f"Response from deployed model: {response}")

            # Keep server running if requested
            if keep_running:
                print("Server running. Press Ctrl+C to stop.")
                while True:
                    await asyncio.sleep(1)

            # Undeploy the test artifact
            await deployment_manager.undeploy_artifact(artifact_id)

            # Deploy again
            await deployment_manager.deploy_artifact(artifact_id)

            # Clean up deployments
            await deployment_manager.cleanup_deployments()

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e
        finally:
            await autoscaler.shutdown_cluster()

    # Run the test
    asyncio.run(test_deployment_manager(keep_running=True))

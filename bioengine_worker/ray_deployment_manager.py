import asyncio
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import ray
from ray import serve
import numpy as np

from bioengine_worker import __version__
from bioengine_worker.ray_autoscaler import RayAutoscaler
from bioengine_worker.utils import create_logger, format_time


class RayDeploymentManager:
    """Manages Ray Serve deployments using Hypha artifact manager

    This class integrates with Hypha artifact manager to deploy
    artifacts from Hypha as Ray Serve deployments.
    """

    def __init__(
        self,
        service_id: str = "bioengine-worker-deployments",
        autoscaler: Optional[RayAutoscaler] = None,
        # Logger
        logger: Optional[logging.Logger] = None,
        log_file: Optional[str] = None,
        _debug: bool = False,
    ):
        """Initialize the Ray Deployment Manager

        Args:
            service_id: ID to use for the Hypha service exposing deployed models
            autoscaler: Optional RayAutoscaler instance
            logger: Optional logger instance
            _debug: Enable debug logging
        """
        # Set up logging
        self.logger = logger or create_logger(
            name="RayDeploymentManager",
            level=logging.DEBUG if _debug else logging.INFO,
            log_file=log_file,
        )

        # Store parameters
        self._service_id = service_id
        self.autoscaler = autoscaler

        # Initialize state variables
        self.server = None
        self.artifact_manager = None
        self.service_info = None
        self._deployed_artifacts = {}

    async def _update_services(self) -> None:
        """Update Hypha services based on currently deployed models"""
        try:
            # Ensure server connection
            if not self.server:
                raise RuntimeError("Hypha server connection not available")

            # Create service functions for each deployment
            deployments = await self._get_deployment_names()
            service_functions = {}

            async def create_deployment_function(
                application_name, method, *args, context=None, **kwargs
            ):
                try:
                    # TODO: add user authorization
                    user = context.get("user") if context else None
                    self.logger.info(
                        f"User {user['id']} is calling deployment '{application_name}' with method '{method}'"
                    )
                    app_handle = serve.get_app_handle(name=application_name)

                    # Recursively put args and kwargs into ray object storage
                    args = [
                        ray.put(arg) if isinstance(arg, np.ndarray) else arg
                        for arg in args
                    ]
                    kwargs = {
                        k: ray.put(v) if isinstance(v, np.ndarray) else v
                        for k, v in kwargs.items()
                    }

                    result = await getattr(app_handle, method).remote(*args, **kwargs)
                    return result
                except Exception as e:
                    self.logger.error(
                        f"Failed to call deployment '{application_name}': {e}"
                    )
                    raise e

            for deployment_name in deployments:
                artifact_id = await self._get_artifact_id_from_deployment_name(
                    deployment_name
                )
                manifest = self._deployed_artifacts[artifact_id]
                service_functions[deployment_name] = {}
                class_config = manifest["deployment_class"]
                if class_config.get("exposed_methods"):
                    for method in class_config["exposed_methods"].keys():
                        service_functions[deployment_name][method] = partial(
                            create_deployment_function, deployment_name, method
                        )
                else:
                    service_functions[deployment_name] = partial(
                        create_deployment_function, deployment_name, "__call__"
                    )

            # Register all model functions as a single service
            service_info = await self.server.register_service(
                {
                    "id": self._service_id,
                    "name": "BioEngine Worker Deployments",
                    "type": "ray-deployment",
                    "description": "Deployed Ray Serve models",
                    "config": {"visibility": "public", "require_context": True},
                    **service_functions,
                },
                {"overwrite": True},
            )
            if len(service_functions) > 0:
                self.logger.info("Successfully registered deployment service")
                server_url = self.server.config.public_base_url
                workspace, sid = service_info.id.split("/")
                service_url = f"{server_url}/{workspace}/services/{sid}"
                for deployment_name in service_functions.keys():
                    self.logger.info(
                        f"Access the deployment service at: {service_url}/{deployment_name}"
                    )
                self.service_info = service_info
            else:
                self.logger.info("No deployments to register as services")
                self.service_info = None

        except Exception as e:
            self.logger.error(f"Error updating services: {e}")
            raise e

    async def _load_deployment_code(
        self,
        class_config: dict,
        artifact_id: str,
        version=None,
        timeout: int = 30,
    ) -> Any:
        """Load and execute deployment code from an artifact directly in memory

        Args:
            class_config: Configuration for the class to load
            artifact_id: ID of the artifact
            version: Optional version of the artifact
            timeout: Timeout in seconds for network requests (default: 30)

        Returns:
            Any: The loaded class or None if not found
        """
        try:
            # Get download URL for the file
            download_url = await self.artifact_manager.get_file(
                artifact_id=artifact_id,
                version=version,
                file_path=class_config["python_file"],
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
            if class_config["class_name"] not in safe_globals:
                raise ValueError(
                    f"{class_config['class_name']} not found in {artifact_id}"
                )
            model = safe_globals[class_config["class_name"]]
            if not model:
                raise RuntimeError(
                    f"Error loading {class_config['class_name']} from {artifact_id}"
                )

            if "multiplexed" in class_config:
                # Add @serve.multiplexed decorator to specified class method
                method_name = class_config["multiplexed"]["method_name"]
                max_num_models_per_replica = class_config["multiplexed"][
                    "max_num_models_per_replica"
                ]

                orig_method = getattr(model, method_name)
                decorated_method = serve.multiplexed(
                    orig_method, max_num_models_per_replica=max_num_models_per_replica
                )
                setattr(model, method_name, decorated_method)

            self.logger.info(
                f"Loaded class '{class_config['class_name']}' from {artifact_id}"
            )
            return model

        except Exception as e:
            self.logger.error(f"Error loading deployment code for {artifact_id}: {e}")
            raise e

    def _get_deployment_name(
        self, artifact_id: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
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

    async def _get_deployment_names(self) -> Dict[str, Any]:
        return {
            self._get_deployment_name(artifact_id)
            for artifact_id in self._deployed_artifacts
        }

    async def _get_artifact_id_from_deployment_name(
        self, deployment_name: str
    ) -> Optional[str]:
        """Get the artifact ID from a deployment name

        Args:
            deployment_name: The deployment name to convert

        Returns:
            str: The converted artifact ID
        """
        for artifact_id in self._deployed_artifacts.keys():
            if self._get_deployment_name(artifact_id) == deployment_name:
                return artifact_id
        return None

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

    async def create_artifact(self, artifact_id: str, files: List[dict]) -> None:
        """
        Create a deployment artifact
        """
        raise NotImplementedError

    async def deploy_artifact(
        self,
        artifact_id: str,
        mode: str = None,
        version: str = None,
        context: Optional[Dict[str, Any]] = None,
        _skip_update=False,
    ) -> str:
        """
        Deploy a single artifact to Ray Serve

        Args:
            artifact_id: ID of the artifact to deploy
            version: Optional version of the artifact
            context: Context for Hypha service
            _skip_update: Skip updating services after deployment

        Returns:
            str: Deployment name
        """
        try:
            # Verify client is connected to Hypha server
            if not self.server:
                raise RuntimeError(
                    "Hypha server connection not available. Call initialize() first."
                )

            # Ensure artifact manager is available
            if not self.artifact_manager:
                raise RuntimeError(
                    "Artifact manager not initialized. Call initialize() first."
                )

            # Verify Ray is initialized
            if not ray.is_initialized():
                raise RuntimeError(
                    "Ray cluster is not running. Call initialize() first."
                )

            if "/" not in artifact_id:
                artifact_id = f"{self.server.config.workspace}/{artifact_id}"
            self.logger.info(f"Deploying artifact '{artifact_id}'...")

            # Read the manifest to get deployment configuration
            artifact = await self.artifact_manager.read(artifact_id, version=version)
            manifest = artifact["manifest"]

            # Get the deployment configuration
            deployment_config = manifest["deployment_config"]

            # Load the deployment code
            class_config = manifest["deployment_class"]
            deployment_config["name"] = class_config["class_name"]

            model = await self._load_deployment_code(
                class_config,
                artifact_id,
                version=version,
            )

            # Check if different modes are supported
            modes = deployment_config.pop("modes", None)
            if modes:
                mode_key = mode or list(modes.keys())[0]
                mode_key = mode_key.lower()
                if mode_key not in modes:
                    raise ValueError(
                        f"Mode '{mode_key}' not found in artifact '{artifact_id}'"
                    )

                # Update deployment config with the selected mode
                for key, value in modes[mode_key].items():
                    deployment_config[key] = value

                self.logger.info(
                    f"Using mode '{mode_key}' for deployment '{artifact_id}'"
                )

            # Create the Ray Serve deployment
            model_deployment = serve.deployment(**deployment_config)(model)

            # Bind the arguments to the deployment and return an Application
            app = model_deployment.bind()

            # Deploy the application in a separate thread to avoid blocking
            deployment_name = self._get_deployment_name(artifact_id)
            task = asyncio.to_thread(
                serve.run, app, name=deployment_name, route_prefix=None
            )
            if self.autoscaler:
                # Wait for the deployment to be ready
                await asyncio.sleep(1)
                # Notify the autoscaler of the new deployment
                await self.autoscaler.notify()

            # Wait for the deployment to complete
            await task

            # Add the deployment to the tracked artifacts
            self._deployed_artifacts[artifact_id] = manifest

            self.logger.info(f"Successfully deployed {artifact_id}")

            if not _skip_update:
                await self._update_services()

            return deployment_name

        except Exception as e:
            self.logger.error(f"Error deploying '{artifact_id}': {e}")
            raise e

    async def undeploy_artifact(
        self,
        artifact_id: str,
        context: Optional[Dict[str, Any]] = None,
        _skip_update=False,
    ) -> None:
        """Remove a deployment from Ray Serve

        Args:
            artifact_id: ID of the artifact to undeploy
            context: Context for Hypha service
            _skip_update: Skip updating services after undeployment
        """
        self.logger.info(f"Undeploying artifact '{artifact_id}'...")

        try:
            # Verify client is connected to Hypha server
            if not self.server:
                raise RuntimeError("Hypha server connection not available")

            # Verify Ray is initialized
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            deployment_name = self._get_deployment_name(artifact_id)

            deployments = await self._get_deployment_names()
            if deployment_name not in deployments:
                raise ValueError(f"Deployment '{deployment_name}' not found")

            # Delete the deployment in a separate thread
            await asyncio.to_thread(serve.delete, deployment_name)

            # Check if the deployment was successfully undeployed
            serve_status = serve.status()
            if deployment_name in serve_status.applications.keys():
                raise RuntimeError(
                    f"Failed to undeploy '{deployment_name}'. It still exists."
                )

            # Remove the deployment from the tracked artifacts
            del self._deployed_artifacts[artifact_id]

            self.logger.info(f"Successfully undeployed {artifact_id}")

            if not _skip_update:
                await self._update_services()

        except Exception as e:
            self.logger.error(f"Failed to undeploy {artifact_id}: {e}")
            raise e

    async def get_status(self) -> Dict[str, Any]:
        """Get a dictionary of currently deployed models"""
        if not ray.is_initialized():
            self.logger.error("Can not get deployments - Ray cluster is not running")
            raise RuntimeError("Ray cluster is not running")

        output = {}
        if self.service_info:
            output["service_id"] = self.service_info.id
        else:
            output["service_id"] = None
            output["note"] = (
                "No deployment service registered. Deploy an artifact first."
            )

            return output

        serve_status = serve.status()  # TODO: run async
        self.logger.debug(
            f"Current deployments: {list(serve_status.applications.keys())}"
        )

        for application_name, application in serve_status.applications.items():
            found = False
            for artifact_id in self._deployed_artifacts.keys():
                if self._get_deployment_name(artifact_id) == application_name:
                    found = True
                    break
            if not found:
                # Skip Ray Serve applications that were not deployed by this manager
                continue
            formatted_time = format_time(application.last_deployed_time_s)
            output[application_name] = {
                "artifact_id": artifact_id,
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

        return output

    async def deploy_all_artifacts(
        self,
        deployment_collection_id: str = "ray-deployments",
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Deploy all artifacts in the deployment collection to Ray Serve

        Args:
            deployment_collection_id: Artifact collection ID for deployments
            context: Context for Hypha service

        Returns:
            list: List of artifact IDs that were successfully deployed
        """
        self.logger.info(
            f"Deploying all artifacts in collection '{deployment_collection_id}'..."
        )
        deployments = []
        try:
            # Ensure artifact manager is available
            if not self.artifact_manager:
                raise ValueError("Artifact manager not initialized")

            # Get all artifacts in the collection
            artifacts = await self.artifact_manager.list(
                parent_id=deployment_collection_id
            )

            # Deploy each artifact
            for artifact in artifacts:
                try:
                    artifact_id = artifact["id"]
                    deployment_name = await self.deploy_artifact(
                        artifact_id, _skip_update=True
                    )
                    deployments.append(deployment_name)
                except Exception as e:
                    self.logger.error(f"Failed to deploy {artifact_id}: {e}")
                    raise e

            # Update services after all deployments
            await self._update_services()

            return deployments

        except Exception as e:
            self.logger.error(f"Error deploying all artifacts: {e}")
            raise e

    async def cleanup_deployments(
        self, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cleanup Ray Serve deployments"""
        self.logger.info("Cleaning up all deployments...")
        try:
            # Ensure Ray is initialized
            if not ray.is_initialized():
                raise RuntimeError("Ray cluster is not running")

            artifact_ids = list(self._deployed_artifacts.keys())
            failed_attempts = 0
            for artifact_id in artifact_ids:
                try:
                    await self.undeploy_artifact(artifact_id)
                except Exception as e:
                    failed_attempts += 1

            if failed_attempts != 0:
                self.logger.warning(
                    f"Failed to clean up all deployments, {failed_attempts} remaining."
                )

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise e


if __name__ == "__main__":
    """Test the RayDeploymentManager functionality with a real Ray cluster and model deployment."""

    from hypha_rpc import connect_to_server, login

    from bioengine_worker.ray_cluster_manager import RayClusterManager

    print("\n===== Testing RayDeploymentManager =====\n")

    # Create and start the autoscaler with shorter thresholds for quicker testing
    cluster_manager = RayClusterManager(
        head_num_cpus=4,
        head_num_gpus=1,
        ray_temp_dir=f"/tmp/ray/{os.environ['USER']}",
        image=str(
            Path(__file__).parent.parent
            / f"apptainer_images/bioengine-worker_{__version__}.sif"
        ),
        worker_data_dir=str(Path(__file__).parent.parent / "data"),
        slurm_log_dir=str(Path(__file__).parent.parent / "logs"),
        _debug=True,
    )
    cluster_manager.start_cluster(force_clean_up=True)

    if cluster_manager.ray_cluster_config["mode"] == "slurm":
        autoscaler = RayAutoscaler(
            cluster_manager=cluster_manager,
            # Use shorter times for faster testing
            default_time_limit="00:10:00",
            max_workers=1,
            metrics_interval_seconds=10,
            _debug=True,
        )
    else:
        autoscaler = None

    async def test_deployment_manager(
        server_url="https://hypha.aicell.io", keep_running=False
    ):
        try:
            if autoscaler:
                # Start autoscaler
                await autoscaler.start()

            # Create deployment manager
            deployment_manager = RayDeploymentManager(
                autoscaler=autoscaler, _debug=True
            )

            # Connect to Hypha server using token from environment
            token = os.environ.get("HYPHA_TOKEN") or await login(
                {"server_url": server_url}
            )
            server = await connect_to_server({"server_url": server_url, "token": token})

            # Initialize deployment manager
            await deployment_manager.initialize(server)

            # Deploy the example deployment
            artifact_id = "example-deployment"
            deployment_name = await deployment_manager.deploy_artifact(artifact_id)

            deployment_status = await deployment_manager.get_status()
            assert deployment_name in deployment_status

            # Test registered Hypha service
            deployment_service_id = deployment_status["service_id"]
            deployment_service = await server.get_service(deployment_service_id)

            # Call the deployed model
            response = await deployment_service[deployment_name]["ping"]()
            deployment_manager.logger.info(f"Response from deployed model: {response}")

            response = await deployment_service[deployment_name]["train"]()
            deployment_manager.logger.info(f"Response from deployed model: {response}")

            # Keep server running if requested
            if keep_running:
                print("Server running. Press Ctrl+C to stop.")
                await server.serve()

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
            if autoscaler:
                await autoscaler.shutdown_cluster()
            else:
                cluster_manager.shutdown_cluster()

    # Run the test
    asyncio.run(test_deployment_manager(keep_running=True))

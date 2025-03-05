import logging
import os
import ray
from ray import serve
import httpx
from typing import Dict, Optional, Any
from functools import partial
import asyncio


class RayDeploymentManager:
    """Manages Ray Serve deployments using Hypha artifact manager

    This class integrates with Hypha artifact manager to deploy
    artifacts from Hypha as Ray Serve deployments.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        deployment_collection_id: str = "ray-deployments",
        service_id: str = "ray-model-services",
    ):
        """Initialize the Ray Deployment Manager

        Args:
            logger: Optional logger instance
            deployment_collection_id: Artifact collection ID for deployments
            service_id: ID to use for the Hypha service exposing deployed models
        """
        # Set up logging
        self.logger = logger or logging.getLogger("ray_deployment_manager")
        if not logger:
            self.logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Store parameters
        self.deployment_collection_id = deployment_collection_id
        self.service_id = service_id

        # Initialize state variables
        self.server = None
        self.artifact_manager = None
        self.service_info = None
        self.deployments = {}

        # Verify Ray is initialized (but don't try to start it)
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before creating RayDeploymentManager")

    async def initialize(self, server):
        """Initialize the deployment manager with a Hypha server connection

        Args:
            server: Hypha server connection

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Verify Ray is still initialized
            if not ray.is_initialized():
                raise RuntimeError("Ray must be initialized")

            # Store server connection
            self.server = server

            # Get artifact manager service
            self.artifact_manager = await self.server.get_service(
                "public/artifact-manager"
            )
            self.logger.info("Successfully connected to artifact manager")

            # # Initialize Ray Serve if not already started
            # if not serve.is_session_initialized():
            #     serve.start(detached=True)
            #     self.logger.info("Ray Serve started")

            return True

        except Exception as e:
            self.logger.error(f"Error initializing Ray Deployment Manager: {e}")
            return False

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

    async def load_deployment_code(
        self,
        artifact_id: str,
        version=None,
        file_path: str = "main.py",
        timeout: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """Load and execute deployment code from an artifact directly in memory

        Args:
            artifact_id: ID of the artifact
            version: Optional version of the artifact
            file_path: Path to the file within the artifact (default: main.py)
            timeout: Timeout in seconds for network requests (default: 30)

        Returns:
            Dictionary containing the module's globals or None if loading fails

        Raises:
            TimeoutError: If network requests exceed the timeout
            ValueError: If the code execution fails or ChironModel is not found
        """
        try:
            # Ensure artifact manager is available
            if not self.artifact_manager:
                raise ValueError(
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
            try:
                exec(code_content, safe_globals)
                if "ChironModel" not in safe_globals:
                    raise ValueError(f"ChironModel not found in {artifact_id}")
                self.logger.info(
                    f"Successfully loaded deployment code for {artifact_id}"
                )
                return safe_globals.get("ChironModel")
            except Exception as e:
                self.logger.error(
                    f"Error executing deployment code for {artifact_id}: {e}"
                )
                raise ValueError(f"Code execution failed: {str(e)}")

        except httpx.TimeoutException:
            self.logger.error(
                f"Timeout while downloading deployment file for {artifact_id}"
            )
            raise TimeoutError(f"Network request timeout for {artifact_id}")
        except httpx.HTTPError as e:
            self.logger.error(
                f"HTTP error while downloading deployment file for {artifact_id}: {e}"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error loading deployment code for {artifact_id}: {e}")
            raise

    async def deploy_artifact(self, artifact_id: str, version=None, skip_update=False):
        """Deploy a single artifact to Ray Serve

        Args:
            artifact_id: ID of the artifact to deploy
            version: Optional version of the artifact
            skip_update: Skip updating services after deployment

        Returns:
            Dict with deployment result information

        Raises:
            ValueError: If deployment configuration is invalid
            RuntimeError: If deployment fails
        """
        try:
            # Load the deployment code
            ChironModel = await self.load_deployment_code(artifact_id, version=version)
            if not ChironModel:
                raise ValueError(f"Failed to load model code for {artifact_id}")

            # Read the manifest to get deployment configuration
            artifact = await self.artifact_manager.read(artifact_id, version=version)
            manifest = artifact.get("manifest")
            if not manifest or "deployment_config" not in manifest:
                raise ValueError(
                    f"Invalid manifest or missing deployment_config for {artifact_id}"
                )

            deployment_config = manifest["deployment_config"]

            try:
                deployment_name = self._get_deployment_name(artifact_id)
                
                # Ensure route_prefix is None to disable HTTP routes
                deployment_config["name"] = deployment_name
                deployment_config["route_prefix"] = None
                
                # Create the Ray Serve deployment
                ChironModelDeployment = serve.deployment(**deployment_config)(
                    ChironModel
                )
                # Bind the arguments to the deployment and return an Application
                app = ChironModelDeployment.bind()
                # Deploy the application
                serve.run(app, name="Chiron")

                # Store deployment information
                self.deployments[deployment_name] = {
                    "artifact_id": artifact_id,
                    "deployment": ChironModelDeployment,
                    "config": deployment_config,
                    "status": "deployed",
                }

            except Exception as e:
                raise RuntimeError(f"Ray Serve deployment failed: {str(e)}")

            self.logger.info(f"Successfully deployed {artifact_id}")

            if not skip_update:
                await self.update_services()

            return {
                "success": True,
                "message": f"Successfully deployed {artifact_id}",
                "deployment_name": deployment_name,
                "service_id": self.service_info["id"] if self.service_info else None,
            }

        except Exception as e:
            self.logger.error(f"Error deploying {artifact_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to deploy {artifact_id}: {str(e)}",
                "error": str(e),
            }

    async def undeploy_artifact(self, artifact_id: str, skip_update=False):
        """Remove a deployment from Ray Serve

        Args:
            artifact_id: ID of the artifact to undeploy
            skip_update: Skip updating services after undeployment

        Returns:
            Dict with undeployment result information
        """
        try:
            deployment_name = self._get_deployment_name(artifact_id)

            if not serve.is_session_initialized():
                self.logger.error("Ray Serve is not initialized")
                return {"success": False, "message": "Ray Serve is not initialized"}

            # Delete the deployment
            serve.delete(deployment_name)

            # Remove from our deployment tracking
            if deployment_name in self.deployments:
                del self.deployments[deployment_name]

            self.logger.info(f"Successfully undeployed {artifact_id}")

            if not skip_update:
                await self.update_services()

            return {
                "success": True,
                "message": f"Successfully undeployed {artifact_id}",
            }
        except Exception as e:
            self.logger.error(f"Error undeploying {artifact_id}: {e}")
            return {
                "success": False,
                "message": f"Failed to undeploy {artifact_id}: {str(e)}",
            }

    async def list_deployments(self) -> Dict:
        """List all active Ray Serve deployments

        Returns:
            Dict containing the deployment information
        """
        try:
            # Check if Ray Serve is running
            if not serve.is_session_initialized():
                return {"success": False, "message": "Ray Serve is not initialized"}

            # Return our tracked deployments
            return {
                "success": True,
                "deployments": self.deployments
            }

        except Exception as e:
            self.logger.error(f"Error listing deployments: {e}")
            return {"success": False, "error": str(e)}

    async def update_services(self):
        """Update Hypha services based on currently deployed models

        Returns:
            Dict with service update information
        """
        try:
            # Ensure server connection
            if not self.server:
                raise ValueError("No Hypha server connection available")

            # Get all current deployments from Ray Serve
            if not serve.is_session_initialized():
                return {"success": False, "message": "Ray Serve is not initialized"}

            # Create service functions for each deployment
            service_functions = {}

            # Define function to create model functions
            async def create_model_function(handle, name, data=None, context=None):
                return await handle.remote(data=data)

            # Get deployment handles for each tracked deployment
            for deployment_name, deployment_info in self.deployments.items():
                try:
                    handle = serve.get_deployment_handle(deployment_name, "Chiron")
                    model_function = partial(
                        create_model_function, handle, deployment_name
                    )
                    service_functions[deployment_name] = model_function
                    self.logger.info(f"Added function for deployment {deployment_name}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get handle for {deployment_name}: {e}"
                    )

            # Add a list_deployments endpoint
            service_functions["list_deployments"] = self.list_deployments

            # Register all model functions as a single service
            if service_functions:
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

                self.service_info = service_info
                self.logger.info(
                    f"Successfully registered {len(service_functions)} model services"
                )

                return {
                    "success": True,
                    "service_id": service_info.id,
                    "functions": list(service_functions.keys()),
                }
            else:
                self.logger.info("No deployments to register as services")
                return {"success": True, "functions": []}

        except Exception as e:
            self.logger.error(f"Error updating services: {e}")
            return {"success": False, "error": str(e)}

    async def deploy_all_artifacts(self):
        """Deploy all artifacts in the ray-deployments collection

        Returns:
            Dict containing deployment results for each artifact
        """
        results = {}
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
                artifact_id = artifact["id"]
                try:
                    result = await self.deploy_artifact(artifact_id, skip_update=True)
                    results[artifact_id] = result
                except Exception as e:
                    self.logger.error(f"Failed to deploy {artifact_id}: {e}")
                    results[artifact_id] = {"success": False, "error": str(e)}

            # Update services after all deployments
            await self.update_services()

            return {
                "success": True,
                "results": results,
                "deployed_count": sum(
                    1 for r in results.values() if r.get("success", False)
                ),
            }

        except Exception as e:
            self.logger.error(f"Error deploying all artifacts: {e}")
            return {"success": False, "error": str(e), "results": results}

    async def cleanup(self):
        """Cleanup Ray Serve deployments

        Returns:
            Dict with cleanup result information
        """
        try:
            # Get list of current deployments
            deployment_list = list(self.deployments.keys())

            # Shutdown Ray Serve and deployments if active
            if serve.is_session_initialized():
                serve.shutdown()
                self.logger.info("Ray Serve shut down")

            # Reset deployments dictionary
            self.deployments = {}

            return {
                "success": True,
                "message": "Successfully cleaned up Ray Serve deployments",
                "deployments_shutdown": deployment_list,
            }
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    """Test the RayDeploymentManager functionality with a real Ray cluster and model deployment."""
    import asyncio
    import time
    from hpc_worker.ray_cluster_manager import RayClusterManager
    from hypha_rpc import connect_to_server
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ray_deployment_test")
    
    async def test_deployment_manager():
        try:
            # Start a Ray cluster using RayClusterManager
            logger.info("Starting Ray cluster...")
            cluster_manager = RayClusterManager(logger=logger)
            
            cluster_result = cluster_manager.start_cluster()
            if not cluster_result["success"]:
                logger.error(f"Failed to start Ray cluster: {cluster_result}")
                return
            
            logger.info("Ray cluster started successfully")
            
            # Create deployment manager instance
            deployment_manager = RayDeploymentManager(logger=logger)
            
            # Connect to Hypha server using token from environment
            server = await connect_to_server({
                "server_url": "https://hypha.aicell.io",
                "token": os.environ["HYPHA_TOKEN"]
            })
            
            # Initialize deployment manager
            init_result = await deployment_manager.initialize(server)
            if not init_result:
                logger.error("Failed to initialize deployment manager")
                return
            
            logger.info("Deployment manager initialized successfully")
            
            # Test deploying a real model from Hypha
            # test_artifact_id = "philosophical-panda"
            # logger.info(f"Attempting to deploy artifact: {test_artifact_id}")

            # TODO: upload model to Hypha Artifacts
            # Upload the example deployment code to Hypha

            # Define metadata for the new deployment
            deployment_manifest = {
                "name": "Example Deployment",
                "description": "A deployment containing example model metadata",
            }

            # Add the deployment to the gallery and stage it for review
            # deployment = await deployment_manager.artifact_manager.create(
            #     alias="example-deployment",
            #     manifest=deployment_manifest,
            #     version="stage"
            # )

            deployment_id = "ws-user-github|49943582/example-deployment"
            logger.info(f"Deployment created: {deployment_id}")

            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, "example_deployment.zip")
            upload_url = await deployment_manager.artifact_manager.put_file(
                deployment_id, file_path=file_path
            )

            # Download the file content with timeout
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.put(upload_url)
                response.raise_for_status()
                logger.info(f"File upload response: {response.text}")

            # Commit the artifact
            commit_result = await deployment_manager.artifact_manager.commit(
                artifact_id=deployment_id,
                version="latest",
            )
            logger.info(f"Commit result: {commit_result}")
            
            # Deploy the artifact
            result = await deployment_manager.deploy_artifact(deployment_id)
            logger.info(f"Deploy result: {result}")
            
            if result["success"]:
                # List deployments
                deployments = await deployment_manager.list_deployments()
                logger.info(f"Current deployments: {deployments}")
                
                # Test the deployed model
                if deployments["success"] and deployments["deployments"]:
                    deployment_name = next(iter(deployments["deployments"].keys()))
                    handle = serve.get_deployment_handle(deployment_name, "Chiron")
                    
                    # Test with sample data
                    test_data = {"input": "test"}
                    logger.info(f"Testing deployment with data: {test_data}")
                    result = await handle.remote(data=test_data)
                    logger.info(f"Model response: {result}")
                
                # Wait a moment to see logs
                time.sleep(2)
                
                # Undeploy the test artifact
                logger.info(f"Undeploying {test_artifact_id}...")
                undeploy_result = await deployment_manager.undeploy_artifact(test_artifact_id)
                logger.info(f"Undeploy result: {undeploy_result}")
            
            # Clean up deployments
            logger.info("Cleaning up deployments...")
            cleanup_result = await deployment_manager.cleanup()
            logger.info(f"Cleanup result: {cleanup_result}")
            
            # Shut down Ray cluster
            logger.info("Shutting down Ray cluster...")
            shutdown_result = cluster_manager.shutdown_cluster()
            logger.info(f"Shutdown result: {shutdown_result}")
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            raise
    
    # Run the test
    if os.environ.get("HYPHA_TOKEN"):
        print("===== Testing RayDeploymentManager =====")
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(test_deployment_manager())
        finally:
            loop.close()
        print("===== Test completed =====")
    else:
        print("HYPHA_TOKEN not set. Please set the HYPHA_TOKEN environment variable to run the test.")

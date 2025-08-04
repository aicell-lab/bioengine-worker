"""
BioEngine Worker Multi-Deployment Application Example

This demonstrates how to create applications with multiple deployments that work together.
BioEngine Worker applications can consist of multiple Ray Serve deployments that are
combined into a single application by the BioEngine Worker.

MULTI-DEPLOYMENT ARCHITECTURE:
- Applications can have multiple deployments listed in the manifest.yaml under 'deployments'
- Each deployment is a separate Ray Serve deployment with its own resource allocation
- Deployments can communicate with each other through DeploymentHandle parameters
- The BioEngine Worker automatically wires deployment dependencies during startup
- All deployments in an application share the same authorized users and lifecycle

DEPLOYMENT PARAMETER NAMING CONVENTION:
- In the manifest.yaml, deployments are defined as 'python_file:ClassName' format
- DeploymentHandle parameters in the entry deployment's __init__ method MUST match
    the python file name (without .py extension) from the manifest
- Example: If manifest has 'deployment1:Deployment1', the parameter must be named 'deployment1'
- This naming convention allows BioEngine to automatically wire dependencies between deployments

MIXED RESOURCE ALLOCATION:
- Different deployments within the same application can have different resource requirements
- Some deployments can be GPU-enabled while others are CPU-only
- This example demonstrates mixed allocation: CompositionDeployment (CPU-only) orchestrates
    Deployment1 (CPU-only) and Deployment2 (potentially GPU-enabled)
- GPU allocation is managed by the AppBuilder which overwrites the num_gpus parameter
    in the ray actor options based on the disable_gpu parameter (default: False)
- Resource allocation is validated per deployment against available cluster resources

DEPLOYMENT REQUIREMENTS:
1. Use @ray.serve.deployment decorator with appropriate resource configuration
2. Implement standard lifecycle methods (if needed)
3. Handle GPU allocation through BioEngine's resource management system
4. Follow proper import patterns for external dependencies

IMPORT HANDLING:
- Standard Python libraries and libraries that are part of the BioEngine can be
    imported at the top of this file. Take a look at the requirements.txt file to see
    which libraries are part of the BioEngine:
    https://github.com/aicell-lab/bioengine-worker/blob/main/requirements.txt
- All libraries that are not part of the standard python library or the BioEngine
    need to be specified in the runtime environment of the deployment and imported
    in each method where they are used (see 'pandas' in the example below).

GPU allocation is managed by the AppBuilder which overwrites the num_gpus parameter
in the ray actor options based on the disable_gpu parameter (default: False) passed to
deploy_application().

Resource allocation is validated against available cluster resources before deployment.
See bioengine_worker/apps_manager.py for the deployment orchestration logic.

Ray Serve deployment parameters: https://docs.ray.io/en/latest/serve/api/doc/ray.serve.deployment_decorator.html
BioEngine app deployment guide: See project README for artifact structure requirements.
"""

import asyncio
import os

from hypha_rpc.utils.schema import schema_method
from ray import serve
from ray.serve.handle import DeploymentHandle
from pydantic import Field


@serve.deployment(
    ray_actor_options={
        # Number of CPUs to allocate for the deployment
        # This deployment acts only as a composition of other deployments, so it does not require any CPU resources.
        "num_cpus": 0,
        # Number of GPUs to allocate for the deployment
        "num_gpus": 0,
        # Memory limit for the deployment (1 GB)
        "memory": 1024 * 1024 * 1024,
        # Runtime environment for the deployment (e.g., dependencies, environment variables)
        "runtime_env": {
            "pip": [
                "pandas",  # Example dependency, can be replaced with any other library
            ],
            "env_vars": {
                "EXAMPLE_ENV_VAR": "example_value",  # Example environment variable
            },
        },
    },
    # Maximum number of ongoing requests to the deployment
    max_ongoing_requests=5,
)
class CompositionDeployment:
    def __init__(
        self,
        demo_input: str,
        deployment1: DeploymentHandle,
        deployment2: DeploymentHandle,
    ) -> None:
        """
        Initialize the composition deployment with the given arguments.
        """
        self.demo_input = demo_input
        self.deployment1 = deployment1
        self.deployment2 = deployment2

    # === BioEngine App Methods - will be called when the deployment is started ===

    async def async_init(self) -> None:
        """
        An optional async initialization method for the deployment. If defined, it will be called
        when the deployment is started.

        Requirements:
        - Must be an async method.
        - Must not accept any arguments.
        - Must not return any value.
        """
        # Mock initialization logic
        print("Initializing CompositionDeployment...")
        await asyncio.sleep(0.01)

    async def test_deployment(self) -> bool:
        """
        An optional method to test the deployment. If defined, it will be called when the deployment
        is started to check if the deployment is working correctly.

        Requirements:
        - Must be an async method.
        - Must not accept any arguments.
        - Must return a boolean value indicating whether the deployment is working correctly.
        """
        # Test importing a library set in the runtime environment
        import pandas

        # Test accessing an environment variable set in the runtime environment
        os.environ["EXAMPLE_ENV_VAR"]

        # Test the application methods
        result_1 = await self.deployment1.elapsed_time.remote()

        result_2 = await self.deployment2.add.remote(number=10)

    # === Exposed BioEngine App Methods - all methods decorated with @schema_method will be exposed as API endpoints ===
    # Note: Parameter type hints and docstrings will be used to generate the API documentation.

    @schema_method
    async def calculate_result(self, number: int = Field(..., description="The number to add to the start value of Deployment2.")) -> str:
        """
        Calculate the result by adding the given number to the start value of Deployment2.

        Returns:
            str: A string containing the uptime of Deployment1 and the result of the addition
                 from Deployment2.
        """
        uptime = await self.deployment1.elapsed_time.remote()
        await asyncio.sleep(1)  # Simulate some processing delay
        result = await self.deployment2.add.remote(number)
        return f"Uptime: {uptime}, Result: {result}, Demo string: {self.demo_input}"

    @schema_method
    async def ping(self) -> str:
        """
        Ping the application to test connectivity.

        Args:
            None

        Returns:
            str: A simple response string to confirm the model is reachable.
        """
        return "pong"

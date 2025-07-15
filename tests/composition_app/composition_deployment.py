# All standard python libraries and libraries that are part of the BioEngine can be
# imported at the top of this file. Take a look at the requirements.txt file to see
# which libraries are part of the BioEngine:
# https://github.com/aicell-lab/bioengine-worker/blob/main/requirements.txt)
import asyncio
import os

from hypha_rpc.utils.schema import schema_method
from ray import serve
from ray.serve.handle import DeploymentHandle

# All libraries that are not part of the standard python library or the BioEngine
# need to be specified in the runtime environment of the deployment and imported
# in each method where they are used (see 'pandas' in the example below).


# The deployment class must be decorated with the @ray.serve.deployment decorator.
# If environment variables such as 'NUM_CPUS', 'NUM_GPUS' and 'MEMORY'
# are used, they can be set when deploying the application using the BioEngine. See all
# deployment parameters here:
# https://docs.ray.io/en/latest/serve/api/doc/ray.serve.deployment_decorator.html
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
        Initialize the composition deployment with the given arguments. Make sure that each model
        com
        """
        self.demo_input = demo_input
        self.deployment1 = deployment1
        self.deployment2 = deployment2

    # === Internal BioEngine App Methods - will be called by the BioEngine when the deployment is started ===

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
        try:
            # Test importing a library set in the runtime environment
            import pandas

            # Test accessing an environment variable set in the runtime environment
            os.environ["EXAMPLE_ENV_VAR"]

            # Test the application methods
            result_1 = await self.deployment1_handle.elapsed_time.remote()

            result_2 = await self.deployment2_handle.add.remote(number=10)

            print("Deployment test passed")

            return True
        except Exception as e:
            print(f"Deployment test failed: {e}")
            return False

    # === Exposed BioEngine App Methods - docstrings will be used to generate the API documentation ===

    @schema_method
    async def calculate_result(self, number: int) -> str:
        """
        Calculate the result by adding the given number to the start value of Deployment2.

        Args:
            number (int): The number to add to the start value of Deployment2.

        Returns:
            str: A string containing the uptime of Deployment1 and the result of the addition
                 from Deployment2.
        """
        uptime = await self.deployment1_handle.elapsed_time.remote()
        result = await self.deployment2_handle.add.remote(number)
        return f"Uptime: {uptime}, Result: {result}, Demo string: {self.demo_input}"

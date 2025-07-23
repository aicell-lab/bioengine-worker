# All standard python libraries and libraries that are part of the BioEngine can be
# imported at the top of this file. Take a look at the requirements.txt file to see
# which libraries are part of the BioEngine:
# https://github.com/aicell-lab/bioengine-worker/blob/main/requirements.txt)
import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Union

from hypha_rpc.utils.schema import schema_method
from ray import serve

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
        "num_cpus": os.environ.get("NUM_CPUS", 1),
        # Number of GPUs to allocate for the deployment
        "num_gpus": os.environ.get("NUM_GPUS", 0),
        # Memory limit for the deployment (1 GB)
        "memory": os.environ.get("MEMORY", 1024 * 1024 * 1024),
        # Runtime environment for the deployment (e.g., dependencies, environment variables)
        "runtime_env": {
            "pip": [
                "pandas",  # Example dependency, can be replaced with any other library
            ],
            "env_vars": {
                "EXAMPLE_ENV_VAR": "example_value",  # Example environment variable
            },
        },
    }
)
class DemoDeployment:
    def __init__(self):
        """Initialize the application."""
        self.start_time = time.time()

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
        print("Initializing DemoDeployment...")
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
            model = await self._get_model("test_model")

            ping_response = await self.ping()

            ascii_art_response = await self.ascii_art()

            print("Deployment test passed")

            return True
        except Exception as e:
            print(f"Deployment test failed: {e}")
            return False

    # === Internal Methods ===

    @serve.multiplexed(max_num_models_per_replica=3)
    async def _get_model(self, model_id: str) -> Any:
        """
        An optional method to load multiplexed models in a replica. If defined, the decorator
        `@ray.serve.multiplexed` will be added to the method.

        Requirements:
        - Must be an async method.
        - Must accept exactly one argument: model_id (str).
        - Must return the loaded model (can be any type, e.g., a machine learning model).
        - Can not be called using a keyword argument.

        The parameter `max_num_models_per_replica` can be used to change the default maximum
        number of models per replica (default is 3).
        """
        # Mock model loading logic
        print(f"Loading model with ID: {model_id}")
        model = None

        return model

    # === Exposed BioEngine App Methods - all methods decorated with @schema_method will be exposed as API endpoints ===
    # Note: Parameter type hints and docstrings will be used to generate the API documentation.

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float]]:
        """
        Ping the model to test connectivity.

        Args:
            None

        Returns:
            Dict[str, Union[str, float]]: A dictionary containing the 'status', 'message', 'timestamp', and 'uptime'.
        """
        return {
            "status": "ok",
            "message": "Hello from the DemoDeployment!",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
        }

    @schema_method
    async def ascii_art(self) -> List[str]:
        """
        Get an ASCII art representation of the word 'Bioengine'.

        Args:
            None

        Returns:
            List[str]: A list of strings representing the ASCII art.
        """

        ascii_art = [
            """|================================================================================================|""",
            """|                                                                                                |""",
            """|                                                                                                |""",
            """|  oooooooooo.   o8o            oooooooooooo                         o8o                         |""",
            """|  `888'   `Y8b  `''            `888'     `8                         `''                         |""",
            """|   888     888 oooo   .ooooo.   888         ooo. .oo.    .oooooooo oooo  ooo. .oo.    .ooooo.   |""",
            """|   888oooo888' `888  d88' `88b  888oooo8    `888P'Y88b  888' `88b  `888  `888P'Y88b  d88' `88b  |""",
            """|   888    `88b  888  888   888  888          888   888  888   888   888   888   888  888ooo888  |""",
            """|   888    .88P  888  888   888  888       o  888   888  `88bod8P'   888   888   888  888    .o  |""",
            """|  o888bood8P'  o888o `Y8bod8P' o888ooooood8 o888o o888o `8oooooo.  o888o o888o o888o `Y8bod8P'  |""",
            """|                                                        d'     YD                               |""",
            """|                                                        'Y88888P'                               |""",
            """|                                                                                                |""",
            """|================================================================================================|""",
        ]
        return ascii_art

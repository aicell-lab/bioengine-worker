"""
BioEngine Worker Application Deployment Example

This demonstrates the standard deployment pattern for BioEngine Worker applications.
All deployments must follow these conventions:

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
run_application().

Resource allocation is validated against available cluster resources before deployment.
See bioengine/applications/apps_manager.py for the deployment orchestration logic.

Ray Serve deployment parameters: https://docs.ray.io/en/latest/serve/api/doc/ray.serve.deployment_decorator.html
BioEngine app deployment guide: See project README for artifact structure requirements.
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Union
import logging

from hypha_rpc.utils.schema import schema_method
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        # Number of CPUs to allocate for the deployment
        "num_cpus": 1,
        # Number of GPUs to allocate for the deployment
        # This can be set to 0 is the parameter `disable_gpu` is set to True when deploying
        "num_gpus": 1,
        # Memory limit for the deployment (0.5 GB)
        "memory": 0.5 * 1024**3,
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
    def __init__(self, test_param: str = "default_value") -> None:
        """Initialize the application."""
        self.start_time = time.time()
        self.fail_health_check = False

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
        await asyncio.sleep(0.01)

    async def test_deployment(self) -> None:
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
        model = await self._get_model("test_model")

        ping_response = await self.ping()

        ascii_art_response = await self.ascii_art()

        datasets = await self.list_datasets()

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
        logger.info(f"Loading model with ID: {model_id}")
        model = None

        return model

    async def check_health(self) -> None:
        """
        An optional method to check the health of the deployment. If defined, it will be called
        periodically by Ray Serve to monitor the health of the deployment.

        Requirements:
        - Must be an async method.
        - Must not accept any arguments.
        """
        if self.fail_health_check:
            raise Exception("Simulated health check failure.")

    # === Exposed BioEngine App Methods - all methods decorated with @schema_method will be exposed as API endpoints ===
    # Note: Parameter type hints and docstrings will be used to generate the API documentation.

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float]]:
        """
        Ping the model to test connectivity.

        Returns:
            Dict[str, Union[str, float]]: A dictionary containing the 'status', 'message', 'timestamp', and 'uptime'.
        """
        return {
            "status": "ok",
            "message": "Hello from the DemoDeployment!",
            "timestamp": datetime.now().isoformat(),
            "timezone": time.tzname[0],
            "uptime": time.time() - self.start_time,
        }

    @schema_method
    async def ascii_art(self) -> List[str]:
        """
        Get an ASCII art representation of the word 'Bioengine'.

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
    
    @schema_method
    async def list_datasets(self) -> Dict[str, List[str]]:
        """
        List available datasets and their files from BioEngine Datasets.

        Returns:
            Dict[str, List[str]]: A dictionary with dataset IDs as keys and lists of file names as values.
        """
        # Test BioEngine datasets
        available_datasets = await self.bioengine_datasets.list_datasets()
        logger.info(f"Available datasets: {list(available_datasets.keys())}")

        data_files = {}
        for dataset_id in available_datasets.keys():
            file_names = await self.bioengine_datasets.list_files(dataset_id)
            data_files[dataset_id] = file_names
            logger.info(f"Files in dataset {dataset_id}: {file_names}")

        return data_files

    @schema_method
    async def set_fail_health_check(self) -> None:
        """
        Set the deployment to fail the health check.
        """
        self.fail_health_check = True

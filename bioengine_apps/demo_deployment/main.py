# All standard python libraries can be imported at the top of this file.
import asyncio
import time
from datetime import datetime
from typing import Any

# All non standard libraries need to be imported in each method where they are used.


class DemoDeployment(object):
    def __init__(self):
        pass

    # === Internal Bioengine App Methods ===

    async def _async_init(self) -> None:
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

    async def _get_model(self, model_id: str) -> Any:
        """
        An optional method to load multiplexed models in a replica. If defined, the decorator
        `@ray.serve.multiplexed` will be added to the method.

        Requirements:
        - Must be an async method.
        - Must accept exactly one argument: model_id (str).
        - Must return the loaded model (can be any type, e.g., a machine learning model).

        The entry `deployment_class.max_num_models_per_replica = <int>` in the manifest
        can be used to change the default maximum number of models per replica (default is 3).
        """
        # Mock model loading logic
        print(f"Loading model with ID: {model_id}")
        model = None

        return model

    async def _test_deployment(self) -> bool:
        """
        An optional method to test the deployment. If defined, it will be called when the deployment
        is started to check if the deployment is working correctly.

        Requirements:
        - Must be an async method.
        - Must not accept any arguments.
        - Must return a boolean value indicating whether the deployment is working correctly.
        """
        try:
            model = await self._get_model(model_id="test_model")
            return True
        except Exception as e:
            return False

    # === Replace with your own methods ===

    async def ping(self) -> str:
        """An example method to test connectivity."""
        return {
            "status": "ok",
            "message": "Hello from the DemoDeployment!",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time
        }

    async def ascii_art(self) -> dict:
        """An example method that returns ASCII art of the word 'Bioengine'."""

        ascii_art_str = [
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
        return ascii_art_str

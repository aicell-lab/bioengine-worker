"""Single-deployment BioEngine hello-world app."""
import logging
import time
from datetime import datetime
from typing import Dict, Union

from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 512 * 1024**2,
        "runtime_env": {
            # No extra pip packages needed — pure stdlib
            "pip": [],
        },
    },
    max_ongoing_requests=10,
)
class HelloDeployment:
    def __init__(self) -> None:
        self.start_time = time.time()

    async def async_init(self) -> None:
        """Async setup."""
        logger.info("HelloDeployment async_init complete")

    async def test_deployment(self) -> None:
        """Smoke test."""
        result = await self.ping()
        assert result["status"] == "ok", f"ping failed: {result}"
        reversed_result = await self.reverse_text(text="hello")
        assert reversed_result["reversed"] == "olleh", f"reverse_text failed: {reversed_result}"
        logger.info("HelloDeployment test_deployment passed")

    async def check_health(self) -> None:
        """Periodic health check."""
        pass

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float]]:
        """Ping the service.

        Returns:
            dict: status, message, timestamp, uptime_seconds
        """
        return {
            "status": "ok",
            "message": "Hello from BioEngine!",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
        }

    @schema_method
    async def reverse_text(
        self,
        text: str = Field(..., description="Text to reverse"),
    ) -> dict:
        """Reverse a string and return metadata.

        Returns:
            dict: original, reversed, length
        """
        return {
            "original": text,
            "reversed": text[::-1],
            "length": len(text),
        }

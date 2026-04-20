"""Entry deployment — orchestrates RuntimeA, RuntimeB, RuntimeC."""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Union

from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve
from ray.serve.handle import DeploymentHandle

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        # Entry deployment only routes requests — no CPU/GPU needed.
        "num_cpus": 0,
        "num_gpus": 0,
        "memory": 256 * 1024**2,
        "runtime_env": {
            "pip": [],
        },
    },
    max_ongoing_requests=20,
)
class EntryDeployment:
    def __init__(
        self,
        runtime_a: DeploymentHandle,   # must match filename "runtime_a" in manifest
        runtime_b: DeploymentHandle,   # must match filename "runtime_b"
        runtime_c: DeploymentHandle,   # must match filename "runtime_c"
    ) -> None:
        self.runtime_a = runtime_a
        self.runtime_b = runtime_b
        self.runtime_c = runtime_c
        self.start_time = time.time()

    async def async_init(self) -> None:
        logger.info("EntryDeployment async_init complete")

    async def test_deployment(self) -> None:
        """Test that all runtimes respond."""
        ping_a = await self.runtime_a.ping.remote()
        ping_b = await self.runtime_b.ping.remote()
        ping_c = await self.runtime_c.ping.remote()
        assert ping_a == "pong", f"runtime_a ping failed: {ping_a}"
        assert ping_b == "pong", f"runtime_b ping failed: {ping_b}"
        assert ping_c == "pong", f"runtime_c ping failed: {ping_c}"
        logger.info("EntryDeployment test_deployment passed")

    @schema_method
    async def status(self) -> dict:
        """Get status of entry and all runtimes.

        Returns:
            dict: status from entry and each runtime
        """
        a, b, c = await asyncio.gather(
            self.runtime_a.get_status.remote(),
            self.runtime_b.get_status.remote(),
            self.runtime_c.get_status.remote(),
        )
        return {
            "entry_uptime": time.time() - self.start_time,
            "runtime_a": a,
            "runtime_b": b,
            "runtime_c": c,
        }

    @schema_method
    async def process_text(
        self,
        text: str = Field(..., description="Text to process"),
    ) -> dict:
        """Process text through RuntimeA (text operations).

        Returns:
            dict: word count, char count, reversed, upper/lower case
        """
        return await self.runtime_a.process_text.remote(text)

    @schema_method
    async def analyze_numbers(
        self,
        values: list = Field(..., description="List of numbers to analyze"),
    ) -> dict:
        """Run statistical analysis through RuntimeB (numpy).

        Returns:
            dict: mean, std, min, max, sum
        """
        return await self.runtime_b.analyze.remote(values)

    @schema_method
    async def time_operations(
        self,
        count: int = Field(5, description="Number of timestamps to generate"),
    ) -> dict:
        """Get time-based string operations through RuntimeC.

        Returns:
            dict: timestamps and formatted strings
        """
        return await self.runtime_c.time_ops.remote(count)

    @schema_method
    async def run_all(
        self,
        text: str = Field("hello bioengine", description="Text input for RuntimeA"),
        values: list = Field(None, description="Numbers for RuntimeB (defaults to [1,2,3,4,5])"),
        count: int = Field(3, description="Count for RuntimeC"),
    ) -> dict:
        """Run all three runtimes in parallel and combine results.

        Returns:
            dict: combined results from all runtimes
        """
        if values is None:
            values = [1, 2, 3, 4, 5]
        text_result, data_result, time_result = await asyncio.gather(
            self.runtime_a.process_text.remote(text),
            self.runtime_b.analyze.remote(values),
            self.runtime_c.time_ops.remote(count),
        )
        return {
            "text_result": text_result,
            "data_result": data_result,
            "time_result": time_result,
        }

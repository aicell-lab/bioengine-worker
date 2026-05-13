"""RuntimeC — string/time operations (no extra pip packages)."""
import logging
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 512 * 1024**2,
        "runtime_env": {
            "pip": [
                # Match the BioEngine driver's pydantic-core (see SKILL.md).
                "pydantic==2.11.0",
            ],
        },
    },
    max_ongoing_requests=5,
)
class RuntimeC:
    def __init__(self) -> None:
        pass

    async def async_init(self) -> None:
        logger.info("RuntimeC ready")

    async def test_deployment(self) -> None:
        result = await self.time_ops(3)
        assert "timestamps" in result, f"Expected timestamps in result: {result}"

    async def ping(self) -> str:
        return "pong"

    async def get_status(self) -> dict:
        import datetime
        return {
            "name": "runtime_c",
            "status": "ok",
            "current_time": datetime.datetime.now().isoformat(),
        }

    async def time_ops(self, count: int = 5) -> dict:
        """Generate timestamps and formatted date strings."""
        import datetime
        import time

        now = datetime.datetime.now()
        timestamps = []
        for i in range(count):
            dt = now - datetime.timedelta(hours=i)
            timestamps.append(dt.isoformat())

        return {
            "current_timestamp": now.isoformat(),
            "unix_timestamp": time.time(),
            "timestamps": timestamps,
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "count": count,
        }

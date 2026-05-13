"""RuntimeB — math/stats operations with numpy."""
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
                "numpy==1.26.4",
            ],
        },
    },
    max_ongoing_requests=5,
)
class RuntimeB:
    def __init__(self) -> None:
        pass

    async def async_init(self) -> None:
        import numpy as np
        logger.info(f"RuntimeB ready (numpy {np.__version__})")

    async def test_deployment(self) -> None:
        result = await self.analyze([1, 2, 3, 4, 5])
        assert "mean" in result, f"Expected mean in result: {result}"

    async def ping(self) -> str:
        return "pong"

    async def get_status(self) -> dict:
        import numpy as np
        return {"name": "runtime_b", "status": "ok", "numpy_version": np.__version__}

    async def analyze(self, values: list) -> dict:
        """Run statistical analysis on a list of numbers."""
        import numpy as np
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sum": float(np.sum(arr)),
            "count": len(arr),
            "sorted": sorted(values),
        }

import time

from ray import serve


@serve.deployment(
    ray_actor_options={
        # Number of CPUs to allocate for the deployment
        "num_cpus": 1,
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
    }
)
class Deployment1:
    def __init__(self) -> None:
        self.start_time = time.time()

    async def elapsed_time(self) -> str:
        return time.time() - self.start_time

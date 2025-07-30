import os

from ray import serve


@serve.deployment(
    ray_actor_options={
        # Number of CPUs to allocate for the deployment
        "num_cpus": 1,
        # Number of GPUs to allocate for the deployment
        "num_gpus": 1 if os.environ["BIOENGINE_ENABLE_GPU"] else 0,
        # Memory limit for the deployment (0.5 GB)
        "memory": 0.5 * 1024 * 1024 * 1024,
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
class Deployment2:
    def __init__(self, start_number: int) -> None:
        self.start_number = start_number

    async def add(self, number: int) -> int:
        return self.start_number + number

import time

from ray import serve

@serve.deployment
class Deployment1:
    def __init__(self) -> None:
        self.start_time = time.time()

    async def elapsed_time(self) -> str:
        return time.time() - self.start_time
from ray import serve

@serve.deployment
class Deployment2:
    def __init__(self, start: int) -> None:
        self.start = start

    async def add(self, number: int) -> int:
        return self.start + number
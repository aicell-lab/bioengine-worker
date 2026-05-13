"""RuntimeA — text operations (no extra pip packages)."""
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
class RuntimeA:
    def __init__(self) -> None:
        pass

    async def async_init(self) -> None:
        logger.info("RuntimeA ready")

    async def test_deployment(self) -> None:
        result = await self.process_text("hello world")
        assert "word_count" in result, f"Expected word_count in result: {result}"

    async def ping(self) -> str:
        return "pong"

    async def get_status(self) -> dict:
        return {"name": "runtime_a", "status": "ok", "capabilities": ["text_processing"]}

    async def process_text(self, text: str) -> dict:
        """Process text — split, count, transform."""
        words = text.split()
        return {
            "word_count": len(words),
            "char_count": len(text),
            "char_count_no_spaces": len(text.replace(" ", "")),
            "words": words,
            "reversed": text[::-1],
            "upper": text.upper(),
            "lower": text.lower(),
            "title": text.title(),
        }

"""Start a training session for debugging."""
import asyncio
import os

from hypha_rpc import connect_to_server, login


async def start_training() -> str:
    """Start a training session and return the session ID."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:  # type: ignore[generalTypeIssues]
        cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")

        # Start training
        result = await cellpose_service.start_training(
            artifact="ri-scale/zarr-demo",
            metadata_path="rdf.yaml",
            model="nuclei",
            n_samples=2,
        )
        print(f"Training started with session ID: {result['session_id']}")  # noqa: T201
        return result["session_id"]


if __name__ == "__main__":
    asyncio.run(start_training())

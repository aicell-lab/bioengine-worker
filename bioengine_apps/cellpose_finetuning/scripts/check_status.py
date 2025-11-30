"""Check the status of a training session."""
import asyncio
import os
import sys

from hypha_rpc import connect_to_server, login


async def check_status(session_id: str) -> None:
    """Check the status of a training session."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:  # type: ignore[generalTypeIssues]
        cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
        status = await cellpose_service.get_training_status(session_id)
        print(f"Status: {status}")  # noqa: T201


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_status.py <session_id>")  # noqa: T201
        sys.exit(1)
    asyncio.run(check_status(sys.argv[1]))

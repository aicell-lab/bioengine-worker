"""Debug task information."""
import asyncio
import os
import sys

from hypha_rpc import connect_to_server, login


async def debug_task(session_id: str) -> None:
    """Get debug info about a training task."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:  # type: ignore[generalTypeIssues]
        cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
        info = await cellpose_service.debug_task_info(session_id)
        print(f"Task Info: {info}")  # noqa: T201


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_task.py <session_id>")  # noqa: T201
        sys.exit(1)
    asyncio.run(debug_task(sys.argv[1]))

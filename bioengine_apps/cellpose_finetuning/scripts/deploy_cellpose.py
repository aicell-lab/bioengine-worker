"""Deploy the cellpose finetuning application to BioEngine."""

import argparse
import asyncio
import os

from hypha_rpc import connect_to_server, login


async def deploy(artifact_id: str, application_id: str):
    """Deploy the cellpose finetuning application."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})
    hypha = await connect_to_server({"server_url": server_url, "token": token})
    workspace = hypha.config.workspace
    print(f"Using workspace: {workspace}")

    worker = await hypha.get_service("bioimage-io/bioengine-worker")

    app_id = await worker.run_application(
        artifact_id=artifact_id,
        application_id=application_id,
        hypha_token=token,
        version=None,  # latest
        disable_gpu=False,  # set True to force CPU-only
        max_ongoing_requests=1,  # keep at 1 for GPU
    )
    print(f"App ID: {app_id}")

    # Wait for services to become available
    print("Waiting for services to start...")
    for _ in range(30):
        app_status = await worker.get_application_status(application_ids=[app_id])
        service_ids = app_status.get("service_ids", [])
        if service_ids:
            print(f"Services: {service_ids}")
            return service_ids
        await asyncio.sleep(5)
    print("Warning: Services not yet available. The deployment may still be starting up.")
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy the cellpose finetuning application to BioEngine.",
    )
    parser.add_argument(
        "--artifact-id",
        type=str,
        default="bioimage-io/cellpose-finetuning",
        help="Artifact ID to deploy (default: bioimage-io/cellpose-finetuning)",
    )
    parser.add_argument(
        "--application-id",
        type=str,
        default="cellpose-finetuning",
        help="Application ID to deploy (default: cellpose-finetuning)",
    )
    args = parser.parse_args()
    asyncio.run(deploy(artifact_id=args.artifact_id, application_id=args.application_id))

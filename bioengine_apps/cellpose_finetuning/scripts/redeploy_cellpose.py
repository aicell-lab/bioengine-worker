"""Redeploy the cellpose finetuning application with the latest code."""

import argparse
import asyncio
import os

from hypha_rpc import connect_to_server, login


async def redeploy(artifact_id: str, application_id: str):
    """Stop the old deployment and start a new one with the latest artifact."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})
    hypha = await connect_to_server({"server_url": server_url, "token": token})
    workspace = hypha.config.workspace
    print(f"Using workspace: {workspace}")

    worker = await hypha.get_service("bioimage-io/bioengine-worker")

    # Stop the existing application if it's running
    try:
        print("Stopping existing application...")
        await worker.stop_application(application_id)
        print("Existing application stopped")
        # Wait a bit for cleanup
        await asyncio.sleep(5)
    except Exception as e:
        print(f"Note: Could not stop existing application (might not be running): {e}")

    # Start the new deployment
    print("Starting new deployment with latest artifact...")
    app_id = await worker.run_application(
        artifact_id=artifact_id,
        application_id=application_id,
        hypha_token=token,
        version=None,  # latest version
        disable_gpu=False,  # set True to force CPU-only
        max_ongoing_requests=1,  # keep at 1 for GPU
    )
    print(f"App ID: {app_id}")

    # Get the new service IDs
    app_status = await worker.get_application_status(application_ids=[app_id])
    service_ids = app_status["service_ids"]
    print(f"Services: {service_ids}")
    return service_ids


if __name__ == "__main__":
    # add argparse argument artifact Id and application id
    parser = argparse.ArgumentParser(
        description="Redeploy the cellpose finetuning application with the latest code.",
    )
    parser.add_argument(
        "--artifact-id",
        type=str,
        default="bioimage-io/cellpose-finetuning-test",
        help="Artifact ID to deploy (default: bioimage-io/cellpose-finetuning-test)",
    )
    parser.add_argument(
        "--application-id",
        type=str,
        default="cellpose-finetuning-test",
        help="Application ID to deploy (default: cellpose-finetuning-test)",
    )
    args = parser.parse_args()
    asyncio.run(
        redeploy(
            artifact_id=args.artifact_id,
            application_id=args.application_id,
        )
    )

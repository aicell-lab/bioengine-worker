"""Test that dataset_artifact_id is returned in training status."""

import asyncio
import os
from hypha_rpc import connect_to_server

async def main():
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN")

    if not token:
        print("ERROR: HYPHA_TOKEN not set")
        return

    print(f"Connecting to {server_url}...")
    async with connect_to_server({"server_url": server_url, "token": token}) as server:
        print("Connected! Getting service...")
        cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
        print("Got service! Starting training...")

        # Start a minimal training session
        session_status = await cellpose_service.start_training(
            artifact="ri-scale/zarr-demo",
            train_images="images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif",
            train_annotations="annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif",
            model="cpsam",
            n_epochs=1,
            n_samples=1,
        )

        session_id = session_status["session_id"]
        print(f"Training started: {session_id}")

        # Immediately check status to see if dataset_artifact_id is present
        status = await cellpose_service.get_training_status(session_id)

        print("\n=== Status Response ===")
        print(f"Status type: {status.get('status_type')}")
        print(f"Message: {status.get('message')}")
        print(f"Dataset artifact ID: {status.get('dataset_artifact_id', 'NOT FOUND')}")

        if 'dataset_artifact_id' in status:
            print(f"\n✅ SUCCESS! dataset_artifact_id is present: {status['dataset_artifact_id']}")
        else:
            print("\n❌ FAILURE! dataset_artifact_id is missing from status")
            print("\nFull status:")
            print(status)

if __name__ == "__main__":
    asyncio.run(main())

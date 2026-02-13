"""Test real-time epoch callbacks during training."""

import asyncio
import os
from hypha_rpc import connect_to_server, login


async def test_callbacks():
    """Test training with multiple epochs to verify callback functionality."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:
        cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")

        print("Starting training with 5 epochs to test real-time callbacks...")
        session_status = await cellpose_service.start_training(
            artifact="ri-scale/zarr-demo",
            train_images="images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif",
            train_annotations="annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif",
            n_epochs=5,
            n_samples=2,
            min_train_masks=1,
        )
        session_id = session_status["session_id"]
        print(f"Training session started: {session_id}")

        # Monitor progress with 1-second updates
        while True:
            status = await cellpose_service.get_training_status(session_id)

            # Build status message
            msg = f"[{status['status_type']}] {status['message']}"

            # Add metrics if available
            if "current_epoch" in status and "total_epochs" in status:
                msg += f" | Epoch: {status['current_epoch']}/{status['total_epochs']}"

            if "elapsed_seconds" in status:
                msg += f" | Time: {status['elapsed_seconds']:.1f}s"

            if "train_losses" in status and status["train_losses"]:
                losses = [l for l in status["train_losses"] if l is not None and l > 0]
                if losses:
                    msg += f" | Train Loss: {losses[-1]:.4f}"

            print(msg)

            if status["status_type"] in ("completed", "failed"):
                break

            await asyncio.sleep(1)

        print("\n" + "=" * 80)
        print("Final Status:")
        print("=" * 80)
        if "train_losses" in status:
            print(f"Training losses: {[f'{l:.4f}' for l in status['train_losses'] if l is not None and l > 0]}")
        if "test_losses" in status:
            print(f"Test losses: {[f'{l:.4f}' for l in status['test_losses'] if l is not None]}")


if __name__ == "__main__":
    asyncio.run(test_callbacks())

"""Test the Cellpose Fine-Tuning service."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from hypha_rpc import connect_to_server, login

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService

# load .env

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def infer(
    cellpose_service: RemoteService,
    model: str | None = None,
) -> None:
    """Test inference functionality of the Cellpose Fine-Tuning service."""
    inference_result = await cellpose_service.infer(
        model=model or "cpsam",
        artifact="ri-scale/zarr-demo",
        diameter=40,
        image_paths=["images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif"],
    )
    logger.info("Inference done! Result: %s", str(inference_result)[:500] + "...")
    arr = inference_result[0]["output"]
    logger.info("Output array shape: %s", arr.shape)


async def infer_with_numpy(
    cellpose_service: RemoteService,
    model: str | None = None,
) -> None:
    """Test inference with numpy array input."""
    import numpy as np

    # Create a test image as numpy array (3 channels, 512x512)
    test_image = np.random.randint(0, 255, (3, 512, 512), dtype=np.uint8)

    logger.info("Testing inference with numpy array input (shape: %s)", test_image.shape)

    inference_result = await cellpose_service.infer(
        model=model or "cpsam",
        input_arrays=[test_image],
        diameter=40,
    )

    logger.info("Numpy inference done! Result: %s", str(inference_result)[:500] + "...")
    output_mask = inference_result[0]["output"]
    logger.info("Output mask shape: %s", output_mask.shape)
    logger.info("Unique mask values: %d", len(np.unique(output_mask)))


async def get_cellpose_service(server: RemoteService) -> RemoteService:
    """Obtain the Cellpose Fine-Tuning service from the Hypha server."""
    workspace = server.config.workspace
    logger.info("Using workspace: %s", workspace)
    cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
    logger.info("Obtained Cellpose Fine-Tuning service!")
    return cellpose_service


async def start_training(
    cellpose_service: RemoteService,
    artifact: str,
    n_epochs: int,
    n_samples: int = 2,
) -> str:
    """Start a training session on the Cellpose Fine-Tuning service."""
    session_status = await cellpose_service.start_training(
        artifact=artifact,
        n_epochs=n_epochs,
        n_samples=n_samples,
        test_indices=[1],  # Use second sample for testing
        min_train_masks=1,  # Allow training with samples that have at least 1 mask (for testing)
    )
    session_id = session_status["session_id"]
    logger.info("Started training with session ID: %s", session_id)
    return session_id


async def monitor_training(
    cellpose_service: RemoteService,
    session_id: str,
) -> None:
    """Monitor the training session until completion."""
    status = None
    current_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{current_time}] Starting training monitoring...")  # noqa: T201
    while True:
        status = await cellpose_service.get_training_status(session_id)
        current_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

        # Build status message with metrics if available
        message = f"\r[{current_time}] {status['message']}"

        # Add current training metrics if available
        if "train_losses" in status and status["train_losses"]:
            train_losses = status["train_losses"]
            # Find the last non-zero train loss
            current_train_loss = next((loss for loss in reversed(train_losses) if loss > 0), None)
            if current_train_loss is not None:
                message += f" | Train Loss: {current_train_loss:.4f}"

        if "test_losses" in status and status["test_losses"]:
            test_losses = status["test_losses"]
            # Find the last non-zero test loss
            current_test_loss = next((loss for loss in reversed(test_losses) if loss > 0), None)
            if current_test_loss is not None:
                message += f" | Test Loss: {current_test_loss:.4f}"

        print(message, end="")  # noqa: T201

        if status["status_type"] in ("completed", "failed"):
            print()  # noqa: T201 # Add newline after completion
            break
        await asyncio.sleep(1)

    # Print final training metrics summary if available
    if status and status["status_type"] == "completed":
        if "train_losses" in status and status["train_losses"]:
            train_losses = status["train_losses"]
            non_zero_train = [loss for loss in train_losses if loss > 0]
            if non_zero_train:
                print(f"\nTraining Metrics Summary:")  # noqa: T201
                print(f"  Epochs: {len(non_zero_train)}")  # noqa: T201
                print(f"  Initial Train Loss: {non_zero_train[0]:.4f}")  # noqa: T201
                print(f"  Final Train Loss: {non_zero_train[-1]:.4f}")  # noqa: T201

                if "test_losses" in status and status["test_losses"]:
                    test_losses = status["test_losses"]
                    non_zero_test = [loss for loss in test_losses if loss > 0]
                    if non_zero_test:
                        print(f"  Final Test Loss: {non_zero_test[-1]:.4f}")  # noqa: T201
                        print(f"  Test evaluations: {len(non_zero_test)}")  # noqa: T201


async def main(session_id: str | None = None) -> None:
    """Test the Cellpose Fine-Tuning service."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:  # type: ignore[generalTypeIssues]
        cellpose_service = await get_cellpose_service(server)

        if session_id is None:
            await infer(cellpose_service, model="cpsam")

            # Test numpy array inference
            await infer_with_numpy(cellpose_service, model="cpsam")

            session_id = await start_training(
                cellpose_service,
                artifact="ri-scale/zarr-demo",
                n_epochs=1,
            )

        await monitor_training(cellpose_service, session_id)

        await infer(cellpose_service, model=session_id)

        # Test numpy array inference with fine-tuned model
        await infer_with_numpy(cellpose_service, model=session_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the Cellpose Fine-Tuning service.",
    )
    parser.add_argument(
        "--session",
        "-s",
        default=None,
        type=str,
        help=(
            "Training session ID to monitor (if not provided,"
            " a new training session will be started)."
        ),
    )
    args = parser.parse_args()
    session_id = args.session

    asyncio.run(main(session_id=session_id))
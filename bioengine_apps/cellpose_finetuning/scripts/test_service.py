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
        train_images="images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif",
        train_annotations="annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif",
        n_epochs=n_epochs,
        n_samples=n_samples,
        min_train_masks=1,  # Allow training with samples that have at least 1 mask (for testing)
    )
    session_id = session_status["session_id"]
    logger.info("Started training with session ID: %s", session_id)
    return session_id


async def monitor_training(
    cellpose_service: RemoteService,
    session_id: str,
) -> None:
    """Monitor the training session until completion with enhanced status display."""
    status = None
    current_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{current_time}] Starting training monitoring...")  # noqa: T201
    while True:
        status = await cellpose_service.get_training_status(session_id)

        # Build comprehensive status message with all available metrics
        msg_parts = [f"[{status['status_type']}]", status['message']]

        # Add dataset info (if available and at start)
        if "n_train" in status and status.get("current_epoch", 0) == 0:
            msg_parts.append(f"| Train: {status['n_train']} samples")
            if "n_test" in status and status.get('n_test', 0) > 0:
                msg_parts[-1] += f", Test: {status['n_test']} samples"

        # Add epoch progress
        if "current_epoch" in status and "total_epochs" in status:
            msg_parts.append(f"| Epoch: {status['current_epoch']}/{status['total_epochs']}")

        # Add elapsed time
        if "elapsed_seconds" in status:
            msg_parts.append(f"| Time: {status['elapsed_seconds']:.1f}s")

        # Add current training metrics if available
        if "train_losses" in status and status["train_losses"]:
            train_losses = status["train_losses"]
            # Find the last non-zero train loss
            current_train_loss = next((loss for loss in reversed(train_losses) if loss > 0), None)
            if current_train_loss is not None:
                msg_parts.append(f"| Train Loss: {current_train_loss:.4f}")

        if "test_losses" in status and status["test_losses"]:
            test_losses = status["test_losses"]
            # Find the last non-zero test loss
            current_test_loss = next((loss for loss in reversed(test_losses) if loss > 0), None)
            if current_test_loss is not None:
                msg_parts.append(f"| Test Loss: {current_test_loss:.4f}")

        message = f"\r{' '.join(msg_parts)}"
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


async def export_model(
    cellpose_service: RemoteService,
    server: RemoteService,
    session_id: str,
) -> str:
    """Export the trained model and download cover image."""
    import httpx

    logger.info("Exporting model from session: %s", session_id)

    # Export the model
    export_result = await cellpose_service.export_model(
        session_id=session_id,
        model_name=f"test-model-{session_id[:8]}",
        collection="bioimage-io/colab-annotations",
    )

    artifact_id = export_result["artifact_id"]
    artifact_url = export_result["artifact_url"]
    download_url = export_result["download_url"]

    logger.info("Model exported successfully!")
    logger.info("  Artifact ID: %s", artifact_id)
    logger.info("  Artifact URL: %s", artifact_url)
    logger.info("  Download URL: %s", download_url)

    # Get artifact manager and download cover image to verify
    artifact_manager = await server.get_service("public/artifact-manager")

    cover_url = await artifact_manager.get_file(artifact_id, "cover.png")

    async with httpx.AsyncClient() as client:
        resp = await client.get(cover_url)
        cover_path = f"exported_cover_{session_id[:8]}.png"
        with open(cover_path, "wb") as f:
            f.write(resp.content)

    logger.info("  Downloaded cover image: %s", cover_path)

    return artifact_id


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

        # Test model export
        artifact_id = await export_model(cellpose_service, server, session_id)
        logger.info("All tests passed! Model artifact: %s", artifact_id)


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
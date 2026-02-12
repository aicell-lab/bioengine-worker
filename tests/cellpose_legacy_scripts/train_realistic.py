"""Realistic long-running training test for Cellpose Fine-Tuning service.

This script performs a full training run with all available samples and
displays comprehensive real-time progress tracking including:
- Dataset information (train/test sample counts)
- Epoch progress (current/total)
- Training and test losses
- Elapsed time
- Start time
- Per-epoch validation with IoU score tracking
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
from dotenv import load_dotenv
from hypha_rpc import connect_to_server, login

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute mean IoU score between prediction and ground truth masks.

    Args:
        pred_mask: Predicted segmentation mask (H, W) with instance labels
        gt_mask: Ground truth segmentation mask (H, W) with instance labels

    Returns:
        Mean IoU score across all instances
    """
    # Get unique instances (excluding background=0)
    pred_instances = set(np.unique(pred_mask)) - {0}
    gt_instances = set(np.unique(gt_mask)) - {0}

    if len(gt_instances) == 0:
        logger.warning("No ground truth instances found")
        return 0.0

    # For each ground truth instance, find best matching predicted instance
    ious = []
    for gt_id in gt_instances:
        gt_binary = (gt_mask == gt_id)

        # Find best matching prediction
        best_iou = 0.0
        for pred_id in pred_instances:
            pred_binary = (pred_mask == pred_id)

            intersection = np.logical_and(gt_binary, pred_binary).sum()
            union = np.logical_or(gt_binary, pred_binary).sum()

            if union > 0:
                iou = intersection / union
                best_iou = max(best_iou, iou)

        ious.append(best_iou)

    return float(np.mean(ious)) if ious else 0.0


async def download_image(url: str, token: str | None = None) -> np.ndarray:
    """Download image from URL and return as numpy array.

    Args:
        url: URL to download image from
        token: Optional authentication token (passed as Bearer token in header)

    Returns:
        Downloaded image as numpy array
    """
    try:
        # Import tifffile here to avoid requiring it for all users
        from tifffile import imread

        # Build headers with authentication if token provided
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            # Save to temp file and load with tifffile
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            img = imread(tmp_path)
            os.unlink(tmp_path)
            return img

    except ImportError:
        logger.error("tifffile package required for loading images. Install with: pip install tifffile")
        raise
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


async def run_validation_inference(
    cellpose_service: RemoteService,
    model_id: str,
    artifact: str,
    image_path: str,
    gt_mask: np.ndarray | None = None,
    save_path: Path | None = None,
    epoch: int | None = None,
) -> dict:
    """Run inference on validation image and compute metrics.

    Args:
        cellpose_service: Cellpose service
        model_id: Model ID to use for inference
        artifact: Artifact ID containing the image
        image_path: Path to image within artifact
        gt_mask: Ground truth mask for computing IoU (optional)
        save_path: Path to save results (optional)
        epoch: Current epoch number (optional)

    Returns:
        Dict with 'mask', 'iou' (if gt_mask provided), and metadata
    """
    try:
        logger.info(f"Running validation inference (epoch {epoch})...")

        result = await cellpose_service.infer(
            model=model_id,
            artifact=artifact,
            image_paths=[image_path],
            diameter=40,
        )

        pred_mask = result[0]["output"]

        # Compute IoU if ground truth provided
        iou = None
        if gt_mask is not None:
            iou = compute_iou(pred_mask, gt_mask)
            logger.info(f"Validation IoU: {iou:.4f}")

        # Save results if requested
        if save_path is not None:
            result_data = {
                "epoch": epoch,
                "model_id": model_id,
                "image_path": image_path,
                "mask_shape": pred_mask.shape,
                "unique_instances": int(len(np.unique(pred_mask)) - 1),  # Exclude background
                "iou": float(iou) if iou is not None else None,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }

            # Save metadata
            save_path.mkdir(parents=True, exist_ok=True)
            epoch_str = f"epoch_{epoch:04d}" if epoch is not None else "initial"
            with open(save_path / f"{epoch_str}_metadata.json", "w") as f:
                json.dump(result_data, f, indent=2)

            # Save mask
            try:
                from tifffile import imwrite
                imwrite(save_path / f"{epoch_str}_mask.tif", pred_mask.astype(np.uint16))
            except ImportError:
                logger.warning("tifffile not available, skipping mask save")

        return {
            "mask": pred_mask,
            "iou": iou,
            "num_instances": int(len(np.unique(pred_mask)) - 1),
        }

    except Exception as e:
        logger.error(f"Validation inference failed: {e}")
        return {"mask": None, "iou": None, "num_instances": 0, "error": str(e)}


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
    train_images: str,
    train_annotations: str,
    test_images: str | None = None,
    test_annotations: str | None = None,
    n_epochs: int = 50,
    model: str = "cpsam",
    learning_rate: float = 1e-6,
    weight_decay: float = 0.0001,
) -> str:
    """Start a realistic training session with all available samples."""
    logger.info("=" * 80)
    logger.info("Starting Realistic Training")
    logger.info("=" * 80)
    logger.info("Artifact: %s", artifact)
    logger.info("Model: %s", model)
    logger.info("Epochs: %d", n_epochs)
    logger.info("Learning rate: %.2e", learning_rate)
    logger.info("Weight decay: %.4f", weight_decay)
    logger.info("Train images pattern: %s", train_images)
    logger.info("Train annotations pattern: %s", train_annotations)
    if test_images and test_annotations:
        logger.info("Test images pattern: %s", test_images)
        logger.info("Test annotations pattern: %s", test_annotations)
    logger.info("=" * 80)

    # Calculate save_every to save only ~10 times during training (min 100 epochs between saves)
    save_every = max(100, n_epochs // 10)

    logger.info("Checkpoint save interval: every %d epochs", save_every)

    session_status = await cellpose_service.start_training(
        artifact=artifact,
        train_images=train_images,
        train_annotations=train_annotations,
        test_images=test_images,
        test_annotations=test_annotations,
        model=model,
        n_epochs=n_epochs,
        n_samples=None,  # Use all available samples
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        min_train_masks=5,  # Require at least 5 masks per image for better training
        save_every=save_every,  # Save checkpoints less frequently to conserve disk space
    )
    session_id = session_status["session_id"]
    logger.info("Training session started: %s", session_id)
    return session_id


async def monitor_training(
    cellpose_service: RemoteService,
    session_id: str,
    update_interval: float = 2.0,
    validation_artifact: str | None = None,
    validation_image: str | None = None,
    validation_gt_mask: np.ndarray | None = None,
    validation_save_path: Path | None = None,
) -> tuple[dict, list[float]]:
    """Monitor training with comprehensive real-time status updates and per-epoch validation.

    Returns:
        Tuple of (final_status, iou_history)
    """
    logger.info("\nMonitoring training progress with real-time updates...\n")

    status = None
    last_epoch = 0
    iou_history = []

    while True:
        status = await cellpose_service.get_training_status(session_id)

        # Build comprehensive progress message
        msg_parts = []

        # Status and message
        msg_parts.append(f"[{status['status_type']}]")
        msg_parts.append(status['message'])

        # Dataset info (only show once when available)
        if "n_train" in status and status.get("current_epoch", 0) == 0:
            msg_parts.append(f"| Train: {status['n_train']} samples")
            if "n_test" in status and status.get('n_test', 0) > 0:
                msg_parts[-1] += f", Test: {status['n_test']} samples"

        # Epoch progress
        if "current_epoch" in status and "total_epochs" in status:
            current = status['current_epoch']
            total = status['total_epochs']
            progress_pct = (current / total * 100) if total > 0 else 0
            msg_parts.append(f"| Epoch: {current}/{total} ({progress_pct:.1f}%)")

        # Elapsed time
        if "elapsed_seconds" in status:
            elapsed = status['elapsed_seconds']
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            msg_parts.append(f"| Time: {time_str}")

        # Latest training loss
        if "train_losses" in status and status["train_losses"]:
            losses = [l for l in status["train_losses"] if l > 0]
            if losses:
                msg_parts.append(f"| Train Loss: {losses[-1]:.4f}")

        # Latest test loss (if available)
        if "test_losses" in status and status["test_losses"]:
            test_losses = [l for l in status["test_losses"] if l > 0]
            if test_losses:
                msg_parts.append(f"| Test Loss: {test_losses[-1]:.4f}")

        # Latest IoU (if available)
        if iou_history:
            msg_parts.append(f"| Val IoU: {iou_history[-1]:.4f}")

        # Print the message
        msg = " ".join(msg_parts)
        print(f"\r{msg}", end="", flush=True)

        # Check if epoch changed - run validation
        current_epoch = status.get("current_epoch", 0)
        if current_epoch > last_epoch and current_epoch > 0:
            print()  # New line for completed epoch

            # Run validation if configured
            if validation_artifact and validation_image:
                val_result = await run_validation_inference(
                    cellpose_service,
                    model_id=session_id,
                    artifact=validation_artifact,
                    image_path=validation_image,
                    gt_mask=validation_gt_mask,
                    save_path=validation_save_path,
                    epoch=current_epoch,
                )

                if val_result["iou"] is not None:
                    iou_history.append(val_result["iou"])

            last_epoch = current_epoch

        if status["status_type"] in ("completed", "failed"):
            print("\n")  # Final newline
            break

        await asyncio.sleep(update_interval)

    return status, iou_history


def print_training_summary(status: dict, iou_history: list[float] | None = None) -> None:
    """Print a comprehensive summary of the training results."""
    print("=" * 80)
    if status["status_type"] == "completed":
        print("Training Completed Successfully!")
    else:
        print(f"Training Ended: {status['status_type']}")
    print("=" * 80)

    # Dataset information
    if "n_train" in status:
        print(f"\nDataset:")
        print(f"  Training samples: {status['n_train']}")
        if "n_test" in status and status.get('n_test', 0) > 0:
            print(f"  Test samples: {status['n_test']}")

    # Timing information
    if "start_time" in status:
        start_time = datetime.fromisoformat(status['start_time'])
        print(f"\nTiming:")
        print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        if "elapsed_seconds" in status:
            elapsed = status['elapsed_seconds']
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            print(f"  Duration: {hours}h {minutes}m {seconds}s ({elapsed:.1f}s total)")

    # Training metrics
    if "train_losses" in status and status["train_losses"]:
        train_losses = [l for l in status["train_losses"] if l > 0]
        if train_losses:
            print(f"\nTraining Metrics:")
            print(f"  Epochs completed: {len(train_losses)}")
            print(f"  Initial train loss: {train_losses[0]:.4f}")
            print(f"  Final train loss: {train_losses[-1]:.4f}")

            # Calculate improvement
            if train_losses[0] > 0:
                improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
                print(f"  Loss improvement: {improvement:.1f}%")

            # Show loss progression for first/last few epochs
            if len(train_losses) > 6:
                print(f"\n  Loss progression:")
                print(f"    Epochs 1-3: {', '.join(f'{l:.4f}' for l in train_losses[:3])}")
                print(f"    Epochs {len(train_losses)-2}-{len(train_losses)}: {', '.join(f'{l:.4f}' for l in train_losses[-3:])}")

    # Test metrics
    if "test_losses" in status and status["test_losses"]:
        test_losses = [l for l in status["test_losses"] if l > 0]
        if test_losses:
            print(f"\nTest Metrics:")
            print(f"  Test evaluations: {len(test_losses)}")
            print(f"  Initial test loss: {test_losses[0]:.4f}")
            print(f"  Final test loss: {test_losses[-1]:.4f}")

            # Calculate improvement
            if test_losses[0] > 0:
                improvement = ((test_losses[0] - test_losses[-1]) / test_losses[0]) * 100
                print(f"  Loss improvement: {improvement:.1f}%")

    # Validation IoU metrics
    if iou_history:
        print(f"\nValidation IoU Metrics:")
        print(f"  Epochs validated: {len(iou_history)}")
        print(f"  Initial IoU (epoch 1): {iou_history[0]:.4f}")
        print(f"  Final IoU (epoch {len(iou_history)}): {iou_history[-1]:.4f}")

        # Calculate improvement
        if iou_history[0] > 0:
            improvement = ((iou_history[-1] - iou_history[0]) / iou_history[0]) * 100
            print(f"  IoU improvement: {improvement:+.1f}%")

        # Show best IoU
        best_iou = max(iou_history)
        best_epoch = iou_history.index(best_iou) + 1
        print(f"  Best IoU: {best_iou:.4f} (epoch {best_epoch})")

        # Show IoU progression for first/last few epochs
        if len(iou_history) > 6:
            print(f"\n  IoU progression:")
            print(f"    Epochs 1-3: {', '.join(f'{iou:.4f}' for iou in iou_history[:3])}")
            print(f"    Epochs {len(iou_history)-2}-{len(iou_history)}: {', '.join(f'{iou:.4f}' for iou in iou_history[-3:])}")

    print("=" * 80)


async def run_inference_test(
    cellpose_service: RemoteService,
    model_id: str,
    artifact: str,
    image_path: str,
) -> None:
    """Run a quick inference test with the trained model."""
    logger.info("\nTesting inference with trained model...")

    try:
        result = await cellpose_service.infer(
            model=model_id,
            artifact=artifact,
            image_paths=[image_path],
            diameter=40,
        )

        mask = result[0]["output"]
        logger.info("Inference successful!")
        logger.info("  Output shape: %s", mask.shape)
        logger.info("  Unique segments: %d", len(set(mask.flatten())))
    except Exception as e:
        logger.error("Inference failed: %s", str(e))


async def main(
    artifact: str = "ri-scale/zarr-demo",
    train_images: str = "images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif",
    train_annotations: str = "annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif",
    test_images: str | None = None,
    test_annotations: str | None = None,
    validation_image_url: str | None = None,
    validation_mask_url: str | None = None,
    validation_save_dir: str | None = None,
    n_epochs: int = 50,
    model: str = "cpsam",
    learning_rate: float = 1e-6,
    weight_decay: float = 0.0001,
    session_id: str | None = None,
    skip_inference: bool = False,
) -> None:
    """Run realistic long-duration training test with per-epoch validation."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:  # type: ignore[generalTypeIssues]
        cellpose_service = await get_cellpose_service(server)

        # Download validation ground truth mask if provided
        validation_gt_mask = None
        if validation_mask_url:
            logger.info("Downloading validation ground truth mask...")
            validation_gt_mask = await download_image(validation_mask_url, token=token)
            logger.info("  Mask shape: %s", validation_gt_mask.shape)
            logger.info("  Instances: %d", len(np.unique(validation_gt_mask)) - 1)

        # Setup validation save path
        validation_save_path = None
        if validation_save_dir:
            validation_save_path = Path(validation_save_dir)
            logger.info("Validation results will be saved to: %s", validation_save_path)

        # Extract validation image path from URL for artifact-based inference
        validation_image_path = None
        if validation_image_url:
            # Extract path after "/files/"
            if "/files/" in validation_image_url:
                validation_image_path = validation_image_url.split("/files/")[1]
                logger.info("Validation image path: %s", validation_image_path)

        # Start training if session_id not provided
        if session_id is None:
            session_id = await start_training(
                cellpose_service,
                artifact=artifact,
                train_images=train_images,
                train_annotations=train_annotations,
                test_images=test_images,
                test_annotations=test_annotations,
                n_epochs=n_epochs,
                model=model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )

            # Run initial validation before training starts
            if validation_image_path:
                logger.info("\n" + "=" * 80)
                logger.info("Running initial validation with pretrained model...")
                logger.info("=" * 80)
                initial_result = await run_validation_inference(
                    cellpose_service,
                    model_id=model,  # Use pretrained model
                    artifact=artifact,
                    image_path=validation_image_path,
                    gt_mask=validation_gt_mask,
                    save_path=validation_save_path,
                    epoch=0,
                )
                if initial_result["iou"] is not None:
                    logger.info("Initial IoU (pretrained): %.4f\n", initial_result["iou"])

        # Monitor training progress with validation
        status, iou_history = await monitor_training(
            cellpose_service,
            session_id,
            validation_artifact=artifact if validation_image_path else None,
            validation_image=validation_image_path,
            validation_gt_mask=validation_gt_mask,
            validation_save_path=validation_save_path,
        )

        # Print summary
        print_training_summary(status, iou_history)

        # Test inference if training succeeded
        if status["status_type"] == "completed" and not skip_inference:
            await run_inference_test(
                cellpose_service,
                model_id=session_id,
                artifact=artifact,
                image_path=train_images.replace("*.ome.tif", "t0000.ome.tif"),
            )

        logger.info("\nSession ID: %s", session_id)
        logger.info("Use this session ID to run inference with the trained model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realistic long-running training test for Cellpose Fine-Tuning.",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default="ri-scale/zarr-demo",
        help="Artifact ID containing training data",
    )
    parser.add_argument(
        "--train-images",
        type=str,
        default="images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif",
        help="Path pattern for training images",
    )
    parser.add_argument(
        "--train-annotations",
        type=str,
        default="annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif",
        help="Path pattern for training annotations",
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default=None,
        help="Path pattern for test images (optional)",
    )
    parser.add_argument(
        "--test-annotations",
        type=str,
        default=None,
        help="Path pattern for test annotations (optional)",
    )
    parser.add_argument(
        "--validation-image-url",
        type=str,
        default="https://hypha.aicell.io/ri-scale/artifacts/zarr-demo/files/images/8fdc2f09-2529-42e0-935e-0b678d2b2623/t0055.ome.tif",
        help="URL to validation image for per-epoch IoU tracking",
    )
    parser.add_argument(
        "--validation-mask-url",
        type=str,
        default="https://hypha.aicell.io/ri-scale/artifacts/zarr-demo/files/annotations/8fdc2f09-2529-42e0-935e-0b678d2b2623/t0055_mask.ome.tif",
        help="URL to validation ground truth mask",
    )
    parser.add_argument(
        "--validation-save-dir",
        type=str,
        default="./validation_results",
        help="Directory to save validation results (default: ./validation_results)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="cpsam",
        help="Pretrained model to start from (default: cpsam)",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-6,
        help="Learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=0.0001,
        help="Weight decay (default: 0.0001)",
    )
    parser.add_argument(
        "--session",
        "-s",
        type=str,
        default=None,
        help="Resume monitoring existing session ID",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference test after training",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            artifact=args.artifact,
            train_images=args.train_images,
            train_annotations=args.train_annotations,
            test_images=args.test_images,
            test_annotations=args.test_annotations,
            validation_image_url=args.validation_image_url,
            validation_mask_url=args.validation_mask_url,
            validation_save_dir=args.validation_save_dir,
            n_epochs=args.epochs,
            model=args.model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            session_id=args.session,
            skip_inference=args.skip_inference,
        )
    )

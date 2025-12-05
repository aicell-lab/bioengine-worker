"""Test Cellpose inference with flow outputs."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from hypha_rpc import connect_to_server, login

# load .env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def test_flows(
    image_path: str,
    model: str = "cpsam",
    diameter: int = None,
    output_dir: str = None,
) -> None:
    """Run inference with flow outputs and visualize results.

    Args:
        image_path: Path to the local image file
        model: Cellpose model to use (default: cpsam)
        diameter: Expected cell diameter in pixels
        output_dir: Directory to save outputs (default: same as input)
    """
    # Check if image file exists
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load image
    logger.info("Loading image from: %s", image_path)
    img = Image.open(image_path)
    logger.info("Image loaded: size=%s, mode=%s", img.size, img.mode)

    # Convert to numpy array
    img_array = np.array(img)
    logger.info("Image array shape: %s", img_array.shape)

    # Resize for processing
    max_size = 512
    if max(img_array.shape[:2]) > max_size:
        scale = max_size / max(img_array.shape[:2])
        new_height = int(img_array.shape[0] * scale)
        new_width = int(img_array.shape[1] * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        logger.info("Resized image to: %s (scale: %.2f)", img_array.shape, scale)

    # Transpose if RGB
    if len(img_array.shape) == 3:
        img_array = np.transpose(img_array, (2, 0, 1))
        logger.info("Transposed to (C, H, W): %s", img_array.shape)

    # Connect to Hypha server
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:  # type: ignore[generalTypeIssues]
        workspace = server.config.workspace
        logger.info("Connected to workspace: %s", workspace)

        # Get the Cellpose service
        cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
        logger.info("Obtained Cellpose Fine-Tuning service")

        # Test 1: Regular inference without flows
        logger.info("\n=== Test 1: Inference WITHOUT flows ===")
        result_no_flows = await cellpose_service.infer(
            model=model,
            input_arrays=[img_array],
            diameter=diameter,
            flow_threshold=0.4,
            cellprob_threshold=0,
            niter=250,
            return_flows=False,
        )
        logger.info("Result keys (no flows): %s", list(result_no_flows[0].keys()))
        logger.info("Mask shape: %s", result_no_flows[0]["output"].shape)
        logger.info("Cells detected: %d", len(np.unique(result_no_flows[0]["output"])) - 1)

        # Test 2: Inference with flows
        logger.info("\n=== Test 2: Inference WITH flows ===")
        result_with_flows = await cellpose_service.infer(
            model=model,
            input_arrays=[img_array],
            diameter=diameter,
            flow_threshold=0.4,
            cellprob_threshold=0,
            niter=250,
            return_flows=True,
        )
        logger.info("Result keys (with flows): %s", list(result_with_flows[0].keys()))
        logger.info("Mask shape: %s", result_with_flows[0]["output"].shape)

        if "flows" in result_with_flows[0]:
            flows = result_with_flows[0]["flows"]
            logger.info("Flows received: %d items", len(flows))
            for i, flow in enumerate(flows):
                logger.info("  Flow[%d] shape: %s, dtype: %s", i, flow.shape, flow.dtype)
        else:
            logger.warning("No flows in result!")

        # Determine output directory
        if output_dir is None:
            output_dir = image_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Save visualizations
        from matplotlib import pyplot as plt

        # Save masks
        output_base = output_dir / f"{image_path.stem}_flows"

        # Save the mask
        mask = result_with_flows[0]["output"]
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap="tab20")
        plt.axis("off")
        plt.title(f"Cellpose Mask ({len(np.unique(mask))-1} cells)")
        plt.tight_layout()
        mask_path = f"{output_base}_mask.png"
        plt.savefig(mask_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved mask to: %s", mask_path)

        # Save flows if available
        if "flows" in result_with_flows[0]:
            flows = result_with_flows[0]["flows"]

            # Flow 0: HSV flow visualization
            if len(flows) > 0:
                hsv_flow = flows[0]
                plt.figure(figsize=(8, 8))
                if hsv_flow.ndim == 3:
                    plt.imshow(hsv_flow)
                else:
                    plt.imshow(hsv_flow, cmap="jet")
                plt.axis("off")
                plt.title("HSV Flow Visualization")
                plt.tight_layout()
                hsv_path = f"{output_base}_hsv.png"
                plt.savefig(hsv_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info("Saved HSV flow to: %s", hsv_path)

            # Flow 1: XY flows (2 channels: dY, dX)
            if len(flows) > 1:
                xy_flow = flows[1]
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                if xy_flow.ndim == 3 and xy_flow.shape[0] == 2:
                    axes[0].imshow(xy_flow[0], cmap="RdBu_r")
                    axes[0].set_title("Y Flow (dY)")
                    axes[0].axis("off")
                    axes[1].imshow(xy_flow[1], cmap="RdBu_r")
                    axes[1].set_title("X Flow (dX)")
                    axes[1].axis("off")
                else:
                    axes[0].imshow(xy_flow, cmap="RdBu_r")
                    axes[0].set_title("XY Flow")
                    axes[0].axis("off")
                    axes[1].axis("off")
                plt.tight_layout()
                xy_path = f"{output_base}_xy_flow.png"
                plt.savefig(xy_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info("Saved XY flow to: %s", xy_path)

            # Flow 2: Cell probability
            if len(flows) > 2:
                cellprob = flows[2]
                plt.figure(figsize=(8, 8))
                plt.imshow(cellprob, cmap="gray")
                plt.axis("off")
                plt.title("Cell Probability Map")
                plt.colorbar(label="Probability")
                plt.tight_layout()
                prob_path = f"{output_base}_cellprob.png"
                plt.savefig(prob_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info("Saved cell probability to: %s", prob_path)

            # Flow 3: Final pixel locations (optional)
            if len(flows) > 3:
                final_locs = flows[3]
                logger.info("Final pixel locations shape: %s", final_locs.shape)

        logger.info("\n=== Summary ===")
        logger.info("Test completed successfully!")
        logger.info("Output directory: %s", output_dir)
        logger.info("Base name: %s", output_base.name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Cellpose inference with flow outputs.",
    )
    parser.add_argument(
        "image_path",
        help="Path to the local image file",
    )
    parser.add_argument(
        "--model",
        default="cpsam",
        help="Cellpose model to use (default: cpsam)",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=None,
        help="Expected cell diameter in pixels (default: None for auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs (default: same as input)",
    )

    args = parser.parse_args()

    asyncio.run(
        test_flows(
            image_path=args.image_path,
            model=args.model,
            diameter=args.diameter,
            output_dir=args.output_dir,
        ),
    )

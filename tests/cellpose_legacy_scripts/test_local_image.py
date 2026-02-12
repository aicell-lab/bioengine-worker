"""Test Cellpose inference on a local image file."""

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


async def test_image_inference(
    image_path: str,
    model: str = "cpsam",
    diameter: int = None,
    output_path: str = None,
) -> None:
    """Run inference on a local image file and save the mask.

    Args:
        image_path: Path to the local image file
        model: Cellpose model to use (default: cpsam)
        diameter: Expected cell diameter in pixels
        output_path: Path to save the output mask (default: same directory as input)
    """
    # Check if image file exists
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load image as numpy array
    logger.info("Loading image from: %s", image_path)
    img = Image.open(image_path)
    logger.info("Image loaded: size=%s, mode=%s", img.size, img.mode)

    # Convert to numpy array
    img_array = np.array(img)
    logger.info("Image array shape: %s", img_array.shape)

    # Resize image to max 512 pixels (HuggingFace setting) to reduce GPU memory
    max_size = 512
    if max(img_array.shape[:2]) > max_size:
        scale = max_size / max(img_array.shape[:2])
        new_height = int(img_array.shape[0] * scale)
        new_width = int(img_array.shape[1] * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        logger.info("Resized image to: %s (scale: %.2f)", img_array.shape, scale)

    # If RGB image, prepare channels for Cellpose
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

        # Run inference with numpy array
        # Cellpose-SAM 4.0.7 is channel-order invariant, no channel specification needed
        # Using HuggingFace settings for best results:
        # - flow_threshold=0.4 (default, good for most cases)
        # - cellprob_threshold=0 (default, finds all cells)
        # - niter=250 (higher iterations for better convergence)
        logger.info("Running inference with model=%s, diameter=%s, niter=250...", model, str(diameter))
        inference_result = await cellpose_service.infer(
            model=model,
            input_arrays=[img_array],
            diameter=diameter,
            flow_threshold=0.4,
            cellprob_threshold=0,
            niter=250,
        )

        # Extract the mask
        output_mask = inference_result[0]["output"]
        logger.info("Inference complete! Mask shape: %s", output_mask.shape)
        logger.info("Unique mask values (cells): %d", len(np.unique(output_mask)))

        # Determine output paths
        if output_path is None:
            output_path = str(image_path.parent / f"{image_path.stem}_mask.png")
        else:
            output_path = str(Path(output_path))

        # Save the mask as PNG
        # Normalize mask to 0-255 range for visualization
        mask_normalized = (output_mask > 0).astype(np.uint8) * 255

        # Create a colorful visualization where each cell has a different color
        # Use the mask values directly for better visualization
        if output_mask.max() > 0:
            # Create a color-mapped version
            mask_vis = (output_mask % 256).astype(np.uint8)
        else:
            mask_vis = mask_normalized

        mask_img = Image.fromarray(mask_vis)
        mask_img.save(output_path)
        logger.info("Mask saved to: %s", output_path)

        # Also save the original image for comparison
        original_path = output_path.replace(".png", "_original.png")
        img.save(original_path)
        logger.info("Original image saved to: %s", original_path)

        # Save a colored mask for better visualization
        colored_mask_path = output_path.replace(".png", "_colored.png")
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.imshow(output_mask, cmap="tab20")
        plt.axis("off")
        plt.title(f"Cellpose Segmentation (Model: {model}, Diameter: {diameter})")
        plt.tight_layout()
        plt.savefig(colored_mask_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Colored mask saved to: %s", colored_mask_path)

        logger.info("\nSummary:")
        logger.info("  - Input image: %s", image_path)
        logger.info("  - Original image (resized): %s", original_path)
        logger.info("  - Binary mask: %s", output_path)
        logger.info("  - Colored mask: %s", colored_mask_path)
        logger.info("  - Total cells detected: %d", len(np.unique(output_mask)) - 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Cellpose inference on a local image file.",
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
        "--output",
        default=None,
        help="Path to save the output mask (default: same directory as input with _mask suffix)",
    )

    args = parser.parse_args()

    asyncio.run(
        test_image_inference(
            image_path=args.image_path,
            model=args.model,
            diameter=args.diameter,
            output_path=args.output,
        ),
    )

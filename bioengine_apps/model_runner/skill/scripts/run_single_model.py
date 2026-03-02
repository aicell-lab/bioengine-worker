"""
Run inference on a single known bioimage.io model.

Usage:
    python run_single_model.py --model-id affable-shark --image /path/to/image.npy --output result.npy

This script demonstrates the complete workflow for running a bioimage.io model:
1. Connect to the Hypha server
2. Get model metadata (RDF) to understand input/output specs
3. Load and prepare the input image
4. Upload the image and run inference
5. Download and save the result

Requirements:
    pip install hypha-rpc httpx numpy
"""

import argparse
import asyncio
import json
import sys
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"
SERVICE_ID = "bioimage-io/model-runner"


def parse_rdf_input_info(rdf: dict) -> dict:
    """Extract input specification from any RDF format version."""
    inp = rdf["inputs"][0]
    axes = inp["axes"]

    if isinstance(axes, str):
        # Format 0.4.x
        return {
            "format": "0.4",
            "axes_str": axes,
            "n_dims": len(axes),
            "shape": inp.get("shape"),
            "input_key": inp.get("name", "input0"),
            "data_type": inp.get("data_type", "float32"),
            "data_range": inp.get("data_range"),
        }
    else:
        # Format 0.5.x
        axis_ids = []
        spatial_constraints = {}
        n_channels = 1
        for ax in axes:
            ax_id = ax.get("id", ax.get("type"))
            axis_ids.append(ax_id)
            if ax.get("type") == "space":
                size_spec = ax.get("size", {})
                if isinstance(size_spec, dict):
                    spatial_constraints[ax_id] = {
                        "min": size_spec.get("min", 1),
                        "step": size_spec.get("step", 1),
                    }
            if ax.get("type") == "channel" and "channel_names" in ax:
                n_channels = len(ax["channel_names"])

        return {
            "format": "0.5",
            "axes": axis_ids,
            "n_dims": len(axes),
            "spatial_constraints": spatial_constraints,
            "n_channels": n_channels,
            "input_key": inp.get("id", "input0"),
            "data_type": inp.get("data", {}).get("type", "float32"),
        }


def get_output_key(rdf: dict) -> str:
    """Get the output tensor key from the RDF."""
    out = rdf["outputs"][0]
    return out.get("id", out.get("name", "output0"))


def prepare_image(image: np.ndarray, input_info: dict) -> np.ndarray:
    """Prepare an image to match model input expectations."""
    img = image.astype(np.float32)
    n_dims = input_info["n_dims"]

    # Add dimensions to reach expected n_dims
    while img.ndim < n_dims:
        img = img[np.newaxis, ...]

    # Trim dimensions if too many
    while img.ndim > n_dims:
        img = img[0]

    return img


async def run_single_model(model_id: str, image_path: str, output_path: str):
    """Run inference on a single model."""
    print(f"Connecting to {SERVER_URL}...")
    server = await connect_to_server(
        {"server_url": SERVER_URL, "method_timeout": 300}
    )
    mr = await server.get_service(SERVICE_ID)

    # 1. Get model metadata
    print(f"\n1. Getting metadata for model '{model_id}'...")
    rdf = await mr.get_model_rdf(model_id=model_id)
    print(f"   Model: {rdf['name']}")
    print(f"   Description: {rdf['description']}")

    input_info = parse_rdf_input_info(rdf)
    output_key = get_output_key(rdf)
    print(f"   Input: {input_info}")
    print(f"   Output key: {output_key}")

    # 2. Load and prepare image
    print(f"\n2. Loading image from '{image_path}'...")
    if image_path.endswith(".npy"):
        image = np.load(image_path)
    else:
        import imageio.v3 as iio

        image = iio.imread(image_path).astype(np.float32)

    print(f"   Raw image: shape={image.shape}, dtype={image.dtype}")

    image = prepare_image(image, input_info)
    print(f"   Prepared: shape={image.shape}, dtype={image.dtype}")

    # 3. Upload image
    print("\n3. Uploading image...")
    upload_info = await mr.get_upload_url(file_type=".npy")
    buffer = BytesIO()
    np.save(buffer, image)
    async with httpx.AsyncClient() as client:
        resp = await client.put(upload_info["upload_url"], content=buffer.getvalue())
        print(f"   Upload status: {resp.status_code}")

    # 4. Run inference
    print(f"\n4. Running inference with model '{model_id}'...")
    result = await mr.infer(
        model_id=model_id,
        inputs=upload_info["file_path"],
        return_download_url=True,
    )
    print(f"   Result keys: {list(result.keys())}")

    # 5. Download result
    print("\n5. Downloading result...")
    async with httpx.AsyncClient() as client:
        for key, url in result.items():
            resp = await client.get(url)
            arr = np.load(BytesIO(resp.content))
            print(f"   {key}: shape={arr.shape}, dtype={arr.dtype}")

            # Save output
            out_file = output_path if key == output_key else f"{Path(output_path).stem}_{key}.npy"
            np.save(out_file, arr)
            print(f"   Saved to: {out_file}")

    await server.disconnect()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Run a bioimage.io model on an input image")
    parser.add_argument("--model-id", required=True, help="Model identifier (e.g., 'affable-shark')")
    parser.add_argument("--image", required=True, help="Path to input image (.npy, .png, .tiff)")
    parser.add_argument("--output", default="result.npy", help="Output file path (default: result.npy)")
    args = parser.parse_args()

    asyncio.run(run_single_model(args.model_id, args.image, args.output))


if __name__ == "__main__":
    main()

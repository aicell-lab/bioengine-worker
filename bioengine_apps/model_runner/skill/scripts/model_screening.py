"""
Model screening and comparison workflow for bioimage.io models.

This script searches for models matching a task description, runs inference
on each candidate model, compares results against ground truth, and creates
visual comparison outputs.

Usage:
    python model_screening.py \
        --task "segment nuclei in fluorescence microscopy images" \
        --input /path/to/input.npy \
        --ground-truth /path/to/gt.npy \
        --output-dir ./comparison_results \
        --max-models 7

Requirements:
    pip install hypha-rpc httpx numpy matplotlib
"""

import argparse
import asyncio
import json
import traceback
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hypha_rpc import connect_to_server

matplotlib.use("Agg")

SERVER_URL = "https://hypha.aicell.io"
SERVICE_ID = "bioimage-io/model-runner"


# =============================================================================
# Utility functions
# =============================================================================


def extract_keywords(task_description: str) -> List[str]:
    """Extract search keywords from a task description.

    This is a simple heuristic. For better results, override with explicit keywords.
    """
    # Common bioimage analysis terms to look for
    keyword_map = {
        "segment": "segmentation",
        "nuclei": "nuclei",
        "nucleus": "nuclei",
        "cell": "cell",
        "membrane": "membrane",
        "mitochond": "mitochondria",
        "neuron": "neuron",
        "denois": "denoising",
        "restor": "restoration",
        "enhance": "enhancement",
        "detect": "detection",
        "fluoresc": "fluorescence",
        "electron": "electron-microscopy",
        "em": "electron-microscopy",
        "phase": "phase-contrast",
        "confocal": "confocal",
        "3d": "3d",
        "2d": "2d",
        "stardist": "stardist",
        "unet": "unet",
    }

    task_lower = task_description.lower()
    keywords = []
    for trigger, keyword in keyword_map.items():
        if trigger in task_lower and keyword not in keywords:
            keywords.append(keyword)

    # If no keywords found, split the task description
    if not keywords:
        words = task_lower.split()
        # Filter common stop words
        stop_words = {"in", "on", "of", "the", "a", "an", "for", "with", "from", "to", "and", "or", "images", "image"}
        keywords = [w for w in words if w not in stop_words and len(w) > 2][:5]

    return keywords


def get_input_info(rdf: dict) -> dict:
    """Extract input shape information from any RDF format version."""
    inp = rdf["inputs"][0]
    axes = inp["axes"]

    if isinstance(axes, str):
        return {
            "format": "0.4",
            "axes_str": axes,
            "n_dims": len(axes),
            "shape": inp.get("shape"),
            "input_key": inp.get("name", "input0"),
        }
    else:
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
        }


def get_output_key(rdf: dict) -> str:
    """Get the output tensor key from the RDF."""
    out = rdf["outputs"][0]
    return out.get("id", out.get("name", "output0"))


def prepare_image(image: np.ndarray, rdf: dict) -> np.ndarray:
    """
    Prepare an image to match model input expectations.

    This function handles dimension expansion/reduction and spatial size adjustments.
    It does NOT discard models based on size constraints — the model-runner handles
    tiling for large images and many models accept sizes beyond their stated minimums.
    """
    input_info = get_input_info(rdf)
    img = image.astype(np.float32).copy()
    n_dims = input_info["n_dims"]

    # Add batch/channel dimensions as needed
    while img.ndim < n_dims:
        img = img[np.newaxis, ...]

    # Trim extra dimensions
    while img.ndim > n_dims:
        img = img[0]

    # For 0.5.x format: ensure spatial dims meet minimum and step constraints
    if input_info["format"] == "0.5":
        for ax_id, constraints in input_info.get("spatial_constraints", {}).items():
            # Find axis index
            ax_idx = input_info["axes"].index(ax_id)
            current = img.shape[ax_idx]
            min_size = constraints["min"]
            step = constraints["step"]

            if current < min_size:
                # Pad to minimum size using reflection
                pad_width = [(0, 0)] * img.ndim
                pad_width[ax_idx] = (0, min_size - current)
                img = np.pad(img, pad_width, mode="reflect")
            else:
                # Round down to nearest valid size: min + N*step
                n = (current - min_size) // step
                valid = min_size + n * step
                if valid < current:
                    slices = [slice(None)] * img.ndim
                    slices[ax_idx] = slice(0, valid)
                    img = img[tuple(slices)]

    return img


def to_2d_display(arr: np.ndarray) -> np.ndarray:
    """Reduce an array to 2D for display (squeeze batch and channel dims)."""
    img = arr.copy()
    while img.ndim > 2:
        if img.shape[0] == 1:
            img = img[0]
        else:
            # Multiple channels — take first channel
            img = img[0]
    return img


# =============================================================================
# Metrics
# =============================================================================


def compute_iou(prediction: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Intersection over Union for binary masks."""
    pred_binary = prediction > threshold
    gt_binary = ground_truth > threshold
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(prediction: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Dice coefficient for binary masks."""
    pred_binary = prediction > threshold
    gt_binary = ground_truth > threshold
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    total = pred_binary.sum() + gt_binary.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection / total)


def compute_psnr(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((prediction.astype(np.float64) - ground_truth.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_val = max(ground_truth.max(), 1.0)
    return float(10 * np.log10(max_val**2 / mse))


# =============================================================================
# Visualization
# =============================================================================


def create_montage(
    input_image: np.ndarray,
    ground_truth: np.ndarray,
    model_results: Dict[str, np.ndarray],
    save_path: str,
):
    """Create a visual comparison montage of all model results.

    First row: input image and ground truth, centered.
    Subsequent rows: model results.
    """
    n_models = len(model_results)
    n_cols = min(max(n_models, 2), 5)
    n_model_rows = (n_models + n_cols - 1) // n_cols
    n_rows = 1 + n_model_rows  # first row for input+GT, rest for models

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for ax in axes.flatten():
        ax.axis("off")

    # First row: input and ground truth, centered
    left_pad = (n_cols - 2) // 2
    axes[0, left_pad].imshow(to_2d_display(input_image), cmap="gray")
    axes[0, left_pad].set_title("Input", fontsize=11, fontweight="bold")
    axes[0, left_pad + 1].imshow(to_2d_display(ground_truth), cmap="gray")
    axes[0, left_pad + 1].set_title("Ground Truth", fontsize=11, fontweight="bold")

    # Subsequent rows: model results
    for i, (model_id, result) in enumerate(model_results.items()):
        row = 1 + i // n_cols
        col = i % n_cols
        axes[row, col].imshow(to_2d_display(result), cmap="gray")
        axes[row, col].set_title(model_id, fontsize=9)

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Montage saved: {save_path}")


def create_metrics_plot(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str,
    save_path: str,
):
    """Create a bar chart comparing model metrics."""
    models = list(metrics.keys())
    values = [metrics[m].get(metric_name, 0.0) for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(range(len(models)), values, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"Model Comparison — {metric_name}", fontsize=14, fontweight="bold")

    max_val = max(values) if values else 1.0
    ax.set_ylim(0, max(max_val * 1.15, 0.1))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metrics plot saved: {save_path}")


# =============================================================================
# Main screening workflow
# =============================================================================


async def screen_models(
    task_description: str,
    input_image: np.ndarray,
    ground_truth: np.ndarray,
    output_dir: Path,
    max_models: int = 7,
    keywords: Optional[List[str]] = None,
):
    """
    Screen multiple bioimage.io models on the same input and compare to ground truth.

    Args:
        task_description: Natural language description of the task
        input_image: Input image as numpy array
        ground_truth: Expected output as numpy array
        output_dir: Directory to save comparison results
        max_models: Maximum number of models to compare (5-10 recommended)
        keywords: Optional explicit keywords (overrides auto-extraction from task)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect
    print(f"Connecting to {SERVER_URL}...")
    server = await connect_to_server(
        {"server_url": SERVER_URL, "method_timeout": 300}
    )
    mr = await server.get_service(SERVICE_ID)

    # --- Step 1: Search for candidate models ---
    if keywords is None:
        keywords = extract_keywords(task_description)
    print(f"\n1. Searching models with keywords: {keywords}")

    candidates = await mr.search_models(keywords=keywords, limit=max_models)
    print(f"   Found {len(candidates)} candidate models:")
    for c in candidates:
        print(f"     - {c['model_id']}: {c['description'][:70]}")

    if not candidates:
        print("   No models found. Try broader keywords.")
        await server.disconnect()
        return

    # --- Step 2: Run inference on each model ---
    print(f"\n2. Running inference on {len(candidates)} models...")

    model_results: Dict[str, np.ndarray] = {}
    model_errors: Dict[str, str] = {}

    async with httpx.AsyncClient() as client:
        for i, candidate in enumerate(candidates):
            model_id = candidate["model_id"]
            print(f"\n   [{i+1}/{len(candidates)}] Model: {model_id}")

            try:
                # Get model metadata
                rdf = await mr.get_model_rdf(model_id=model_id)
                input_info = get_input_info(rdf)
                output_key = get_output_key(rdf)
                print(f"     Input spec: {input_info['n_dims']}D, output key: '{output_key}'")

                # Prepare image for this model
                prepared = prepare_image(input_image, rdf)
                print(f"     Prepared input: {prepared.shape}")

                # Upload image
                upload_info = await mr.get_upload_url(file_type=".npy")
                buffer = BytesIO()
                np.save(buffer, prepared)
                await client.put(upload_info["upload_url"], content=buffer.getvalue())

                # Run inference
                result = await mr.infer(
                    model_id=model_id,
                    inputs=upload_info["file_path"],
                    return_download_url=True,
                )

                # Download result
                if output_key in result:
                    url = result[output_key]
                else:
                    # Use first available key
                    output_key = list(result.keys())[0]
                    url = result[output_key]

                resp = await client.get(url)
                arr = np.load(BytesIO(resp.content))

                print(f"     Output: {arr.shape}, dtype={arr.dtype}")
                model_results[model_id] = arr

                # Save individual result
                np.save(output_dir / f"{model_id}_result.npy", arr)

            except Exception as e:
                error_msg = str(e)
                print(f"     ERROR: {error_msg[:120]}")
                model_errors[model_id] = error_msg
                continue

    print(f"\n   Successful: {len(model_results)}/{len(candidates)}")
    if model_errors:
        print(f"   Failed: {len(model_errors)}")
        for mid, err in model_errors.items():
            print(f"     - {mid}: {err[:100]}")

    if not model_results:
        print("\nNo models produced results. Cannot create comparison.")
        await server.disconnect()
        return

    # --- Step 3: Compute metrics ---
    print("\n3. Computing comparison metrics...")

    gt_2d = to_2d_display(ground_truth)
    metrics: Dict[str, Dict[str, float]] = {}

    for model_id, result in model_results.items():
        result_2d = to_2d_display(result)

        # Resize result to match ground truth if needed
        if result_2d.shape != gt_2d.shape:
            # Simple center crop or pad to match
            result_matched = np.zeros_like(gt_2d)
            h = min(result_2d.shape[0], gt_2d.shape[0])
            w = min(result_2d.shape[1], gt_2d.shape[1])
            result_matched[:h, :w] = result_2d[:h, :w]
            result_2d = result_matched

        # Normalize both to [0, 1] for comparison
        if result_2d.max() > result_2d.min():
            result_norm = (result_2d - result_2d.min()) / (result_2d.max() - result_2d.min())
        else:
            result_norm = result_2d

        if gt_2d.max() > gt_2d.min():
            gt_norm = (gt_2d - gt_2d.min()) / (gt_2d.max() - gt_2d.min())
        else:
            gt_norm = gt_2d

        iou = compute_iou(result_norm, gt_norm)
        dice = compute_dice(result_norm, gt_norm)
        psnr = compute_psnr(result_norm, gt_norm)

        metrics[model_id] = {"iou": iou, "dice": dice, "psnr": psnr}
        print(f"   {model_id}: IoU={iou:.4f}, Dice={dice:.4f}, PSNR={psnr:.2f}")

    # --- Step 4: Create visualizations ---
    print("\n4. Creating comparison visualizations...")

    create_montage(
        input_image=input_image,
        ground_truth=ground_truth,
        model_results=model_results,
        save_path=str(output_dir / "model_comparison_montage.png"),
    )

    create_metrics_plot(
        metrics=metrics,
        metric_name="iou",
        save_path=str(output_dir / "model_comparison_iou.png"),
    )

    create_metrics_plot(
        metrics=metrics,
        metric_name="dice",
        save_path=str(output_dir / "model_comparison_dice.png"),
    )

    # --- Step 5: Save summary ---
    summary = {
        "task": task_description,
        "keywords": keywords,
        "candidates": [c["model_id"] for c in candidates],
        "successful_models": list(model_results.keys()),
        "failed_models": {k: v[:200] for k, v in model_errors.items()},
        "metrics": metrics,
        "best_model_iou": max(metrics, key=lambda m: metrics[m]["iou"]) if metrics else None,
        "best_model_dice": max(metrics, key=lambda m: metrics[m]["dice"]) if metrics else None,
    }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")

    # Print final recommendation
    if summary["best_model_iou"]:
        best = summary["best_model_iou"]
        best_iou = metrics[best]["iou"]
        print(f"\n{'='*60}")
        print(f"BEST MODEL (by IoU): {best} (IoU={best_iou:.4f})")
        print(f"{'='*60}")

    await server.disconnect()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Screen bioimage.io models for a task")
    parser.add_argument("--task", required=True, help="Task description (e.g., 'segment nuclei in fluorescence images')")
    parser.add_argument("--input", required=True, help="Path to input image (.npy)")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth output (.npy)")
    parser.add_argument("--output-dir", default="./comparison_results", help="Output directory")
    parser.add_argument("--max-models", type=int, default=7, help="Max models to compare (default: 7)")
    parser.add_argument("--keywords", nargs="+", help="Override auto-extracted keywords")
    args = parser.parse_args()

    input_image = np.load(args.input)
    ground_truth = np.load(args.ground_truth)

    asyncio.run(
        screen_models(
            task_description=args.task,
            input_image=input_image,
            ground_truth=ground_truth,
            output_dir=Path(args.output_dir),
            max_models=args.max_models,
            keywords=args.keywords,
        )
    )


if __name__ == "__main__":
    main()

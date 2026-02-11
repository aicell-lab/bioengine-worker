# Cellpose Fine-Tuning Service

A BioEngine service for running Cellpose-SAM 4.0.7 inference and fine-tuning cell segmentation models.

## Features

- **Inference**: Run cell segmentation on images using Cellpose-SAM
- **Fine-tuning**: Train custom models on your annotated datasets
- **Model Export**: Export trained models to BioImage.IO format for sharing and reuse
- **Async API**: Monitor training progress in real-time
- **GPU-accelerated**: Optimized for fast inference and training

## Cellpose 4.0.7 (Cellpose-SAM)

This service uses Cellpose 4.0.7, which features the Cellpose-SAM model - a transformer-based segmentation model with the following improvements:

- **Channel-order invariant**: No need to specify channel order
- **Better performance**: Improved accuracy over previous versions
- **Single model**: Only `cpsam` model is available (replaces all previous models)
- **Advanced parameters**: Fine-tune results with `flow_threshold`, `cellprob_threshold`, and `niter`

## Quick Start

### 1. Connect to the Service

```python
from hypha_rpc import connect_to_server, login
import os

# Connect to Hypha server
server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

server = await connect_to_server({"server_url": server_url, "token": token})

# Get the Cellpose service
cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
```

### 2. Run Inference on a Single Image

```python
import numpy as np
from PIL import Image
import httpx

# Download an image
image_url = "https://images.proteinatlas.org/41082/733_D4_2_blue_red_green.jpg"
async with httpx.AsyncClient() as client:
    response = await client.get(image_url)
    img = Image.open(BytesIO(response.content))

# Convert to numpy array (C, H, W) format
img_array = np.array(img)
if len(img_array.shape) == 3:
    img_array = np.transpose(img_array, (2, 0, 1))

# Run inference with HuggingFace-recommended settings
result = await cellpose_service.infer(
    model="cpsam",
    input_arrays=[img_array],
    diameter=None,  # Auto-detect
    flow_threshold=0.4,  # Default, good for most cases
    cellprob_threshold=0,  # Find all cells
    niter=250,  # Higher iterations for better convergence
)

# Get the segmentation mask
mask = result[0]["output"]
print(f"Detected {len(np.unique(mask)) - 1} cells")
```

### 3. Fine-Tune a Model

```python
# Start training asynchronously
# Option 1: Using folder paths (assumes same filenames)
session_status = await cellpose_service.start_training(
    artifact="your-workspace/your-dataset",
    train_images="images/experiment1/",  # Folder with images
    train_annotations="annotations/experiment1/",  # Folder with annotations (same filenames)
    model="cpsam",
    n_epochs=10,
)

# Option 2: Using path patterns (for different naming conventions)
session_status = await cellpose_service.start_training(
    artifact="your-workspace/your-dataset",
    train_images="images/experiment1/*.ome.tif",  # Pattern for image files
    train_annotations="annotations/experiment1/*_mask.ome.tif",  # Pattern for annotation files
    model="cpsam",
    n_epochs=10,
    n_samples=None,  # Use all matched samples
    # Optional: specify test set
    test_images="images/test/*.ome.tif",
    test_annotations="annotations/test/*_mask.ome.tif",
)

session_id = session_status["session_id"]
print(f"Training started: {session_id}")

# Monitor training progress with real-time updates
while True:
    status = await cellpose_service.get_training_status(session_id)

    # Build progress message with all available info
    msg = f"[{status['status_type']}] {status['message']}"

    # Add epoch progress if available
    if "current_epoch" in status and "total_epochs" in status:
        msg += f" | Epoch: {status['current_epoch']}/{status['total_epochs']}"

    # Add elapsed time if available
    if "elapsed_seconds" in status:
        msg += f" | Time: {status['elapsed_seconds']:.1f}s"

    # Add latest training loss if available
    if "train_losses" in status and status["train_losses"]:
        losses = [l for l in status["train_losses"] if l is not None and l > 0]
        if losses:
            msg += f" | Train Loss: {losses[-1]:.4f}"

    # Add validation metrics if available
    if "test_metrics" in status and status["test_metrics"]:
        last_metrics = next((m for m in reversed(status["test_metrics"]) if m), None)
        if last_metrics:
            msg += f" | Val F1: {last_metrics['f1']:.4f}, IoU: {last_metrics['iou']:.4f}"

    print(msg)

    if status["status_type"] in ("completed", "failed"):
        break

    await asyncio.sleep(2)

# Use the fine-tuned model for inference
result = await cellpose_service.infer(
    model=session_id,  # Use the session ID as model
    artifact="your-workspace/your-dataset",
    image_paths=["path/to/image.tif"],
    diameter=40,
)
```

## API Reference

### `infer()`

Run inference on images.

**Parameters:**
- `model` (str): Model to use ("cpsam" for pretrained, or session ID for fine-tuned)
- `artifact` (str, optional): Artifact ID containing images
- `image_paths` (list, optional): List of image paths within artifact
- `input_arrays` (list, optional): List of numpy arrays (C, H, W) format
- `diameter` (float, optional): Expected cell diameter (None for auto-detect)
- `flow_threshold` (float): Flow error threshold (default: 0.4)
- `cellprob_threshold` (float): Cell probability threshold (default: 0.0)
- `niter` (int, optional): Number of iterations for dynamics (None for auto)

**Returns:** List of dicts with `"output"` key containing segmentation masks

### `start_training()`

Start asynchronous model fine-tuning.

**Parameters:**
- `artifact` (str): Artifact ID containing training data
- `train_images` (str): **Required** - Path to training images. Can be:
  - Folder path ending with '/': `"images/folder/"` (assumes same filenames as annotations)
  - Path pattern with wildcard: `"images/folder/*.ome.tif"`
- `train_annotations` (str): **Required** - Path to training annotations. Can be:
  - Folder path ending with '/': `"annotations/folder/"` (assumes same filenames as images)
  - Path pattern with wildcard: `"annotations/folder/*_mask.ome.tif"`
- `test_images` (str, optional): Optional test images path (same format as train_images). Providing test data enables per-epoch pixel-level validation metrics and end-of-training instance segmentation metrics (AP@0.5/0.75/0.9).
- `test_annotations` (str, optional): Optional test annotations (label masks). Required together with `test_images`.
- `model` (str): Pretrained model to start from (default: "cpsam")
- `n_epochs` (int): Number of training epochs (default: 10)
- `n_samples` (int, optional): Limit number of samples to use
- `learning_rate` (float): Learning rate (default: 1e-6)
- `weight_decay` (float): Weight decay (default: 0.0001)
- `validation_interval` (int, optional): Epochs between validation evaluations. Always validates on the first epoch. Default (None) validates every 10 epochs. Set to 1 for every epoch. Requires `test_images` and `test_annotations`.

**Returns:** Dict with `"session_id"` key

**Path Formats:**

1. **Folder paths** (assumes identical filenames):
   - `train_images="images/folder/"` and `train_annotations="annotations/folder/"`
   - Matches all files with same names in both folders

2. **Pattern paths** (for different naming conventions):
   - `train_images="images/folder/*.ome.tif"` and `train_annotations="annotations/folder/*_mask.ome.tif"`
   - The `*` part must match between images and annotations
   - Example: `t0000.ome.tif` ↔ `t0000_mask.ome.tif`

**Limitations:**
- Currently limited to 1000 files per folder (via artifact.ls()). Future versions will support pagination for larger directories.

### `get_training_status()`

Monitor training progress and retrieve training metrics with real-time updates.

**Parameters:**
- `session_id` (str): Training session ID

**Returns:** Dict with the following keys:
- `status_type` (str): Status of the training ("waiting", "preparing", "running", "completed", "failed")
- `message` (str): Human-readable status message
- `dataset_artifact_id` (str): Training dataset artifact ID (e.g., "bioimage-io/your-dataset")
- `train_losses` (list[float], optional): Per-epoch training loss values (updated in real-time)
- `test_losses` (list[float | None], optional): Per-epoch test loss values. `None` for epochs where validation was skipped.
- `test_metrics` (list[ValidationMetrics | None], optional): Per-epoch pixel-level validation metrics. `None` for epochs where validation was skipped. Controlled by `validation_interval`.
- `instance_metrics` (InstanceMetrics | None, optional): Instance segmentation metrics computed at the end of training by running full Cellpose inference on the test set. Only present when test data was provided.
- `n_train` (int, optional): Number of training samples
- `n_test` (int, optional): Number of test samples
- `start_time` (str, optional): Training start time in ISO 8601 format
- `current_epoch` (int, optional): Current epoch number (1-indexed)
- `total_epochs` (int, optional): Total number of epochs
- `elapsed_seconds` (float, optional): Elapsed time since training started

**ValidationMetrics fields** (pixel-level binary foreground/background — NOT instance segmentation):
- `pixel_accuracy` (float): (TP+TN) / total pixels
- `precision` (float): TP / (TP+FP) — fraction of predicted foreground that is correct
- `recall` (float): TP / (TP+FN) — fraction of true foreground that is detected
- `f1` (float): Harmonic mean of precision and recall (same as Dice coefficient)
- `iou` (float): TP / (TP+FP+FN) — Jaccard index on binary foreground mask

**InstanceMetrics fields** (computed at end of training using Hungarian matching):
- `ap_0_5` (float): Average precision at IoU >= 0.5
- `ap_0_75` (float): Average precision at IoU >= 0.75
- `ap_0_9` (float): Average precision at IoU >= 0.9
- `n_true` (int): Total ground-truth instances in test set
- `n_pred` (int): Total predicted instances in test set

### `export_model()`

Export a trained model to BioImage.IO format for sharing and reuse.

**Parameters:**
- `session_id` (str): Training session ID of the model to export
- `model_name` (str): Name for the exported model artifact
- `collection` (str): Collection to save the model to (e.g., "bioimage-io/colab-annotations")

**Returns:** Dict with the following keys:
- `artifact_id` (str): Full artifact ID (e.g., "bioimage-io/test-model-abc123")
- `model_name` (str): Model name
- `status` (str): Export status ("exported")
- `artifact_url` (str): URL to view artifact (e.g., "https://hypha.aicell.io/bioimage-io/artifacts/test-model-abc123")
- `download_url` (str): URL to download model as zip (e.g., "https://hypha.aicell.io/bioimage-io/artifacts/test-model-abc123/create-zip-file")
- `files` (list): List of exported files

**Example:**
```python
# Train a model
session_status = await cellpose_service.start_training(
    artifact="bioimage-io/your-dataset",
    train_images="images/*.tif",
    train_annotations="annotations/*_mask.tif",
    n_epochs=10,
)
session_id = session_status["session_id"]

# Wait for training to complete...
# (monitor with get_training_status)

# Export the trained model
export_result = await cellpose_service.export_model(
    session_id=session_id,
    model_name="my-cell-model-v1",
    collection="bioimage-io/colab-annotations",
)

print(f"Model exported: {export_result['artifact_url']}")
print(f"Download: {export_result['download_url']}")
```

**Exported Files:**
The exported model includes 7 files in BioImage.IO format:
1. `rdf.yaml` - Model metadata and specification (includes `parent` field for lineage tracking)
2. `model.py` - Model architecture wrapper
3. `model_weights.pth` - Trained weights
4. `input_sample.npy` - Sample input for testing
5. `output_sample.npy` - Sample output for testing
6. `cover.png` - Visualization showing input and segmentation
7. `doc.md` - Documentation with training details

**Model Lineage:**
The RDF YAML includes a `parent` field that tracks the base model used for fine-tuning (e.g., "cpsam"). Additionally, each exported model stores the `training_dataset_id` in its artifact config, enabling you to find all models trained on a specific dataset.

### `list_models_by_dataset()`

List all models trained on a specific dataset.

**Parameters:**
- `dataset_id` (str): Dataset artifact ID to find models trained on it
- `collection` (str): Collection to search in (default: "bioimage-io/colab-annotations")

**Returns:** List of dicts, each containing:
- `id` (str): Model artifact ID
- `name` (str): Model name
- `created_at` (int): Creation timestamp
- `url` (str): URL to view the model

**Example:**
```python
# Find all models trained on a specific dataset
models = await cellpose_service.list_models_by_dataset(
    dataset_id="bioimage-io/your-dataset",
    collection="bioimage-io/colab-annotations"
)

for model in models:
    print(f"Model: {model['name']}")
    print(f"  ID: {model['id']}")
    print(f"  URL: {model['url']}")
    print(f"  Created: {model['created_at']}")
```

### `list_pretrained_models()`

Get available pretrained models.

**Returns:** List of model names (currently only ["cpsam"])

## Inference Parameters Guide

### `flow_threshold` (default: 0.4)
- **Higher values** (e.g., 0.6): More lenient, returns more masks (may include poorly-shaped ones)
- **Lower values** (e.g., 0.2): More strict, fewer masks but better quality
- **Recommended**: 0.4 for most cases

### `cellprob_threshold` (default: 0.0)
- **Higher values** (e.g., 1.0): Fewer masks, filters out dim areas
- **Lower values** (e.g., -1.0): More masks, includes dimmer cells
- **Recommended**: 0.0 to find all cells

### `niter` (default: None)
- **Higher values** (e.g., 250): Better convergence, recommended for HuggingFace-quality results
- **None/0**: Auto-set based on diameter
- **Recommended**: 250 for best results

## Deploying the Service

The cellpose fine-tuning service runs as a Ray Serve deployment on BioEngine. To deploy or redeploy it, you need a valid `HYPHA_TOKEN` with `bioimage-io` workspace access.

### Prerequisites

1. A valid Hypha token with read/write access to the `bioimage-io` workspace. Generate one from [hypha.aicell.io](https://hypha.aicell.io/).
2. Store the token in `.env` at the repository root:
   ```
   HYPHA_TOKEN=eyJhbG...your_token_here
   ```

### Upload the Application Code

Before deploying, upload the latest application code to the artifact store:

```bash
source .env
python scripts/save_application.py \
    --directory "bioengine_apps/cellpose_finetuning" \
    --server_url "https://hypha.aicell.io" \
    --workspace "bioimage-io" \
    --token "$HYPHA_TOKEN"
```

### Deploy (First Time)

```bash
source .env
python bioengine_apps/cellpose_finetuning/scripts/deploy_cellpose.py

# Deploy to a test target instead:
python bioengine_apps/cellpose_finetuning/scripts/deploy_cellpose.py \
    --artifact-id bioimage-io/cellpose-finetuning-test \
    --application-id cellpose-finetuning-test
```

This calls `run_application()` with `hypha_token=token`, which injects `HYPHA_TOKEN` into the Ray worker's environment as a secret env var. The token is required for the service to access artifacts (datasets and models) on behalf of users.

### Redeploy (Update Running Service)

```bash
source .env
python bioengine_apps/cellpose_finetuning/scripts/redeploy_cellpose.py

# Redeploy a test target instead:
python bioengine_apps/cellpose_finetuning/scripts/redeploy_cellpose.py \
    --artifact-id bioimage-io/cellpose-finetuning-test \
    --application-id cellpose-finetuning-test
```

This stops the existing deployment and starts a fresh one with the latest artifact code and a new token.

### When to Redeploy

You must redeploy the service when:
- **The token expires**: The `HYPHA_TOKEN` baked into the Ray worker at deploy time has a fixed expiry. If training fails with `HYPHA_TOKEN environment variable is not set`, the token has likely expired. Generate a fresh token, update `.env`, and redeploy.
- **Code changes**: After updating `main.py` or other files, upload the new code with `save_application.py` and then redeploy.

### Troubleshooting

**"HYPHA_TOKEN environment variable is not set" during training**

This error occurs on the server side (Ray worker), not locally. It means the token passed at deploy time is missing or expired. Fix by redeploying with a fresh token.

**Multiple tokens in `.env`**

Ensure `.env` contains only one `HYPHA_TOKEN` entry. If there are duplicates, the last one wins when sourced, which may point to the wrong workspace or an expired token.

## Example Scripts

This repository includes several example scripts in `bioengine_apps/cellpose_finetuning/scripts/`:

- **`deploy_cellpose.py`**: First-time deployment of the service
- **`redeploy_cellpose.py`**: Stop and redeploy the service with updated code/token
- **`test_service.py`**: Full workflow (training, monitoring, inference, export) with argparse
- **`test_dataset_artifact_id.py`**: Verify `dataset_artifact_id` is returned in training status

Run them with:

```bash
source .env

# Full training + inference test (with validation every epoch)
python bioengine_apps/cellpose_finetuning/scripts/test_service.py \
    --dataset-artifact ri-scale/zarr-demo \
    --train-images 'images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif' \
    --train-annotations 'annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif' \
    --test-images 'images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif' \
    --test-annotations 'annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif' \
    --n-epochs 5 --validation-interval 1

# Resume monitoring an existing training session
python bioengine_apps/cellpose_finetuning/scripts/test_service.py \
    --dataset-artifact ri-scale/zarr-demo \
    --train-images 'images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif' \
    --train-annotations 'annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif' \
    --session <session-id>

# Export a completed model
python bioengine_apps/cellpose_finetuning/scripts/test_service.py \
    --dataset-artifact ri-scale/zarr-demo \
    --train-images 'images/108bb69d-2e52-4382-8100-e96173db24ee/*.ome.tif' \
    --train-annotations 'annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif' \
    --session <session-id> --export --download-cover
```

## Real-Time Training Progress

The service provides comprehensive real-time progress tracking with epoch callbacks. You can monitor:

1. **Real-time metrics during training**: Epoch-by-epoch updates with losses, timing, and progress
2. **Dataset information**: Training/test sample counts available immediately
3. **Detailed timing**: Start time and elapsed seconds updated after each epoch
4. **Training history**: Full loss history available during and after training

Example of accessing all status information:

```python
status = await cellpose_service.get_training_status(session_id)

# Dataset information
print(f"Training samples: {status.get('n_train', 'N/A')}")
print(f"Test samples: {status.get('n_test', 'N/A')}")

# Training progress
if "current_epoch" in status:
    print(f"Progress: {status['current_epoch']}/{status['total_epochs']} epochs")
    print(f"Elapsed time: {status['elapsed_seconds']:.1f}s")

# Loss history
if "train_losses" in status and status["train_losses"]:
    train_losses = status["train_losses"]
    test_losses = status["test_losses"]

    # Filter non-zero losses
    valid_losses = [l for l in train_losses if l > 0]
    if valid_losses:
        print(f"Latest training loss: {valid_losses[-1]:.4f}")
        print(f"Initial loss: {valid_losses[0]:.4f}")

    # test_losses uses None for skipped epochs
    valid_test = [l for l in test_losses if l is not None]
    if valid_test:
        print(f"Latest val loss: {valid_test[-1]:.4f}")

# Instance metrics (computed at end of training)
if status.get("instance_metrics"):
    im = status["instance_metrics"]
    print(f"AP@0.5={im['ap_0_5']}, AP@0.75={im['ap_0_75']}, AP@0.9={im['ap_0_9']}")
    print(f"Ground-truth cells: {im['n_true']}, Predicted cells: {im['n_pred']}")
```

## Tutorial Notebook

For a complete tutorial, see `tutorial_cellpose_finetuning.ipynb` which covers:
- Dataset preparation
- Training a custom model
- Comparing pretrained vs fine-tuned results

## Requirements

- Python 3.8+
- hypha-rpc
- numpy
- httpx (for URL-based images)

## Support

For issues or questions, please open an issue on the [BioEngine GitHub repository](https://github.com/aicell-lab/bioengine-worker).

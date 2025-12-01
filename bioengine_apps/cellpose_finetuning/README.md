# Cellpose Fine-Tuning Service

A BioEngine service for running Cellpose-SAM 4.0.7 inference and fine-tuning cell segmentation models.

## Features

- **Inference**: Run cell segmentation on images using Cellpose-SAM
- **Fine-tuning**: Train custom models on your annotated datasets
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

# Monitor training progress
while True:
    status = await cellpose_service.get_training_status(session_id)
    print(f"{status['message']}")

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
- `test_images` (str, optional): Optional test images path (same format as train_images)
- `test_annotations` (str, optional): Optional test annotations path (same format as train_annotations)
- `model` (str): Pretrained model to start from (default: "cpsam")
- `n_epochs` (int): Number of training epochs (default: 10)
- `n_samples` (int, optional): Limit number of samples to use
- `learning_rate` (float): Learning rate (default: 1e-6)
- `weight_decay` (float): Weight decay (default: 0.0001)

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

Monitor training progress and retrieve training metrics.

**Parameters:**
- `session_id` (str): Training session ID

**Returns:** Dict with the following keys:
- `status_type` (str): Status of the training ("waiting", "preparing", "running", "completed", "failed")
- `message` (str): Human-readable status message
- `train_losses` (list[float], optional): Per-epoch training loss values
- `test_losses` (list[float], optional): Per-epoch test loss values (computed periodically)

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

## Example Scripts

This repository includes several example scripts:

- **`scripts/test_single_image.py`**: Run inference on a single image from URL
- **`scripts/test_service.py`**: Full workflow including training and inference with metrics display
- **`scripts/check_status.py`**: Check training status and view metrics for a session

Run them with:

```bash
# Set your Hypha token
export HYPHA_TOKEN="your-token-here"

# Test single image inference
python scripts/test_single_image.py

# Test full training workflow
python scripts/test_service.py

# Check status of a training session
python scripts/check_status.py <session_id>
```

## Training Metrics

The service now tracks and returns training metrics including per-epoch training and test losses. You can:

1. **Monitor during training**: The `test_service.py` script displays real-time loss values during training
2. **Query after completion**: Use `get_training_status()` to retrieve full training history
3. **Visualize progress**: Training and test losses are returned as lists for easy plotting

Example of accessing metrics:

```python
status = await cellpose_service.get_training_status(session_id)

if "train_losses" in status:
    train_losses = status["train_losses"]
    test_losses = status["test_losses"]

    # Plot or analyze the losses
    print(f"Initial loss: {train_losses[0]:.4f}")
    print(f"Final loss: {train_losses[-1]:.4f}")
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

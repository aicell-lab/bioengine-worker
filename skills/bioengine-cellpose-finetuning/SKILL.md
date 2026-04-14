---
name: bioengine-cellpose-finetuning
description: Fine-tunes Cellpose-SAM on user-provided annotated microscopy images via the BioEngine cellpose-finetuning service. Use when the user wants to adapt Cellpose to a novel cell morphology, train on their own annotated data, monitor training progress, evaluate per-epoch metrics, or export a trained model to the BioImage.IO Model Zoo.
license: MIT
metadata:
  service-id: bioimage-io/cellpose-finetuning
  server: https://hypha.aicell.io
  model: Cellpose-SAM (cpsam), Cellpose 4.0.7
  gpu: 4× A40 (BioEngine GPU cluster)
---

# BioEngine Cellpose Fine-Tuning

Fine-tune Cellpose-SAM on your own annotated microscopy images — no local GPU, no code, entirely browser- and API-accessible via BioEngine.

## What this skill does

| Task | API method |
|---|---|
| Start fine-tuning on a dataset | `start_training(artifact, train_images, train_annotations, ...)` |
| Monitor training progress (live IoU curve) | `get_training_status(session_id)` |
| Stop a running session | `stop_training(session_id)` |
| Export trained model to BioImage.IO | `export_model(session_id, model_name, authors, ...)` |
| Run inference with trained model | `infer(model=session_id, input_arrays=[np.ndarray])` |

## Quick start

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({"server_url": "https://hypha.aicell.io", "token": "<HYPHA_TOKEN>"})
svc = await server.get_service("bioimage-io/cellpose-finetuning")

# 1. Start fine-tuning
session = await svc.start_training(
    artifact="your-workspace/your-dataset",
    train_images="train/*_image.ome.tif",
    train_annotations="train/*_mask.ome.tif",
    test_images="test/*_image.ome.tif",        # optional but recommended
    test_annotations="test/*_mask.ome.tif",
    n_epochs=1000,
    learning_rate=1e-5,
    validation_interval=10,     # compute IoU every 10 epochs
    min_train_masks=5,
)
session_id = session["session_id"]

# 2. Poll training progress
status = await svc.get_training_status(session_id)
# status.status_type: "preparing" | "running" | "completed" | "stopped" | "error"
# status.current_epoch, status.total_epochs
# status.test_metrics: [{pixel_iou, f1, precision, recall}, ...] (one per checkpoint)
# status.train_losses: [float, ...]

# 3. Run inference with the fine-tuned model
import numpy as np
test_image = np.load("my_test_image.npy")  # 2D grayscale or 3D (H, W, C)
result = await svc.infer(
    model=session_id,           # session_id is passed as the 'model' parameter
    input_arrays=[test_image]   # list of numpy arrays (NOT Python lists)
)
mask = result[0]["output"]      # integer instance mask (0=background, 1..N=cells)

# 4. Export when done
result = await svc.export_model(
    session_id=session_id,
    model_name="my-cellpose-model",
    description="Fine-tuned on phase-contrast HeLa cells, 80 annotated images",
    authors=[{"name": "Alice Smith", "affiliation": "My University"}],
    collection="bioimage-io/colab-annotations",
)
# result["url"] -> BioImage.IO model page
```

## Dataset format

Data must be stored in a Hypha artifact as OME-TIFF pairs:

```
your-dataset/
├── train/
│   ├── 001_image.ome.tif        # microscopy image
│   ├── 001_mask.ome.tif         # integer label mask (0 = background)
│   ├── 002_image.ome.tif
│   └── ...
└── test/
    ├── 001_image.ome.tif
    ├── 001_mask.ome.tif
    └── ...
```

Or use the `metadata_dir` parameter with a JSON index of image/mask paths.

**Minimum recommended**: 10+ annotated images for fine-tuning to have any effect. For best results: 50–200 images.

### Brightfield / phase-contrast images

For non-fluorescence imaging, preprocessing is required before training and inference:
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): expands low-contrast images to fill the full dynamic range. Without CLAHE, Cellpose-SAM cannot detect cells at all on typical brightfield images (pixel values span only ~10 out of 255).
2. **Downscale to target diameter**: resize images so cells appear ~30–60px in diameter (matching Cellpose-SAM's expected scale). For example, 1008×1008px images with ~112px cells → downscale to 270×270px.

```python
import cv2, numpy as np

def preprocess_brightfield(img_uint8):
    """CLAHE + no rescaling needed if already at correct diameter."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    return clahe.apply(img_uint8)
```

The service does **not** apply preprocessing automatically — apply it before calling `start_training` or `infer`.

## start_training parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `artifact` | str | required | Hypha artifact ID `workspace/alias` containing images |
| `train_images` | str | required | Glob pattern for training images (e.g. `train/*_image.ome.tif`) |
| `train_annotations` | str | required | Glob pattern for training masks |
| `test_images` | str | None | Test images for per-epoch IoU evaluation |
| `test_annotations` | str | None | Test masks |
| `metadata_dir` | str | None | Alternative: JSON metadata index directory |
| `n_epochs` | int | 500 | Total training epochs |
| `learning_rate` | float | 1e-5 | Initial learning rate |
| `validation_interval` | int | 10 | Compute test metrics every N epochs |
| `min_train_masks` | int | 5 | Skip images with fewer than N annotated instances |

## get_training_status return fields

```json
{
  "status_type": "running",
  "session_id": "8bcd26bb-...",
  "current_epoch": 100,
  "total_epochs": 1000,
  "elapsed_seconds": 345,
  "train_losses": [1.14, 0.74, 0.68, ...],
  "test_metrics": [
    {"iou": 0.453, "f1": 0.623, "precision": 0.457, "recall": 0.979, "pixel_accuracy": 0.91},
    null, null, null,
    {"iou": 0.473, "f1": 0.642, "precision": 0.478, "recall": 0.981, "pixel_accuracy": 0.92},
    null, null, null, null,
    {"iou": 0.501, "f1": 0.667, "precision": 0.511, "recall": 0.964, "pixel_accuracy": 0.93}
  ],
  "instance_metrics": null  // populated at end of training
  // NOTE: test_metrics is a list with one entry per epoch, non-validated epochs are null
}
```

## export_model parameters

| Parameter | Description |
|---|---|
| `session_id` | ID of a **completed** training session |
| `model_name` | Custom model name (defaults to `cellpose-{session_id}`) |
| `description` | Text appended to the BioImage.IO RDF description |
| `authors` | List of `{"name": "...", "affiliation": "..."}` dicts |
| `uploader` | `{"name": "...", "email": "..."}` for BioImage.IO uploader field |
| `collection` | Hypha artifact collection to upload to (default: `bioimage-io/colab-annotations`) |

Returns `{"artifact_id": "...", "model_name": "...", "status": "exported", "url": "https://..."}`.

## Real experimental results (Session A, 2026-04-08)

Dataset: `ri-scale/cellpose-test` (80 train / 20 test images, OME-TIFF, fluorescence microscopy)
Model: Cellpose-SAM (cpsam), 1000 epochs, lr=1e-5

| Epoch | IoU (pixel) | F1 | Precision | Recall |
|---|---|---|---|---|
| 1 (baseline) | 0.430 | 0.601 | 0.438 | 0.961 |
| 10 | 0.453 | 0.623 | 0.457 | 0.979 |
| 20 | 0.473 | 0.642 | 0.478 | 0.981 |
| 100 | 0.501 | 0.667 | 0.511 | 0.964 |

**Interpretation**: Baseline (epoch 1) IoU = 0.43 with high recall (finds most cells) but low precision (over-segments). Fine-tuning improves precision while maintaining recall, increasing IoU by ~17% at epoch 100.

## Authentication

This service requires a Hypha token. Obtain one with:

```python
# Browser-based login (interactive)
from hypha_rpc import login
token = await login(server_url="https://hypha.aicell.io")

# Or use an existing token from environment
import os
token = os.environ["HYPHA_TOKEN"]
```

Public read access to `ri-scale/cellpose-test` and `ri-scale/zarr-demo` datasets is available. Writing (starting training sessions) requires authentication.

## Integration with BioImage.IO Colab

The preferred annotation workflow:
1. Open BioImage.IO Colab in browser — no installation
2. Mount your dataset from Hypha artifact storage
3. Use Cellpose-SAM for initial pre-segmentation
4. Correct annotations interactively (multiple annotators, any device)
5. Call `start_training()` with the annotated dataset
6. Monitor with `get_training_status()` until IoU plateaus
7. Call `export_model()` to publish to BioImage.IO Model Zoo

No local GPU, no command line, no software installation.

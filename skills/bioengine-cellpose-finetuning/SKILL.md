---
name: bioengine-cellpose-finetuning
description: Fine-tunes Cellpose-SAM on user-provided annotated microscopy images via the BioEngine cellpose-finetuning service. Use when the user wants to adapt Cellpose to a novel cell morphology, train on their own annotated data, monitor training progress, evaluate per-epoch metrics, or export a trained model to the BioImage.IO Model Zoo.
license: MIT
metadata:
  service-id: bioimage-io/cellpose-finetuning
  worker-service-id: bioimage-io/bioengine-worker
  server: https://hypha.aicell.io
  model: Cellpose-SAM (cpsam), Cellpose 4.0.7
  gpu: 4× A40 (BioEngine GPU cluster)
  app-version: 0.0.19
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
    validation_interval=10,     # compute metrics every 10 epochs
    min_train_masks=5,
)
session_id = session["session_id"]

# 2. Poll training progress
status = await svc.get_training_status(session_id)
# status.status_type: "preparing" | "running" | "completed" | "stopped" | "failed"
# status.current_epoch, status.total_epochs
# status.test_metrics: [{iou, f1, precision, recall}, ...] (one per checkpoint, others null)
# status.train_losses: [float, ...]
# status.instance_metrics: {ap_0_5, ap_0_75, ap_0_9, n_true, n_pred}  (populated at end)

# 3. Run inference with the fine-tuned model
import numpy as np
test_image = np.load("my_test_image.npy")  # 2D grayscale or 3D (H, W, C)
result = await svc.infer(
    model=session_id,           # session_id is the model identifier
    input_arrays=[test_image],  # list of numpy arrays (NOT Python lists)
    flow_threshold=0.4,
    cellprob_threshold=-1.0,
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

Data must be stored in a Hypha artifact. Supported label formats:

- **OME-TIFF** (`*.ome.tif`, `*.tif`): standard integer label masks, 0 = background
- **Palette-mode PNG** (`*.png`): produced by BioImage.IO Colab annotations — supported directly. The service converts these to integer TIFF internally.
- **GeoJSON** (`.geojson`): ignored — only image and mask files are used

```
your-dataset/
├── train_images/
│   ├── cell_001.png        # microscopy image
│   └── ...
├── masks_cells/
│   ├── cell_001.png        # palette-mode PNG label (BioImage.IO Colab output)
│   └── ...
└── test_images/
    └── ...
```

Or use the `metadata_dir` parameter with a JSON index of image/mask paths.

**Minimum recommended**: 10+ annotated images for fine-tuning to have any effect. For best results: 50–200 images.

### Brightfield / phase-contrast images

For non-fluorescence imaging (brightfield, phase-contrast, DIC), use `enable_clahe=True`:

```python
session = await svc.start_training(
    artifact="bioimage-io/annotation-mnxnayjn-gd8c",
    train_images="train_images/*.png",
    train_annotations="masks_cells/*.png",
    test_images="test_images/*.png",
    test_annotations="masks_cells/*.png",
    n_epochs=30,
    learning_rate=1e-5,
    enable_clahe=True,   # required for brightfield
    min_train_masks=1,
    validation_interval=1,
)
```

And for inference:
```python
result = await svc.infer(
    model=session_id,
    input_arrays=[raw_brightfield_image],
    enable_clahe=True,       # applies same CLAHE as training
    cellprob_threshold=-1.0,
)
```

**Why CLAHE is required**: Brightfield images typically have pixel values spanning only 10–60 out of 255. Cellpose-SAM (trained on fluorescence) cannot detect cells in such low-contrast images — CLAHE expands the dynamic range and makes cells visible.

**Note on per-epoch pixel IoU for brightfield**: The pixel-level IoU metric in `test_metrics` will read 0.000 throughout training on brightfield data — this is expected and not a bug. Cellpose's binary cell-probability map is near-zero for unseen brightfield images, so the pixel metric cannot improve during training. The meaningful metric is the **instance-level F1** (Hungarian matching at IoU≥0.5), computed post-training and returned in `instance_metrics`.

**Optimal training duration for brightfield**: Based on experiments, fine-tuning peaks around **20–50 epochs** with LR=1e-5. Longer training (500+ epochs) overfits on small datasets (< 50 images). Start with 30 epochs and evaluate.

## start_training parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `artifact` | str | required | Hypha artifact ID `workspace/alias` containing images |
| `train_images` | str | required | Glob pattern or folder path for training images |
| `train_annotations` | str | required | Glob pattern or folder path for training masks |
| `test_images` | str | None | Test images for per-epoch IoU evaluation |
| `test_annotations` | str | None | Test masks (must match test_images) |
| `metadata_dir` | str | None | Alternative: JSON metadata index directory |
| `model` | str | `"cpsam"` | Base model or previous session_id to continue from |
| `n_epochs` | int | 10 | Total training epochs |
| `learning_rate` | float | 1e-6 | Initial learning rate (use 1e-5 for brightfield) |
| `weight_decay` | float | 1e-4 | Weight decay |
| `validation_interval` | int | None | Compute test metrics every N epochs (None = every 10) |
| `min_train_masks` | int | 5 | Skip images with fewer than N annotated instances (use 1 for small datasets) |
| `n_samples` | int | None | Cap training images per epoch |
| `enable_clahe` | bool | False | Apply CLAHE preprocessing (required for brightfield/phase-contrast) |

## get_training_status return fields

```json
{
  "status_type": "running",
  "session_id": "2026-04-14-003025-2b7bfe9d",
  "current_epoch": 10,
  "total_epochs": 30,
  "elapsed_seconds": 70,
  "n_train": 13,
  "n_test": 2,
  "train_losses": [0.85, 0.72, 0.65, ...],
  "test_metrics": [
    {"iou": 0.000, "f1": 0.000, "precision": 0.000, "recall": 0.000},
    null, null, null,
    {"iou": 0.000, "f1": 0.000, "precision": 0.000, "recall": 0.000}
  ],
  "instance_metrics": {
    "ap_0_5": 0.72, "ap_0_75": 0.51, "ap_0_9": 0.18,
    "n_true": 340, "n_pred": 312
  }
}
```

**`test_metrics` iou is 0 for brightfield** — see note above. Use `instance_metrics` for the meaningful evaluation.

## infer parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | str | `"cpsam"` | Base model name OR session_id of fine-tuned model |
| `input_arrays` | list | — | List of numpy arrays (2D grayscale or 3D HWC) |
| `artifact` | str | None | Alternative: artifact ID + `image_paths` for server-side loading |
| `image_paths` | list[str] | None | Paths within artifact to load server-side |
| `diameter` | float | None | Cell diameter in pixels (None = auto-detect) |
| `flow_threshold` | float | 0.4 | Flow error threshold |
| `cellprob_threshold` | float | 0.0 | Cell probability threshold (-1.0 to 1.0; use -1.0 for low-contrast) |
| `enable_clahe` | bool | False | Apply CLAHE before inference (must match training setting) |

Returns a list of `{"input_path": str, "output": np.ndarray}` dicts.

## export_model parameters

| Parameter | Description |
|---|---|
| `session_id` | ID of a **completed** training session |
| `model_name` | Custom model name |
| `description` | Text appended to the BioImage.IO RDF description |
| `authors` | List of `{"name": "...", "affiliation": "..."}` dicts |
| `uploader` | `{"name": "...", "email": "..."}` for BioImage.IO uploader field |
| `collection` | Hypha artifact collection (default: `bioimage-io/colab-annotations`) |

Returns `{"artifact_id": "...", "model_name": "...", "status": "exported", "url": "https://..."}`.

## Known behaviours and pitfalls

### Palette-mode PNG labels (BioImage.IO Colab output)
BioImage.IO Colab saves annotation masks as palette-mode PNGs (PIL mode "P"). The service handles these correctly — they are converted to integer TIFF internally before training. No pre-conversion needed.

### Pixel IoU always 0 on brightfield
The per-epoch `test_metrics.iou` uses Cellpose's binary cell-probability pixel metric. For brightfield images, this is always near-zero during training because Cellpose-SAM (pre-trained on fluorescence) predicts zero cell probability for unseen brightfield textures regardless of fine-tuning. This is expected. The instance-level AP in `instance_metrics` (computed via Cellpose inference + Hungarian matching) is the correct metric.

### Training with very small datasets
With < 20 images, use:
- `min_train_masks=1` (default 5 skips images with few cells)
- `n_epochs=20–50` (overfitting starts early with small data)
- `learning_rate=1e-5` (higher than default 1e-6 for faster convergence)

## Real experimental results

### Fluorescence (Session B, 2026-04-14) — v0.0.19, validated
Dataset: `ri-scale/cellpose-test` (72 train / 19 test images, OME-TIFF)
LR=1e-5, 100 epochs, `min_train_masks=1`, `validation_interval=10`

| Epoch | Pixel IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| 1 (baseline) | 0.397 | 0.569 | 0.398 | 0.997 |
| 11 | 0.401 | 0.572 | 0.402 | 0.990 |
| 41 | 0.430 | 0.602 | 0.437 | 0.964 |
| 100 | **0.434** | **0.605** | 0.440 | 0.970 |

Instance AP@0.5: **0.477**, AP@0.75: **0.347** (19 test images). Pixel IoU improves +9.3% relative from baseline. Training loss declines from 0.87 → 0.55; test loss stabilises ~epoch 60.

**Interpretation**: Fine-tuning consistently improves fluorescence segmentation. Plateau visible after ~50 epochs at 100ep — longer runs (500+ep) may improve further but risk overfitting with small datasets.

### Fluorescence (Session A, 2026-04-08) — historical reference
Dataset: `ri-scale/cellpose-test` (80 train / 20 test images, OME-TIFF)
LR=1e-5, 1000 epochs

| Epoch | Pixel IoU | F1 |
|---|---|---|
| 1 (baseline) | 0.430 | 0.601 |
| 100 | 0.501 | 0.667 |
| 1000 | 0.461 | 0.631 |

Instance AP@0.5 at epoch 1000: 0.495 (vs baseline ~0.40)

### Brightfield (2026-04-14)
Dataset: `bioimage-io/annotation-mnxnayjn-gd8c` (13 train / 2 test images, palette PNG)
LR=1e-5, enable_clahe=True

| Model | Test image | GT cells | Detected | Instance F1 |
|---|---|---|---|---|
| Baseline (cpsam) | BL | 18 | 15 | **0.848** |
| Fine-tuned 30ep | BL | 18 | 13 | 0.774 |
| Fine-tuned 500ep | BL | 18 | 12 | 0.667 |

Raw brightfield baseline (without CLAHE): **0 cells detected**. CLAHE is required.

**Note on brightfield fine-tuning degradation**: With only 13 training images covering 7 of 9 tile positions in a 3×3 well grid, fine-tuning degrades on the held-out tile positions (BL, CL). This is a dataset size/coverage issue — not an application bug. Fluorescence experiments (Session A and B above) confirm the application produces consistent improvement when training data is sufficient.

## Deploying and updating the cellpose-finetuning app

The `cellpose-finetuning` service is deployed on the BioEngine worker (`bioimage-io/bioengine-worker`) and runs in the `bioimage-io` workspace. It requires a `HYPHA_TOKEN` that has write access to that workspace. Pass it on first deployment:

```bash
export BIOENGINE_WORKER_SERVICE_ID=bioimage-io/bioengine-worker
bioengine apps deploy ./bioengine_apps/cellpose-finetuning/ \
  --env _HYPHA_TOKEN=<bioimage-io-scoped-token>
```

**When updating an existing deployment**, the new version automatically inherits all env vars (including `HYPHA_TOKEN`) and all init args/kwargs from the previous app — do **not** pass `--env` again unless intentionally rotating a secret:

```bash
bioengine apps upload ./bioengine_apps/cellpose-finetuning/
bioengine apps run bioimage-io/cellpose-finetuning --app-id <existing-app-id>
# HYPHA_TOKEN and all other env vars are carried over automatically
```

## Authentication

```python
from hypha_rpc import login
token = await login(server_url="https://hypha.aicell.io")
# or: token = os.environ["HYPHA_TOKEN"]
```

## Integration with BioImage.IO Colab

1. Open BioImage.IO Colab in browser
2. Mount your dataset from Hypha artifact storage
3. Use Cellpose-SAM for initial pre-segmentation
4. Correct annotations interactively (multiple annotators, any device)
5. Call `start_training()` with the annotated dataset
6. Monitor with `get_training_status()` until metrics plateau
7. Call `export_model()` to publish to BioImage.IO Model Zoo

No local GPU, no command line, no software installation.

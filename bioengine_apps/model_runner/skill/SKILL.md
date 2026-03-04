# BioImage.IO Model Runner Skill

## Metadata

- **Name**: BioImage.IO Model Runner
- **Version**: 1.0.0
- **Description**: Run deep learning models from the BioImage Model Zoo for bioimage analysis tasks including segmentation, denoising, super-resolution, and object detection. Supports single-model inference and multi-model screening/comparison.
- **Domain**: Bioimage analysis, microscopy, biological imaging
- **MCP Server**: `https://hypha.aicell.io/bioimage-io/mcp/model-runner`
- **HTTP Base URL**: `https://hypha.aicell.io/bioimage-io/services/model-runner`
- **Hypha Service ID**: `bioimage-io/model-runner`

---

## When to Use This Skill

Use this skill when the user wants to:

- Run a deep learning model on a microscopy or biological image
- Find suitable models for a bioimage analysis task (segmentation, denoising, restoration, detection)
- Compare multiple models on the same input image
- Validate or test a BioImage.IO model
- Get metadata about a model from the BioImage Model Zoo
- Screen models for a specific application (e.g., "segment nuclei in fluorescence images")

**Typical image types**: fluorescence microscopy, electron microscopy (EM), phase-contrast microscopy, confocal microscopy, cryoEM, histopathology, plant tissue imaging, agarose gel images.

---

## API Reference

### Endpoints

All endpoints are available via:

| Method | URL Pattern |
|--------|------------|
| **HTTP GET/POST** | `https://hypha.aicell.io/bioimage-io/services/model-runner/<function>` |
| **MCP** | `https://hypha.aicell.io/bioimage-io/mcp/model-runner` |
| **Hypha RPC** | Service ID: `bioimage-io/model-runner` |

### Functions

#### `search_models`

Search for models in the BioImage.IO collection.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `keywords` | `list[str]` | No | `null` | Keywords to filter models (e.g., `["nuclei", "segmentation"]`) |
| `limit` | `int` | No | `10` | Maximum number of models to return |
| `ignore_checks` | `bool` | No | `false` | If `true`, return all models including those that have not passed BioEngine inference checks |

**Returns**: List of `{"model_id": str, "description": str}`.

**Important**: By default (`ignore_checks=false`), only models that have passed BioEngine inference checks are returned. These models are confirmed to work with the model-runner. Setting `ignore_checks=true` returns all models, but some may fail during inference.

**HTTP example**:
```
GET https://hypha.aicell.io/bioimage-io/services/model-runner/search_models?keywords=nuclei,segmentation&limit=5
```

#### `get_model_rdf`

Retrieve the Resource Description Framework (RDF) metadata for a model.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str` | **Yes** | — | Model identifier (e.g., `"affable-shark"`) |
| `stage` | `bool` | No | `false` | Use staged (draft) version instead of committed |

**Returns**: Dictionary with full model metadata including `inputs`, `outputs`, `name`, `description`, `tags`, `weights`, etc.

**IMPORTANT — Known Limitation**: The HTTP GET endpoint for `get_model_rdf` currently fails with `"ValueError: Out of range float values are not JSON compliant"` for some models. **Use the Hypha RPC SDK or fetch the RDF YAML directly instead**:
```
GET https://hypha.aicell.io/bioimage-io/artifacts/<model_id>/files/rdf.yaml
```
Parse the returned YAML content. The Hypha RPC SDK (`hypha-rpc`) does not have this limitation.

#### `get_upload_url`

Get a presigned URL for uploading an input image to temporary S3 storage.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_type` | `str` | **Yes** | — | One of: `.npy`, `.png`, `.tiff`, `.tif`, `.jpeg`, `.jpg` |

**Returns**: `{"upload_url": str, "file_path": str}`

- `upload_url`: Presigned URL — upload the file via HTTP PUT
- `file_path`: Temporary path to pass as `inputs` to `infer`

**TTL**: Upload URLs and uploaded files expire after **1 hour**.

**HTTP example**:
```
GET https://hypha.aicell.io/bioimage-io/services/model-runner/get_upload_url?file_type=.npy
```

#### `infer`

Run inference on a model with provided input data.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str` | **Yes** | — | Model identifier |
| `inputs` | `ndarray \| str \| dict` | **Yes** | — | Input image as numpy array, file path from `get_upload_url`, or HTTP URL |
| `return_download_url` | `bool` | No | `false` | If `true`, return presigned download URLs instead of raw arrays |
| `weights_format` | `str` | No | `null` | Preferred weights: `"pytorch_state_dict"`, `"torchscript"`, `"onnx"`, `"tensorflow_saved_model"` |
| `device` | `str` | No | `null` | `"cuda"` or `"cpu"` (auto-detected if null) |
| `skip_cache` | `bool` | No | `false` | Force re-download of model files |

**Returns**: Dictionary mapping output tensor names to numpy arrays (or download URLs if `return_download_url=true`).

**Example return**: `{"output0": <np.ndarray or URL>}` — the key name comes from the model's RDF output specification.

**HTTP examples** (with file path from `get_upload_url`):
```
# GET (simpler for agents)
GET https://hypha.aicell.io/bioimage-io/services/model-runner/infer?model_id=affable-shark&inputs=temp/abc123-def456.npy&return_download_url=true

# POST (also works)
POST https://hypha.aicell.io/bioimage-io/services/model-runner/infer
Content-Type: application/json

{
    "model_id": "affable-shark",
    "inputs": "temp/abc123-def456.npy",
    "return_download_url": true
}
```

**IMPORTANT**: Always set `return_download_url=true` when using HTTP endpoints. Without it, the endpoint tries to serialize numpy arrays to JSON, which fails.

#### `test`

Run the official BioImage.IO test suite on a model.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_id` | `str` | **Yes** | — | Model identifier |
| `stage` | `bool` | No | `false` | Use staged version |
| `skip_cache` | `bool` | No | `false` | Force re-download before testing |

**Returns**: Comprehensive test report dictionary.

#### `validate`

Validate an RDF dictionary against BioImage.IO specifications.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `rdf_dict` | `dict` | **Yes** | — | Complete RDF dictionary to validate |

**Returns**: `{"success": bool, "details": str}`

---

## Workflow 1: Run a Known Model

Use this when the user specifies a model ID or a clear model name.

### Steps

1. **Get model metadata** — call `get_model_rdf(model_id=...)` to understand input/output specs
2. **Prepare the input image** — reshape/normalize to match model expectations
3. **Upload the image** — call `get_upload_url`, then PUT the file to the returned URL
4. **Run inference** — call `infer(model_id=..., inputs=<file_path>, return_download_url=true)`
5. **Download and display results** — GET the returned download URLs, load as numpy arrays

### Python Code

```python
import asyncio
import numpy as np
import httpx
from io import BytesIO
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"

async def run_model(model_id: str, image: np.ndarray):
    """Run a bioimage.io model on an input image."""
    server = await connect_to_server(
        {"server_url": SERVER_URL, "method_timeout": 180}
    )
    mr = await server.get_service("bioimage-io/model-runner")

    # 1. Get model metadata to understand expected input format
    rdf = await mr.get_model_rdf(model_id=model_id)
    print(f"Model: {rdf['name']}")
    print(f"Description: {rdf['description']}")

    # 2. Upload image
    upload_info = await mr.get_upload_url(file_type=".npy")
    buffer = BytesIO()
    np.save(buffer, image.astype(np.float32))
    async with httpx.AsyncClient() as client:
        await client.put(upload_info["upload_url"], content=buffer.getvalue())

    # 3. Run inference
    result = await mr.infer(
        model_id=model_id,
        inputs=upload_info["file_path"],
        return_download_url=True,
    )

    # 4. Download results
    outputs = {}
    async with httpx.AsyncClient() as client:
        for key, url in result.items():
            resp = await client.get(url)
            outputs[key] = np.load(BytesIO(resp.content))
            print(f"Output '{key}': shape={outputs[key].shape}")

    await server.disconnect()
    return outputs
```

---

## Workflow 2: Model Screening and Comparison

Use this when the user wants to find the best model for their task. The user provides:
- An input image
- A ground truth output (for quantitative comparison)
- A description of the task (e.g., "segment nuclei in fluorescence images")

### Steps

1. **Search for candidate models** — call `search_models` with task-relevant keywords
2. **Select 5–10 models** for comparison based on descriptions
3. **For each model**:
   a. Get model RDF to understand input requirements
   b. Prepare the input image (resize/crop/normalize as needed)
   c. Upload the image and run inference
   d. Download the result
   e. Handle errors gracefully (input shape mismatch, model incompatibility)
4. **Compare results** — compute quantitative metrics (e.g., IoU for segmentation)
5. **Visualize** — create a montage image and a metrics bar chart
6. **Save files** — save the montage and metrics plot for the user

### Searching for Models

Choose keywords based on the user's task:
- Segmentation tasks: `["segmentation", "<target>"]` (e.g., `["segmentation", "nuclei"]`)
- Denoising/restoration: `["denoising"]`, `["restoration"]`, `["enhancement"]`
- Detection: `["detection"]`, `["stardist"]`
- Modality-specific: `["electron-microscopy"]`, `["fluorescence"]`, `["phase-contrast"]`

### Handling Input Shape Mismatches

Models expect specific input shapes. When inference fails, the most common cause is an input shape mismatch. **Always try inference first before discarding a model** — the RDF dimensions can be conservative and models often accept other sizes.

**Reading input specs from the RDF**:

```python
# Format 0.5.x (axes is a list of dicts)
axes = rdf["inputs"][0]["axes"]
# Example: [{"type": "batch"}, {"id": "channel", "type": "channel", ...}, {"id": "y", "type": "space", "size": {"min": 64, "step": 16}}, ...]

# Format 0.4.x (axes is a string like "bcyx")
axes = rdf["inputs"][0]["axes"]  # e.g., "bcyx"
shape = rdf["inputs"][0].get("shape")  # e.g., [1, 1, 128, 128]
```

**Common adaptations when inference fails**:

```python
import numpy as np

def prepare_image_for_model(image: np.ndarray, rdf: dict) -> np.ndarray:
    """Adapt image dimensions to match model expectations."""
    axes = rdf["inputs"][0]["axes"]
    
    # Determine expected number of dimensions
    if isinstance(axes, str):
        n_dims = len(axes)
    else:
        n_dims = len(axes)
    
    img = image.copy().astype(np.float32)
    
    # Add batch dimension if needed
    if img.ndim == n_dims - 1:
        img = img[np.newaxis, ...]
    
    # Add channel dimension if needed (2D image -> 4D: bcyx)
    if img.ndim == 2 and n_dims == 4:
        img = img[np.newaxis, np.newaxis, ...]
    elif img.ndim == 3 and n_dims == 4:
        # Could be (c, y, x) or (y, x, c) — check last dim
        if img.shape[-1] in (1, 3, 4):  # likely channels-last
            img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]
    
    # Ensure spatial dims meet min/step constraints (for 0.5.x format)
    # Valid sizes are: min + N*step (N >= 0), e.g., min=64, step=16 → 64, 80, 96, ...
    if isinstance(axes, list):
        for i, ax in enumerate(axes):
            if isinstance(ax, dict) and ax.get("type") == "space":
                size_spec = ax.get("size", {})
                if isinstance(size_spec, dict):
                    step = size_spec.get("step", 1)
                    min_size = size_spec.get("min", 0)
                    current = img.shape[i]
                    if current < min_size:
                        # Pad to minimum size
                        pad_width = [(0, 0)] * img.ndim
                        pad_width[i] = (0, min_size - current)
                        img = np.pad(img, pad_width, mode="reflect")
                    else:
                        # Round down to nearest valid size: min + N*step
                        n = (current - min_size) // step
                        valid = min_size + n * step
                        if valid < current:
                            slices = [slice(None)] * img.ndim
                            slices[i] = slice(0, valid)
                            img = img[tuple(slices)]
    
    return img
```

### Computing Comparison Metrics

For segmentation tasks, use IoU (Intersection over Union):

```python
def compute_iou(prediction: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5) -> float:
    """Compute IoU between prediction and ground truth masks."""
    pred_binary = prediction > threshold
    gt_binary = ground_truth > threshold
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)
```

For other tasks, choose appropriate metrics:
- **Denoising/Restoration**: PSNR (`10 * log10(MAX² / MSE)`), SSIM
- **Detection**: Precision, Recall, F1
- **Super-resolution**: PSNR, SSIM

### Creating the Comparison Montage

```python
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("Agg")  # Non-interactive backend

def create_montage(
    input_image: np.ndarray,
    ground_truth: np.ndarray,
    model_results: dict,  # {model_id: np.ndarray}
    save_path: str = "model_comparison_montage.png",
):
    """Create a visual comparison montage.

    First row: input image and ground truth, centered.
    Subsequent rows: model results.
    """
    n_models = len(model_results)
    n_cols = min(max(n_models, 2), 6)  # max 6 per row
    n_model_rows = (n_models + n_cols - 1) // n_cols
    n_rows = 1 + n_model_rows  # first row for input+GT, rest for models

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Hide all axes
    for ax in axes.flatten():
        ax.axis("off")

    # First row: input and ground truth, centered
    left_pad = (n_cols - 2) // 2
    axes[0, left_pad].imshow(input_image, cmap="gray")
    axes[0, left_pad].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0, left_pad + 1].imshow(ground_truth, cmap="gray")
    axes[0, left_pad + 1].set_title("Ground Truth", fontsize=11, fontweight="bold")

    # Subsequent rows: model results
    for i, (model_id, result) in enumerate(model_results.items()):
        row = 1 + i // n_cols
        col = i % n_cols
        axes[row, col].imshow(result, cmap="gray")
        axes[row, col].set_title(model_id, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Montage saved to: {save_path}")
```

### Creating the Metrics Plot

```python
def create_metrics_plot(
    metrics: dict,  # {model_id: {"iou": float, ...}}
    metric_name: str = "IoU",
    save_path: str = "model_comparison_metrics.png",
):
    """Create a bar chart comparing model metrics."""
    models = list(metrics.keys())
    values = [metrics[m][metric_name.lower()] for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    bars = ax.bar(range(len(models)), values, color="steelblue")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Model Comparison — {metric_name}")
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics plot saved to: {save_path}")
```

### Full Screening Script

See [scripts/model_screening.py](scripts/model_screening.py) for the complete, runnable model screening workflow.

---

## Workflow 3: Validate and Test a Model

Use this when a user wants to check if a model is valid or passes tests.

```python
# Validate model format
rdf = await mr.get_model_rdf(model_id="affable-shark")
validation = await mr.validate(rdf_dict=rdf)
print(f"Valid: {validation['success']}")
print(validation['details'])

# Run full test suite
test_report = await mr.test(model_id="affable-shark")
print(f"Status: {test_report['status']}")
```

---

## Critical Rules and Pitfalls

### DO — Correct Patterns

1. **Always use `return_download_url=true` for HTTP/MCP workflows** — raw numpy arrays cannot be transferred over HTTP. Only the Hypha RPC SDK can transfer numpy arrays directly.

2. **Always upload images via `get_upload_url` + HTTP PUT for HTTP/MCP workflows** — you cannot pass raw array data in a JSON request body.

3. **Try inference before discarding a model** — the RDF may list conservative input dimensions. Many models accept a range of sizes. Only adjust dimensions after an inference error.

4. **Cast images to `float32` before uploading** — models expect float32 input.

5. **Use `.npy` format for upload when possible** — it preserves array shape and dtype exactly. Use `.png`/`.tiff` only for 2D single-channel or RGB images.

6. **Handle both RDF format versions** — format 0.4.x uses `axes: "bcyx"` (string) and `shape: [1,1,128,128]`; format 0.5.x uses `axes: [{type: "batch"}, {id: "channel", ...}, ...]` (list of dicts).

7. **Squeeze display dimensions** — model outputs are typically 4D (batch, channel, y, x). Squeeze or index to 2D for visualization: `result[0, 0, :, :]`.

8. **Use `method_timeout=180` (or higher) for Hypha RPC connections** — inference can take 30–120 seconds for large models on first run (includes model download + inference).

### DO NOT — Common Mistakes

1. **DO NOT pass numpy arrays in HTTP JSON requests** — JSON cannot encode numpy arrays. Use `get_upload_url` to upload, then pass the `file_path` string.

    ```python
    # WRONG — will fail
    requests.post(f"{BASE}/infer", json={"model_id": "affable-shark", "inputs": image.tolist()})

    # CORRECT — upload first, then pass file_path
    upload_info = requests.get(f"{BASE}/get_upload_url?file_type=.npy").json()
    requests.put(upload_info["upload_url"], data=npy_bytes)
    requests.post(f"{BASE}/infer", json={"model_id": "affable-shark", "inputs": upload_info["file_path"], "return_download_url": True})
    ```

2. **DO NOT use `get_model_rdf` via HTTP GET** — it fails with `"ValueError: Out of range float values are not JSON compliant"` for many models because some RDF fields contain `inf`/`-inf` values. Fetch the RDF YAML directly:

    ```python
    # WRONG — fails for models with inf values
    requests.get(f"{BASE}/get_model_rdf?model_id=ambitious-ant")

    # CORRECT — fetch YAML directly
    import yaml
    resp = requests.get(f"https://hypha.aicell.io/bioimage-io/artifacts/ambitious-ant/files/rdf.yaml")
    rdf = yaml.safe_load(resp.text)

    # ALSO CORRECT — use Hypha RPC SDK (handles non-JSON types)
    mr = await server.get_service("bioimage-io/model-runner")
    rdf = await mr.get_model_rdf(model_id="ambitious-ant")
    ```

3. **DO NOT assume all searched models will work** — inference can fail due to:
    - Input shape mismatch (most common — try resizing/padding)
    - Model requires specific preprocessing
    - 3D models receiving 2D input
    - Model weights incompatibility with available hardware
    
    Always wrap inference in try/except and continue to the next model.

4. **DO NOT discard a model based solely on RDF shape specs** — the shape in the RDF is often the minimum or test shape. A model with `shape: [1, 1, 128, 128]` usually works with `[1, 1, 256, 256]` or `[1, 1, 512, 512]` too.

5. **DO NOT forget to normalize images** — if uploading as `.png`/`.jpg`, the image must be uint8 [0, 255]. For `.npy`, use float32 and consider normalizing to [0, 1] if the model expects it (check `data_range` in the RDF inputs).

6. **DO NOT use very large images without tiling** — the model-runner handles tiling automatically for supported models, but very large images (>4096px) may still cause memory issues. Consider downscaling or cropping first.

7. **DO NOT compare model outputs directly without matching shapes** — different models may output different numbers of channels or spatial sizes. Always check output shape and crop/resize when comparing.

8. **DO NOT pass `inputs` as a URL string when using the Hypha RPC SDK with a local numpy array** — pass the array directly. The `inputs` parameter accepts numpy arrays natively via RPC.

    ```python
    # WRONG with hypha-rpc — unnecessary upload step
    upload_info = await mr.get_upload_url(file_type=".npy")
    # ... upload steps ...
    result = await mr.infer(model_id="affable-shark", inputs=upload_info["file_path"])

    # CORRECT with hypha-rpc — pass array directly
    result = await mr.infer(model_id="affable-shark", inputs=image)

    # ALSO CORRECT — pass array directly AND get download URLs back
    result = await mr.infer(model_id="affable-shark", inputs=image, return_download_url=True)
    ```
    Use `get_upload_url` only when: (a) using HTTP endpoints, or (b) using MCP. The `return_download_url` parameter works with all input types including direct numpy arrays via RPC.

9. **DO NOT use `image.tolist()` or `json.dumps(array)` to serialize arrays** — always use `np.save()` to a BytesIO buffer for `.npy` format.

10. **DO NOT forget that `infer` via HTTP GET requires `return_download_url=true`** — without it, the response will try to serialize numpy arrays as JSON, which fails.

    ```
    # WRONG — missing return_download_url, response cannot serialize numpy arrays
    GET /infer?model_id=affable-shark&inputs=temp/abc.npy

    # CORRECT — return_download_url=true returns URLs instead of arrays
    GET /infer?model_id=affable-shark&inputs=temp/abc.npy&return_download_url=true

    # ALSO CORRECT — POST with JSON body
    POST /infer
    Content-Type: application/json
    {"model_id": "affable-shark", "inputs": "temp/abc.npy", "return_download_url": true}
    ```

---

## RDF Format Quick Reference

### Format 0.4.x (older models)

```yaml
inputs:
  - axes: bcyx                    # String: batch, channel, y, x
    data_type: float32
    shape: [1, 1, 128, 128]      # Fixed shape
    data_range: [0.0, 1.0]
    name: input0
outputs:
  - axes: bcyx
    data_type: float32
    shape: [1, 1, 256, 256]
    name: output0
test_inputs: [test-input.npy]     # Test data file names
test_outputs: [test-output.npy]
```

### Format 0.5.x (newer models)

```yaml
inputs:
  - id: input0
    axes:
      - type: batch
      - id: channel
        type: channel
        channel_names: [channel0]
      - id: y
        type: space
        size: {min: 64, step: 16}  # Flexible size with constraints
      - id: x
        type: space
        size: {min: 64, step: 16}
    test_tensor:
      source: test_input_0.npy
    data:
      type: float32
outputs:
  - id: output0
    axes:
      - type: batch
      - id: channel
        type: channel
        channel_names: [channel0, channel1]
      - id: y
        type: space
      - id: x
        type: space
    test_tensor:
      source: test_output_0.npy
```

### Reading the Input Key

To determine which key name to use for multi-input models:

```python
# 0.4.x: use "name" field
input_name = rdf["inputs"][0]["name"]    # e.g., "input0"

# 0.5.x: use "id" field
input_name = rdf["inputs"][0]["id"]      # e.g., "input0"
```

### Reading the Output Key

The output dictionary key from `infer` matches the output name/id:

```python
# 0.4.x
output_key = rdf["outputs"][0]["name"]   # e.g., "output0"

# 0.5.x
output_key = rdf["outputs"][0]["id"]     # e.g., "output0"
```

---

## File Reference

| File | Description |
|------|-------------|
| [assets/search_keywords.yaml](assets/search_keywords.yaml) | Common search keywords by task, target structure, modality, and architecture |
| [scripts/model_screening.py](scripts/model_screening.py) | Complete model screening and comparison workflow |
| [scripts/run_single_model.py](scripts/run_single_model.py) | Run inference with a single known model |
| [scripts/test_api.py](scripts/test_api.py) | API endpoint verification test script |
| [references/api_reference.md](references/api_reference.md) | Detailed API parameter and response reference |
| [references/rdf_format.md](references/rdf_format.md) | BioImage.IO RDF format documentation |

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `"Out of range float values are not JSON compliant"` | Model RDF contains `inf`/`-inf` values | Fetch RDF as YAML from artifacts API instead of using HTTP `get_model_rdf` |
| `"Failed to run inference"` with shape error | Input shape doesn't match model | Check RDF axes, add batch/channel dims, resize spatial dims to valid size |
| Timeout on `infer` call | First run downloads model (~10–200 MB) | Increase `method_timeout` to 300; subsequent calls use cache |
| `"Source does not exist or has expired"` | Uploaded file TTL (1 hour) expired | Re-upload the file with a fresh `get_upload_url` call |
| Empty or all-zero output | Input not normalized correctly | Check `data_range` in RDF inputs; normalize to [0,1] if required |
| `"model_id is a URL"` error | Passed a URL instead of model ID | Use the short model ID like `"affable-shark"`, not a full URL |
| 3D model fails on 2D input | Model expects z-dimension | Skip this model or add a z-dimension `img[np.newaxis, ...]` |
| Output has unexpected channels | Model outputs multiple classes | Check `outputs[0].axes` for channel count; select relevant channel index |

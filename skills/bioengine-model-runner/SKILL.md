---
name: bioengine-model-runner
description: Searches, validates, tests, and runs inference on BioImage.IO models via the BioEngine model-runner service. Use when the user wants to run microscopy image analysis, find segmentation/denoising/restoration/detection models, compare multiple models against ground truth, validate a model RDF, or test BioImage.IO model compliance.
license: MIT
metadata:
  service-id: bioimage-io/model-runner
  worker-service-id: bioimage-io/bioengine-worker
  http-base-url: https://hypha.aicell.io/bioimage-io/services/model-runner
  mcp-server: https://hypha.aicell.io/bioimage-io/mcp/model-runner
  cli-package: bioengine (pip install bioengine)
---

# BioEngine Model Runner

## Quick start

The CLI is bundled in this skill at `bioengine_cli/`. Install its dependencies once:

```bash
pip install hypha-rpc httpx numpy tifffile Pillow click
```

Then use the bundled CLI directly from the skill root:

```bash
python -m bioengine_cli runner search --keywords nuclei segmentation --limit 5
python -m bioengine_cli runner info affable-shark
python -m bioengine_cli runner infer affable-shark --input cells.tif --output mask.npy
```

Alternatively, install as a package for the shorter `bioengine` command:

```bash
pip install -e bioengine_cli/   # installs from bundled source
# or: pip install bioengine     # install from PyPI
bioengine runner search --keywords nuclei segmentation --limit 5
```

**No authentication required** — the model-runner service is public.
**No local GPU required** — computation runs on BioEngine remote workers.

## CLI reference

| Command | Purpose |
|---|---|
| `bioengine runner search [--keywords ...] [--limit N]` | Find runnable models in BioImage.IO |
| `bioengine runner info <model-id>` | Model metadata: inputs, outputs, weights |
| `bioengine runner validate <rdf.yaml>` | Validate RDF format against bioimage.io spec |
| `bioengine runner test <model-id>` | Run official BioImage.IO test suite |
| `bioengine runner infer <model-id> --input <file> --output <file>` | Run inference |
| `bioengine runner compare <id1> <id2> ... --input <file>` | Run multiple models, save all outputs |

All commands accept `--json` for machine-parseable output and `--server-url` to override the default server.

## Default operating mode

- Use the CLI for all operations — it handles upload, RPC connection, and download automatically.
- Input formats: `.npy` (lossless, preferred), `.tif`/`.tiff`, `.png`.
- Output format: `.npy` by default; `.tif` if output path ends in `.tif`.
- Default to models that pass BioImage.IO checks (omit `--ignore-checks` unless necessary).
- **Output keys vary by model** — read `outputs[0].id` from the RDF via `bioengine runner info`, not assume `"output0"`.
- **Search keywords**: AND-matched against model tags. If a keyword like `"denoising"` returns few results, try synonyms: `"restoration"`, `"noise"`. Use `assets/search_keywords.yaml` for known working presets.
- **RDF objects**: `get_model_rdf` via RPC returns `ObjectProxy` (not plain dict). JSON-serialize with `json.dumps(rdf, default=str)` if needed.

## Single model inference workflow

```text
- [ ] Step 1: Search for models — bioengine runner search --keywords <task>
- [ ] Step 2: Inspect the best candidate — bioengine runner info <model-id>
- [ ] Step 3: Read model documentation — get_model_documentation(model_id) — verify domain compatibility
- [ ] Step 4: Run inference — bioengine runner infer <model-id> --input image.tif --output result.npy
- [ ] Step 5: Validate output — load result.npy with numpy, check shape and values
```

```bash
# Full example
bioengine runner search --keywords nuclei segmentation --limit 5 --json
bioengine runner info affable-shark
bioengine runner infer affable-shark --input cells.tif --output mask.npy
```

Use `scripts/utils.py` helpers for normalization and evaluation — do not rewrite tensor logic:

```python
from scripts.utils import prepare_image_for_model, normalize_image

image = normalize_image(raw_image)               # percentile normalization (pmin=1, pmax=99.8)
tensor = prepare_image_for_model(image, axes)    # reshape to model input axes
```

Output key: `outputs[0].name` (RDF 0.4.x) or `outputs[0].id` (RDF 0.5.x). On shape errors: inspect the RDF (`bioengine runner info`), adapt dimensions, retry before discarding the model.

## Model screening / comparison workflow

```text
- [ ] Step 1: Clarify task type (segmentation / denoising / restoration / detection)
- [ ] Step 2: Search models — use keywords from assets/search_keywords.yaml
- [ ] Step 3: For each candidate — call get_model_documentation to read the README
- [ ] Step 4: Filter candidates — discard domain mismatches based on documentation
- [ ] Step 5: Run all suitable models on the same input — bioengine runner compare
- [ ] Step 6: Score models — utils.evaluate_segmentation() for IoU/Dice
- [ ] Step 7: Save all artifacts to comparison_results/
- [ ] Step 8: Generate Illustration 1 (ranked barplot), Illustration 2 (montage), Illustration 3 if instance segmentation (object count)
- [ ] Step 9: Write comparison_summary.json
- [ ] Step 10: Generate HTML report
```

```bash
# Run multiple models and save all outputs
bioengine runner compare affable-shark ambitious-ant chatty-frog \
  --input cells.tif \
  --output-dir comparison_results/
```

### Step 3: Read model documentation before running

**Always call `get_model_documentation` for every candidate before running inference.** This fetches the model's README markdown file, which contains:
- Training data domain (brightfield, fluorescence, H&E, electron microscopy, etc.)
- Required input channels and expected staining protocols
- Recommended preprocessing steps
- Known limitations and magnification constraints

**HTTP endpoint**:
```
GET https://hypha.aicell.io/bioimage-io/services/model-runner/get_model_documentation?model_id={model_id}
```

**Python (RPC)**:
```python
from hypha_rpc import connect_to_server
server = await connect_to_server(server_url="https://hypha.aicell.io")
runner = await server.get_service("bioimage-io/model-runner")
doc = await runner.get_model_documentation(model_id="affable-shark")
# Returns: Markdown string or None if no documentation file exists
```

**HTTP (requests)**:
```python
import requests
r = requests.get(
    "https://hypha.aicell.io/bioimage-io/services/model-runner/get_model_documentation",
    params={"model_id": "affable-shark"}
)
doc = r.json()  # Markdown string or null
```

**Decision rules after reading documentation**:
- Model trained on H&E/brightfield → skip if input is fluorescence (and vice versa)
- Model requires 3+ channels → skip if only 1 channel is available (unless you can provide all required channels)
- Model trained on whole-slide-imaging at 40× → skip if your image is at a very different magnification
- If documentation is None → fall back to RDF tags and test tensor inspection (see domain mismatch section)

Also check the RDF `tags` and `description` fields from `bioengine runner info` as a secondary signal.

**Artifact layout** (always save here — create folder if missing):

```
comparison_results/
├── {model_id}_output.npy              # raw prediction per model
├── illustration1_barplot.pdf          # Illustration 1: ranked metric barplot
├── illustration1_barplot.png          # same at 300 DPI
├── illustration2_montage.pdf          # Illustration 2: input/GT/predictions montage
├── illustration2_montage.png          # same at 300 DPI
├── illustration3_counts.pdf           # Illustration 3: object count (instance seg only)
├── illustration3_counts.png           # same at 300 DPI
└── comparison_summary.json
```

### Required illustrations

Every screening run **must produce at least two illustrations**. Generate them as publication-quality figures (Nature/Cell style):
- Figure width: 7.0 inches (Nature Methods single-column)
- Font: Arial or Helvetica, 7–8 pt for axis labels, 6 pt for tick labels
- Resolution: 300 DPI PNG + PDF with embedded fonts (`pdf.fonttype=42`, `ps.fonttype=42`)
- No chartjunk: remove top/right spines, use subtle gridlines (#e0e0e0)
- Colors: use colorblind-friendly palettes (e.g. ColorBrewer diverging/sequential)
- All labels must be legible at print size — minimum 6 pt

---

#### Illustration 1 — Ranked metric barplot

**Skip only if there is exactly one suitable model** (a single-candidate result). In that case, embed the metric value as a text annotation in Illustration 2 instead.

**For segmentation tasks** (instance or semantic): use **F1 score (IoU ≥ 0.5)** as the primary metric. Show both F1 and mean IoU as grouped or stacked bars if space allows.

**For other tasks**: choose the most appropriate metric for the domain (e.g., PSNR/SSIM for denoising, AP for detection) and label the axis accordingly.

**Layout rules**:
- Horizontal barplot, models sorted **best at top, worst at bottom**
- Color bars by performance category (e.g. blue = good, purple = acceptable, red = poor, grey/hatched = domain mismatch)
- Annotate each bar with its numeric metric value (right of bar, 5.5 pt)
- Add italic annotation for domain-mismatch or wrong-task models (e.g. "(domain mismatch)", "(wrong task)")
- Include a brief legend for the color categories
- x-axis range: 0 to 1 for F1/IoU; add dashed reference line at x=1.0
- y-axis: model IDs (use the bioimage.io slug, not the full name)
- Panel label: bold "a" at top-left

**Python implementation** (use matplotlib; do NOT use seaborn):

```python
import matplotlib
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(3.5, 0.6 * n_models + 0.8))
bars = ax.barh(y_pos, f1_scores, color=colors, height=0.55)
ax.set_xlim(0, 1.25)
ax.axvline(1.0, color="#999999", lw=0.5, ls="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig("illustration1_barplot.pdf", bbox_inches="tight")
fig.savefig("illustration1_barplot.png", bbox_inches="tight", dpi=300)
```

---

#### Illustration 2 — Input / GT / Predictions montage

Show all screened models in a single figure. **Models are ordered top-to-bottom (or left-to-right) in the same rank order as Illustration 1** (best first).

**Layout**:
- Row 1: Input image | Ground truth (if available) — centered, with empty columns as padding if needed
- Remaining rows: one panel per model prediction, in ranked order
- If ≤ 6 models: use a 2-column grid (input/GT in cols 1-2 of row 1; pairs of models per row below)
- If > 6 models: use a 3-column grid
- Each panel: title = model ID (6 pt), subtitle = metric value e.g. "F1 = 0.909" (5.5 pt, grey) — unless single-candidate (then add metric directly to title)
- Scale bar in bottom-left of each panel (white, with µm label if pixel size is known)
- For segmentation outputs: render as colored instance overlay (tab20 colormap, dark background `#141414`). Do NOT show raw probability maps — always postprocess to instance labels first (watershed for UNet, NMS polygon for StarDist).
- For the input panel: apply CLAHE and display as grayscale (cmap="gray")
- For GT: render as colored instance overlay, same style as predictions
- Panel letter "b" at top-left of figure

**Python**:
```python
fig.savefig("illustration2_montage.pdf", bbox_inches="tight")
fig.savefig("illustration2_montage.png", bbox_inches="tight", dpi=300)
```

---

#### Illustration 3 — Object count comparison (instance segmentation only)

Only generate this figure for **instance segmentation tasks** (cell segmentation, nucleus counting, etc.).

**Layout**:
- Horizontal barplot, same model order as Illustration 1 (best at top)
- Bars show predicted object count per model
- **Red vertical reference line** at the ground truth count (annotate with "GT (n=N)")
- Annotate each bar with its numeric count
- Same color scheme as Illustration 1
- Concise x-axis label: "Predicted cell count" (or "nucleus count", etc.)
- No y-axis tick labels (shared with Illustration 1 via proximity in the paper)
- Panel label: bold "c" at top-left

---

**comparison_summary.json schema**:

```json
{
  "task": "nuclei segmentation",
  "dataset": "Human Protein Atlas field 3235, 512x512 center crop",
  "input_channel": "405nm (DAPI/nucleus)",
  "keywords": ["nuclei", "fluorescence"],
  "candidates": ["model-id-1", "model-id-2"],
  "excluded": {"model-id-x": "domain mismatch: trained on H&E brightfield"},
  "metrics": {
    "model-id-1": {"f1": 0.909, "iou": 0.85, "tp": 10, "n_pred": 10, "n_gt": 12}
  },
  "failed_models": {"model-id-3": "shape mismatch error message"},
  "best_model": "model-id-1",
  "ground_truth_n_cells": 12,
  "notes": {"model-id-2": "HPA model: requires 3 channels"}
}
```

## Validation / testing workflow

```text
- [ ] Step 1: bioengine runner validate rdf.yaml — format compliance check
- [ ] Step 2: bioengine runner test <model-id> — runs official BioImage.IO test suite (may be cached)
- [ ] Step 3: Review output — check status (passed/failed) and details
```

```bash
bioengine runner validate ./my-model/rdf.yaml
bioengine runner test ambitious-ant
bioengine runner test ambitious-ant --skip-cache  # force re-download and re-test
```

## Validation loop (quality-critical runs)

Run inference → if failure, inspect error and RDF constraints (`bioengine runner info`) → adjust dimensions or normalization → rerun → repeat until success or clear incompatibility → record what changed and why.

## Output interpretation guide

Model outputs are **raw tensors** — not ready-to-use instance labels. Always read the RDF to understand channel semantics before interpreting results.

### Common output formats

| Model type | Output shape | Interpretation |
|---|---|---|
| UNet nucleus (e.g. affable-shark) | `(1, 2, H, W)` | ch0=foreground prob, ch1=boundary prob. Use `foreground - boundary` → watershed |
| UNet softmax (e.g. conscientious-seashell) | `(1, 3, H, W)` | Softmax probabilities across 3 classes (background/boundary/nucleus). Sum=1 per pixel. Identify nucleus channel by lowest mean. |
| StarDist | `(1, H, W, 33)` | ch0=object probability, ch1–32=star polygon radii. Requires NMS postprocessing — do NOT threshold naively. |
| Restoration (e.g. dazzling-spider) | `(1, 1, H, W)` | Direct output; output key is `"prediction"` not `"output0"` |

### HPA multi-channel models (conscientious-seashell, loyal-parrot)

These models expect **3 channels** (not just nucleus staining):
- **Channel 0**: 405nm (DAPI/nucleus staining, normalized to [0,1])
- **Channel 1**: 488nm (ER or microtubule channel, normalized to [0,1])  
- **Channel 2**: 638nm (protein of interest, normalized to [0,1])

Input shape: `(1, 3, 512, 512)` float32. Normalize each channel independently with percentile normalization (p1=1, p99=99.8). Using only the nucleus channel gives degraded results — the model was trained on all 3.

### StarDist postprocessing

StarDist output ch0 is the **object probability map** — values are typically very sparse (>99% of pixels near 0 even for images with many nuclei). Apply NMS using the 32 radii channels:

```python
import numpy as np
from skimage.feature import peak_local_max
from skimage.draw import polygon

def stardist_nms(prob, radii, prob_thresh=0.5, nms_thresh=0.3, min_radius=5):
    H, W = prob.shape
    n_rays = radii.shape[-1]
    angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    lm = peak_local_max(prob, min_distance=5, threshold_abs=prob_thresh)
    if len(lm) == 0:
        return np.zeros((H, W), dtype=int)
    # Sort by score, apply greedy NMS
    order = np.argsort(-prob[lm[:, 0], lm[:, 1]])
    lm = lm[order]
    labels = np.zeros((H, W), dtype=int)
    kept = []; inst_id = 0
    for y, x in lm:
        r = np.abs(radii[y, x])
        if r.mean() < min_radius:
            continue
        poly_y = np.clip(y + r * np.sin(angles), 0, H-1)
        poly_x = np.clip(x + r * np.cos(angles), 0, W-1)
        rr, cc = polygon(poly_y, poly_x, shape=(H, W))
        if len(rr) < 20:
            continue
        cur = np.zeros((H, W), dtype=bool); cur[rr, cc] = True
        if any((np.logical_and(cur, p).sum() / (np.logical_or(cur, p).sum()+1e-6)) > nms_thresh for p in kept):
            continue
        inst_id += 1; labels[rr, cc] = inst_id; kept.append(cur)
    return labels
```

### Resolution sensitivity

StarDist models tagged `whole-slide-imaging` are trained at specific magnifications. If the detected cell radii (mean of ch1–ch32 at top-probability pixels) are much larger or smaller than actual cell sizes in your image, the model is mismatched to your resolution. Switch to a model trained at your acquisition settings.

### Domain mismatch — brightfield vs fluorescence

**Critical**: Some models tagged `nuclei` or `segmentation` are trained on H&E brightfield images, not fluorescence. In brightfield/H&E, nuclei appear **dark on a bright background**; in fluorescence, nuclei are **bright on a dark background**. Running a brightfield-trained model on fluorescence data (or vice versa) produces detections with zero overlap with actual nuclei — F1=0.

**Prevention (do this first)**: Call `get_model_documentation(model_id)` before running any model. The README describes the training domain explicitly. Exclude models with incompatible domains before running inference — this saves time and prevents misleading results.

**Fallback detection** (if documentation is None or ambiguous):
1. Check the model's test input: `inputs[0].test_tensor.source` — download it and inspect visually
2. If test input has high mean pixel values (e.g. >80 for 8-bit) across all channels → brightfield
3. If test input has low mean with bright spots → fluorescence
4. Check RDF `tags` for `"brightfield"`, `"histopathology"`, `"H&E"` → skip for fluorescence tasks
5. After inference: if probability at ground-truth nucleus centroids is all ≈ 0.000 → domain is wrong

**Input scale is NOT the cause**: StarDist models (`chatty-frog`, `fearless-crab`) internally normalize their input. Sending [0,1] vs [0,255] values produces identical outputs (correlation=1.000). The domain difference (fluorescence vs brightfield) is the actual problem.

**chatty-frog** (`whole-slide-imaging`, StarDist RGB): trained on brightfield/H&E. Do NOT use for fluorescence microscopy. Use **fearless-crab** instead for fluorescence single-channel nuclei.

### Search keyword guide for HPA images

For Human Protein Atlas fluorescence images, use these keywords in order of reliability:

| Task | Best keywords |
|---|---|
| Nucleus segmentation | `["nuclei", "segmentation"]` |
| Cell segmentation | `["cell", "segmentation", "fluorescence"]` |
| HPA-specific | `["HPA"]` or search conscientious-seashell / loyal-parrot directly |
| Denoising | `["restoration", "2D"]` (not `"denoising"` — may return 0 results) |

`conscientious-seashell` = HPA nucleus model (3-channel). `loyal-parrot` = HPA cell body model (3-channel).

## Deploying and updating the model-runner app

The `model-runner` service is deployed on the BioEngine worker (`bioimage-io/bioengine-worker`) and runs in the `bioimage-io` workspace. It requires a `HYPHA_TOKEN` that has write access to that workspace. Pass it on first deployment:

```bash
export BIOENGINE_WORKER_SERVICE_ID=bioimage-io/bioengine-worker
bioengine apps deploy ./bioengine_apps/model-runner/ \
  --env _HYPHA_TOKEN=<bioimage-io-scoped-token>
```

**When updating an existing deployment**, the new version automatically inherits all env vars (including `HYPHA_TOKEN`) and all init args/kwargs from the previous app — do **not** pass `--env` again unless intentionally rotating a secret:

```bash
bioengine apps upload ./bioengine_apps/model-runner/
bioengine apps run bioimage-io/model-runner --app-id <existing-app-id>
# HYPHA_TOKEN and all other env vars are carried over automatically
```

## References

- Full API endpoint docs and examples: [references/api_reference.md](references/api_reference.md)
- RDF format spec (0.4.x vs 0.5.x axes, output keys): [references/rdf_format.md](references/rdf_format.md)
- Task/modality keyword presets: [assets/search_keywords.yaml](assets/search_keywords.yaml)
- CLI source and advanced usage: [references/cli_reference.md](references/cli_reference.md)

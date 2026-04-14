---
name: bioengine-model-runner
description: Searches, validates, tests, and runs inference on BioImage.IO models via the BioEngine model-runner service. Use when the user wants to run microscopy image analysis, find segmentation/denoising/restoration/detection models, compare multiple models against ground truth, validate a model RDF, or test BioImage.IO model compliance.
license: MIT
metadata:
  service-id: bioimage-io/model-runner
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
- [ ] Step 3: Run inference — bioengine runner infer <model-id> --input image.tif --output result.npy
- [ ] Step 4: Validate output — load result.npy with numpy, check shape and values
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
- [ ] Step 3: Run all models on the same input — bioengine runner compare
- [ ] Step 4: Score models — utils.evaluate_segmentation() for IoU/Dice
- [ ] Step 5: Save all artifacts to comparison_results/
- [ ] Step 6: Generate montage SVG and barplot SVG
- [ ] Step 7: Write comparison_summary.json
- [ ] Step 8: Generate HTML report
```

```bash
# Run multiple models and save all outputs
bioengine runner compare affable-shark ambitious-ant chatty-frog \
  --input cells.tif \
  --output-dir comparison_results/
```

**Artifact layout** (always save here — create folder if missing):

```
comparison_results/
├── {model_id}_{output_key}.npy        # raw prediction per model
├── model_comparison_montage.svg       # row 1: input+GT centered; rows below: predictions
├── model_comparison_barplot.svg       # grouped metrics bar chart per model
└── comparison_summary.json
```

**Montage layout rule**: input and ground truth in the middle of row 1 (e.g., 4 columns → skip col 1, input col 2, GT col 3, skip col 4). Max 4 columns unless user specifies otherwise. No metric text in image titles.

**Generate HTML report**:

```bash
python scripts/generate_report.py \
  --summary comparison_results/comparison_summary.json \
  --montage comparison_results/model_comparison_montage.svg \
  --barplot comparison_results/model_comparison_barplot.svg
```

**comparison_summary.json schema**:

```json
{
  "task": "nuclei segmentation",
  "keywords": ["nuclei", "fluorescence"],
  "candidates": ["model-id-1", "model-id-2"],
  "metrics": {"model-id-1": {"iou": 0.82, "dice": 0.87}},
  "failed_models": {"model-id-3": "shape mismatch"},
  "best_model": "model-id-1"
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

**How to detect domain mismatch**:
1. Check the model's test input: `inputs[0].test_tensor.source` — download it and inspect visually
2. If test input has high mean pixel values (e.g. >80 for 8-bit) across all channels → brightfield
3. If test input has low mean with bright spots → fluorescence
4. After inference: check probability at ground-truth nucleus centroids — if all ≈ 0.000, domain is wrong

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

The `model-runner` service runs in the `bioimage-io` workspace and requires a `HYPHA_TOKEN` that has write access to that workspace. Pass it on first deployment:

```bash
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

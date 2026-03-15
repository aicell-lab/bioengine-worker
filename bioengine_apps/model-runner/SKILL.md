---
name: model-runner
description: Runs and compares BioImage.IO models through the BioEngine model-runner service. Use when users need microscopy image inference, model search, model screening, RDF inspection, or model validation/testing.
license: MIT
compatibility: Requires network access to hypha.aicell.io and Python packages including hypha-rpc, httpx, numpy, and matplotlib.
metadata:
  service-id: bioimage-io/model-runner
  http-base-url: https://hypha.aicell.io/bioimage-io/services/model-runner
  mcp-server: https://hypha.aicell.io/bioimage-io/mcp/model-runner
---

# BioImage.IO Model Runner

## Use this skill when

- The user wants to run a known BioImage.IO model on an image.
- The user wants to find candidate models for a task (for example nuclei segmentation).
- The user wants to compare multiple models against ground truth.
- The user wants to validate a model RDF or run BioImage.IO tests.

## Default operating mode

- Prefer Hypha RPC for reliability and direct numpy transfer.
- Use HTTP endpoints when RPC is unavailable.
- For HTTP or MCP inference, always set `return_download_url=true`.
- Use `.npy` uploads by default for lossless dtype/shape handling.

## Primary workflow

Copy this checklist and mark progress:

```text
Model Runner Progress:
- [ ] Step 1: Clarify task and expected output
- [ ] Step 2: Choose candidate model(s)
- [ ] Step 3: Prepare and upload input
- [ ] Step 4: Run inference
- [ ] Step 5: Validate outputs and report
```

### Step 1: Clarify task and output

- Confirm task type: segmentation, denoising, restoration, detection, super-resolution.
- Confirm expected output format: mask, probability map, restored image, labels, metrics.

### Step 2: Choose model(s)

- If model ID is known: retrieve RDF first.
- If unknown: call `search_models` with targeted keywords from [assets/search_keywords.yaml](assets/search_keywords.yaml).
- Default to models that pass checks (`ignore_checks=false`).

### Step 3: Prepare and upload input

- Use the utility script provided in `scripts/utils.py`.
- Use `utils.prepare_image_for_model(image, axes_str)` to align input dimensions to the required target array structure before passing the dataset.
- Cast image arrays to `float32` automatically and handle any min/max scaling logic using `utils.normalize_image(image)`.

### Step 4: Run inference

- Always provide `model_id` and `inputs`.
- Use the provided `utils.infer_http(model_id, input_tensor)` to circumvent complex async HTTP logic. It immediately returns the inferred ndarray.
- On shape errors, adapt dimensions and retry before discarding a model.

### Step 5: Validate outputs and model screening

- Confirm output key(s) from RDF (`outputs[].name` in 0.4.x, `outputs[].id` in 0.5.x).
- Verify output shape/channel interpretation before scoring.
- When performing a model screening/comparison task on multiple models, **you should construct a python script to run inference dynamically.**
- However, do not write networking or boilerplate tensor logic from scratch. Import and rely on functions inside `scripts/utils.py` (e.g. `infer_http`, `get_model_rdf`, `get_input_axes_info`, and `evaluate_segmentation`).
  - Choose appropriate metrics for the task by relying on `utils.evaluate_segmentation` and adapt if doing instance vs semantic segmentation.
- All model outputs, generated files, plots, and summaries must be saved to a `comparison_results` folder. Create this directory if it does not exist.
- Save the raw output prediction of each model inference as a `.npy` file inside the `comparison_results` folder (e.g. `comparison_results/model_id_output.npy`).
- Use Python libraries (e.g., matplotlib) to generate and save a visual montage comparing the outputs of the multiple models. **Always have the input and ground truth in the middle of the first row (e.g. for 4 columns, skip 1, put input on 2, ground truth on 3, skip 4) and all model predictions in rows below that. Try to create complete rows. The max number of columns is 4 if not otherwise defined by the user. Do not add the metrics in the montage titles.** Save the montage only as `.svg` inside the `comparison_results` folder (i.e. `comparison_results/model_comparison_montage.svg`).
- Also generate a single grouped comparison barplot showing all chosen metrics side by side for each model, and save it only as `.svg` inside the `comparison_results` folder (`comparison_results/model_comparison_barplot.svg`).
- Output your summary results to a `comparison_results/comparison_summary.json` file. Ensure this file contains: `task` (str), `keywords` (list), `metrics` (a dict where keys are model ids and values are dict of metric values like {"iou": 0.5}), `failed_models` (dict of errors if any), `candidates` (list of model ids evaluated), and the best model properties based on metrics.
- Finally, utilize the available UI generation script by referencing its absolute path (e.g. `python scripts/generate_report.py --summary comparison_results/comparison_summary.json --montage comparison_results/model_comparison_montage.svg --barplot comparison_results/model_comparison_barplot.svg`) to generate a standalone HTML report.
- Wrap per-model inference in error handling and continue screening runs.

## Performing API Verification or Single Model Inference

- If you need to perform an API sanity check, or run inference for a single model:
  - You must write the necessary scripts or Python code to execute these endpoints yourself.
  - Refer to the detailed usages directly inside your code. No pre-built Python scripts exist wrapper to do this; investigate the documented [API Reference](references/api_reference.md).

## Validation loop

Use this loop for quality-critical runs:

1. Run inference with current preprocessing.
2. If failure: inspect error and RDF constraints.
3. Adjust dimensions/normalization and rerun.
4. Repeat until success or clear incompatibility.
5. Record what was changed and why.

## References

- API details and endpoint examples: [references/api_reference.md](references/api_reference.md)
- RDF format differences and parsing rules: [references/rdf_format.md](references/rdf_format.md)
- Keyword presets by task/modality: [assets/search_keywords.yaml](assets/search_keywords.yaml)

## Notes for maintainers

- Keep this file short and task-oriented; move detailed examples to `references/` or `scripts/`.
- Use forward-slash paths only.
- If this skill directory is renamed, update frontmatter `name` accordingly.
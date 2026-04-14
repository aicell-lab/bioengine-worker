"""
bioengine runner — search, inspect, test, and run BioImage.IO models.

All commands connect to the BioEngine model-runner service at
https://hypha.aicell.io (or BIOENGINE_SERVER_URL). No authentication is
required for any command in this group; the service is public.

API signatures verified against:
  bioengine-worker/bioengine_apps/model-runner/entry_deployment.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from bioengine_cli.utils import (
    async_command,
    connect_model_runner,
    download_array,
    error_exit,
    get_server_url,
    get_token,
    print_json,
    print_table,
    read_image,
    upload_array,
    write_image,
)


@click.group("runner")
def runner_group():
    """
    Search and run BioImage.IO deep learning models.

    Models are executed on BioEngine workers (GPU-enabled servers). No local
    GPU is required — computation runs remotely. Results are returned as image
    files or numpy arrays.

    \b
    Typical workflow:
      bioengine runner search --keywords nuclei segmentation
      bioengine runner info <model-id>
      bioengine runner infer <model-id> --input image.tif --output result.npy
      bioengine runner test <model-id>

    The model-runner service is public; no login is required.
    """


# ── search ───────────────────────────────────────────────────────────────────

@runner_group.command("search")
@click.option(
    "--keywords", "-k",
    multiple=True,
    metavar="WORD",
    help=(
        "Keywords to filter models. Repeat the flag for multiple keywords: "
        "--keywords nuclei --keywords segmentation. "
        "Matches against model name, description, and tags. "
        "If omitted, returns all runnable models."
    ),
)
@click.option(
    "--limit", "-n",
    default=10,
    show_default=True,
    metavar="N",
    help="Maximum number of results to return (1–100).",
)
@click.option(
    "--ignore-checks",
    is_flag=True,
    default=False,
    help=(
        "Include models that have NOT passed BioEngine inference tests. "
        "By default only validated models are shown."
    ),
)
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON instead of a table.")
@click.option("--server-url", envvar="BIOENGINE_SERVER_URL", default=None, hidden=True)
def search(keywords, limit, ignore_checks, as_json, server_url):
    """
    Search for runnable models in the BioImage.IO collection.

    Searches by keyword across model names, descriptions, and tags. Returns
    model IDs and short descriptions for all matching models that have been
    validated on BioEngine workers.

    \b
    Examples:
      bioengine runner search --keywords nuclei segmentation --limit 5
      bioengine runner search --keywords denoising fluorescence
      bioengine runner search --limit 20 --json
    """
    server_url = get_server_url(server_url)

    async def _run():
        try:
            service = await connect_model_runner(server_url)
            kw_list = list(keywords) if keywords else None
            results = await service.search_models(
                keywords=kw_list,
                limit=limit,
                ignore_checks=ignore_checks,
            )
        except Exception as exc:
            error_exit(f"Search failed: {exc}", "Check your network connection and server URL.")

        if not results:
            click.echo("No models found matching your query.")
            return

        if as_json:
            print_json(results)
        else:
            rows = [(m.get("model_id", "?"), m.get("description", "")) for m in results]
            click.echo(f"\nFound {len(results)} model(s):\n")
            print_table(rows, ["MODEL ID", "DESCRIPTION"])
            click.echo(
                f"\nRun `bioengine runner info <model-id>` to see input/output specs."
            )

    asyncio.run(_run())


# ── info ─────────────────────────────────────────────────────────────────────

@runner_group.command("info")
@click.argument("model_id")
@click.option(
    "--stage",
    is_flag=True,
    default=False,
    help="Get the staged (draft) version of the model instead of the published version.",
)
@click.option("--json", "as_json", is_flag=True, help="Output full RDF as JSON.")
@click.option("--server-url", envvar="BIOENGINE_SERVER_URL", default=None, hidden=True)
def info(model_id, stage, as_json, server_url):
    """
    Show metadata and input/output specification for a model.

    MODEL_ID is the BioImage.IO model identifier (e.g. 'ambitious-ant').
    Displays model name, description, tags, input tensor axes and shapes,
    output tensor axes, and available weight formats.

    \b
    Examples:
      bioengine runner info ambitious-ant
      bioengine runner info affable-shark --json
    """
    server_url = get_server_url(server_url)

    async def _run():
        try:
            service = await connect_model_runner(server_url)
            rdf = await service.get_model_rdf(model_id=model_id, stage=stage)
        except Exception as exc:
            error_exit(
                f"Could not fetch model '{model_id}': {exc}",
                "Check that the model ID is correct. "
                "Use `bioengine runner search` to find valid IDs.",
            )

        if as_json:
            print_json(rdf)
            return

        # Human-friendly summary
        click.echo(f"\n{'='*60}")
        click.echo(f"Model:       {rdf.get('name', model_id)}")
        click.echo(f"ID:          {model_id}")
        click.echo(f"Description: {rdf.get('description', 'N/A')}")
        click.echo(f"Format:      {rdf.get('format_version', 'N/A')}")

        tags = rdf.get("tags", [])
        if tags:
            click.echo(f"Tags:        {', '.join(tags[:10])}")

        # Weights
        weights = rdf.get("weights", {})
        if weights:
            click.echo(f"Weights:     {', '.join(weights.keys())}")

        # Inputs
        inputs = rdf.get("inputs", [])
        if inputs:
            click.echo(f"\nInputs ({len(inputs)}):")
            for inp in inputs:
                inp_id = inp.get("id") or inp.get("name", "?")
                axes = inp.get("axes", [])
                axes_str = _format_axes(axes)
                dtype = inp.get("data", {}).get("type", "?") if isinstance(inp.get("data"), dict) else inp.get("data_type", "?")
                click.echo(f"  [{inp_id}]  axes={axes_str}  dtype={dtype}")

        # Outputs
        outputs = rdf.get("outputs", [])
        if outputs:
            click.echo(f"\nOutputs ({len(outputs)}):")
            for out in outputs:
                out_id = out.get("id") or out.get("name", "?")
                axes = out.get("axes", [])
                axes_str = _format_axes(axes)
                dtype = out.get("data", {}).get("type", "?") if isinstance(out.get("data"), dict) else out.get("data_type", "?")
                click.echo(f"  [{out_id}]  axes={axes_str}  dtype={dtype}")

        click.echo(f"{'='*60}\n")
        click.echo(f"Run inference:  bioengine runner infer {model_id} --input <image> --output result.npy")
        click.echo(f"Run tests:      bioengine runner test {model_id}")

    asyncio.run(_run())


def _format_axes(axes) -> str:
    """Format RDF axes (0.4.x list of dicts or 0.5.x list of dicts)."""
    if not axes:
        return "?"
    parts = []
    for ax in axes:
        if isinstance(ax, dict):
            ax_type = ax.get("type", ax.get("id", "?"))
            parts.append(ax_type[0] if ax_type else "?")
        elif isinstance(ax, str):
            parts.append(ax)
    return "".join(parts)


# ── test ─────────────────────────────────────────────────────────────────────

@runner_group.command("test")
@click.argument("model_id")
@click.option(
    "--skip-cache",
    is_flag=True,
    default=False,
    help=(
        "Force a complete model re-download and re-run of tests, "
        "bypassing cached results. Useful if the model was recently updated."
    ),
)
@click.option(
    "--stage",
    is_flag=True,
    default=False,
    help="Test the staged (draft) version of the model instead of the published version.",
)
@click.option(
    "--extra-packages",
    multiple=True,
    metavar="PKG",
    help=(
        "Additional pip packages to install in the test environment "
        "(e.g. --extra-packages scipy>=1.7.0). Repeat for multiple packages."
    ),
)
@click.option("--json", "as_json", is_flag=True, help="Output test report as JSON.")
@click.option("--server-url", envvar="BIOENGINE_SERVER_URL", default=None, hidden=True)
def test(model_id, skip_cache, stage, extra_packages, as_json, server_url):
    """
    Run the official BioImage.IO test suite on a model.

    Tests download the model, run inference on its embedded test inputs, and
    verify outputs match the expected values. Results are cached; subsequent
    calls are fast unless --skip-cache is passed.

    The test may take several minutes on first run (model download + inference).

    \b
    Examples:
      bioengine runner test ambitious-ant
      bioengine runner test ambitious-ant --skip-cache
      bioengine runner test my-draft-model --stage
    """
    server_url = get_server_url(server_url)

    async def _run():
        click.echo(f"Testing model '{model_id}'... (may take a few minutes on first run)")
        try:
            service = await connect_model_runner(server_url)
            extra = list(extra_packages) if extra_packages else None
            report = await service.test(
                model_id=model_id,
                stage=stage,
                additional_requirements=extra,
                skip_cache=skip_cache,
            )
        except Exception as exc:
            error_exit(
                f"Test failed for model '{model_id}': {exc}",
                "If this is a network error, retry. "
                "If the model is unsupported, try --ignore-checks in search.",
            )

        if as_json:
            print_json(report)
            return

        status = report.get("status", "unknown")
        icon = "PASSED" if status == "passed" else "FAILED"
        click.echo(f"\n[{icon}] Model '{model_id}' — status: {status}")

        details = report.get("details", "")
        if details:
            click.echo(f"\n{details}")

        if status != "passed":
            sys.exit(1)

    asyncio.run(_run())


# ── infer ─────────────────────────────────────────────────────────────────────

@runner_group.command("infer")
@click.argument("model_id")
@click.option(
    "--input", "-i", "input_path",
    required=True,
    metavar="PATH",
    help=(
        "Input image file. Supported formats: .npy (numpy, lossless), "
        ".tif/.tiff (TIFF), .png. "
        "The array is uploaded to BioEngine temporary storage and passed to the model."
    ),
)
@click.option(
    "--output", "-o", "output_path",
    default="result.npy",
    show_default=True,
    metavar="PATH",
    help=(
        "Output file path. Extension determines format: "
        ".npy (default, lossless), .tif/.tiff, .png. "
        "For multi-output models, additional outputs are saved as <name>_<key>.npy."
    ),
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"], case_sensitive=False),
    default=None,
    help=(
        "Computation device. 'cuda' for GPU, 'cpu' for CPU-only. "
        "Default: auto-select based on model and hardware availability."
    ),
)
@click.option(
    "--weights-format",
    type=click.Choice(
        ["pytorch_state_dict", "torchscript", "onnx", "tensorflow_saved_model"],
        case_sensitive=False,
    ),
    default=None,
    help=(
        "Preferred model weight format. "
        "Default: automatically select the best available format."
    ),
)
@click.option(
    "--skip-cache",
    is_flag=True,
    default=False,
    help="Force re-download of the model before inference.",
)
@click.option(
    "--blocksize",
    "blocksize",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Override the tiling block size for memory management. "
        "Larger values may be faster but use more GPU/CPU memory."
    ),
)
@click.option("--json", "as_json", is_flag=True, help="Print output metadata as JSON instead of saving.")
@click.option("--server-url", envvar="BIOENGINE_SERVER_URL", default=None, hidden=True)
def infer(model_id, input_path, output_path, device, weights_format, skip_cache, blocksize, as_json, server_url):
    """
    Run inference on a BioImage.IO model with a local image file.

    Reads the input image, uploads it to BioEngine temporary storage (1-hour
    TTL), runs the model remotely, downloads the result, and saves it locally.

    No local GPU is needed — all computation runs on BioEngine workers.

    \b
    Examples:
      bioengine runner infer ambitious-ant --input cells.tif --output mask.npy
      bioengine runner infer affable-shark --input image.tif --output result.tif --device cuda
      bioengine runner infer my-model --input data.npy --output out.npy --skip-cache
    """
    server_url = get_server_url(server_url)

    async def _run():
        # 1. Read input image
        try:
            array = read_image(input_path)
        except click.ClickException:
            raise
        except Exception as exc:
            error_exit(f"Cannot read input file '{input_path}': {exc}")

        click.echo(
            f"Input: {Path(input_path).name}  shape={array.shape}  dtype={array.dtype}"
        )

        # 2. Connect and upload
        try:
            service = await connect_model_runner(server_url)
        except Exception as exc:
            error_exit(f"Cannot connect to BioEngine: {exc}")

        click.echo("Uploading input to BioEngine...")
        try:
            file_path = await upload_array(service, array)
        except Exception as exc:
            error_exit(f"Upload failed: {exc}")

        # 3. Run inference
        click.echo(f"Running inference with model '{model_id}'... (may take a minute)")
        try:
            result = await service.infer(
                model_id=model_id,
                inputs=file_path,
                weights_format=weights_format,
                device=device,
                default_blocksize_parameter=blocksize,
                skip_cache=skip_cache,
                return_download_url=True,  # always use download URLs (works via HTTP)
            )
        except Exception as exc:
            error_exit(
                f"Inference failed: {exc}",
                "Check that the input shape matches the model's expected axes. "
                "Run `bioengine runner info {model_id}` to see the input spec.".format(
                    model_id=model_id
                ),
            )

        # 4. Download and save results
        if as_json:
            # Just print the download URLs and shapes as JSON
            meta = {}
            for key, url in result.items():
                meta[key] = {"download_url": url}
            print_json(meta)
            return

        output_p = Path(output_path)
        saved = []
        for idx, (key, url) in enumerate(result.items()):
            click.echo(f"Downloading output '{key}'...")
            try:
                out_array = await download_array(url)
            except Exception as exc:
                error_exit(f"Failed to download output '{key}': {exc}")

            # Determine save path: first output → output_path, rest → suffixed
            if idx == 0:
                save_path = str(output_p)
            else:
                save_path = str(output_p.parent / f"{output_p.stem}_{key}{output_p.suffix}")

            write_image(out_array, save_path)
            saved.append((key, save_path, out_array.shape, str(out_array.dtype)))
            click.echo(
                f"  Saved: {save_path}  shape={out_array.shape}  dtype={out_array.dtype}"
            )

        click.echo(f"\nDone. {len(saved)} output(s) saved.")

    asyncio.run(_run())


# ── validate ──────────────────────────────────────────────────────────────────

@runner_group.command("validate")
@click.argument("rdf_path")
@click.option("--json", "as_json", is_flag=True, help="Output validation result as JSON.")
@click.option("--server-url", envvar="BIOENGINE_SERVER_URL", default=None, hidden=True)
def validate(rdf_path, as_json, server_url):
    """
    Validate a BioImage.IO RDF (rdf.yaml) file against the specification.

    RDF_PATH is the local path to an rdf.yaml file. Useful when developing a
    new model and checking that its metadata is correctly formatted before
    uploading to BioImage.IO.

    \b
    Examples:
      bioengine runner validate ./my-model/rdf.yaml
      bioengine runner validate rdf.yaml --json
    """
    server_url = get_server_url(server_url)

    import yaml

    async def _run():
        try:
            with open(rdf_path) as f:
                rdf_dict = yaml.safe_load(f)
        except Exception as exc:
            error_exit(f"Cannot read RDF file '{rdf_path}': {exc}")

        try:
            service = await connect_model_runner(server_url)
            result = await service.validate(rdf_dict=rdf_dict)
        except Exception as exc:
            error_exit(f"Validation call failed: {exc}")

        if as_json:
            print_json(result)
            return

        success = result.get("success", False)
        icon = "VALID" if success else "INVALID"
        click.echo(f"[{icon}] {rdf_path}")
        details = result.get("details", "")
        if details:
            click.echo(f"\n{details}")

        if not success:
            sys.exit(1)

    asyncio.run(_run())


# ── compare ───────────────────────────────────────────────────────────────────

@runner_group.command("compare")
@click.argument("model_ids", nargs=-1, required=True, metavar="MODEL_ID...")
@click.option(
    "--input", "-i", "input_path",
    required=True,
    metavar="PATH",
    help="Input image file (.npy, .tif, .png) shared across all models.",
)
@click.option(
    "--output-dir",
    default="comparison_results",
    show_default=True,
    metavar="DIR",
    help="Directory where per-model outputs are saved.",
)
@click.option("--device", type=click.Choice(["cuda", "cpu"], case_sensitive=False), default=None)
@click.option("--json", "as_json", is_flag=True, help="Output summary as JSON.")
@click.option("--server-url", envvar="BIOENGINE_SERVER_URL", default=None, hidden=True)
def compare(model_ids, input_path, output_dir, device, as_json, server_url):
    """
    Run the same input image through multiple models and save all outputs.

    Useful for evaluating which model works best for your data. Each model's
    output is saved to output-dir/<model-id>_output.npy. A summary JSON is
    written to output-dir/comparison_summary.json.

    \b
    Examples:
      bioengine runner compare ambitious-ant affable-shark --input cells.tif
      bioengine runner compare model-a model-b model-c --input image.npy --output-dir results/
    """
    server_url = get_server_url(server_url)
    import asyncio as _asyncio

    async def _run():
        array = read_image(input_path)
        click.echo(f"Input: shape={array.shape}  dtype={array.dtype}")

        service = await connect_model_runner(server_url)
        click.echo("Uploading input...")
        file_path = await upload_array(service, array)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "input": input_path,
            "candidates": list(model_ids),
            "results": {},
            "failed_models": {},
        }

        for model_id in model_ids:
            click.echo(f"\nRunning '{model_id}'...")
            try:
                result = await service.infer(
                    model_id=model_id,
                    inputs=file_path,
                    device=device,
                    return_download_url=True,
                )
                saved_paths = []
                for key, url in result.items():
                    out_array = await download_array(url)
                    save_path = str(out_dir / f"{model_id}_{key}.npy")
                    write_image(out_array, save_path)
                    saved_paths.append(save_path)
                    click.echo(f"  Saved: {save_path}  shape={out_array.shape}")
                summary["results"][model_id] = {"outputs": saved_paths}
            except Exception as exc:
                click.echo(f"  FAILED: {exc}", err=True)
                summary["failed_models"][model_id] = str(exc)

        # Write summary
        summary_path = out_dir / "comparison_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        click.echo(f"\nSummary saved: {summary_path}")

        if as_json:
            print_json(summary)

    asyncio.run(_run())

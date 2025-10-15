"""Cellpose finetuning service using AsyncHyphaArtifact and simplified flows."""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pydantic import Field
from ray import serve

from hypha_rpc import login
from hypha_rpc.utils.schema import schema_function

if TYPE_CHECKING:  # typing-only import
    from cellpose.models import CellposeModel
    from hypha_artifact import AsyncHyphaArtifact

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Avoid typing issues with Ray Serve decorators in absence of full stubs
serve = cast("Any", serve)

GB = 1024**3
NDIM_3D_THRESHOLD = 2


def _ensure_model_dir() -> Path:
    d = Path.cwd() / "model"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _channels(train_channel: str, second_train_channel: str) -> list[int]:
    lut = {"Grayscale": 0, "Red": 1, "Green": 2, "Blue": 3}
    return [lut[train_channel], lut[second_train_channel]]


def _artifact_cache_dir(artifact_id: str) -> Path:
    """Return persistent cache dir for an artifact, creating it if needed.

    Example: ./data_cache/workspace__alias
    """
    safe = artifact_id.replace("/", "__")
    cache_dir = Path.cwd() / "data_cache" / safe
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _encode_ndarray(arr: np.ndarray) -> dict[str, Any]:
    r"""Encode a numpy array into a JSON-serializable dict (NPY-in-base64).

    Reconstruction examples:
    - Python:
        data = base64.b64decode(obj["data"])\n
        arr = np.load(
            io.BytesIO(data),
            allow_pickle=False,
        )
    - JS/TS: decode base64 to ArrayBuffer and parse with an NPY reader
      (or handle server-side and send a PNG if visualization only is needed).
    """
    buf = io.BytesIO()
    # NPY stores dtype and shape
    np.save(buf, arr, allow_pickle=False)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "encoding": "npy_base64",
        "data": payload,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


async def _make_artifact_client(
    artifact_id: str,
    token: str,
    server_url: str,
) -> AsyncHyphaArtifact:
    from hypha_artifact import AsyncHyphaArtifact

    if "/" not in artifact_id:
        msg = "artifact_id must be of form 'workspace/alias'"
        raise ValueError(msg)
    workspace, _alias = artifact_id.split("/", 1)
    return AsyncHyphaArtifact(
        artifact_id=artifact_id,
        workspace=workspace,
        server_url=server_url,
        token=token,
    )


def _split_train_test(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    idx = np.arange(len(pairs))
    rng = np.random.default_rng()
    rng.shuffle(idx)
    tsize = max(1, int(len(idx) * train_ratio))
    tr, te = idx[:tsize], idx[tsize:]
    train_files = [pairs[i][0] for i in tr]
    train_labels = [pairs[i][1] for i in tr]
    test_files = [pairs[i][0] for i in te]
    test_labels = [pairs[i][1] for i in te]
    return train_files, train_labels, test_files, test_labels


def _train(
    save_dir: Path,
    model: CellposeModel,
    initial_model: str,
    train_files: list[Path],
    train_labels_files: list[Path],
    test_files: list[Path],
    test_labels_files: list[Path],
    channels: list[int],
    n_epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> Path:
    from cellpose import train

    new_model_path, _train_losses, _test_losses = train.train_seg(
        model.net,
        train_files=train_files,
        train_labels_files=train_labels_files,
        test_files=test_files,
        test_labels_files=test_labels_files,
        channels=channels,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        SGD=True,
        nimg_per_epoch=1,
        save_path=save_dir,
        model_name=f"finetuned_{initial_model}",
        min_train_masks=1,
    )
    return Path(new_model_path)


def _load_local_model(initial_model: str) -> CellposeModel:
    from cellpose import core, models

    return models.CellposeModel(
        gpu=core.use_gpu(),
        model_type=initial_model,
    )


def _load_model(identifier: str) -> CellposeModel:
    """Load a Cellpose model by builtin name or by local file path.

    If `identifier` points to an existing file, it is treated as a path to a
    finetuned model. Otherwise, it is treated as a builtin model name
    (e.g., "cyto3").
    """
    from cellpose import core, models

    use_gpu = core.use_gpu()
    p = Path(identifier)
    if p.exists():
        return models.CellposeModel(gpu=use_gpu, model_type=str(p))
    return models.CellposeModel(gpu=use_gpu, model_type=identifier)


@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 1,
        "memory": 4 * GB,
        "runtime_env": {
            "pip": [
                "cellpose==3.1.1.1",
                "numpy==1.26.4",
                "tifffile",
                "hypha-artifact==0.1.2",
                "hypha-rpc",
            ],
        },
    },
    max_ongoing_requests=3,
    max_queued_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 2,
        "target_num_ongoing_requests_per_replica": 0.8,
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        "downscale_delay_s": 300,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class CellposeFinetune:
    """Ray Serve deployment for finetuning and inference with Cellpose.

    Downloads per-timepoint TIFFs from a dataset artifact, finetunes a Cellpose
    model locally, and runs inference using the saved model stored in ./model.

    Based on cellpose 2.0 finetune notebook:
    https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb#scrollTo=Q7c7V4yEqDc_
    """

    def __init__(self) -> None:
        """Initialize the deployment and prepare local directories."""
        _ensure_model_dir()
        os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(Path.cwd() / "models"))
        # Provided by BioEngine runtime; annotated for type checkers
        self.bioengine_hypha_client: Any | None = None
        self.pretrained_models = [
            "cyto",
            "cyto3",
            "nuclei",
            "tissuenet_cp3",
            "livecell_cp3",
            "yeast_PhC_cp3",
            "yeast_BF_cp3",
            "bact_phase_cp3",
            "bact_fluor_cp3",
            "deepbacs_cp3",
        ]

    async def list_pretrained_models(self) -> list[str]:
        """Return available pretrained model identifiers."""
        return self.pretrained_models

    def _get_missing_paths(
        self,
        dests: list[Path],
        out_dir: Path,
    ) -> tuple[list[str], list[str]]:
        """Get lists of a set of remote & local paths that are missing locally."""
        missing_dests: list[Path] = [
            dest for dest in dests if not (dest.exists() and dest.stat().st_size > 0)
        ]
        for dest in missing_dests:
            dest.parent.mkdir(parents=True, exist_ok=True)

        remote_paths = [str(dest) for dest in missing_dests]
        local_paths = [str(out_dir / dest) for dest in missing_dests]

        return remote_paths, local_paths

    async def _download_pairs_from_artifact(
        self,
        artifact: AsyncHyphaArtifact,
        out_dir: Path,
    ) -> list[tuple[Path, Path]]:
        """Download dataset files in batches and return local (img, ann) pairs."""
        metadata_dir = out_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        await artifact.get(
            "metadata/",
            str(metadata_dir),
            recursive=True,
            on_error="ignore",
        )

        image_paths, annotation_paths = self._image_annotation_paths(out_dir)
        if not image_paths or not annotation_paths:
            msg = "No metadata/*.json found or no valid image/annotation pairs"
            raise FileNotFoundError(msg)

        missing_rpaths, missing_lpaths = self._get_missing_paths(
            image_paths + annotation_paths,
            out_dir,
        )

        if missing_rpaths:
            await artifact.get(missing_rpaths, missing_lpaths, on_error="ignore")

        return list(zip(image_paths, annotation_paths))

    def _image_annotation_paths(
        self,
        cache_root: Path,
    ) -> tuple[list[Path], list[Path]]:
        """Parse local metadata files to image/annotation Path pairs.

        Reads JSON files from cache_root/metadata and extracts
        uploaded_images/uploaded_annotations lists fields.
        Malformed JSON files are skipped with a warning.
        """
        import json

        all_image_paths: list[Path] = []
        all_annotation_paths: list[Path] = []
        for mf in (cache_root / "metadata").glob("*.json"):
            try:
                text = mf.read_text(encoding="utf-8")
                meta = json.loads(text)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("Skipping metadata file %s due to error: %s", mf, exc)
                continue

            meta_img_paths: list[Path] = [
                Path(item) for item in meta.get("uploaded_images", [])
            ]
            meta_ann_paths: list[Path] = [
                Path(item) for item in meta.get("uploaded_annotations", [])
            ]
            if (
                meta_img_paths
                and meta_ann_paths
                and len(meta_img_paths) == len(meta_ann_paths)
            ):
                all_image_paths.extend(meta_img_paths)
                all_annotation_paths.extend(meta_ann_paths)

        return all_image_paths, all_annotation_paths

    @schema_function(skip_self=True, arbitrary_types_allowed=True)
    async def train(
        self,
        data_artifact_id: str = Field(
            ...,
            description="Artifact id 'workspace/alias' containing dataset TIFFs",
            examples=["ri-scale/zarr-demo"],
        ),
        model: str = Field(
            "cyto3",
            description="Base Cellpose model to finetune",
            examples=["cyto", "cyto3", "nuclei"],
        ),
        train_channel: str = Field(
            "Grayscale",
            description="First channel identifier for Cellpose",
        ),
        second_train_channel: str = Field(
            "Grayscale",
            description="Second channel identifier for Cellpose",
        ),
        train_ratio: float = Field(
            0.8,
            ge=0.0,
            le=1.0,
            description="Fraction of pairs used for training (rest for testing)",
        ),
        n_epochs: int = Field(10, ge=1, description="Training epochs"),
        learning_rate: float = Field(
            1e-6,
            gt=0,
            description="Optimizer learning rate",
        ),
        weight_decay: float = Field(
            1e-4,
            ge=0,
            description="Optimizer weight decay",
        ),
        server_url: str = Field(
            "https://hypha.aicell.io",
            description="Hypha server URL",
        ),
    ) -> dict[str, Any]:
        """Train using images saved in artifact in per-timepoint TIFF layout."""
        if model not in self.pretrained_models:
            msg = f"Invalid base model: {model}"
            raise ValueError(msg)

        env_token = os.environ.get("HYPHA_TOKEN")
        token = env_token if env_token else await login({"server_url": server_url})

        artifact = await _make_artifact_client(
            data_artifact_id,
            token,
            server_url,
        )
        # Use persistent cache directory for dataset
        cache_dir = _artifact_cache_dir(data_artifact_id)
        pairs = await self._download_pairs_from_artifact(
            artifact,
            cache_dir,
        )
        train_files, train_labels, test_files, test_labels = _split_train_test(
            pairs,
            train_ratio,
        )

        channels = _channels(train_channel, second_train_channel)

        cp_model = _load_local_model(model)
        new_model_path = _train(
            save_dir=cache_dir,
            model=cp_model,
            initial_model=model,
            train_files=train_files,
            train_labels_files=train_labels,
            test_files=test_files,
            test_labels_files=test_labels,
            channels=channels,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        # Save to local model/ directory
        model_dir = _ensure_model_dir()
        final_path = model_dir / new_model_path.name
        final_path.write_bytes(new_model_path.read_bytes())

        return {"status": "success", "model_path": str(final_path)}

    @schema_function(skip_self=True, arbitrary_types_allowed=True)
    async def infer(
        self,
        artifact_id: str = Field(
            ...,
            description="Artifact id 'workspace/alias' containing source images",
        ),
        image_paths: list[str] = Field(
            ...,
            description="List of image paths within the artifact",
            min_length=1,
        ),
        model: str = Field(
            "cyto3",
            description=(
                "Model identifier: either a local finetuned model file path (*.pth) "
                "or the name of a builtin pretrained model (e.g., 'cyto3')."
            ),
        ),
        train_channel: str = Field(
            "Grayscale",
            description="First channel identifier for Cellpose",
        ),
        second_train_channel: str = Field(
            "Grayscale",
            description="Second channel identifier for Cellpose",
        ),
        diameter: float | None = Field(None, description="Diameter for Cellpose", ge=0),
        server_url: str = Field(
            "https://hypha.aicell.io",
            description="Hypha server URL",
        ),
    ) -> dict[str, Any]:
        """Run inference and return encoded mask arrays instead of file paths.

        Returns a dict with a `predictions` list, where each item contains the
        original artifact path and an encoded array payload describing the mask.
        """
        from tifffile import imread

        env_token = os.environ.get("HYPHA_TOKEN")
        token = env_token if env_token else await login({"server_url": server_url})

        # Resolve model identifier (local path or builtin name)
        model_identifier = model
        # If a path-like identifier is provided but doesn't exist, error early
        if model and Path(model).suffix and not Path(model).exists():
            msg = f"Model path not found: {model}"
            raise FileNotFoundError(msg)

        model_obj = _load_model(model_identifier)
        channels = _channels(train_channel, second_train_channel)

        artifact = await _make_artifact_client(
            artifact_id,
            token,
            server_url,
        )

        # Use the same persistent cache for inference; download missing files
        cache_dir = _artifact_cache_dir(artifact_id)
        out_items: list[dict[str, Any]] = []
        for p in image_paths:
            local_img = cache_dir / p
            local_img.parent.mkdir(parents=True, exist_ok=True)
            if not (local_img.exists() and local_img.stat().st_size > 0):
                async with artifact.open(p, "rb") as fh:
                    bytes_or_str = await fh.read()
                    data_bytes = (
                        bytes_or_str.encode("utf-8")
                        if isinstance(bytes_or_str, str)
                        else bytes_or_str
                    )
                    local_img.write_bytes(data_bytes)
            img = imread(local_img)
            masks = model_obj.eval([img], channels=channels, diameter=diameter)[0]
            # Cellpose returns labeled mask as 2D; if not, take first plane
            mask_np = masks if isinstance(masks, np.ndarray) else np.asarray(masks)
            mask_np = mask_np[0] if mask_np.ndim > NDIM_3D_THRESHOLD else mask_np
            # Ensure integer type for segmentation masks
            if not np.issubdtype(mask_np.dtype, np.integer):
                mask_np = mask_np.astype(np.int32, copy=False)
            out_items.append(
                {
                    "artifact_path": p,
                    "array": _encode_ndarray(mask_np),
                },
            )

        return {"predictions": out_items}


def _generate_test_image(size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Generate a simple synthetic grayscale image with a few bright blobs.

    No external dependencies; draws 2-3 disks on a dark background.
    """
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=np.float32)
    # Three disks with different centers/radii
    disks = [
        ((h * 0.3, w * 0.35), min(h, w) * 0.10, 0.8),
        ((h * 0.6, w * 0.6), min(h, w) * 0.12, 1.0),
        ((h * 0.7, w * 0.3), min(h, w) * 0.08, 0.6),
    ]
    for (cy, cx), r, val in disks:
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        img[mask] = np.maximum(img[mask], val)
    # Add mild noise
    rng = np.random.default_rng(42)
    img = np.clip(
        img + 0.05 * rng.standard_normal(img.shape, dtype=np.float32),
        0.0,
        1.0,
    )
    return (img * 255).astype(np.uint8)


def _run_local_model_test() -> dict[str, str]:
    """Run a local smoke test: save synthetic image, run model, save prediction.

    Returns a dict with paths to input and output files.
    """
    from tifffile import imread, imwrite

    base_dir = Path.cwd() / "local_tests"
    in_dir = base_dir / "inputs"
    out_dir = base_dir / "outputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = in_dir / "synthetic_256.ome.tif"
    if not input_path.exists():
        img = _generate_test_image()
        imwrite(input_path, img)

    # Choose model: prefer finetuned model in ./model, else fallback to pretrained
    model_dir = _ensure_model_dir()
    # Prefer a finetuned model if present (optional); we still use initial_model
    _finetuned_candidates = sorted(model_dir.glob("*.pth"))
    initial_model = "cyto3"
    model = _load_local_model(initial_model)

    channels = _channels("Grayscale", "Grayscale")
    img = imread(input_path)
    masks = model.eval([img], channels=channels)[0]
    masks_np = masks if isinstance(masks, np.ndarray) else np.asarray(masks)
    output_path = out_dir / "synthetic_256_pred.tif"
    imwrite(
        output_path,
        masks_np[0] if masks_np.ndim > NDIM_3D_THRESHOLD else masks_np,
    )

    return {"input": str(input_path), "output": str(output_path)}


if __name__ == "__main__":
    # Simple local smoke test: generate an input, run model, store outputs locally.
    paths = _run_local_model_test()
    logger.info(
        "Local model test completed. Input: %s Output: %s",
        paths["input"],
        paths["output"],
    )
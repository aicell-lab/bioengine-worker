"""Cellpose finetuning Ray Serve service with Hypha Artifact IO.

This service downloads training data from a Hypha Artifact, fine-tunes a
Cellpose model, and exposes training control and inference functions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict, TypeGuard
from urllib.parse import urlparse

import numpy as np
from pydantic import Field
from ray import serve

from hypha_rpc.utils.schema import schema_method

if TYPE_CHECKING:
    from cellpose.models import CellposeModel
    from hypha_artifact import AsyncHyphaArtifact


# ---------------------------------------------------------------------------
# Constants and logging
# ---------------------------------------------------------------------------
DEFAULT_SERVER_URL = "https://hypha.aicell.io"
ENCODING_NPY_BASE64 = "npy_base64"
METADATA_DIRNAME = "metadata"
NDIM_3D_THRESHOLD = 3
GB = 1024**3

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TrainingParams(TypedDict):
    """Grouped training parameters for scheduling and logging."""

    artifact_id: str
    metadata_path: str
    model: str | Path
    channel1: str
    channel2: str
    ratio: float
    n_epochs: int
    learning_rate: float
    weight_decay: float
    server_url: str
    n_samples: int | None
    session_id: str
    save_every: int


class DatasetSplit(TypedDict):
    """Grouped dataset split paths for Cellpose using file-based inputs.

    Cellpose ``train_seg`` accepts file lists via ``train_files``/
    ``train_labels_files`` and ``test_files``/``test_labels_files``.
    """

    train_files: list[Path]
    train_labels_files: list[Path]
    test_files: list[Path]
    test_labels_files: list[Path]


# ---------------------------------------------------------------------------
# Helper types and enums
# ---------------------------------------------------------------------------
class PretrainedModel(str, Enum):
    """Builtin Cellpose models with member-attached descriptions.

    Note: str-based Enum ensures JSON serializability in schema parameters and
    example lists.
    """

    description: str

    CYTO = (
        "cyto",
        "Original Cellpose cytoplasm model (2D cell segmentation).",
    )
    CYTO3 = (
        "cyto3",
        "Cellpose 3 cytoplasm model (latest general-purpose cytoplasm).",
    )
    NUCLEI = (
        "nuclei",
        "Cellpose nuclei model for nuclear segmentation.",
    )
    TISSUENET_CP3 = (
        "tissuenet_cp3",
        "Cellpose 3 model trained on TissueNet dataset (tissue images).",
    )
    LIVECELL_CP3 = (
        "livecell_cp3",
        "Cellpose 3 model trained on LiveCell dataset (live-cell imaging).",
    )
    YEAST_PHC_CP3 = (
        "yeast_PhC_cp3",
        "Cellpose 3 model for yeast phase-contrast microscopy.",
    )
    YEAST_BF_CP3 = (
        "yeast_BF_cp3",
        "Cellpose 3 model for yeast bright-field microscopy.",
    )
    BACT_PHASE_CP3 = (
        "bact_phase_cp3",
        "Cellpose 3 model for bacterial phase-contrast microscopy.",
    )
    BACT_FLUOR_CP3 = (
        "bact_fluor_cp3",
        "Cellpose 3 model for bacterial fluorescence microscopy.",
    )
    DEEPBACS_CP3 = (
        "deepbacs_cp3",
        "Cellpose 3 model trained on DeepBACS dataset (bacteria).",
    )

    def __new__(cls, value: str, description: str) -> Self:
        """Create a str-backed enum member and attach a description attribute."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    @classmethod
    def values(cls) -> list[str]:
        """Return a list of available model identifiers."""
        return [m.value for m in cls]

    @classmethod
    def info_list(cls) -> list[dict[str, str]]:
        """Return list of entries with keys 'id' and 'description'."""
        return [
            {"id": m.value, "description": getattr(m, "description", m.value)}
            for m in cls
        ]


class Channel(int, Enum):
    """Channel enumeration for Cellpose channel indices."""

    GRAYSCALE = 0
    RED = 1
    GREEN = 2
    BLUE = 3


_NAME_TO_CHANNEL: dict[str, Channel] = {
    "grayscale": Channel.GRAYSCALE,
    "gray": Channel.GRAYSCALE,
    "grey": Channel.GRAYSCALE,
    "red": Channel.RED,
    "green": Channel.GREEN,
    "blue": Channel.BLUE,
}


class StatusType(str, Enum):
    """Status state for background training sessions."""

    WAITING = "waiting"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class TrainingPair(TypedDict):
    """Grouped image/annotation training pairs."""

    image: Path
    annotation: Path


# ---------------------------------------------------------------------------
# Path helpers and encoding utilities
# ---------------------------------------------------------------------------


def strip_leading_slash(p: Path) -> Path:
    """Return ``p`` without any leading slash as a POSIX-style path."""
    return Path(p.as_posix().lstrip("/"))


def to_local_path(root: Path, rel: Path) -> Path:
    """Join root with a normalized relative path."""
    return root / strip_leading_slash(rel)


def get_sessions_path() -> Path:
    """Return the path to the directory where training sessions are stored."""
    return Path.home() / "sessions"


def get_model_path(session_id: str) -> Path:
    """Get the path to the model file for a training session."""
    return get_session_path(session_id) / "models" / "model"


def artifact_cache_dir(artifact_id: str) -> Path:
    """Return local cache dir for an artifact id, ensuring it exists."""
    safe = artifact_id.replace("/", "__")
    d = Path.home() / "data_cache" / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


# Type guard to check that object is list[np.ndarray]:
def is_ndarray(potential_arrays: list[object]) -> TypeGuard[list[np.ndarray]]:
    """Check if the input is a list of numpy ndarrays."""
    return all(isinstance(array, np.ndarray) for array in potential_arrays)


class PredictionItemModel(TypedDict):
    """A single prediction mapping an input identifier to an encoded mask."""

    input_path: str
    output: np.ndarray


class SessionStatus(TypedDict):
    """Status and message for a background training session."""

    status_type: StatusType
    message: str


class SessionStatusWithId(TypedDict):
    """Session status including the associated session identifier."""

    status_type: StatusType
    message: str
    session_id: str


async def make_artifact_client(
    artifact_id: str,
    server_url: str,
) -> AsyncHyphaArtifact:
    """Construct an async Hypha Artifact client."""
    from hypha_artifact import AsyncHyphaArtifact

    token = os.environ.get("HYPHA_TOKEN")

    if not token:
        error_msg = "HYPHA_TOKEN environment variable is not set."
        raise RuntimeError(error_msg)

    if "/" not in artifact_id:
        msg = "artifact_id must be of form 'workspace/alias'"
        raise ValueError(msg)
    workspace, _alias = artifact_id.split("/", 1)

    return AsyncHyphaArtifact(
        artifact_id=artifact_id,
        token=token,
        server_url=server_url,
    )


def get_url_and_artifact_id(artifact_id: str) -> tuple[str, str]:
    """Parse artifact into server URL and artifact ID components."""
    parsed = urlparse(artifact_id)
    if parsed.scheme in ("http", "https"):
        path_parts = parsed.path.lstrip("/").split("/")
        if path_parts[1] != "artifacts":
            msg = (
                "When providing a full URL for artifact, it must be "
                "of the form 'https://<server>/<workspace>/artifacts/<alias>'"
            )
            raise ValueError(msg)
        workspace = path_parts[0]
        alias = path_parts[2]
        artifact_id = f"{workspace}/{alias}"
        server_url = f"{parsed.scheme}://{parsed.netloc}"
    else:
        server_url = DEFAULT_SERVER_URL

    return server_url, artifact_id


# ---------------------------------------------------------------------------
# Core: training and dataset prep
# ---------------------------------------------------------------------------


def _channel_index(name: str) -> int:
    """Parse a user-provided channel name into a Channel index."""
    ch = _NAME_TO_CHANNEL.get(name.strip().lower())
    return ch.value if ch is not None else Channel.GRAYSCALE.value


def get_channels(first: str, second: str) -> list[int]:
    """Map channel names to Cellpose channel indices."""
    return [_channel_index(first), _channel_index(second)]


def get_session_path(session_id: str) -> Path:
    """Get the path to the directory for a training session."""
    return get_sessions_path() / session_id


def get_status_path(session_id: str) -> Path:
    """Get the path to the status.json file for a training session."""
    return get_session_path(session_id) / "status.json"


def update_status(session_id: str, status_type: StatusType, message: str) -> None:
    """Update the status of a training session."""
    status_path = get_status_path(session_id)
    with status_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        status = json.dumps(
            SessionStatus(
                status_type=status_type,
                message=message,
            ),
        )
        f.write(status)

    append_info(
        session_id,
        f"Status: {status_type}. Message: {message}",
    )


def append_info(session_id: str, info: str, *, with_time: bool = False) -> None:
    """Append information to the info.txt file for a training session."""
    with (get_session_path(session_id) / "info.txt").open(
        "a",
        encoding="utf-8",
    ) as f:
        if with_time:
            current_time = datetime.now(tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S %Z",
            )
            f.write(f"[{current_time}] {info}\n")
        else:
            f.write(info + "\n")


def get_status(session_id: str) -> SessionStatus:
    """Get the current status of a training session."""
    status_path = get_status_path(session_id)

    if not status_path.exists():
        return SessionStatus(
            status_type=StatusType.WAITING,
            message="Waiting for training to start...",
        )

    with status_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def run_blocking_task(
    cellpose_model: CellposeModel,
    model_save_path: Path,
    channels: list[int],
    dataset_split: DatasetSplit,
    training_params: TrainingParams,
) -> tuple[Path, list[float], list[float]]:
    """Run the blocking training task."""
    from cellpose import train as cellpose_train

    session_id = training_params["session_id"]

    update_status(session_id, StatusType.RUNNING, "Training running")

    try:
        seg_result = cellpose_train.train_seg(
            cellpose_model.net,
            **dataset_split,
            channels=channels,
            save_path=model_save_path,
            n_epochs=training_params["n_epochs"],
            learning_rate=training_params["learning_rate"],
            weight_decay=training_params["weight_decay"],
            save_every=training_params["save_every"],
            model_name="model",
            min_train_masks=1,
        )
    except Exception as e:
        update_status(
            session_id,
            StatusType.FAILED,
            f"Training failed with exception: {e}",
        )
        raise

    update_status(
        session_id,
        StatusType.COMPLETED,
        "Training completed successfully",
    )

    append_info(
        session_id,
        "Training completed successfully.",
        with_time=True,
    )

    return seg_result


async def finetune_cellpose(
    training_params: TrainingParams,
    executor: ThreadPoolExecutor,
) -> tuple[Path, list[float], list[float]]:
    """Prepare data, build model, and call Cellpose train_seg asynchronously."""
    session_id = training_params["session_id"]
    data_save_path = artifact_cache_dir(training_params["artifact_id"])
    pairs = await make_training_pairs(training_params, data_save_path)
    dataset_split = split_train_test(
        pairs,
        training_params["ratio"],
    )

    model_save_path = get_session_path(session_id)
    model_save_path.mkdir(parents=True, exist_ok=True)

    param_str = str(training_params)
    append_info(
        session_id,
        f"Training started. Parameters:\n{param_str}",
        with_time=True,
    )

    channels = get_channels(training_params["channel1"], training_params["channel2"])
    cellpose_model = load_model(training_params["model"])

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor,
        run_blocking_task,
        cellpose_model,
        model_save_path,
        channels,
        dataset_split,
        training_params,
    )


def launch_training_task(
    training_params: TrainingParams,
    executor: ThreadPoolExecutor,
) -> asyncio.Task:
    """Launch the Cellpose finetuning task asynchronously."""
    return asyncio.create_task(finetune_cellpose(training_params, executor))


def load_model(identifier: str | Path) -> CellposeModel:
    """Load a Cellpose model by builtin name or by local file path.

    If `identifier` points to an existing file, it is treated as a path to a
    finetuned model. Otherwise, it is treated as a builtin model name
    (e.g., "cyto3").
    """
    from cellpose import core, models

    use_gpu = core.use_gpu()

    if isinstance(identifier, Path):
        return models.CellposeModel(gpu=use_gpu, pretrained_model=str(identifier))

    return models.CellposeModel(gpu=use_gpu, model_type=identifier)


def is_missing(path: Path, *, check_nonempty: bool) -> bool:
    """Return True if path is missing or empty (if check_nonempty=True)."""
    if not path.exists():
        return True
    if not check_nonempty:
        return False
    try:
        return path.stat().st_size <= 0
    except OSError:
        return True


def get_missing_paths(
    dests: list[Path],
    out_dir: Path,
    *,
    check_nonempty: bool = True,
) -> tuple[list[str], list[str]]:
    """Return remote/local paths still missing in cache for the given dests."""
    normalized_paths = [strip_leading_slash(dest) for dest in dests]
    localized_paths = [to_local_path(out_dir, dest) for dest in dests]
    missing_flags = [
        is_missing(local_path, check_nonempty=check_nonempty)
        for local_path in localized_paths
    ]

    missing_remote_paths = [
        remote_path
        for remote_path, missing_flag in zip(normalized_paths, missing_flags)
        if missing_flag
    ]
    missing_local_paths = [
        local_path
        for local_path, missing_flag in zip(localized_paths, missing_flags)
        if missing_flag
    ]

    for local_path in missing_local_paths:
        local_path.parent.mkdir(parents=True, exist_ok=True)

    return [str(missing_remote_path) for missing_remote_path in missing_remote_paths], [
        str(missing_local_path) for missing_local_path in missing_local_paths
    ]


def image_annotation_paths(
    cache_root: Path,
    metadata_path: Path | str | None = None,
) -> tuple[list[Path], list[Path]]:
    """Resolve metadata location (folder or file) and extract path pairs.

    Behavior
    - If ``metadata_path`` is None: use ``cache_root / "metadata"``.
    - If it's a directory: read all ``*.json`` files in that directory.
    - If it's a single JSON file: read only that file.
    - Relative paths are resolved against ``cache_root``; absolute paths are used
      as-is.

    Expects each JSON to contain two lists: ``uploaded_images`` and
    ``uploaded_annotations`` of equal length. Any malformed files are skipped
    with a warning.
    """
    if metadata_path is None:
        resolved = cache_root / METADATA_DIRNAME
    else:
        p = Path(metadata_path)
        resolved = p if p.is_absolute() else (cache_root / p)

    imgs: list[Path] = []
    anns: list[Path] = []

    if resolved.is_dir():
        json_files = sorted(resolved.glob("*.json"))
    elif resolved.is_file():
        json_files = [resolved]
    else:
        logger.warning("Metadata path does not exist: %s", resolved)
        return imgs, anns

    for mf in json_files:
        try:
            text = mf.read_text(encoding="utf-8")
            meta = json.loads(text)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Skipping metadata file %s due to error: %s", mf, exc)
            continue

        m_imgs = [Path(item) for item in meta.get("uploaded_images", [])]
        m_anns = [Path(item) for item in meta.get("uploaded_annotations", [])]
        if m_imgs and m_anns and len(m_imgs) == len(m_anns):
            imgs.extend(m_imgs)
            anns.extend(m_anns)

    return imgs, anns


async def download_pairs_from_artifact(
    artifact: AsyncHyphaArtifact,
    out_dir: Path,
    image_paths: list[Path],
    annotation_paths: list[Path],
) -> list[TrainingPair]:
    """Download dataset files in batches and return local (img, ann) pairs.

    The returned paths are absolute local paths within ``out_dir``. Inputs are
    treated as artifact-relative paths; any leading slashes are ignored.
    """
    missing_rpaths, missing_lpaths = get_missing_paths(
        [*image_paths, *annotation_paths],
        out_dir,
    )
    if missing_rpaths:
        await artifact.get(missing_rpaths, missing_lpaths, on_error="ignore")

    local_imgs = [to_local_path(out_dir, p) for p in image_paths]
    local_anns = [to_local_path(out_dir, p) for p in annotation_paths]

    missing_after = [
        str(p)
        for p in [*local_imgs, *local_anns]
        if not p.exists() or p.stat().st_size <= 0
    ]
    if missing_after:
        sample = ", ".join(missing_after[:5])
        msg = (
            "Some dataset files were not found after download: "
            f"{len(missing_after)} missing (e.g., {sample}). "
            "Check your metadata JSON paths and artifact contents."
        )
        raise RuntimeError(msg)

    return [
        TrainingPair(image=img, annotation=ann)
        for img, ann in zip(local_imgs, local_anns)
    ]


async def download_metadata_from_artifact(
    artifact: AsyncHyphaArtifact,
    cache_root: Path,
    meta_remote: str,
) -> Path:
    """Ensure metadata (folder or single file) is present in the local cache.

    - If ``meta_remote`` points to a single ``.json`` file, download only if missing
        using batched get with computed remote/local paths.
    - If it points to a directory, perform a best-effort recursive fetch of the
        directory contents (limited by the artifact API capabilities).

    Returns the local path to the metadata folder or file under ``cache_root``.
    """
    local = cache_root / meta_remote
    if Path(meta_remote).suffix.lower() == ".json":
        dests = [Path(meta_remote)]
        missing_remote, missing_local = get_missing_paths(dests, cache_root)
        if missing_remote:
            await artifact.get(missing_remote, missing_local, on_error="ignore")
        return local

    local.mkdir(parents=True, exist_ok=True)
    await artifact.get(
        f"{meta_remote.rstrip('/')}/",
        str(local),
        recursive=True,
        on_error="ignore",
    )
    return local


def split_train_test(
    pairs: list[TrainingPair],
    ratio: float,
) -> DatasetSplit:
    """Split (image, label) pairs into train/test according to ratio."""
    if not pairs or len(pairs) < 1:
        error_msg = "No (image, label) pairs found for train/test split."
        raise ValueError(error_msg)
    if ratio <= 0 or ratio >= 1:
        error_msg = "Train/test split ratio must be in the interval (0, 1)."
        raise ValueError(error_msg)

    rng = np.random.default_rng()
    shuffled_indices = rng.permutation(len(pairs))
    cut = max(int(len(pairs) * ratio), 1)
    train_indices = shuffled_indices[:cut]
    test_indices = shuffled_indices[cut:]
    return DatasetSplit(
        train_files=[pairs[train_i]["image"] for train_i in train_indices],
        train_labels_files=[pairs[train_i]["annotation"] for train_i in train_indices],
        test_files=[pairs[test_i]["image"] for test_i in test_indices],
        test_labels_files=[pairs[test_i]["annotation"] for test_i in test_indices],
    )


def get_training_subset(
    image_paths: list[Path],
    annotation_paths: list[Path],
    n_samples: int,
) -> tuple[list[Path], list[Path]]:
    """Select a subset of the dataset if n_samples is specified."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(image_paths))[:n_samples]
    image_paths = [image_paths[i] for i in idx]
    annotation_paths = [annotation_paths[i] for i in idx]
    return image_paths, annotation_paths


async def make_training_pairs(
    config: TrainingParams,
    save_path: Path,
) -> list[TrainingPair]:
    """Download metadata and dataset pairs from artifact according to config."""
    artifact = await make_artifact_client(
        config["artifact_id"],
        config["server_url"],
    )
    # TODO: don't download metadata if already present
    await download_metadata_from_artifact(artifact, save_path, config["metadata_path"])

    image_paths, annotation_paths = image_annotation_paths(
        cache_root=save_path,
        metadata_path=config["metadata_path"],
    )

    if config["n_samples"] is not None:
        image_paths, annotation_paths = get_training_subset(
            image_paths,
            annotation_paths,
            config["n_samples"],
        )

    return await download_pairs_from_artifact(
        artifact,
        save_path,
        image_paths,
        annotation_paths,
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


async def _images_from_artifact(
    artifact_id: str,
    image_paths: list[str],
    server_url: str,
) -> list[np.ndarray]:
    """Download missing images into cache and load as numpy arrays."""
    from tifffile import imread

    artifact = await make_artifact_client(artifact_id, server_url)
    cache_dir = artifact_cache_dir(artifact_id)
    dests = [Path(p) for p in image_paths]
    missing_remote, missing_local = get_missing_paths(
        dests,
        cache_dir,
        check_nonempty=False,
    )
    if missing_remote:
        await artifact.get(missing_remote, missing_local, on_error="ignore")
    imgs: list[np.ndarray] = []
    for rel in image_paths:
        local_img = cache_dir / rel
        imgs.append(imread(local_img))
    return imgs


def _predict_and_encode(
    *,
    model: CellposeModel,
    images: list[np.ndarray],
    image_paths: list[str],
    channels: list[int],
    diameter: float | None,
) -> list[PredictionItemModel]:
    """Run model on images and return encoded mask payloads."""
    out: list[PredictionItemModel] = []
    for image, path in zip(images, image_paths):
        masks = model.eval(
            [image],
            channels=channels,
            diameter=diameter,
        )[0]
        mask_np = masks if isinstance(masks, np.ndarray) else np.asarray(masks)
        if mask_np.ndim >= NDIM_3D_THRESHOLD and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        if not np.issubdtype(mask_np.dtype, np.integer):
            mask_np = mask_np.astype(np.int32, copy=False)
        out_item = PredictionItemModel(
            input_path=path,
            output=mask_np,
        )
        out.append(out_item)
    return out


# ---------------------------------------------------------------------------
# Ray Serve deployment
# ---------------------------------------------------------------------------
@serve.deployment(
    ray_actor_options={
        "num_gpus": 0.25,
        "num_cpus": 4,
        "memory": 12 * GB,
        "runtime_env": {
            "pip": [
                "cellpose==3.1.1.1",
                "numpy==1.26.4",
                "tifffile",
                "hypha-artifact==0.1.2",
            ],
        },
    },
    max_ongoing_requests=1,
    max_queued_requests=10,
    # autoscaling_config={
    #     "min_replicas": 1,
    #     "initial_replicas": 1,
    #     "max_replicas": 2,
    #     "target_num_ongoing_requests_per_replica": 0.8,
    #     "metrics_interval_s": 2.0,
    #     "look_back_period_s": 10.0,
    #     "downscale_delay_s": 300,
    #     "upscale_delay_s": 0.0,
    # },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class CellposeFinetune:
    """Ray Serve deployment for finetuning and inference with Cellpose.

    Downloads per-timepoint TIFFs from a dataset artifact, finetunes a Cellpose
    model locally, and runs inference using the saved model stored in ./model.
    """

    pretrained_models: list[str]
    executors: dict[str, ThreadPoolExecutor]
    _session_lock: asyncio.Lock

    def __init__(self) -> None:
        """Initialize directories and defaults for the service."""
        get_sessions_path().mkdir(parents=True, exist_ok=True)
        self.pretrained_models = PretrainedModel.values()
        self.executors = {}
        self._session_lock = asyncio.Lock()

    def get_model_id(self, model: str) -> str | Path:
        """Return a model identifier suitable for loading by Cellpose."""
        # print all contents of get_session_path(model):
        if model in self.pretrained_models:
            return model
        if get_model_path(model).exists():
            return Path(model)
        msg = (
            f"Model identifier '{model}' is not a known pretrained model "
            "or a valid session ID of a previously finetuned model."
        )
        raise ValueError(msg)

    @schema_method(arbitrary_types_allowed=True)
    async def start_training(
        self,
        artifact: str = Field(
            description=(
                "Artifact identifier 'workspace/alias' containing TIFF images and "
                "annotations for training."
            ),
            examples=["ri-scale/zarr-demo"],
        ),
        metadata_path: str = Field(
            "metadata/",
            description=(
                "Path within the artifact to either a folder of *.json files "
                "or a single metadata JSON. Each JSON must contain two lists "
                "of equal length: 'uploaded_images' and 'uploaded_annotations' "
                "(paths relative to the artifact)."
            ),
            examples=["metadata/experiment1.json", "metadata/"],
        ),
        model: str = Field(
            PretrainedModel.CYTO3.value,
            description=(
                "Name of Cellpose model to finetune."
                " Must be a builtin pretrained Cellpose model or"
                " The session ID of a previously finetuned model file."
            ),
            examples=[
                "abc123ef-4567-890a-bcde-f1234567890a",
                *PretrainedModel.values(),
            ],
        ),
        channel1: str = Field(
            "Grayscale",
            description=(
                "First channel spec for Cellpose. One of {Grayscale, Red, "
                "Green, Blue}."
            ),
        ),
        channel2: str = Field(
            "Grayscale",
            description=(
                "Second channel spec for Cellpose. One of {Grayscale, Red, "
                "Green, Blue}."
            ),
        ),
        ratio: float = Field(
            0.8,
            description=(
                "Fraction of pairs used for training; the remainder is used "
                "for testing"
            ),
        ),
        n_samples: int | None = Field(
            None,
            description=(
                "Optional number of samples to use from the dataset. If None, "
                "all available samples are used."
            ),
        ),
        n_epochs: int = Field(10, description="Number of training epochs"),
        learning_rate: float = Field(1e-6, description="Learning rate"),
        weight_decay: float = Field(1e-4, description="Weight decay"),
        save_every: int = Field(
            100,
            description="Frequency (in epochs) to save the model during training.",
        ),
    ) -> SessionStatusWithId:
        """Start asynchronous finetuning of a Cellpose model on an artifact dataset.

        This downloads metadata and the referenced image/annotation files from the
        given Hypha artifact to a local cache and launches training in the
        background. Use ``get_training_status`` to poll progress or ``stop_training``
        to cancel.
        """
        from uuid import uuid4

        server_url, artifact_id = get_url_and_artifact_id(artifact)
        model_id = self.get_model_id(model)
        session_id = str(uuid4())
        get_session_path(session_id).mkdir(parents=True, exist_ok=True)

        update_status(
            session_id=session_id,
            status_type=StatusType.PREPARING,
            message="Preparing for training...",
        )

        training_params = TrainingParams(
            artifact_id=artifact_id,
            metadata_path=metadata_path,
            model=model_id,
            channel1=channel1,
            channel2=channel2,
            ratio=ratio,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            server_url=server_url,
            n_samples=n_samples,
            session_id=session_id,
            save_every=save_every,
        )

        async with self._session_lock:
            executor = ThreadPoolExecutor(max_workers=1)
            self.executors[session_id] = executor
            launch_training_task(training_params, executor)

        status = get_status(session_id)
        return SessionStatusWithId(**status, session_id=session_id)

    # TODO: implement
    @schema_method(arbitrary_types_allowed=True)
    async def stop_training(
        self,
        session_id: str = Field(
            description="Identifier returned by ``start_training``.",
        ),
    ) -> SessionStatus:
        """Stop an ongoing training session."""
        executor = self.executors.get(session_id)
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
            del self.executors[session_id]

        update_status(
            session_id=session_id,
            status_type=StatusType.STOPPED,
            message="Training session stopped by user.",
        )

        return get_status(session_id)

    @schema_method(arbitrary_types_allowed=True)
    async def get_training_status(
        self,
        session_id: str = Field(
            description="Identifier returned by ``start_training``.",
        ),
    ) -> SessionStatus:
        """Retrieve the current status of a training session.

        Use ``get_trained_model_path(session_id)`` to obtain the final saved model
        path once the session is completed.
        """
        return get_status(session_id)

    @schema_method(arbitrary_types_allowed=True)
    async def list_training_sessions(
        self,
        status_types: list[str] | None = Field(
            None,
            description=(
                "Optional list of statuses to include. Values must be among "
                "{running, stopped, completed, failed, unknown}."
            ),
            examples=[["running", "completed"]],
        ),
    ) -> dict[str, SessionStatus]:
        """List all known training sessions with their current or final status.

        Includes running sessions tracked in-memory and finished sessions recorded in
        the session history. For completed sessions, the saved model path is included
        if available.
        """
        if status_types is not None:
            allowed = {s.value for s in StatusType}
            filt = {s for s in status_types if s in allowed}

        all_session_ids = get_sessions_path().iterdir()
        status_paths = [session_dir / "status.json" for session_dir in all_session_ids]
        session_statuses = [json.loads(path.read_text()) for path in status_paths]

        return {
            str(status_path): SessionStatus(**session)
            for status_path, session in zip(status_paths, session_statuses)
            if filt is None or session["status_type"] in filt
        }

    @schema_method(arbitrary_types_allowed=True)
    async def infer(
        self,
        artifact: str | None = Field(
            None,
            description=(
                "Artifact identifier 'workspace/alias' containing the source images."
            ),
        ),
        image_paths: list[str] | None = Field(
            None,
            description="List of artifact-relative image paths to segment",
            min_length=1,
        ),
        input_arrays: list[object] | None = Field(
            None,
            description="List of numpy ndarrays to segment",
            examples=[[[0, 0, 0], [255, 255, 255]]],
            min_length=1,
        ),
        model: str = Field(
            PretrainedModel.CYTO3.value,
            description=(
                "Identifier of the Cellpose model to use for inference. Either a "
                "built-in pretrained model name or the session ID of a finetuned model."
            ),
            examples=["abc123ef-4567-890a-bcde-f1234567890a", PretrainedModel.values()],
        ),
        channel1: str = Field(
            "Grayscale",
            description=(
                "First channel spec for Cellpose. One of {Grayscale, Red, "
                "Green, Blue}."
            ),
        ),
        channel2: str = Field(
            "Grayscale",
            description=(
                "Second channel spec for Cellpose. One of {Grayscale, Red, "
                "Green, Blue}."
            ),
        ),
        diameter: float | None = Field(
            None,
            description=(
                "Approximate object diameter; if None, Cellpose will estimate it"
            ),
            ge=0,
        ),
    ) -> list[PredictionItemModel]:
        """Run Cellpose inference on artifact images and return encoded masks.

        Images are fetched from the specified artifact into a local cache as
        needed. For each input path, the corresponding mask array is returned as
        an NPY-serialized base64 payload suitable for cross-language transport.

        """
        if artifact is not None:
            server_url, artifact_id = get_url_and_artifact_id(artifact)

        if image_paths is None:
            image_paths = []

        if input_arrays is not None and not is_ndarray(input_arrays):
            error_msg = "input_arrays must be a list of numpy ndarrays"
            raise TypeError(error_msg)

        model_id = self.get_model_id(model)
        model_obj = load_model(model_id)
        channels = get_channels(channel1, channel2)

        images: list[np.ndarray]
        if input_arrays is not None:
            image_paths = [f"input_arrays[{i}]" for i in range(len(input_arrays))]
        elif artifact_id is not None and image_paths is not None:
            images = await _images_from_artifact(
                artifact_id=artifact_id,
                image_paths=image_paths,
                server_url=server_url,
            )
        else:
            images = []

        return _predict_and_encode(
            model=model_obj,
            images=images,
            image_paths=image_paths,
            channels=channels,
            diameter=diameter,
        )


async def infer(
    cellpose_tuner: object,
    model: str | None = None,
) -> None:
    """Test inference functionality of the Cellpose Fine-Tuning service."""
    from plotly import express as px

    inference_result = await cellpose_tuner.infer(  # type: ignore[hasAttribute]
        model=model,
        artifact="ri-scale/zarr-demo",
        diameter=40,
        image_paths=["images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif"],
    )
    logger.info("Inference done! Result: %s", str(inference_result)[:500] + "...")
    arr = inference_result[0]["output"]
    px.imshow(arr).show()


async def monitor_training(
    cellpose_tuner: object,
    session_id: str,
) -> None:
    """Monitor the training session until completion."""
    status = None
    current_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{current_time}] Starting training monitoring...")  # noqa: T201
    while True:
        status = await cellpose_tuner.get_training_status(session_id)  # type: ignore[hasAttribute]
        current_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        # print(f"\r[{current_time}] {status['message']}", end="")  #lnoqa: T201
        if status["status_type"] == "completed":
            break

        if status["status_type"] == "failed":
            # print(f"[{current_time}] Training failed: {status['message']}")
            break

        await asyncio.sleep(1)


async def test_cellpose_finetune() -> None:
    """Test the CellposeFinetune Ray Serve deployment end-to-end."""
    cellpose_tuner = CellposeFinetune.func_or_class()  # type: ignore[reportCallIssue]

    await infer(cellpose_tuner, model="cyto3")

    session_status = await cellpose_tuner.start_training(
        artifact="ri-scale/zarr-demo",
        n_epochs=1,
        n_samples=5,
    )

    session_id = session_status["session_id"]

    await monitor_training(cellpose_tuner, session_id)

    await infer(cellpose_tuner, model=session_id)


if __name__ == "__main__":
    asyncio.run(test_cellpose_finetune())
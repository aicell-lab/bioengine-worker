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
from typing import TYPE_CHECKING, Any, Self, TypedDict, TypeGuard, cast
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
from hypha_rpc.utils.schema import schema_method  # type: ignore
from pydantic import Field
from ray import serve

if TYPE_CHECKING:
    import torch  # type: ignore
    from cellpose.models import CellposeModel  # type: ignore
    from hypha_artifact import AsyncHyphaArtifact  # type: ignore


# ---------------------------------------------------------------------------
# Constants and logging
# ---------------------------------------------------------------------------
DEFAULT_SERVER_URL = "https://hypha.aicell.io"
ENCODING_NPY_BASE64 = "npy_base64"
METADATA_DIRNAME = "metadata"
NDIM_3D_THRESHOLD = 3
GB = 1024**3
DOC_FILENAME = "doc.md"
RDF_FILENAME = "rdf.yaml"
TRAINING_PARAMS_FILENAME = "training_params.json"

# Model template for BioImage.io export
MODEL_TEMPLATE_PY = '''"""BioImage.io Model Wrapper for Cellpose 4.0.7 (Cellpose-SAM).

This wrapper provides a PyTorch nn.Module interface for Cellpose 4.0.7 models
that is compatible with the BioImage.io model format.
"""
import numpy as np
import torch
import torch.nn as nn
from cellpose import models as cpmodels
from cellpose.vit_sam import Transformer
from cellpose.core import assign_device


# Prevent mix-up between pytorch module eval and cellpose eval functions
cpmodels.CellposeModel.evaluate = cpmodels.CellposeModel.eval  # type: ignore


class CellposeSAMWrapper(nn.Module, cpmodels.CellposeModel):
    """
    A wrapper around the Cellpose 4.0.7 (Cellpose-SAM) model
    which acts as a PyTorch model compatible with BioImage.io format.

    This wrapper is designed for the Transformer-based Cellpose-SAM architecture.
    """

    def __init__(
        self,
        model_type="cpsam",
        diam_mean=30.0,
        cp_batch_size=8,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        stitch_threshold=0.0,
        estimate_diam=False,
        normalize=True,
        do_3D=False,
        gpu=True,
        use_bfloat16=True,
    ):
        """Initialize the Cellpose-SAM wrapper.

        Args:
            model_type: Model type (default: "cpsam" for Cellpose-SAM)
            diam_mean: Mean diameter of objects (default: 30.0 pixels)
            cp_batch_size: Batch size for cellpose processing (default: 8)
            channels: Channel configuration [cytoplasm, nucleus] (default: [0, 0] for grayscale)
            flow_threshold: Flow error threshold for mask reconstruction (default: 0.4)
            cellprob_threshold: Cell probability threshold (default: 0.0)
            stitch_threshold: Threshold for stitching tiles (default: 0.0)
            estimate_diam: Whether to estimate diameter automatically (default: False)
            normalize: Whether to normalize images (default: True)
            do_3D: Whether to process 3D images (default: False)
            gpu: Whether to use GPU (default: True)
            use_bfloat16: Whether to use bfloat16 precision (default: True)
        """
        nn.Module.__init__(self)

        self.model_type = model_type
        self.diam_mean = diam_mean
        self.cp_batch_size = cp_batch_size
        self.channels = channels
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.stitch_threshold = stitch_threshold
        self.estimate_diam = estimate_diam
        self.normalize = normalize
        self.do_3D = do_3D
        self.use_bfloat16 = use_bfloat16

        # Device assignment
        self.device, self.gpu = assign_device(use_torch=True, gpu=gpu)

        # Create Transformer network (Cellpose-SAM)
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.net = Transformer(dtype=dtype).to(self.device)

        # Set diameter parameters
        self.net.diam_labels = nn.Parameter(torch.tensor([diam_mean]), requires_grad=False)
        self.net.diam_mean = nn.Parameter(torch.tensor([diam_mean]), requires_grad=False)

        # Cellpose model parameters
        self.nclasses = 3
        self.channel_axis = None
        self.invert = False

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load model weights from state dict.

        Args:
            state_dict: Dictionary containing model weights
            strict: Whether to strictly enforce key matching (default: True)
            assign: Whether to assign values (default: False)

        Returns:
            NamedTuple with missing_keys and unexpected_keys
        """
        from collections import namedtuple

        Incompatible = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])

        # Load the state dict into the network
        result = self.net.load_state_dict(state_dict, strict=strict)

        # Update diameter parameters from loaded weights
        if hasattr(self.net, 'diam_mean'):
            self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
        if hasattr(self.net, 'diam_labels'):
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]

        return result

    def eval(self, *args, **kwargs):
        """Evaluate the model.

        This method handles both PyTorch module eval (no args) and
        Cellpose eval (with args) by dispatching appropriately.
        """
        if len(args) == 0 and len(kwargs) == 0:
            # PyTorch module eval
            return self.train(False)
        else:
            # Cellpose model eval
            return self.evaluate(*args, **kwargs)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for BioImage.io compatibility.

        Args:
            x: Input tensor of shape (batch, channel, height, width)

        Returns:
            masks: Segmentation masks of shape (batch, height, width)

        Raises:
            ValueError: If input dimensions are invalid
        """
        if len(x.shape) != 4:
            raise ValueError(
                f"Input image(s) must be 4-dimensional (batch, channel, height, width), "
                f"got shape {x.shape}"
            )

        # Convert torch tensor to list of numpy arrays (Y, X, C format for cellpose)
        image_list = []
        for img in x:
            # Convert from (C, H, W) to (H, W, C)
            img_np = img.permute(1, 2, 0).cpu().numpy()

            # Ensure 3 channels for Cellpose-SAM
            if img_np.shape[2] == 1:
                # Replicate single channel to 3 channels
                img_np = np.concatenate([img_np, img_np, img_np], axis=2)
            elif img_np.shape[2] == 2:
                # Add a zero channel
                img_np = np.concatenate([img_np, np.zeros_like(img_np[:,:,0:1])], axis=2)
            elif img_np.shape[2] > 3:
                # Use first 3 channels
                img_np = img_np[:,:,:3]

            image_list.append(img_np)

        # Run cellpose eval
        masks_list, flows_list, styles_list = self.eval(  # type: ignore
            image_list,
            channels=self.channels,
            channel_axis=self.channel_axis,
            diameter=self.diam_mean,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            stitch_threshold=self.stitch_threshold,
            batch_size=self.cp_batch_size,
            normalize=self.normalize,
            invert=self.invert,
            do_3D=self.do_3D,
        )

        # Convert masks to tensor
        if isinstance(masks_list, list):
            masks = torch.stack([torch.from_numpy(np.array(m, dtype=np.float32)) for m in masks_list])
        else:
            masks = torch.from_numpy(np.array(masks_list, dtype=np.float32))

        # Move to same device as input
        masks = masks.to(x.device)

        # Ensure correct shape (B, H, W)
        if len(masks.shape) == 2:
            masks = masks.unsqueeze(0)

        return masks
'''

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TrainingParams(TypedDict):
    """Grouped training parameters for scheduling and logging."""

    artifact_id: str
    train_images: str
    train_annotations: str
    test_images: str | None
    test_annotations: str | None
    model: str | Path
    n_epochs: int
    learning_rate: float
    weight_decay: float
    server_url: str
    n_samples: int | None
    session_id: str
    min_train_masks: int


class DatasetSplit(TypedDict):
    """Grouped dataset split paths for Cellpose using file-based inputs.

    Cellpose ``train_seg`` accepts file lists via ``train_files``/
    ``train_labels_files`` and ``test_files``/``test_labels_files``.
    Test files can be None to skip test evaluation during training.
    """

    train_files: list[Path]
    train_labels_files: list[Path]
    test_files: list[Path] | None
    test_labels_files: list[Path] | None


# ---------------------------------------------------------------------------
# Helper types and enums
# ---------------------------------------------------------------------------
class PretrainedModel(str, Enum):
    """Builtin Cellpose models with member-attached descriptions.

    Note: str-based Enum ensures JSON serializability in schema parameters and
    example lists.
    """

    description: str

    CPSAM = (
        "cpsam",
        "Cellpose-SAM 4.0 model (transformer-based, channel-order invariant).",
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
def is_ndarray(potential_arrays: list[object]) -> TypeGuard[list[npt.NDArray[Any]]]:
    """Check if the input is a list of numpy ndarrays."""
    return all(isinstance(array, np.ndarray) for array in potential_arrays)


class PredictionItemModel(TypedDict, total=False):
    """A single prediction mapping an input identifier to an encoded mask."""

    input_path: str
    output: npt.NDArray[Any]
    flows: list[
        npt.NDArray[Any]
    ]  # Optional: [HSV flow, XY flows, cellprob, final positions]


class ValidationMetrics(TypedDict, total=False):
    """Validation performance metrics for a segmentation foreground mask.

    Metrics are computed on Cellpose's cell-probability channel as a binary
    foreground/background classification problem.
    """

    pixel_accuracy: float
    precision: float
    recall: float
    f1: float
    iou: float


class SessionStatus(TypedDict, total=False):
    """Status and message for a background training session."""

    status_type: StatusType
    message: str
    dataset_artifact_id: str  # Training dataset artifact ID
    train_losses: list[float]
    test_losses: list[float]
    test_metrics: list[ValidationMetrics | None]
    n_train: int  # Number of training samples
    n_test: int  # Number of test samples
    start_time: str  # Training start time (ISO format)
    current_epoch: int  # Current epoch number (1-indexed)
    total_epochs: int  # Total number of epochs
    elapsed_seconds: float  # Elapsed time in seconds
    current_batch: int  # Current batch number within epoch (0-indexed)
    total_batches: int  # Total number of batches per epoch
    exported_artifact_id: str  # Artifact ID if model has been exported
    model_modified: bool  # Flag indicating if model was modified since last export


class SessionStatusWithId(TypedDict, total=False):
    """Session status including the associated session identifier."""

    status_type: StatusType
    message: str
    train_losses: list[float]
    test_losses: list[float]
    test_metrics: list[ValidationMetrics | None]
    n_train: int  # Number of training samples
    n_test: int  # Number of test samples
    start_time: str  # Training start time (ISO format)
    current_epoch: int  # Current epoch number (1-indexed)
    total_epochs: int  # Total number of epochs
    elapsed_seconds: float  # Elapsed time in seconds
    current_batch: int  # Current batch number within epoch (0-indexed)
    total_batches: int  # Total number of batches per epoch
    exported_artifact_id: str  # Artifact ID if model has been exported
    model_modified: bool  # Flag indicating if model was modified since last export
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
    _, _alias = artifact_id.split("/", 1)

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


def ensure_3_channels(image: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert image to 3-channel format required by Cellpose 4.0.7.

    Cellpose-SAM requires images with exactly 3 channels and is channel-order invariant.

    Args:
        image: Input image array. Can be:
            - (H, W): Single channel grayscale
            - (2, H, W): Two channel image
            - (3, H, W): Three channel image (returned as-is)
            - (n, H, W) where n > 3: First 3 channels used

    Returns:
        Image array with shape (3, H, W)

    Raises:
        ValueError: If image dimensions are invalid
    """
    if image.ndim == 2:
        # Single channel (H, W) -> replicate to (3, H, W)
        return np.stack([image, image, image], axis=0)
    elif image.ndim == 3:
        n_channels = image.shape[0]
        if n_channels == 1:
            # (1, H, W) -> (3, H, W)
            return np.concatenate([image, image, image], axis=0)
        elif n_channels == 2:
            # (2, H, W) -> (3, H, W) by padding with zeros
            zero_channel = np.zeros_like(image[0:1])
            return np.concatenate([image, zero_channel], axis=0)
        elif n_channels == 3:
            # Already 3 channels
            return image
        elif n_channels > 3:
            # More than 3 channels -> use first 3
            logger.warning(
                f"Image has {n_channels} channels, using first 3 for Cellpose-SAM"
            )
            return image[:3]
        else:
            raise ValueError(f"Invalid number of channels: {n_channels}")
    else:
        raise ValueError(
            f"Invalid image dimensions: {image.shape}. "
            "Expected 2D (H, W) or 3D (C, H, W) image."
        )


def ensure_3_channels_batch(images: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Ensure a batch of images is shaped (N, 3, H, W)."""
    if images.ndim == 3:
        images = images[:, None, :, :]
    if images.ndim != 4:
        raise ValueError(f"Invalid batch dimensions: {images.shape}")

    channel_count = images.shape[1]
    if channel_count == 3:
        return images
    if channel_count == 1:
        return np.repeat(images, 3, axis=1)
    if channel_count == 2:
        zero_channel = np.zeros_like(images[:, :1, :, :])
        return np.concatenate([images, zero_channel], axis=1)
    return images[:, :3, :, :]


def get_session_path(session_id: str) -> Path:
    """Get the path to the directory for a training session."""
    return get_sessions_path() / session_id


def get_status_path(session_id: str) -> Path:
    """Get the path to the status.json file for a training session."""
    return get_session_path(session_id) / "status.json"


def update_status(
    session_id: str,
    status_type: StatusType,
    message: str,
    dataset_artifact_id: str | None = None,
    train_losses: list[float] | None = None,
    test_losses: list[float] | None = None,
    test_metrics: list[ValidationMetrics | None] | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    start_time: str | None = None,
    current_epoch: int | None = None,
    total_epochs: int | None = None,
    elapsed_seconds: float | None = None,
    current_batch: int | None = None,
    total_batches: int | None = None,
) -> None:
    """Update the status of a training session."""
    status_path = get_status_path(session_id)

    # Read existing status to preserve fields not being updated
    existing_status: SessionStatus = {}
    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as f:
            try:
                existing_status = json.load(f)
            except json.JSONDecodeError:
                pass  # If file is corrupt, start fresh

    # Start with existing status, then update with new values
    status_dict: SessionStatus = {**existing_status}
    status_dict["status_type"] = status_type
    status_dict["message"] = message

    if dataset_artifact_id is not None:
        status_dict["dataset_artifact_id"] = dataset_artifact_id
    if train_losses is not None:
        status_dict["train_losses"] = train_losses
    if test_losses is not None:
        status_dict["test_losses"] = test_losses
    if test_metrics is not None:
        status_dict["test_metrics"] = test_metrics
    if n_train is not None:
        status_dict["n_train"] = n_train
    if n_test is not None:
        status_dict["n_test"] = n_test
    if start_time is not None:
        status_dict["start_time"] = start_time
    if current_epoch is not None:
        status_dict["current_epoch"] = current_epoch
    if total_epochs is not None:
        status_dict["total_epochs"] = total_epochs
    if elapsed_seconds is not None:
        status_dict["elapsed_seconds"] = elapsed_seconds
    if current_batch is not None:
        status_dict["current_batch"] = current_batch
    if total_batches is not None:
        status_dict["total_batches"] = total_batches

    with status_path.open("w", encoding="utf-8") as f:
        status = json.dumps(status_dict)
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

    try:
        with status_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Handle race condition where file exists but is empty or being written
        return SessionStatus(
            status_type=StatusType.WAITING,
            message="Waiting for training to start...",
        )


# ---------------------------------------------------------------------------
# Custom Cellpose training with callbacks
# ---------------------------------------------------------------------------


def train_seg_with_callbacks(
    net: Any,
    train_data: list[npt.NDArray[Any]] | None = None,
    train_labels: list[npt.NDArray[Any]] | None = None,
    train_files: list[Path] | None = None,
    train_labels_files: list[Path] | None = None,
    train_probs: list[float] | None = None,
    test_data: list[npt.NDArray[Any]] | None = None,
    test_labels: list[npt.NDArray[Any]] | None = None,
    test_files: list[Path] | None = None,
    test_labels_files: list[Path] | None = None,
    test_probs: list[float] | None = None,
    channel_axis: int | None = None,
    load_files: bool = True,
    batch_size: int = 1,
    learning_rate: float = 5e-5,
    SGD: bool = False,
    n_epochs: int = 100,
    weight_decay: float = 0.1,
    normalize: bool | dict[str, Any] = True,
    compute_flows: bool = False,
    save_path: Path | str | None = None,
    nimg_per_epoch: int | None = None,
    nimg_test_per_epoch: int | None = None,
    rescale: bool = False,
    scale_range: float | None = None,
    bsize: int = 256,
    min_train_masks: int = 5,
    model_name: str | None = None,
    class_weights: npt.NDArray[Any] | None = None,
    epoch_callback: Any = None,
    batch_callback: Any = None,
) -> tuple[Path, npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Train the network with images for segmentation (with epoch and batch callbacks).

    This is a modified version of cellpose.train.train_seg that adds support
    for epoch and batch callbacks to enable real-time progress tracking.

    Args:
        net: The network model to train
        ... (all standard parameters same as cellpose.train.train_seg)
        epoch_callback: Optional callback function called after each epoch.
            Signature: callback(epoch, train_loss, test_loss, elapsed_seconds, val_metrics)
            Where val_metrics is a dict with keys like precision/recall/f1, or None
            when validation was not run for that epoch.
        batch_callback: Optional callback function called after each batch.
            Signature: callback(epoch, batch_idx, total_batches, batch_loss, elapsed_seconds, val_metrics)
            Where val_metrics can be None (training) or dict (validation).

    Returns:
        tuple: (model_path, train_losses, test_losses)
    """
    import time

    import torch
    from cellpose import models  # type: ignore
    from cellpose.train import (  # type: ignore
        _get_batch,  # type: ignore
        _loss_fn_class,  # type: ignore
        _loss_fn_seg,  # type: ignore
        _process_train_test,  # type: ignore
        train_logger,  # type: ignore
    )
    from cellpose.transforms import random_rotate_and_resize  # type: ignore

    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")

    device = net.device

    original_net_dtype = None
    if device.type == "mps" and net.dtype == torch.bfloat16:
        original_net_dtype = torch.bfloat16
        train_logger.warning(
            "Training with bfloat16 on MPS is not supported, using float32 network instead"
        )
        net.dtype = torch.float32
        net.to(torch.float32)

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}  # type: ignore
    elif not isinstance(normalize, bool):  # type: ignore
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default  # type: ignore
        normalize_params["normalize"] = normalize

    out = cast(
        Any,
        _process_train_test(
            train_data=train_data,
            train_labels=train_labels,
            train_files=train_files,
            train_labels_files=train_labels_files,
            train_probs=train_probs,
            test_data=test_data,
            test_labels=test_labels,
            test_files=test_files,
            test_labels_files=test_labels_files,
            test_probs=test_probs,
            load_files=load_files,
            min_train_masks=min_train_masks,
            compute_flows=compute_flows,
            channel_axis=channel_axis,
            normalize_params=normalize_params,
            device=net.device,
        ),
    )
    (
        train_data,
        train_labels,
        train_files,
        train_labels_files,
        train_probs,
        diam_train,
        test_data,
        test_labels,
        test_files,
        test_labels_files,
        test_probs,
        diam_test,
        normed,
    ) = out

    # already normalized, do not normalize during training
    if normed:
        kwargs = cast(dict[str, Any], {})
    else:
        kwargs = cast(
            dict[str, Any],
            {"normalize_params": normalize_params, "channel_axis": channel_axis},
        )

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    class_weights_tensor: Any = None
    if class_weights is not None:
        class_weights_tensor = torch.from_numpy(class_weights).to(device).float()  # type: ignore
        print(class_weights_tensor)

    nimg = len(train_data) if train_data is not None else len(train_files)  # type: ignore
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    # Ensure nimg_test is treated as int, defaulting to 0 if None
    nimg_test_val = nimg_test if nimg_test is not None else 0

    nimg_per_epoch_val = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch_val = (
        nimg_test_val if nimg_test_per_epoch is None else nimg_test_per_epoch
    )

    # learning rate schedule
    learning_rate_schedule = np.linspace(0, learning_rate, 10)
    learning_rate_schedule = np.append(
        learning_rate_schedule, learning_rate * np.ones(max(0, n_epochs - 10))
    )
    if n_epochs > 300:
        learning_rate_schedule = learning_rate_schedule[:-100]
        for _ in range(10):
            learning_rate_schedule = np.append(
                learning_rate_schedule, learning_rate_schedule[-1] / 2 * np.ones(10)
            )
    elif n_epochs > 99:
        learning_rate_schedule = learning_rate_schedule[:-50]
        for _ in range(10):
            learning_rate_schedule = np.append(
                learning_rate_schedule, learning_rate_schedule[-1] / 2 * np.ones(5)
            )

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")
    train_logger.info(
        f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
    )
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)

    def _safe_div(n: float, d: float) -> float:
        return float(n / d) if d != 0 else 0.0

    def _compute_binary_metrics(
        pred: "torch.Tensor",  # (B, H, W)
        target: "torch.Tensor",  # (B, H, W)
    ) -> ValidationMetrics:
        """Compute basic binary foreground/background metrics.

        Uses simple thresholds chosen based on value ranges.
        """
        # Choose thresholds robustly for both [0,1] and signed/logit-like ranges.
        tmin = float(target.min().item())
        tmax = float(target.max().item())
        target_thr = 0.5 if (tmin >= 0.0 and tmax <= 1.0) else 0.0

        pmin = float(pred.min().item())
        pmax = float(pred.max().item())
        pred_thr = 0.5 if (pmin >= 0.0 and pmax <= 1.0) else 0.0

        gt = (target > target_thr).to(torch.bool).flatten()
        pr = (pred > pred_thr).to(torch.bool).flatten()

        tp = int(torch.sum(pr & gt).item())
        fp = int(torch.sum(pr & ~gt).item())
        fn = int(torch.sum(~pr & gt).item())
        tn = int(torch.sum(~pr & ~gt).item())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        pixel_accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
        iou = _safe_div(tp, tp + fp + fn)

        return ValidationMetrics(
            pixel_accuracy=pixel_accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            iou=iou,
        )

    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch_val:
            # choose random images for epoch with probability train_probs
            rperm = np.random.choice(
                np.arange(0, nimg), size=(nimg_per_epoch_val,), p=train_probs
            )
        else:
            # otherwise use all images
            rperm = np.random.permutation(np.arange(0, nimg))

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate_schedule[iepoch]  # set learning rate

        net.train()
        for k in range(0, nimg_per_epoch_val, batch_size):
            kend = min(k + batch_size, nimg_per_epoch_val)
            inds = rperm[k:kend]
            imgs, lbls = _get_batch(  # type: ignore
                inds,
                data=train_data,
                labels=train_labels,
                files=train_files,
                labels_files=train_labels_files,
                **kwargs,
            )
            diams = np.array([diam_train[i] for i in inds])
            rsc = (
                diams / net.diam_mean.item()
                if rescale
                else np.ones(len(diams), "float32")
            )
            # augmentations
            imgi, lbl = random_rotate_and_resize(  # type: ignore
                imgs, Y=lbls, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize)
            )[:2]
            imgi = ensure_3_channels_batch(imgi)  # type: ignore
            # network and loss optimization
            x_batch = torch.from_numpy(imgi).to(device)  # type: ignore
            lbl = torch.from_numpy(lbl).to(device)  # type: ignore

            if x_batch.dtype != net.dtype:
                x_batch = x_batch.to(net.dtype)
                lbl = lbl.to(net.dtype)

            y = net(x_batch)[0]
            loss = cast(torch.Tensor, _loss_fn_seg(lbl, y, device))  # type: ignore
            if y.shape[1] > 3:
                loss3 = cast(torch.Tensor, _loss_fn_class(lbl, y, class_weights=class_weights_tensor))  # type: ignore
                loss += loss3
            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()  # type: ignore
            train_loss = loss.item()  # type: ignore
            train_loss *= len(imgi)  # type: ignore

            # keep track of average training loss across epochs
            lavg += train_loss  # type: ignore
            nsum += len(imgi)
            # per epoch training loss
            train_losses[iepoch] += train_loss

            # **CALLBACK: Report batch progress**
            if batch_callback is not None:
                elapsed = time.time() - t0
                batch_idx = k // batch_size
                total_batches = (nimg_per_epoch_val + batch_size - 1) // batch_size
                batch_loss_per_sample = float(
                    loss.item()  # type: ignore
                )  # Loss per sample for this batch
                # Pass None for val_metrics during training
                batch_callback(
                    iepoch + 1,
                    batch_idx,
                    total_batches,
                    batch_loss_per_sample,
                    elapsed,
                    None,
                )

        train_losses[iepoch] /= nimg_per_epoch_val

        # Compute test loss if appropriate
        lavgt = 0.0
        val_metrics: ValidationMetrics | None = None
        # Validate at epoch 0 (first epoch), and then every 10 epochs (10, 20, 30...)
        # iepoch is 0-indexed, so iepoch=9 corresponds to Epoch 10.
        if iepoch == 0 or (iepoch + 1) % 10 == 0:
            if test_data is not None or test_files is not None:
                np.random.seed(42)
                if nimg_test_val != nimg_test_per_epoch_val:
                    rperm = np.random.choice(
                        np.arange(0, nimg_test_val),
                        size=(nimg_test_per_epoch_val,),
                        p=test_probs,
                    )
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test_val))

                # Confusion counts for cellprob foreground/background
                tp = fp = fn = tn = 0

                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm[ibatch : ibatch + batch_size]
                        imgs, lbls = _get_batch(  # type: ignore
                            inds,
                            data=test_data,
                            labels=test_labels,
                            files=test_files,
                            labels_files=test_labels_files,
                            **kwargs,
                        )
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = (
                            diams / net.diam_mean.item()
                            if rescale
                            else np.ones(len(diams), "float32")
                        )
                        imgi, lbl = random_rotate_and_resize(  # type: ignore
                            imgs,
                            Y=lbls,
                            rescale=rsc,
                            scale_range=scale_range,
                            xy=(bsize, bsize),
                        )[:2]
                        imgi = ensure_3_channels_batch(imgi)  # type: ignore
                        x_batch = torch.from_numpy(imgi).to(device)  # type: ignore
                        lbl = torch.from_numpy(lbl).to(device)  # type: ignore

                        if x_batch.dtype != net.dtype:
                            x_batch = x_batch.to(net.dtype)
                            lbl = lbl.to(net.dtype)

                        y = net(x_batch)[0]
                        loss = cast(torch.Tensor, _loss_fn_seg(lbl, y, device))  # type: ignore
                        if y.shape[1] > 3:
                            loss3 = cast(torch.Tensor, _loss_fn_class(lbl, y, class_weights=class_weights_tensor))  # type: ignore
                            loss += loss3

                        # Validation performance on Cellpose cell-prob channel.
                        # Channel layout is typically: [flow_y, flow_x, cellprob, ...]
                        pred_cellprob = y[:, 2].detach().float().cpu()

                        # Handle different label formats (flows vs masks)
                        if lbl.shape[1] >= 3:
                            # Labels are flows: [flow_y, flow_x, cellprob, ...]
                            true_cellprob = lbl[:, 2].detach().float().cpu()
                        elif lbl.shape[1] == 1:
                            # Labels are masks: [mask]
                            # Create binary cellprob from mask (0 is background)
                            true_cellprob = (lbl[:, 0] > 0).float().cpu()
                        else:
                            # Unexpected shape (e.g. 2 channels), skip metrics
                            train_logger.warning(
                                f"Skipping metrics: unexpected lbl shape {lbl.shape}"
                            )
                            continue

                        batch_metrics = _compute_binary_metrics(
                            pred_cellprob, true_cellprob
                        )
                        # Reconstruct confusion from metrics is lossy; instead
                        # accumulate directly here for stability.
                        # Use same thresholding logic as _compute_binary_metrics.
                        tmin = float(true_cellprob.min().item())
                        tmax = float(true_cellprob.max().item())
                        target_thr = 0.5 if (tmin >= 0.0 and tmax <= 1.0) else 0.0
                        pmin = float(pred_cellprob.min().item())
                        pmax = float(pred_cellprob.max().item())
                        pred_thr = 0.5 if (pmin >= 0.0 and pmax <= 1.0) else 0.0
                        gt = (true_cellprob > target_thr).to(torch.bool).flatten()
                        pr = (pred_cellprob > pred_thr).to(torch.bool).flatten()
                        tp += int(torch.sum(pr & gt).item())
                        fp += int(torch.sum(pr & ~gt).item())
                        fn += int(torch.sum(~pr & gt).item())
                        tn += int(torch.sum(~pr & ~gt).item())

                        # **CALLBACK: Report batch progress (during validation)**
                        if batch_callback is not None:
                            elapsed = time.time() - t0
                            # Validation batches - mapping batch_idx
                            batch_idx = ibatch // batch_size
                            total_batches = (len(rperm) + batch_size - 1) // batch_size
                            batch_loss_per_sample = loss.item()  # type: ignore
                            batch_callback(
                                iepoch + 1,
                                batch_idx,
                                total_batches,
                                batch_loss_per_sample,
                                elapsed,
                                batch_metrics,
                            )

                        test_loss = loss.item()  # type: ignore
                        test_loss *= len(imgi)  # type: ignore
                        lavgt += test_loss  # type: ignore

                lavgt /= len(rperm)  # type: ignore
                test_losses[iepoch] = lavgt

                # Finalize metrics if we collected any pixels.
                if (tp + fp + fn + tn) > 0:
                    precision = _safe_div(tp, tp + fp)
                    recall = _safe_div(tp, tp + fn)
                    f1 = _safe_div(2 * precision * recall, precision + recall)
                    pixel_accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
                    iou = _safe_div(tp, tp + fp + fn)
                    val_metrics = ValidationMetrics(
                        pixel_accuracy=pixel_accuracy,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        iou=iou,
                    )

            lavg /= nsum  # type: ignore
            if val_metrics is not None:
                train_logger.info(
                    f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, "
                    f"val_acc={val_metrics.get('pixel_accuracy', 0.0):.4f}, "
                    f"val_p={val_metrics.get('precision', 0.0):.4f}, "
                    f"val_r={val_metrics.get('recall', 0.0):.4f}, "
                    f"val_f1={val_metrics.get('f1', 0.0):.4f}, "
                    f"val_iou={val_metrics.get('iou', 0.0):.4f}, "
                    f"LR={learning_rate_schedule[iepoch]:.6f}, time {time.time()-t0:.2f}s"
                )
            else:
                train_logger.info(
                    f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={learning_rate_schedule[iepoch]:.6f}, time {time.time()-t0:.2f}s"
                )
            lavg, nsum = 0, 0

        # **CALLBACK: Report epoch progress**
        if epoch_callback is not None:
            elapsed = time.time() - t0
            epoch_callback(
                iepoch + 1, train_losses[iepoch], lavgt, elapsed, val_metrics
            )

    # Save final model only (no intermediate snapshots)
    train_logger.info(f"saving final network parameters to {filename}")
    net.save_model(filename)

    if original_net_dtype is not None:
        net.dtype = original_net_dtype
        net.to(original_net_dtype)

    return filename, train_losses, test_losses


def run_blocking_task(
    cellpose_model: CellposeModel,
    model_save_path: Path,
    dataset_split: DatasetSplit,
    training_params: TrainingParams,
) -> tuple[Path, Any, Any]:
    """Run the blocking training task."""
    import time

    session_id = training_params["session_id"]

    # Calculate dataset sizes
    n_train = len(dataset_split["train_files"])
    n_test = (
        len(dataset_split["test_files"])
        if dataset_split["test_files"] is not None
        else 0
    )

    # Record start time
    start_time = datetime.now(tz=timezone.utc)
    start_time_str = start_time.isoformat()
    t0 = time.time()

    # Update status with initial training info
    update_status(
        session_id,
        StatusType.RUNNING,
        f"Training started (epoch 0/{training_params['n_epochs']})",
        n_train=n_train,
        n_test=n_test,
        start_time=start_time_str,
        current_epoch=0,
        total_epochs=training_params["n_epochs"],
        elapsed_seconds=0.0,
    )

    # Lists to accumulate training metrics during epoch callbacks
    accumulated_train_losses: list[float] = []
    accumulated_test_losses: list[float] = []
    accumulated_test_metrics: list[ValidationMetrics | None] = []

    # Define batch callback for within-epoch progress updates
    last_batch_update = [0]  # Use list to allow modification in nested function

    def batch_callback(
        epoch: int,
        batch_idx: int,
        total_batches: int,
        batch_loss: float,
        elapsed_seconds: float,
        val_metrics: ValidationMetrics | None = None,
    ) -> None:
        """Update status after each batch (throttled to every 10 batches)."""
        # Update every 10 batches or on first/last batch to avoid too frequent updates
        if batch_idx % 10 == 0 or batch_idx == 0 or batch_idx == total_batches - 1:
            # Prepare message
            stage = "Validating" if val_metrics is not None else "Training"
            msg = f"{stage} epoch {epoch}/{training_params['n_epochs']} (batch {batch_idx + 1}/{total_batches})"
            if val_metrics:
                msg += f" - Val Acc: {val_metrics.get('pixel_accuracy', 0):.4f}"

            update_status(
                session_id,
                StatusType.RUNNING,
                msg,
                train_losses=(
                    accumulated_train_losses.copy()
                    if accumulated_train_losses
                    else [0.0] * epoch
                ),
                test_losses=(
                    accumulated_test_losses.copy() if accumulated_test_losses else []
                ),
                test_metrics=(
                    accumulated_test_metrics.copy() if accumulated_test_metrics else []
                ),
                n_train=n_train,
                n_test=n_test,
                start_time=start_time_str,
                current_epoch=epoch,
                total_epochs=training_params["n_epochs"],
                elapsed_seconds=elapsed_seconds,
                current_batch=batch_idx,
                total_batches=total_batches,
            )
            last_batch_update[0] = batch_idx

    # Define epoch callback for real-time progress updates
    def epoch_callback(
        epoch: int,
        train_loss: float,
        test_loss: float,
        elapsed_seconds: float,
        test_metrics: ValidationMetrics | None,
    ) -> None:
        """Update status after each epoch."""
        # Accumulate losses
        accumulated_train_losses.append(train_loss)
        accumulated_test_losses.append(test_loss)
        accumulated_test_metrics.append(test_metrics)

        update_status(
            session_id,
            StatusType.RUNNING,
            f"Training in progress (epoch {epoch}/{training_params['n_epochs']})",
            train_losses=accumulated_train_losses.copy(),
            test_losses=accumulated_test_losses.copy(),
            test_metrics=accumulated_test_metrics.copy(),
            n_train=n_train,
            n_test=n_test,
            start_time=start_time_str,
            current_epoch=epoch,
            total_epochs=training_params["n_epochs"],
            elapsed_seconds=elapsed_seconds,
        )

    try:
        seg_result = train_seg_with_callbacks(
            cellpose_model.net,
            **dataset_split,
            save_path=model_save_path,
            n_epochs=training_params["n_epochs"],
            learning_rate=training_params["learning_rate"],
            weight_decay=training_params["weight_decay"],
            model_name="model",
            min_train_masks=training_params["min_train_masks"],
            epoch_callback=epoch_callback,
            batch_callback=batch_callback,
        )
    except Exception as e:
        elapsed = time.time() - t0
        update_status(
            session_id,
            StatusType.FAILED,
            f"Training failed with exception: {e}",
            n_train=n_train,
            n_test=n_test,
            start_time=start_time_str,
            total_epochs=training_params["n_epochs"],
            elapsed_seconds=elapsed,
        )
        raise

    # Calculate elapsed time
    elapsed = time.time() - t0

    # Extract training metrics from the result
    _, train_losses, test_losses = seg_result

    # Convert numpy arrays to lists for JSON serialization
    train_losses_list = (
        train_losses.tolist() if hasattr(train_losses, "tolist") else list(train_losses)
    )
    test_losses_list = (
        test_losses.tolist() if hasattr(test_losses, "tolist") else list(test_losses)
    )

    # Check if we're continuing from a previous session and inherit training history
    model_param = training_params["model"]
    if isinstance(model_param, Path):
        # Model is from a previous session - load and append to previous history
        previous_session_id = model_param.parent.name
        previous_status_path = get_status_path(previous_session_id)
        if previous_status_path.exists():
            try:
                with previous_status_path.open("r", encoding="utf-8") as f:
                    previous_status = json.load(f)
                    previous_train_losses = previous_status.get("train_losses", [])
                    previous_test_losses = previous_status.get("test_losses", [])

                    # Append new losses to previous history
                    train_losses_list = previous_train_losses + train_losses_list
                    test_losses_list = previous_test_losses + test_losses_list

                    logger.info(
                        "Session %s: Inherited %d epochs of training history from session %s",
                        session_id,
                        len(previous_train_losses),
                        previous_session_id,
                    )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    "Session %s: Could not load previous training history: %s",
                    session_id,
                    str(e),
                )

    # Count actual completed epochs (non-zero losses)
    completed_epochs = len([loss for loss in train_losses_list if loss > 0])

    update_status(
        session_id,
        StatusType.COMPLETED,
        "Training completed successfully",
        train_losses=train_losses_list,
        test_losses=test_losses_list,
        test_metrics=accumulated_test_metrics.copy(),
        n_train=n_train,
        n_test=n_test,
        start_time=start_time_str,
        current_epoch=completed_epochs,
        total_epochs=training_params["n_epochs"],
        elapsed_seconds=elapsed,
    )

    # Mark model as modified (needs re-export if previously exported)
    status_path = get_status_path(session_id)
    with status_path.open("r+", encoding="utf-8") as f:
        status_data = json.load(f)
        status_data["model_modified"] = True
        f.seek(0)
        f.write(json.dumps(status_data))
        f.truncate()

    append_info(
        session_id,
        f"Training completed successfully in {elapsed:.1f}s ({completed_epochs} epochs).",
        with_time=True,
    )

    return seg_result


async def finetune_cellpose(
    training_params: TrainingParams,
    executor: ThreadPoolExecutor,
) -> tuple[Path, list[float], list[float]]:
    """Prepare data, build model, and call Cellpose train_seg asynchronously."""
    session_id = training_params["session_id"]
    try:
        logger.info("Session %s: Getting artifact cache directory", session_id)
        data_save_path = artifact_cache_dir(training_params["artifact_id"])

        logger.info("Session %s: Listing and matching training pairs", session_id)
        update_status(
            session_id,
            StatusType.PREPARING,
            "Listing files and matching training pairs from artifact...",
        )
        train_pairs, test_pairs = await make_training_pairs(
            training_params, data_save_path
        )

        logger.info("Session %s: Creating dataset split", session_id)
        update_status(
            session_id,
            StatusType.PREPARING,
            "Creating dataset split...",
        )
        dataset_split = create_dataset_split(train_pairs, test_pairs)

        model_save_path = get_session_path(session_id)
        model_save_path.mkdir(parents=True, exist_ok=True)

        param_str = str(training_params)
        append_info(
            session_id,
            f"Training started. Parameters:\n{param_str}",
            with_time=True,
        )

        logger.info("Session %s: Loading Cellpose model", session_id)
        update_status(
            session_id,
            StatusType.PREPARING,
            "Loading Cellpose model...",
        )
        cellpose_model = load_model(training_params["model"])

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            run_blocking_task,
            cellpose_model,
            model_save_path,
            dataset_split,
            training_params,
        )
    except Exception as e:
        update_status(
            session_id,
            StatusType.FAILED,
            f"Training preparation failed: {str(e)}",
        )
        logger.exception(
            "Training failed during preparation for session %s", session_id
        )
        raise


async def launch_training_task(
    training_params: TrainingParams,
    executor: ThreadPoolExecutor,
) -> None:
    """Launch the Cellpose finetuning task asynchronously."""
    session_id = training_params["session_id"]
    logger.info("launch_training_task started for session %s", session_id)
    try:
        await finetune_cellpose(training_params, executor)
        logger.info(
            "launch_training_task completed successfully for session %s", session_id
        )
    except Exception as e:
        logger.exception(
            "launch_training_task failed for session %s: %s", session_id, str(e)
        )
        raise


def load_model(identifier: str | Path) -> CellposeModel:
    """Load a Cellpose model by builtin name or by local file path.

    If `identifier` points to an existing file, it is treated as a path to a
    finetuned model. Otherwise, it is treated as a builtin model name
    (e.g., "cyto3").
    """
    from cellpose import core, models  # type: ignore

    use_gpu = core.use_gpu()

    if isinstance(identifier, Path):
        return models.CellposeModel(gpu=use_gpu, pretrained_model=str(identifier))  # type: ignore

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


def parse_path_pattern(path_pattern: str) -> tuple[str, str]:
    """Parse a path pattern into folder and filename pattern.

    Args:
        path_pattern: Either a folder path ending with '/' or a path with wildcard pattern

    Returns:
        Tuple of (folder_path, filename_pattern)

    Examples:
        >>> parse_path_pattern("images/folder/")
        ('images/folder/', '*')
        >>> parse_path_pattern("images/folder/*.ome.tif")
        ('images/folder/', '*.ome.tif')
        >>> parse_path_pattern("images/folder/*_mask.ome.tif")
        ('images/folder/', '*_mask.ome.tif')
    """
    if path_pattern.endswith("/"):
        # Folder path - assume all files with same names
        return path_pattern, "*"

    # Extract folder and pattern
    path_obj = Path(path_pattern)
    folder = str(path_obj.parent) + "/"
    pattern = path_obj.name

    return folder, pattern


def extract_pattern_match(filename: str, pattern: str) -> str | None:
    """Extract the wildcard part from a filename based on a pattern.

    Args:
        filename: The filename to match (e.g., "t0000.ome.tif")
        pattern: The pattern with * wildcard (e.g., "*.ome.tif")

    Returns:
        The matched wildcard part (e.g., "t0000"), or None if no match

    Examples:
        >>> extract_pattern_match("t0000.ome.tif", "*.ome.tif")
        't0000'
        >>> extract_pattern_match("t0000_mask.ome.tif", "*_mask.ome.tif")
        't0000'
        >>> extract_pattern_match("image.png", "*.ome.tif")
        None
    """
    import re

    # Convert pattern with * to regex
    # Escape all regex special characters except *
    pattern_escaped = re.escape(pattern)
    # Replace escaped \* with (.+) to capture the wildcard part
    pattern_regex = pattern_escaped.replace(r"\*", "(.+)")
    # Match from start to end
    pattern_regex = f"^{pattern_regex}$"

    match = re.match(pattern_regex, filename)
    if match:
        return match.group(1)
    return None


async def list_artifact_files(
    artifact: AsyncHyphaArtifact,
    folder_path: str,
) -> list[str]:
    """List all files in an artifact folder using AsyncHyphaArtifact.ls().

    Note: Currently limited to first 1000 files. Future versions will support
    pagination for larger directories.

    Args:
        artifact: The AsyncHyphaArtifact client
        folder_path: Path to the folder (should end with '/')

    Returns:
        List of filenames (not full paths, just basenames)
    """
    if not folder_path.endswith("/"):
        folder_path = folder_path + "/"

    try:
        # ls() returns a list of file info dicts
        files = await artifact.ls(folder_path)
        # Extract filenames (basenames only)
        filenames: list[str] = []
        for file_info in files:
            # file_info is typically a dict with 'name' or 'path' key
            # Get the basename
            if isinstance(file_info, dict):
                file_info_dict = cast(dict[str, Any], file_info)
                path = file_info_dict.get("name") or file_info_dict.get("path", "")
            else:
                path = str(file_info)

            # Extract basename
            basename = Path(str(path)).name
            if basename and basename != ".":
                filenames.append(basename)

        logger.info(f"Listed {len(filenames)} files from {folder_path}")
        return filenames
    except Exception as e:
        logger.error(f"Failed to list files from {folder_path}: {e}")
        raise


def match_image_annotation_pairs(
    image_files: list[str],
    annotation_files: list[str],
    image_pattern: str,
    annotation_pattern: str,
) -> list[tuple[str, str]]:
    """Match image and annotation files based on patterns.

    Args:
        image_files: List of image filenames
        annotation_files: List of annotation filenames
        image_pattern: Pattern for images (e.g., "*.ome.tif")
        annotation_pattern: Pattern for annotations (e.g., "*_mask.ome.tif")

    Returns:
        List of (image_filename, annotation_filename) pairs

    Example:
        >>> images = ["t0000.ome.tif", "t0001.ome.tif"]
        >>> annots = ["t0000_mask.ome.tif", "t0001_mask.ome.tif"]
        >>> match_image_annotation_pairs(images, annots, "*.ome.tif", "*_mask.ome.tif")
        [("t0000.ome.tif", "t0000_mask.ome.tif"), ("t0001.ome.tif", "t0001_mask.ome.tif")]
    """
    # Build a dict mapping wildcard matches to annotation files
    annot_map: dict[str, str] = {}
    for annot_file in annotation_files:
        match = extract_pattern_match(annot_file, annotation_pattern)
        if match:
            annot_map[match] = annot_file

    # Match images to annotations
    pairs: list[tuple[str, str]] = []
    for image_file in image_files:
        match = extract_pattern_match(image_file, image_pattern)
        if match and match in annot_map:
            pairs.append((image_file, annot_map[match]))

    logger.info(
        f"Matched {len(pairs)} pairs from {len(image_files)} images "
        f"and {len(annotation_files)} annotations"
    )

    return pairs


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
        logger.info("Downloading %d files from artifact", len(missing_rpaths))
        try:
            # Add a timeout to prevent indefinite hanging (10 minutes)
            await asyncio.wait_for(
                artifact.get(missing_rpaths, missing_lpaths, on_error="ignore"),
                timeout=600,
            )
            logger.info("Download completed successfully")
        except asyncio.TimeoutError:
            logger.error("Download timed out after 600 seconds")
            raise RuntimeError(
                f"Download timed out after 600 seconds. "
                f"Attempted to download {len(missing_rpaths)} files."
            ) from None
        except Exception as e:
            logger.exception("Download failed: %s", str(e))
            raise

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


def create_dataset_split(
    train_pairs: list[TrainingPair],
    test_pairs: list[TrainingPair],
) -> DatasetSplit:
    """Convert train and test pairs into DatasetSplit format for Cellpose.

    Args:
        train_pairs: List of training (image, annotation) pairs
        test_pairs: List of test (image, annotation) pairs (can be empty)

    Returns:
        DatasetSplit with train and test files

    Raises:
        ValueError: If no training pairs are provided
    """
    if not train_pairs or len(train_pairs) < 1:
        error_msg = "No training pairs found. At least one training sample is required."
        raise ValueError(error_msg)

    dataset_split = DatasetSplit(
        train_files=[pair["image"] for pair in train_pairs],
        train_labels_files=[pair["annotation"] for pair in train_pairs],
        test_files=[pair["image"] for pair in test_pairs] if test_pairs else None,
        test_labels_files=(
            [pair["annotation"] for pair in test_pairs] if test_pairs else None
        ),
    )

    logger.info(
        f"Created dataset split: {len(dataset_split['train_files'])} train, "
        f"{len(dataset_split['test_files']) if dataset_split['test_files'] is not None else 0} test"
    )

    return dataset_split


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
) -> tuple[list[TrainingPair], list[TrainingPair]]:
    """List files from artifact folders and match image-annotation pairs.

    Returns:
        Tuple of (train_pairs, test_pairs). test_pairs is empty if test folders not specified.
    """
    artifact = await make_artifact_client(
        config["artifact_id"],
        config["server_url"],
    )

    # Parse training path patterns
    train_img_folder, train_img_pattern = parse_path_pattern(config["train_images"])
    train_ann_folder, train_ann_pattern = parse_path_pattern(
        config["train_annotations"]
    )

    # List training files
    logger.info("Listing training images from %s", train_img_folder)
    train_image_files = await list_artifact_files(artifact, train_img_folder)
    logger.info("Listing training annotations from %s", train_ann_folder)
    train_annotation_files = await list_artifact_files(artifact, train_ann_folder)

    # Match training pairs
    train_matched = match_image_annotation_pairs(
        train_image_files,
        train_annotation_files,
        train_img_pattern,
        train_ann_pattern,
    )

    # Build full paths for training files
    train_image_paths = [Path(train_img_folder) / img for img, _ in train_matched]
    train_annotation_paths = [Path(train_ann_folder) / ann for _, ann in train_matched]

    # Apply n_samples if specified
    if config["n_samples"] is not None and config["n_samples"] < len(train_image_paths):
        train_image_paths, train_annotation_paths = get_training_subset(
            train_image_paths,
            train_annotation_paths,
            config["n_samples"],
        )

    # Download training pairs
    train_pairs = await download_pairs_from_artifact(
        artifact,
        save_path,
        train_image_paths,
        train_annotation_paths,
    )

    # Handle test pairs if specified
    test_pairs: list[TrainingPair] = []
    if config["test_images"] and config["test_annotations"]:
        # Parse test path patterns
        test_img_folder, test_img_pattern = parse_path_pattern(config["test_images"])
        test_ann_folder, test_ann_pattern = parse_path_pattern(
            config["test_annotations"]
        )

        logger.info("Listing test images from %s", test_img_folder)
        test_image_files = await list_artifact_files(artifact, test_img_folder)
        logger.info("Listing test annotations from %s", test_ann_folder)
        test_annotation_files = await list_artifact_files(artifact, test_ann_folder)

        # Match test pairs
        test_matched = match_image_annotation_pairs(
            test_image_files,
            test_annotation_files,
            test_img_pattern,
            test_ann_pattern,
        )

        # Build full paths for test files
        test_image_paths = [Path(test_img_folder) / img for img, _ in test_matched]
        test_annotation_paths = [Path(test_ann_folder) / ann for _, ann in test_matched]

        # Download test pairs
        test_pairs = await download_pairs_from_artifact(
            artifact,
            save_path,
            test_image_paths,
            test_annotation_paths,
        )

    return train_pairs, test_pairs


# ---------------------------------------------------------------------------
# Model Export helpers
# ---------------------------------------------------------------------------


async def create_test_samples(
    session_id: str,
    training_params: TrainingParams,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Create test input and output samples from training data.

    Uses the last training image to generate test samples for model validation.

    Args:
        session_id: Training session ID
        training_params: Training parameters containing artifact and paths

    Returns:
        Tuple of (test_input, test_output) as numpy arrays
    """

    # Get artifact and paths
    artifact = await make_artifact_client(
        training_params["artifact_id"],
        training_params["server_url"],
    )
    save_path = artifact_cache_dir(training_params["artifact_id"])

    # Parse training path patterns
    train_img_folder, train_img_pattern = parse_path_pattern(
        training_params["train_images"]
    )
    train_ann_folder, train_ann_pattern = parse_path_pattern(
        training_params["train_annotations"]
    )

    # List training files
    train_image_files = await list_artifact_files(artifact, train_img_folder)
    train_annotation_files = await list_artifact_files(artifact, train_ann_folder)

    # Match training pairs
    train_matched = match_image_annotation_pairs(
        train_image_files,
        train_annotation_files,
        train_img_pattern,
        train_ann_pattern,
    )

    if not train_matched:
        raise ValueError("No training pairs found for test sample generation")

    # Use the last pair as test sample
    last_img_file, last_ann_file = train_matched[-1]

    # Download files if needed
    img_path = Path(train_img_folder) / last_img_file
    ann_path = Path(train_ann_folder) / last_ann_file

    await download_pairs_from_artifact(
        artifact,
        save_path,
        [img_path],
        [ann_path],
    )

    # Load files
    local_img = to_local_path(save_path, img_path)
    local_ann = to_local_path(save_path, ann_path)

    # Use PIL to support multiple formats (PNG, TIF, etc.)
    from PIL import Image

    pil_img = Image.open(local_img)
    test_input = np.array(pil_img)
    test_output = np.array(Image.open(local_ann))

    logger.info(
        f"Loaded test input from PIL: shape={test_input.shape}, dtype={test_input.dtype}, PIL mode={pil_img.mode}"
    )

    # PIL returns images in (H, W, C) format for RGB/RGBA, (H, W) for grayscale
    # ensure_3_channels expects (C, H, W) format
    if test_input.ndim == 2:
        # Grayscale (H, W) - ensure_3_channels can handle this
        pass
    elif test_input.ndim == 3 and test_input.shape[2] in [1, 3, 4]:
        # Image is in (H, W, C) format, transpose to (C, H, W)
        logger.info(f"Transposing from (H,W,C) to (C,H,W): {test_input.shape} -> ")
        test_input = np.transpose(test_input, (2, 0, 1))
        logger.info(f"{test_input.shape}")

    # Ensure 3 channels for input (required by Cellpose-SAM)
    logger.info(f"Before ensure_3_channels: shape={test_input.shape}")
    test_input = ensure_3_channels(test_input)
    logger.info(f"After ensure_3_channels: shape={test_input.shape}")

    logger.info(
        f"Created test samples: input shape {test_input.shape}, output shape {test_output.shape}"
    )

    return test_input, test_output


async def generate_cover_image(
    test_input: npt.NDArray[Any],
    test_output: npt.NDArray[Any],
    output_path: Path,
    model_name: str = "Cellpose Model",
    session_id: str = "",
    training_info: dict[str, Any] | SessionStatus | None = None,
) -> None:
    """Generate a side-by-side cover image with model metadata in title.

    Args:
        test_input: Test input image (C, H, W)
        test_output: Test output mask (H, W)
        output_path: Path to save cover image
        model_name: Name of the model
        session_id: Training session ID
        training_info: Dictionary with training metadata (epochs, samples, loss, etc.)
    """
    from datetime import datetime

    import matplotlib.pyplot as plt

    logger.info(
        f"Generating cover: input shape {test_input.shape}, output shape {test_output.shape}"
    )

    training_info = training_info or {}

    # Convert input to displayable format (H, W, C)
    if test_input.ndim == 3:
        display_input = np.transpose(test_input, (1, 2, 0))
    else:
        display_input = test_input

    # Normalize input for display
    if display_input.max() > 1:
        display_input = (display_input - display_input.min()) / (
            display_input.max() - display_input.min()
        )

    # Convert to RGB if grayscale
    if display_input.shape[2] == 1:
        display_input = np.repeat(display_input, 3, axis=2)

    # Create colored mask overlay
    mask_colored = np.zeros((*test_output.shape, 4))
    unique_labels = np.unique(test_output)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background

    # Use a colormap for the masks
    cmap = plt.get_cmap("tab20")
    for i, label in enumerate(unique_labels):
        color = cmap(i % 20)
        mask_colored[test_output == label] = color

    # Create figure with side-by-side images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # type: ignore

    # Plot input
    ax1.imshow(display_input)  # type: ignore
    ax1.set_title("Input Image", fontsize=14)  # type: ignore
    ax1.axis("off")  # type: ignore

    # Plot output mask only (no background image)
    ax2.imshow(mask_colored)  # type: ignore
    ax2.set_title(f"Segmentation ({len(unique_labels)} objects)", fontsize=14)  # type: ignore
    ax2.axis("off")  # type: ignore

    # Add overall title with model metadata
    date_str = datetime.now().strftime("%Y-%m-%d")
    short_id = session_id[:8] if session_id else "N/A"

    # Get training metrics
    n_train = training_info.get("n_train", "N/A")
    epochs = training_info.get("total_epochs", "N/A")
    train_losses = training_info.get("train_losses", [])
    if isinstance(train_losses, list) and len(cast(list[Any], train_losses)) > 0:
        final_loss = f"{train_losses[-1]:.4f}"
    else:
        final_loss = "N/A"

    # Create title with model info
    title_lines = [
        f"{model_name}",
        f"Cellpose-SAM | ID: {short_id} | Date: {date_str}",
        f"Samples: {n_train} | Epochs: {epochs} | Loss: {final_loss}",
    ]

    fig.suptitle("\n".join(title_lines), fontsize=12, fontweight="bold", y=0.98)  # type: ignore

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")  # type: ignore
    plt.close()

    logger.info(f"Generated cover image at {output_path}")


def generate_rdf_yaml(
    model_name: str,
    session_id: str,
    model_weights_filename: str,
    model_py_filename: str,
    test_input_filename: str,
    test_output_filename: str,
    training_params: TrainingParams,
    test_input_shape: tuple[int, ...],
    test_output_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Generate BioImage.io RDF YAML structure.

    Args:
        model_name: Name of the model
        session_id: Training session ID
        model_weights_filename: Filename of model weights
        model_py_filename: Filename of model.py
        test_input_filename: Filename of test input
        test_output_filename: Filename of test output
        training_params: Training parameters
        test_input_shape: Shape of test input (C, H, W)
        test_output_shape: Shape of test output (H, W)

    Returns:
        Dictionary representing RDF YAML structure
    """
    import torch

    return {
        "name": model_name,
        "description": f"Cellpose-SAM model fine-tuned on custom dataset (session: {session_id[:8]})",
        "authors": [
            {
                "name": "BioEngine",
                "affiliation": "AI Cell Lab",
            }
        ],
        "cite": [
            {
                "text": "Stringer, C., Wang, T., Michaelos, M. et al. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021).",
                "doi": "10.1038/s41592-020-01018-x",
            },
            {
                "text": "Pachitariu, M., Stringer, C. Cellpose 2.0: how to train your own model. Nat Methods 19, 1634–1641 (2022).",
                "doi": "10.1038/s41592-022-01663-4",
            },
            {
                "text": "Stringer, Carsen, and Marius Pachitariu. Cellpose3: one-click image restoration for improved cellular segmentation. bioRxiv (2024).",
                "doi": "10.1101/2024.02.10.579780",
            },
        ],
        "license": "BSD-3-Clause",
        "tags": [
            "Cellpose",
            "Cellpose-SAM",
            "Cell Segmentation",
            "Segmentation",
            "Fine-tuned",
        ],
        "parent": training_params.get("model", "cpsam"),  # Track model lineage
        "version": "0.1.0",
        "format_version": "0.5.6",
        "type": "model",
        "id": session_id,
        "id_emoji": "🔬",
        "documentation": DOC_FILENAME,
        "inputs": [
            {
                "id": "input",
                "axes": [
                    {"type": "batch"},
                    {
                        "type": "channel",
                        "channel_names": (
                            ["r", "g", "b"] if test_input_shape[0] == 3 else ["channel"]
                        ),
                    },
                    {"size": test_input_shape[1], "id": "y", "type": "space"},
                    {"size": test_input_shape[2], "id": "x", "type": "space"},
                ],
                "test_tensor": {
                    "source": test_input_filename,
                },
            }
        ],
        "outputs": [
            {
                "id": "masks",
                "axes": [
                    {"type": "batch"},
                    {"size": test_output_shape[0], "id": "y", "type": "space"},
                    {"size": test_output_shape[1], "id": "x", "type": "space"},
                ],
                "test_tensor": {
                    "source": test_output_filename,
                },
            }
        ],
        "weights": {
            "pytorch_state_dict": {
                "source": model_weights_filename,
                "architecture": {
                    "source": model_py_filename,
                    "callable": "CellposeSAMWrapper",
                    "kwargs": {
                        "model_type": "cpsam",
                        "diam_mean": float(
                            cast(dict[str, Any], training_params).get("diam_mean", 30.0)
                        ),
                        "cp_batch_size": 8,
                        "channels": [0, 0],
                        "flow_threshold": 0.4,
                        "cellprob_threshold": 0.0,
                        "stitch_threshold": 0.0,
                        "estimate_diam": False,
                        "normalize": True,
                        "do_3D": False,
                        "gpu": False,
                        "use_bfloat16": True,
                    },
                },
                "pytorch_version": str(torch.__version__),
            }
        },
        "config": {
            "bioimageio": {
                "reproducibility_tolerance": [
                    {
                        "relative_tolerance": 0.01,
                        "absolute_tolerance": 0.001,
                        "mismatched_elements_per_million": 20,
                    }
                ]
            }
        },
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


async def _images_from_artifact(
    artifact_id: str,
    image_paths: list[str],
    server_url: str,
) -> list[npt.NDArray[Any]]:
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
    imgs: list[npt.NDArray[Any]] = []
    for rel in image_paths:
        local_img = cache_dir / rel
        imgs.append(imread(local_img))
    return imgs


def _predict_and_encode(
    *,
    model: CellposeModel,
    images: list[npt.NDArray[Any]],
    image_paths: list[str],
    diameter: float | None,
    flow_threshold: float,
    cellprob_threshold: float,
    niter: int | None,
    return_flows: bool = False,
) -> list[PredictionItemModel]:
    """Run model on images and return encoded mask payloads.

    Images are expected to have 3 channels (preprocessed with ensure_3_channels).

    Args:
        model: Cellpose model to use for inference
        images: List of input images
        image_paths: List of image path identifiers
        diameter: Cell diameter for inference
        flow_threshold: Flow error threshold
        cellprob_threshold: Cell probability threshold
        niter: Number of iterations for dynamics
        return_flows: If True, include flows in the output (HSV flow, XY flows, cellprob, final positions)

    Returns:
        List of prediction items with masks and optionally flows
    """
    out: list[PredictionItemModel] = []
    for image, path in zip(images, image_paths):
        # Ensure image has 3 channels (required by Cellpose 4.0.7)
        image_3ch = ensure_3_channels(image)

        # model.eval returns (masks, flows, styles)
        # flows = [HSV flow, XY flows, cellprob, final pixel locations]
        masks, flows, _styles = model.eval(  # type: ignore
            [image_3ch],
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            niter=niter,
        )

        # Cast masks to ndarray for type checking
        masks = cast(npt.NDArray[Any] | list[npt.NDArray[Any]], masks)

        # Process masks
        mask_np = masks[0] if isinstance(masks, list) else masks
        if not isinstance(mask_np, np.ndarray):  # type: ignore
            mask_np = cast(npt.NDArray[Any], np.asarray(mask_np))
        if mask_np.ndim >= NDIM_3D_THRESHOLD and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        if not np.issubdtype(mask_np.dtype, np.integer):
            mask_np = mask_np.astype(np.int32, copy=False)

        # Create output item
        out_item = PredictionItemModel(
            input_path=path,  # type: ignore
        )

        # Optionally add flows
        if return_flows:
            # flows is a list containing [HSV flow, XY flows, cellprob, final positions]
            flow_list = cast(
                list[Any],
                (
                    flows[0]
                    if isinstance(flows, list) and len(cast(list[Any], flows)) > 0
                    else flows
                ),
            )
            out_item["flows"] = flow_list

        out.append(out_item)
    return out


# ---------------------------------------------------------------------------
# Ray Serve deployment
# ---------------------------------------------------------------------------
@serve.deployment(  # type: ignore
    ray_actor_options={
        "num_gpus": 0.75,
        "num_cpus": 4,
        "memory": 12 * GB,
        "runtime_env": {
            "pip": [
                "cellpose==4.0.7",
                "numpy==1.26.4",
                "tifffile",
                "hypha-artifact==0.1.2",
            ],
        },
    },
    max_ongoing_requests=1,
    max_queued_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
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
    """

    pretrained_models: list[str]
    executors: dict[str, ThreadPoolExecutor]
    tasks: dict[str, asyncio.Task[Any]]
    _session_lock: asyncio.Lock

    def __init__(self) -> None:
        """Initialize directories and defaults for the service."""
        get_sessions_path().mkdir(parents=True, exist_ok=True)
        self.pretrained_models = PretrainedModel.values()
        self.executors = {}
        self.tasks = {}
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

    @schema_method(arbitrary_types_allowed=True)  # type: ignore
    async def start_training(
        self,
        artifact: str = Field(
            description=(
                "Artifact identifier 'workspace/alias' containing TIFF images and "
                "annotations for training."
            ),
            examples=["ri-scale/zarr-demo"],
        ),
        train_images: str = Field(
            description=(
                "Path to training images. Can be either:\n"
                "1. Folder path ending with '/' (assumes same filenames as annotations)\n"
                "2. Path pattern with wildcard (e.g., 'images/folder/*.ome.tif')"
            ),
            examples=[
                "images/108bb69d-2e52-4382-8100-e96173db24ee/",
                "images/folder/*.ome.tif",
            ],
        ),
        train_annotations: str = Field(
            description=(
                "Path to training annotations. Can be either:\n"
                "1. Folder path ending with '/' (assumes same filenames as images)\n"
                "2. Path pattern with wildcard (e.g., 'annotations/folder/*_mask.ome.tif')\n"
                "The * part in patterns must match between images and annotations."
            ),
            examples=[
                "annotations/108bb69d-2e52-4382-8100-e96173db24ee/",
                "annotations/folder/*_mask.ome.tif",
            ],
        ),
        test_images: str | None = Field(
            None,
            description=("Optional path to test images. Same format as train_images."),
            examples=["images/test/", "images/test/*.ome.tif"],
        ),
        test_annotations: str | None = Field(
            None,
            description=(
                "Optional path to test annotations. Same format as train_annotations."
            ),
            examples=["annotations/test/", "annotations/test/*_mask.ome.tif"],
        ),
        model: str = Field(
            PretrainedModel.CPSAM.value,
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
        min_train_masks: int = Field(
            5,
            description=(
                "Minimum number of masks per training batch. Lower values speed up "
                "training, useful for quick testing with large sample sets. "
                "Cellpose default is 5."
            ),
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
            dataset_artifact_id=artifact_id,
        )

        training_params = TrainingParams(
            artifact_id=artifact_id,
            train_images=train_images,
            train_annotations=train_annotations,
            test_images=test_images,
            test_annotations=test_annotations,
            model=model_id,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            server_url=server_url,
            n_samples=n_samples,
            session_id=session_id,
            min_train_masks=min_train_masks,
        )

        # Save training parameters for later export
        training_params_path = get_session_path(session_id) / "training_params.json"
        # Convert Path objects to strings for JSON serialization
        params_dict = dict(training_params)
        if "model" in params_dict and isinstance(params_dict["model"], Path):
            params_dict["model"] = str(params_dict["model"])
        training_params_path.write_text(json.dumps(params_dict, indent=2))

        async with self._session_lock:
            executor = ThreadPoolExecutor(max_workers=1)
            self.executors[session_id] = executor
            task = asyncio.create_task(launch_training_task(training_params, executor))
            self.tasks[session_id] = task
            logger.info(
                "Training task created for session %s, task: %s", session_id, task
            )

        status = get_status(session_id)
        return SessionStatusWithId(**status, session_id=session_id)

    @schema_method(arbitrary_types_allowed=True)  # type: ignore
    async def stop_training(
        self,
        session_id: str = Field(
            description="Identifier returned by ``start_training``.",
        ),
    ) -> SessionStatus:
        """Stop an ongoing training session."""
        task = self.tasks.get(session_id)
        if task and not task.done():
            task.cancel()

        executor = self.executors.get(session_id)
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
            del self.executors[session_id]

        if session_id in self.tasks:
            del self.tasks[session_id]

        update_status(
            session_id=session_id,
            status_type=StatusType.STOPPED,
            message="Training session stopped by user.",
        )

        return get_status(session_id)

    @schema_method(arbitrary_types_allowed=True)  # type: ignore
    async def debug_task_info(
        self,
        session_id: str = Field(
            description="Session ID to debug",
        ),
    ) -> dict[str, Any]:
        """Debug information about the training task."""
        task = self.tasks.get(session_id)
        executor = self.executors.get(session_id)

        info: dict[str, Any] = {
            "session_id": session_id,
            "task_exists": task is not None,
            "executor_exists": executor is not None,
            "all_tasks": list(self.tasks.keys()),
            "all_executors": list(self.executors.keys()),
        }

        if task:
            info["task_done"] = task.done()
            info["task_cancelled"] = task.cancelled()
            if task.done():
                try:
                    info["task_result"] = str(task.result())
                except Exception as e:
                    info["task_exception"] = str(e)

        return info

    @schema_method(arbitrary_types_allowed=True)  # type: ignore
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

    @schema_method(arbitrary_types_allowed=True)  # type: ignore
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
        filt: set[str] | None = None
        if status_types is not None:
            allowed = {s.value for s in StatusType}
            filt = {s for s in status_types if s in allowed}

        sessions: dict[str, SessionStatus] = {}
        # Ensure session path exists (might not if no sessions yet)
        sessions_path = get_sessions_path()
        if not sessions_path.exists():
            return sessions

        for session_dir in sessions_path.iterdir():
            if not session_dir.is_dir():
                continue
            status_path = session_dir / "status.json"
            if not status_path.exists():
                continue

            try:
                session_data = json.loads(status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if filt is not None and session_data.get("status_type") not in filt:
                continue

            # Cast to SessionStatus to satisfy type checker
            sessions[str(status_path)] = SessionStatus(
                status_type=session_data.get("status_type"),
                message=session_data.get("message"),
                train_losses=session_data.get("train_losses"),
                test_losses=session_data.get("test_losses"),
                test_metrics=session_data.get("test_metrics"),
                n_train=session_data.get("n_train"),
                n_test=session_data.get("n_test"),
                start_time=session_data.get("start_time"),
                current_epoch=session_data.get("current_epoch"),
                total_epochs=session_data.get("total_epochs"),
                elapsed_seconds=session_data.get("elapsed_seconds"),
                current_batch=session_data.get("current_batch"),
                total_batches=session_data.get("total_batches"),
                exported_artifact_id=session_data.get("exported_artifact_id"),
                model_modified=session_data.get("model_modified"),
            )

        return sessions

    @schema_method(arbitrary_types_allowed=True)  # type: ignore
    async def export_model(
        self,
        session_id: str = Field(description="Training session ID to export"),
        model_name: str | None = Field(
            None,
            description="Optional custom name for the model (defaults to cellpose-{session_id})",
        ),
        collection: str = Field(
            "bioimage-io/colab-annotations",
            description="Collection to upload to (format: workspace/collection)",
        ),
    ) -> dict[str, Any]:
        """Export trained model as BioImage.io package to artifact manager.

        This function packages the trained model with all necessary files for
        BioImage.io compatibility, including model weights, architecture code,
        test samples, cover image, and RDF descriptor.

        Args:
            session_id: ID of completed training session
            model_name: Custom name for the model (optional)
            collection: Target collection in format "workspace/collection"

        Returns:
            Dictionary containing:
                - artifact_id: ID of uploaded artifact
                - model_name: Name of the model
                - status: Export status
                - url: URL to view the model

        Raises:
            ValueError: If session doesn't exist or training not completed
            RuntimeError: If export or upload fails
        """
        import shutil
        import tempfile

        import yaml
        from hypha_rpc import connect_to_server  # type: ignore

        from bioengine.utils import create_file_list_from_directory

        logger.info(f"Starting model export for session {session_id}")

        # Validate session
        status = get_status(session_id)
        if status.get("status_type") != "completed":
            raise ValueError(
                f"Cannot export model from session {session_id}: "
                f"training status is '{status.get('status_type')}', must be 'completed'"
            )

        # Get training parameters
        session_path = get_session_path(session_id)
        training_params_path = session_path / TRAINING_PARAMS_FILENAME
        if not training_params_path.exists():
            raise ValueError(f"Training parameters not found for session {session_id}")

        training_params = json.loads(training_params_path.read_text())

        # Check if model was already exported and hasn't been modified
        exported_artifact_id = status.get("exported_artifact_id")
        if exported_artifact_id and not status.get("model_modified", True):
            logger.info(
                f"Model already exported as {exported_artifact_id}, returning cached result"
            )

            # Reconstruct the result from stored info
            workspace = collection.split("/")[0]
            base_url = training_params.get("server_url", "https://hypha.aicell.io")
            artifact_url = f"{base_url}/{workspace}/artifacts/{exported_artifact_id.split('/')[-1]}"
            download_url = f"{artifact_url}/create-zip-file"

            return {
                "artifact_id": exported_artifact_id,
                "model_name": exported_artifact_id.split("/")[-1],
                "status": "exported",
                "artifact_url": artifact_url,
                "download_url": download_url,
                "files": [
                    "model_weights.pth",
                    "model.py",
                    "input_sample.npy",
                    "output_sample.npy",
                    "cover.png",
                    DOC_FILENAME,
                    RDF_FILENAME,
                ],
                "cached": True,
            }

        # Determine model name
        if model_name is None:
            model_name = f"cellpose-{session_id[:8]}"

        logger.info(f"Exporting model as '{model_name}' to collection '{collection}'")

        # Create temporary export directory
        export_dir = Path(tempfile.mkdtemp(prefix=f"cellpose_export_{session_id}_"))
        logger.info(f"Created export directory: {export_dir}")

        try:
            # 1. Copy model weights
            model_path = get_model_path(session_id)
            if not model_path.exists():
                raise ValueError(f"Model weights not found at {model_path}")

            weights_filename = "model_weights.pth"
            shutil.copy(model_path, export_dir / weights_filename)
            logger.info(f"Copied model weights: {weights_filename}")

            # Copy training history and params
            session_path = get_session_path(session_id)
            shutil.copy(
                session_path / TRAINING_PARAMS_FILENAME,
                export_dir / TRAINING_PARAMS_FILENAME,
            )
            shutil.copy(
                get_status_path(session_id), export_dir / "training_history.json"
            )

            # 2. Write model.py from embedded template
            model_py_filename = "model.py"
            (export_dir / model_py_filename).write_text(MODEL_TEMPLATE_PY)
            logger.info(f"Wrote model architecture: {model_py_filename}")

            # 3. Generate test samples
            logger.info("Generating test input/output samples...")
            test_input, test_output = await create_test_samples(
                session_id, training_params
            )

            # Save test samples as numpy arrays
            test_input_filename = "input_sample.npy"
            test_output_filename = "output_sample.npy"
            np.save(export_dir / test_input_filename, test_input)
            np.save(export_dir / test_output_filename, test_output)
            logger.info(f"Saved test samples: {test_input.shape}, {test_output.shape}")

            # 4. Generate cover image
            logger.info("Generating cover image...")
            cover_filename = "cover.png"
            await generate_cover_image(
                test_input,
                test_output,
                export_dir / cover_filename,
                model_name=model_name,
                session_id=session_id,
                training_info=status,
            )

            # 5. Generate documentation
            train_losses = status.get("train_losses")
            final_loss = (
                f"{train_losses[-1]:.4f}"  # type: ignore
                if train_losses and len(train_losses) > 0  # type: ignore
                else "N/A"
            )
            doc_content = f"""# {model_name}

Cellpose-SAM model fine-tuned on custom dataset.

## Training Information

- **Session ID**: `{session_id}`
- **Training samples**: {status.get('n_train', 'N/A')}
- **Test samples**: {status.get('n_test', 0)}
- **Epochs**: {status.get('total_epochs', 'N/A')}
- **Final training loss**: {final_loss}
- **Training time**: {status.get('elapsed_seconds', 0):.1f} seconds

## Model Parameters

- **Mean diameter**: {training_params.get('diam_mean', 30.0)} pixels
- **Model type**: Cellpose-SAM (Transformer-based)
- **Flow threshold**: 0.4
- **Cell probability threshold**: 0.0

## Dataset

- **Artifact**: {training_params.get('artifact_id', 'N/A')}
- **Train images**: {training_params.get('train_images', 'N/A')}
- **Train annotations**: {training_params.get('train_annotations', 'N/A')}

## Usage

This model can be loaded using the BioImage.io tools or directly with PyTorch:

```python
import torch
from model import CellposeSAMWrapper

# Load the model
model = CellposeSAMWrapper()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Run inference
output_masks = model(input_tensor)
```

## References

See the citations in the RDF file for relevant papers.

## License

BSD-3-Clause (Cellpose license)
"""
            (export_dir / DOC_FILENAME).write_text(doc_content)
            logger.info("Generated documentation")

            # 6. Generate RDF YAML
            logger.info("Generating RDF YAML descriptor...")
            rdf = generate_rdf_yaml(
                model_name=model_name,
                session_id=session_id,
                model_weights_filename=weights_filename,
                model_py_filename=model_py_filename,
                test_input_filename=test_input_filename,
                test_output_filename=test_output_filename,
                training_params=training_params,  # type: ignore
                test_input_shape=test_input.shape,
                test_output_shape=test_output.shape,
            )

            def _write_rdf() -> None:
                with open(export_dir / RDF_FILENAME, "w") as f:
                    yaml.dump(rdf, f)

            await asyncio.to_thread(_write_rdf)
            # 7. Upload to artifact manager
            logger.info(f"Uploading to artifact manager: {collection}")

            # Get workspace from collection
            workspace = collection.split("/")[0]

            # Connect to hypha
            token = os.environ.get("HYPHA_TOKEN")
            if not token:
                raise RuntimeError(
                    "HYPHA_TOKEN environment variable not set. "
                    "Cannot upload to artifact manager."
                )

            server: Any = await connect_to_server(  # type: ignore
                {
                    "server_url": training_params.get(
                        "server_url", "https://hypha.aicell.io"
                    ),
                    "workspace": workspace,
                    "token": token,
                }
            )

            # Cast server to Any to avoid "Unknown type" errors
            server = cast(Any, server)
            artifact_manager = await server.get_service("public/artifact-manager")  # type: ignore

            # Create file list
            files = create_file_list_from_directory(
                directory_path=str(export_dir),
            )
            logger.info(f"Prepared {len(files)} files for upload")

            # Get the collection ID
            collection_alias = (
                collection.split("/")[1] if "/" in collection else collection
            )
            collection_id_str = f"{workspace}/{collection_alias}"
            try:
                collection_info = await artifact_manager.read(collection_id_str)  # type: ignore
                collection_id = collection_info["id"]  # type: ignore
            except Exception as e:
                logger.error(f"Collection {collection_id_str} not found: {e}")
                raise ValueError(
                    f"Collection '{collection_id_str}' does not exist. "
                    f"Please create it first or use an existing collection."
                )

            # Create model artifact
            logger.info(f"Creating model artifact in collection {collection_id}")
            artifact_result = await artifact_manager.create(  # type: ignore
                type="model",
                alias=model_name,
                parent_id=collection_id,
                manifest=rdf,
                stage=True,  # Enable staging mode for file uploads
            )
            val = cast(
                Any,
                (
                    artifact_result["id"]
                    if isinstance(artifact_result, dict)
                    else artifact_result
                ),
            )
            artifact_id = str(val)
            logger.info(f"Created model artifact: {artifact_id}")

            # Upload files
            import base64

            import httpx

            async with httpx.AsyncClient(timeout=120) as client:
                for file_info in files:
                    logger.debug(f"Uploading file: {file_info['name']}")

                    # Get presigned upload URL
                    upload_url = await artifact_manager.put_file(  # type: ignore
                        artifact_id, file_path=file_info["name"]
                    )

                    # Prepare content
                    content = file_info["content"]
                    if file_info.get("type") == "base64":
                        content = base64.b64decode(content)

                    # Upload content to presigned URL
                    response = await client.put(upload_url, content=content)
                    response.raise_for_status()

            # Commit the artifact
            await artifact_manager.commit(artifact_id)  # type: ignore
            logger.info(f"Successfully exported model: {artifact_id}")

            # Update artifact with training dataset ID (stored in artifact config, not in rdf.yaml)
            training_dataset_id = training_params.get("artifact_id")
            if training_dataset_id:
                try:
                    await artifact_manager.edit(  # type: ignore
                        artifact_id, config={"training_dataset_id": training_dataset_id}
                    )
                    logger.info(f"Added training_dataset_id: {training_dataset_id}")
                except Exception as e:
                    logger.warning(f"Failed to add training_dataset_id: {e}")

            # Construct URLs
            base_url = training_params.get("server_url", "https://hypha.aicell.io")
            artifact_url = (
                f"{base_url}/{workspace}/artifacts/{artifact_id.split('/')[-1]}"
            )
            download_url = f"{artifact_url}/create-zip-file"

            result: dict[str, str] = {
                "artifact_id": artifact_id,
                "model_name": model_name,
                "status": "exported",
                "artifact_url": artifact_url,
                "download_url": download_url,
                "files": [
                    weights_filename,
                    model_py_filename,
                    test_input_filename,
                    test_output_filename,
                    cover_filename,
                    DOC_FILENAME,
                    RDF_FILENAME,
                    TRAINING_PARAMS_FILENAME,
                    "training_history.json",
                ],
            }

            # Update status with export info and mark model as not modified
            current_status = get_status(session_id)
            update_status(
                session_id,
                current_status.get("status_type", StatusType.FAILED),  # Default or cast
                current_status.get("message", ""),
                train_losses=current_status.get("train_losses"),
                test_losses=current_status.get("test_losses"),
                test_metrics=current_status.get("test_metrics"),
                n_train=current_status.get("n_train"),
                n_test=current_status.get("n_test"),
                start_time=current_status.get("start_time"),
                current_epoch=current_status.get("current_epoch"),
                total_epochs=current_status.get("total_epochs"),
                elapsed_seconds=current_status.get("elapsed_seconds"),
            )
            # Save export info directly to status file
            status_path = get_status_path(session_id)
            with status_path.open("r+", encoding="utf-8") as f:
                status_data = json.load(f)
                status_data["exported_artifact_id"] = artifact_id
                status_data["model_modified"] = False
                f.seek(0)
                f.write(json.dumps(status_data))
                f.truncate()

            logger.info(f"Model export completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Error during model export: {e}")
            raise RuntimeError(f"Model export failed: {e}") from e

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(export_dir)
                logger.info(f"Cleaned up export directory: {export_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up export directory: {e}")

    @schema_method  # type: ignore
    async def list_models_by_dataset(
        self,
        dataset_id: str = Field(
            description="Dataset artifact ID to find models trained on it"
        ),
        collection: str = Field(
            "bioimage-io/colab-annotations",
            description="Collection to search in (format: workspace/collection)",
        ),
    ) -> list[dict[str, Any]]:
        """List all models trained on a specific dataset.

        This function queries the artifact manager to find all model artifacts
        that were trained on the specified dataset by checking the
        training_dataset_id in the artifact config.

        Args:
            dataset_id: ID of the dataset artifact
            collection: Collection to search in

        Returns:
            List of model artifacts, each containing:
                - id: Model artifact ID
                - name: Model name
                - created_at: Creation timestamp
                - url: URL to view the model

        Raises:
            RuntimeError: If query fails
        """
        import os  # type: ignore

        from hypha_rpc import connect_to_server  # type: ignore

        logger.info(f"Listing models trained on dataset: {dataset_id}")

        try:
            # Get workspace from collection
            workspace = collection.split("/")[0]

            # Connect to hypha
            token = os.environ.get("HYPHA_TOKEN")
            if not token:
                raise RuntimeError(
                    "HYPHA_TOKEN environment variable not set. "
                    "Cannot query artifact manager."
                )

            server: Any = await connect_to_server(  # type: ignore
                {
                    "server_url": "https://hypha.aicell.io",
                    "workspace": workspace,
                    "token": token,
                }
            )

            # Cast server to Any
            server = cast(Any, server)
            artifact_manager = await server.get_service("public/artifact-manager")  # type: ignore

            # Get the collection ID
            collection_alias = (
                collection.split("/")[1] if "/" in collection else collection
            )
            collection_id_str = f"{workspace}/{collection_alias}"

            try:
                collection_info = await artifact_manager.read(collection_id_str)
                collection_id = collection_info["id"]
            except Exception as e:
                logger.error(f"Collection {collection_id_str} not found: {e}")
                raise ValueError(f"Collection '{collection_id_str}' does not exist.")

            # List all artifacts in the collection
            artifacts = await artifact_manager.list(
                parent_id=collection_id, filters={"type": "model"}
            )

            # Filter models by training_dataset_id
            matching_models: list[dict[str, Any]] = []
            for artifact in artifacts:
                config = artifact.get("config", {})
                if config.get("training_dataset_id") == dataset_id:
                    # Build model info
                    model_id = artifact["id"]
                    model_alias = artifact.get("alias", model_id.split("/")[-1])
                    base_url = "https://hypha.aicell.io"
                    artifact_url = (
                        f"{base_url}/{workspace}/artifacts/{model_id.split('/')[-1]}"
                    )

                    matching_models.append(
                        {
                            "id": model_id,
                            "name": model_alias,
                            "created_at": artifact.get("created_at"),
                            "url": artifact_url,
                        }
                    )

            logger.info(f"Found {len(matching_models)} models trained on {dataset_id}")
            return matching_models

        except Exception as e:
            logger.error(f"Error listing models by dataset: {e}")
            raise RuntimeError(f"Failed to list models by dataset: {e}") from e

    @schema_method(arbitrary_types_allowed=True)  # type: ignore
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
            PretrainedModel.CPSAM.value,
            description=(
                "Identifier of the Cellpose model to use for inference. Either a "
                "built-in pretrained model name or the session ID of a finetuned model."
            ),
            examples=["abc123ef-4567-890a-bcde-f1234567890a", PretrainedModel.values()],
        ),
        diameter: float | None = Field(
            None,
            description=(
                "Approximate object diameter; if None, Cellpose will estimate it"
            ),
            ge=0,
        ),
        flow_threshold: float = Field(
            0.4,
            description=(
                "Flow error threshold for dynamics. Higher values allow more masks. "
                "Decrease if too many ill-shaped ROIs are returned."
            ),
            ge=0,
        ),
        cellprob_threshold: float = Field(
            0.0,
            description=(
                "Cell probability threshold. Decrease to find more ROIs, "
                "increase to filter out dim areas."
            ),
        ),
        niter: int | None = Field(
            None,
            description=(
                "Number of iterations for flow dynamics. If None, automatically "
                "set based on diameter. Use higher values (e.g., 250) for better convergence."
            ),
            ge=0,
        ),
        return_flows: bool = Field(
            False,
            description=(
                "If True, return flows in addition to masks. Flows include: "
                "[0] HSV flow visualization, [1] XY flows at each pixel, "
                "[2] cell probability map, [3] final pixel locations after dynamics."
            ),
        ),
    ) -> list[PredictionItemModel]:
        """Run Cellpose inference on artifact images and return encoded masks.

        Images are fetched from the specified artifact into a local cache as
        needed. For each input path, the corresponding mask array is returned as
        an NPY-serialized base64 payload suitable for cross-language transport.

        Optionally, flows can be returned by setting return_flows=True, which includes
        flow visualization, XY flows, cell probability, and final pixel positions.

        """
        # Initialize with defaults to avoid UnboundLocalError
        artifact_id: str | None = None
        server_url: str = DEFAULT_SERVER_URL

        if artifact is not None:
            server_url, artifact_id = get_url_and_artifact_id(artifact)

        if image_paths is None:
            image_paths = []

        if input_arrays is not None and not is_ndarray(input_arrays):
            error_msg = "input_arrays must be a list of numpy ndarrays"
            raise TypeError(error_msg)

        model_id = self.get_model_id(model)
        model_obj = load_model(model_id)

        images: list[npt.NDArray[Any]]
        if input_arrays is not None:
            images = input_arrays
            image_paths = [f"input_arrays[{i}]" for i in range(len(input_arrays))]
        elif artifact_id is not None:
            images = await _images_from_artifact(
                artifact_id=artifact_id,
                image_paths=image_paths,  # type: ignore
                server_url=server_url,
            )
        else:
            images = []

        return _predict_and_encode(
            model=model_obj,
            images=images,
            image_paths=image_paths,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            niter=niter,
            return_flows=return_flows,
        )


async def infer(
    cellpose_tuner: Any,
    model: str | None = None,
) -> None:
    """Test inference functionality of the Cellpose Fine-Tuning service."""
    from plotly import express as px  # type: ignore

    inference_result = await cellpose_tuner.infer(  # type: ignore[hasAttribute]
        model=model,
        artifact="ri-scale/zarr-demo",
        diameter=40,
        image_paths=["images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif"],
    )
    logger.info("Inference done! Result: %s", str(inference_result)[:500] + "...")
    arr = inference_result[0]["output"]
    # px.imshow usually expects image like (H, W) or (H, W, C).
    px.imshow(arr).show()


async def monitor_training(
    cellpose_tuner: Any,
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
    # Note: .func_or_class() or .options().bind() usage depends on Ray version.
    # Assuming .func_or_class() based on previous code but modifying for type safety if needed.
    # Reverting to original call style but casting:
    cellpose_tuner = cast(Any, CellposeFinetune.func_or_class())  # type: ignore

    await infer(cellpose_tuner, model="cyto3")

    session_status = cast(
        dict[str, Any],
        await cellpose_tuner.start_training(
            artifact="ri-scale/zarr-demo",
            n_epochs=1,
            n_samples=5,
        ),
    )

    session_id = session_status["session_id"]

    await monitor_training(cellpose_tuner, session_id)

    await infer(cellpose_tuner, model=session_id)


if __name__ == "__main__":
    asyncio.run(test_cellpose_finetune())

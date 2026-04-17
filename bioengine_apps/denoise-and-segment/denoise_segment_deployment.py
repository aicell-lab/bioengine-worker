"""
Denoise and Segment BioEngine App

Two-stage pipeline for noisy fluorescence microscopy:
  1. Content-aware image restoration using a BioImage.IO denoising model (e.g. hiding-tiger / N2V / CARE)
  2. Cellpose cyto3 instance segmentation

Demonstrates BioEngine model chaining: a BioImage.IO model feeds its output directly
into a Cellpose model within a single Ray Serve deployment.

IMPORT HANDLING:
- Standard Python libraries and BioEngine core libraries are imported at module top level.
- cellpose, bioimageio.core, and tifffile are NOT part of the BioEngine base environment;
  they are specified in runtime_env.pip and imported inside each method that uses them.

References:
- BioImage.IO hiding-tiger: https://bioimage.io/#/?id=hiding-tiger
- Cellpose: https://github.com/mouseland/cellpose
- bioimageio.core: https://github.com/bioimage-io/core-bioimage-io-python
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        # Sufficient CPUs for parallel denoising + segmentation inference
        "num_cpus": 4,
        # CPU-only: bioimageio.core and Cellpose cyto3 run on CPU by default
        "num_gpus": 0,
        # 8 GB: denoising model weights + Cellpose weights + image buffers
        "memory": 8 * 1024**3,
        "runtime_env": {
            "pip": [
                "cellpose>=4.0",
                "bioimageio.core>=0.6",
                "tifffile",
                "numpy",
            ],
        },
    }
)
class DenoiseAndSegmentDeployment:
    def __init__(self) -> None:
        """Initialize the Denoise-and-Segment deployment."""
        self.start_time = time.time()
        self._cellpose_model = None

    async def async_init(self) -> None:
        """
        Load the Cellpose cyto3 model at startup.

        cellpose is specified in runtime_env.pip and must be imported here
        (not at module top level) to respect BioEngine import conventions.
        bioimageio.core is loaded lazily inside each method call because the
        specific denoising model is not known at init time.
        """
        from cellpose import models as cellpose_models

        logger.info("Loading Cellpose cyto3 model...")
        self._cellpose_model = cellpose_models.CellposeModel(model_type="cyto3")
        logger.info("Cellpose cyto3 model loaded successfully.")

    async def check_health(self) -> None:
        """Periodically checked by Ray Serve to monitor deployment health."""
        if self._cellpose_model is None:
            raise RuntimeError("Cellpose model is not loaded.")

    # === Internal helpers ===

    def _run_cellpose(
        self,
        img: "np.ndarray",
        diameter: Optional[float],
        channels: Optional[List[int]],
        flow_threshold: float,
        cellprob_threshold: float,
    ) -> Dict[str, Union[list, int]]:
        """Blocking Cellpose inference — called via run_in_executor."""
        ch = channels if channels is not None else [0, 0]
        masks, _flows, _styles = self._cellpose_model.eval(
            img,
            diameter=diameter,
            channels=ch,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        labels = masks.astype(np.int32)
        n_cells = int(labels.max())
        cell_areas = [
            int(np.sum(labels == cell_id)) for cell_id in range(1, n_cells + 1)
        ]
        return {
            "labels": labels.tolist(),
            "n_cells": n_cells,
            "cell_areas": cell_areas,
        }

    def _run_denoising(self, image_array: "np.ndarray", model_id: str) -> "np.ndarray":
        """
        Blocking bioimageio.core denoising inference — called via run_in_executor.

        Loads the BioImage.IO model identified by *model_id* (e.g. 'hiding-tiger'),
        builds an input Sample from *image_array*, runs blocking prediction, and
        returns the denoised array with the same spatial shape.
        """
        import numpy as np
        import bioimageio.core
        from bioimageio.core import Sample
        from bioimageio.spec.model.v0_5 import TensorId

        logger.info(f"Loading BioImage.IO denoising model: {model_id}")
        rdf = bioimageio.core.load_description(model_id)

        with bioimageio.core.create_prediction_pipeline(rdf) as pp:
            # Determine expected input tensor name from model spec
            input_id = TensorId(pp.input_ids[0] if pp.input_ids else "input")

            # Ensure array has the axes the model expects (bcyx / bczyx).
            # Most 2-D restoration models expect shape [batch, channel, Y, X].
            arr = image_array.astype(np.float32)
            if arr.ndim == 2:
                # [H, W] -> [1, 1, H, W]
                arr = arr[np.newaxis, np.newaxis, ...]
            elif arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
                # [C, H, W] -> [1, C, H, W]
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                # [H, W, C] -> [1, C, H, W]
                arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

            # Build a bioimageio.core Sample
            import xarray as xr
            from bioimageio.core import Sample
            from bioimageio.spec.model.v0_5 import TensorId

            # Axes order for a typical 2-D restoration model: b, c, y, x
            axes = ("b", "c", "y", "x") if arr.ndim == 4 else ("b", "c", "z", "y", "x")
            xr_arr = xr.DataArray(arr, dims=axes)
            sample = Sample(members={input_id: xr_arr}, stat={}, id=None)

            result_sample = pp.predict_sample_with_blocking(sample)

        # Extract the first output tensor and squeeze batch/channel dims back out
        out_id = list(result_sample.members.keys())[0]
        denoised = np.array(result_sample.members[out_id])

        # Remove batch dim if present
        if denoised.ndim >= 4 and denoised.shape[0] == 1:
            denoised = denoised[0]  # [C, H, W] or [H, W]

        # If single-channel, squeeze to [H, W]
        if denoised.ndim == 3 and denoised.shape[0] == 1:
            denoised = denoised[0]

        logger.info(f"Denoising complete. Output shape: {denoised.shape}")
        return denoised

    @staticmethod
    def _prepare_for_cellpose(arr: "np.ndarray") -> "np.ndarray":
        """
        Convert array to the shape Cellpose expects: [H, W] or [H, W, C].

        Cellpose cyto3 works best with [H, W] grayscale or [H, W, C] RGB.
        Input from bioimageio.core is typically [C, H, W] or [H, W].
        """
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
            # [C, H, W] -> [H, W, C]
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
        return arr

    # === Exposed API Methods ===

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float, bool]]:
        """
        Ping the deployment to test connectivity and report model status.

        Returns:
            Dict with 'status', 'message', 'timestamp', 'uptime', and 'model_loaded' fields.
        """
        return {
            "status": "ok",
            "message": "Denoise-and-Segment deployment is ready.",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "model_loaded": self._cellpose_model is not None,
        }

    @schema_method
    async def segment(
        self,
        image: list = Field(
            ...,
            description=(
                "Input image as a nested Python list (use numpy_array.tolist()). "
                "Accepts 2-D grayscale [H, W], 3-D single-channel [1, H, W], "
                "or 3-D multi-channel [C, H, W] arrays."
            ),
        ),
        diameter: Optional[float] = Field(
            None,
            description=(
                "Expected cell diameter in pixels. Pass null/None to let Cellpose "
                "estimate it automatically from the image."
            ),
        ),
        channels: Optional[List[int]] = Field(
            None,
            description=(
                "Cellpose channel list [cytoplasm_channel, nucleus_channel]. "
                "Use [0, 0] for grayscale, [2, 3] for green/blue, etc. "
                "Defaults to [0, 0] (grayscale) when null."
            ),
        ),
        flow_threshold: float = Field(
            0.4,
            description="Flow error threshold for cell mask acceptance (0–1). Higher values accept more cells.",
        ),
        cellprob_threshold: float = Field(
            -0.5,
            description="Cell probability threshold. Lower values detect more (potentially noisy) cells.",
        ),
    ) -> Dict[str, Union[list, int]]:
        """
        Segment cells directly using Cellpose cyto3 (no denoising step).

        Use this method when your image is already clean, or to compare
        segmentation quality before and after denoising.

        Returns:
            Dict containing:
              - 'labels': integer label array (same spatial shape as input) as a nested list.
                Each unique non-zero integer corresponds to one segmented cell.
              - 'n_cells': total number of detected cells.
              - 'cell_areas': list of pixel areas for each cell (ordered by label id).
        """
        if self._cellpose_model is None:
            raise RuntimeError("Cellpose model not initialised. Call async_init first.")

        img = np.array(image, dtype=np.float32)
        img = self._prepare_for_cellpose(img)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._run_cellpose(img, diameter, channels, flow_threshold, cellprob_threshold),
        )
        return result

    @schema_method
    async def denoise_and_segment(
        self,
        image: list = Field(
            ...,
            description=(
                "Input image as a nested Python list (use numpy_array.tolist()). "
                "Accepts 2-D grayscale [H, W], 3-D [C, H, W] or [H, W, C] arrays."
            ),
        ),
        denoising_model_id: str = Field(
            "hiding-tiger",
            description=(
                "BioImage.IO model ID or nickname for the denoising/restoration step. "
                "Examples: 'hiding-tiger' (N2V fluorescence), any CARE model ID from bioimage.io."
            ),
        ),
        diameter: Optional[float] = Field(
            None,
            description=(
                "Expected cell diameter in pixels passed to Cellpose. "
                "Pass null/None to let Cellpose estimate it automatically."
            ),
        ),
        channels: Optional[List[int]] = Field(
            None,
            description=(
                "Cellpose channel list [cytoplasm_channel, nucleus_channel]. "
                "Defaults to [0, 0] (grayscale) when null."
            ),
        ),
        flow_threshold: float = Field(
            0.4,
            description="Flow error threshold for Cellpose mask acceptance (0–1).",
        ),
        cellprob_threshold: float = Field(
            -0.5,
            description="Cell probability threshold for Cellpose. Lower values detect more cells.",
        ),
    ) -> Dict[str, Union[list, int, str]]:
        """
        Run a two-stage denoise-then-segment pipeline on a fluorescence microscopy image.

        Stage 1 — Denoising / image restoration:
            Downloads and runs a BioImage.IO model (default: 'hiding-tiger', an N2V model
            trained on fluorescence data) to suppress photon shot noise while preserving
            structural detail.

        Stage 2 — Instance segmentation:
            Feeds the restored image through Cellpose cyto3 to produce an integer label
            map where each unique non-zero value identifies one cell instance.

        Both stages run in a thread executor to avoid blocking the Ray Serve event loop.

        Returns:
            Dict containing:
              - 'labels': integer label array as a nested list (reconstruct with np.array()).
              - 'n_cells': total number of segmented cells.
              - 'cell_areas': list of pixel areas per cell (ordered by label id).
              - 'denoising_model': the BioImage.IO model ID used for denoising.
              - 'pipeline': human-readable pipeline description string.
        """
        if self._cellpose_model is None:
            raise RuntimeError("Cellpose model not initialised. Call async_init first.")

        img = np.array(image, dtype=np.float32)

        # --- Stage 1: denoising (blocking, run in thread executor) ---
        loop = asyncio.get_event_loop()
        logger.info(f"Stage 1: denoising with model '{denoising_model_id}'")
        denoised = await loop.run_in_executor(
            None,
            lambda: self._run_denoising(img, denoising_model_id),
        )

        # --- Stage 2: Cellpose segmentation (blocking, run in thread executor) ---
        seg_input = self._prepare_for_cellpose(denoised)
        logger.info("Stage 2: Cellpose segmentation on denoised image")
        result = await loop.run_in_executor(
            None,
            lambda: self._run_cellpose(seg_input, diameter, channels, flow_threshold, cellprob_threshold),
        )

        result["denoising_model"] = denoising_model_id
        result["pipeline"] = "denoise→segment"
        return result

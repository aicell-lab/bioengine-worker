"""
Cellpose Segmentation BioEngine App

Provides universal cell and organoid segmentation using the Cellpose cyto3 model.
Supports phase-contrast, brightfield, and fluorescence microscopy images.

Source: https://github.com/mouseland/cellpose
License: BSD 3-Clause

IMPORT HANDLING:
- Standard Python libraries and BioEngine core libraries are imported at module top level.
- cellpose and tifffile are NOT part of the BioEngine base environment; they are specified
  in runtime_env.pip and imported inside each method that uses them.
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
        "num_cpus": 4,
        "num_gpus": 0,
        "memory": 4 * 1024**3,
        "runtime_env": {
            "pip": [
                "cellpose>=4.0",
                "tifffile",
                "numpy",
            ],
        },
    }
)
class CellposeDeployment:
    def __init__(self) -> None:
        """Initialize the Cellpose Segmentation deployment."""
        self.start_time = time.time()
        self._model = None

    async def async_init(self) -> None:
        """
        Load the Cellpose cyto3 model at startup.

        The cellpose package is specified in runtime_env.pip and must be imported here
        (not at module top level) to respect BioEngine import conventions.
        """
        from cellpose import models as cellpose_models

        logger.info("Loading Cellpose cyto3 model...")
        self._model = cellpose_models.CellposeModel(model_type="cyto3")
        logger.info("Cellpose cyto3 model loaded successfully.")

    async def check_health(self) -> None:
        """Periodically checked by Ray Serve to monitor deployment health."""
        if self._model is None:
            raise RuntimeError("Cellpose model is not loaded.")

    # === Exposed API Methods ===

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float]]:
        """
        Ping the deployment to test connectivity and check model status.

        Returns:
            Dict with 'status', 'message', 'timestamp', and 'uptime' fields.
        """
        return {
            "status": "ok",
            "message": "Cellpose cyto3 segmentation deployment is ready.",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "model_loaded": self._model is not None,
        }

    @schema_method
    async def segment(
        self,
        image: list = Field(..., description=(
            "Input image as a nested Python list (use numpy_array.tolist()). "
            "Accepts 2-D grayscale [H, W], 3-D single-channel [1, H, W], "
            "or 3-D multi-channel [C, H, W] arrays."
        )),
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
        Segment cells or organoids in an input image using Cellpose cyto3.

        Accepts images from phase-contrast, brightfield, or fluorescence modalities.
        The image is passed as a nested Python list; convert with ``numpy_array.tolist()``
        on the client side. Returned label arrays can be reconstructed with
        ``numpy.array(result['labels'])``.

        Returns:
            Dict containing:
              - 'labels': integer label array (same spatial shape as input) as a nested list.
                Each unique non-zero integer corresponds to one segmented cell.
              - 'n_cells': total number of detected cells.
              - 'cell_areas': list of pixel areas for each detected cell (ordered by label id).
        """
        import numpy as np
        from cellpose import models as cellpose_models

        if self._model is None:
            raise RuntimeError("Model not initialised. Call async_init first.")

        img = np.array(image, dtype=np.float32)

        # Cellpose expects [H, W] or [H, W, C]; transpose from [C, H, W] if needed
        if img.ndim == 3 and img.shape[0] in (1, 2, 3, 4) and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 1:
                img = img[:, :, 0]

        ch = channels if channels is not None else [0, 0]

        masks, flows, styles = self._model.eval(
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

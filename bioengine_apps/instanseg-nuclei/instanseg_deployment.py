"""
InstanSeg Nuclei BioEngine App

Provides instance segmentation of nuclei and cells in fluorescence microscopy images
using the InstanSeg TorchScript model (fluorescence_nuclei_and_cells variant).

Source: https://github.com/instanseg/instanseg
License: Apache 2.0

IMPORT HANDLING:
- Standard Python libraries and BioEngine core libraries are imported at module top level.
- instanseg and torch are NOT part of the BioEngine base environment; they are specified
  in runtime_env.pip and imported inside each method that uses them.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Union

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        "num_cpus": 4,
        "num_gpus": 0,
        "memory": 6 * 1024**3,
        "runtime_env": {
            "pip": [
                "instanseg-torch>=0.1.1",  # requires Python >= 3.9; pip package: instanseg-torch
                "tifffile",
                "numpy",
                "torch",
            ],
        },
    }
)
class InstanSegDeployment:
    def __init__(self) -> None:
        """Initialize the InstanSeg Nuclei deployment."""
        self.start_time = time.time()
        self._model = None

    async def async_init(self) -> None:
        """
        Download and initialise the InstanSeg fluorescence_nuclei_and_cells model.

        instanseg is specified in runtime_env.pip and must be imported here
        (not at module top level) to respect BioEngine import conventions.
        """
        from instanseg import InstanSeg

        logger.info("Initialising InstanSeg fluorescence_nuclei_and_cells model...")
        self._model = InstanSeg("fluorescence_nuclei_and_cells")
        logger.info("InstanSeg model initialised successfully.")

    async def check_health(self) -> None:
        """Periodically checked by Ray Serve to monitor deployment health."""
        if self._model is None:
            raise RuntimeError("InstanSeg model is not loaded.")

    # === Exposed API Methods ===

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float, bool]]:
        """
        Ping the deployment to test connectivity and check model status.

        Returns:
            Dict with 'status', 'message', 'timestamp', 'uptime', and 'model_loaded' fields.
        """
        return {
            "status": "ok",
            "message": "InstanSeg nuclei/cell segmentation deployment is ready.",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "model_loaded": self._model is not None,
        }

    @schema_method
    async def segment_nuclei(
        self,
        image: list = Field(..., description=(
            "Fluorescence image as a nested Python list (use numpy_array.tolist()). "
            "Expected shape [C, H, W] or [H, W] for single-channel images. "
            "Pixel values should be raw intensity counts or normalised floats."
        )),
        pixel_size: float = Field(
            0.65,
            description=(
                "Pixel size in micrometres (µm/px). InstanSeg uses this to normalise "
                "the input to the expected scale. Default 0.65 µm/px corresponds to a "
                "typical 20x objective with a standard camera."
            ),
        ),
    ) -> Dict[str, Union[list, int]]:
        """
        Segment nuclei (and cells when available) in a fluorescence microscopy image.

        The image is passed as a nested Python list; convert with ``numpy_array.tolist()``
        on the client side. The returned label arrays can be reconstructed with
        ``numpy.array(result['nucleus_labels'])``.

        Returns:
            Dict containing:
              - 'nucleus_labels': integer label array (same H×W spatial shape as input)
                as a nested list. Each non-zero integer identifies one nucleus instance.
              - 'n_nuclei': total number of detected nuclei.
        """
        import numpy as np
        import torch

        if self._model is None:
            raise RuntimeError("Model not initialised. Call async_init first.")

        img = np.array(image, dtype=np.float32)

        # InstanSeg expects a [C, H, W] tensor; add channel dim for 2-D input
        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        img_tensor = torch.from_numpy(img)

        # segment() returns (labeled_output, image_tensor)
        # labeled_output shape: [1, n_classes, H, W] — first class is nuclei
        labeled_output, _ = self._model.eval_small_image(
            img_tensor, pixel_size=pixel_size
        )

        # Output shape: [1, 2, H, W] — channel 0 = cells, channel 1 = nuclei
        nucleus_labels = labeled_output[0, 1].cpu().numpy().astype(np.int32)
        n_nuclei = int(nucleus_labels.max())

        return {
            "nucleus_labels": nucleus_labels.tolist(),
            "n_nuclei": n_nuclei,
        }

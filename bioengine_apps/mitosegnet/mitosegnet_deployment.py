"""
MitoSegNet BioEngine App

Provides mitochondria segmentation in fluorescence live-cell images using the
MitoSegNet U-Net architecture (Fischer et al., iScience 2020, Cell Press).

Pre-trained weights are fetched from Zenodo record 3539340 at startup.

Reference:
  Fischer, C.A. et al. MitoSegNet: Easy-to-use Deep Learning Segmentation
  for Analyzing Mitochondrial Morphology. iScience 23, 101601 (2020).
  https://doi.org/10.1016/j.isci.2020.101601

Weights DOI: https://doi.org/10.5281/zenodo.3539340

IMPORT HANDLING:
- Standard Python libraries and BioEngine core libraries are imported at module top level.
- tensorflow, scikit-image, and tifffile are NOT part of the BioEngine base environment;
  they are specified in runtime_env.pip and imported inside each method that uses them.
"""

import asyncio
import logging
import os
import time
import urllib.request
from datetime import datetime
from typing import Dict, Union

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

logger = logging.getLogger("ray.serve")

_WEIGHTS_URL = "https://zenodo.org/record/3539340/files/MitoSegNet_model.hdf5"
_WEIGHTS_PATH = "/tmp/mitosegnet_weights.hdf5"


@serve.deployment(
    ray_actor_options={
        "num_cpus": 4,
        "num_gpus": 0,
        "memory": 4 * 1024**3,
        "runtime_env": {
            "pip": [
                "tensorflow>=2.0",
                "numpy",
                "tifffile",
                "scikit-image",
            ],
        },
    }
)
class MitoSegNetDeployment:
    def __init__(self) -> None:
        """Initialize the MitoSegNet deployment."""
        self.start_time = time.time()
        self._model = None

    async def async_init(self) -> None:
        """
        Download MitoSegNet weights from Zenodo and load the Keras model.

        tensorflow is specified in runtime_env.pip and must be imported here
        (not at module top level) to respect BioEngine import conventions.
        """
        import tensorflow as tf

        # Download weights if not already cached
        if not os.path.exists(_WEIGHTS_PATH):
            logger.info(f"Downloading MitoSegNet weights from {_WEIGHTS_URL} ...")
            await asyncio.get_event_loop().run_in_executor(
                None,
                urllib.request.urlretrieve,
                _WEIGHTS_URL,
                _WEIGHTS_PATH,
            )
            logger.info(f"Weights saved to {_WEIGHTS_PATH}.")
        else:
            logger.info(f"Using cached MitoSegNet weights at {_WEIGHTS_PATH}.")

        logger.info("Loading MitoSegNet Keras model...")
        self._model = tf.keras.models.load_model(_WEIGHTS_PATH)
        logger.info("MitoSegNet model loaded successfully.")

    async def check_health(self) -> None:
        """Periodically checked by Ray Serve to monitor deployment health."""
        if self._model is None:
            raise RuntimeError("MitoSegNet model is not loaded.")

    # === Internal helpers ===

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Normalise a 2-D float image to [0, 1] and reshape for Keras inference."""
        img = img.astype(np.float32)
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        # Keras expects [batch, H, W, C]
        return img[np.newaxis, :, :, np.newaxis]

    def _postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """Threshold network output to a binary mask and label connected components."""
        from skimage.measure import label as skimage_label

        prob_map = prediction[0, :, :, 0]  # remove batch + channel dims
        binary_mask = (prob_map >= 0.5).astype(np.uint8)
        labeled = skimage_label(binary_mask).astype(np.int32)
        return labeled

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
            "message": "MitoSegNet mitochondria segmentation deployment is ready.",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "model_loaded": self._model is not None,
        }

    @schema_method
    async def segment_mitochondria(
        self,
        image: list = Field(..., description=(
            "2-D fluorescence image of mitochondria as a nested Python list "
            "(use numpy_array.tolist()). Shape should be [H, W]. "
            "Pixel values can be raw intensity counts; normalisation is applied internally."
        )),
    ) -> Dict[str, Union[list, int]]:
        """
        Segment mitochondria in a 2-D fluorescence live-cell image using MitoSegNet.

        The image is passed as a nested Python list; convert with ``numpy_array.tolist()``
        on the client side. The returned mask array can be reconstructed with
        ``numpy.array(result['mask'])``.

        Returns:
            Dict containing:
              - 'mask': integer label array (same H×W shape as input) as a nested list.
                Each non-zero integer identifies one connected mitochondrion instance.
              - 'n_mitochondria': total number of detected mitochondrial objects.
        """
        import numpy as np

        if self._model is None:
            raise RuntimeError("Model not initialised. Call async_init first.")

        img = np.array(image, dtype=np.float32)

        if img.ndim != 2:
            raise ValueError(
                f"Expected a 2-D image [H, W], got shape {img.shape}. "
                "For multi-channel images, pass a single mitochondria channel."
            )

        input_tensor = self._preprocess(img)

        # Run inference in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            None, self._model.predict, input_tensor
        )

        labeled_mask = self._postprocess(prediction)
        n_mitochondria = int(labeled_mask.max())

        return {
            "mask": labeled_mask.tolist(),
            "n_mitochondria": n_mitochondria,
        }

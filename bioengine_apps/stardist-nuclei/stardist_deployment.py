"""
StarDist Nuclei Segmentation BioEngine App

Provides instance segmentation of cell nuclei using StarDist star-convex polygon
detection (Schmidt et al., Nature Methods 2018). Particularly effective for dense,
overlapping nuclei in fluorescence and H&E histology images.

Supports multiple pretrained model variants via Ray Serve model multiplexing:
  - 2D_versatile_fluo  : Fluorescence microscopy (DAPI, Hoechst). General-purpose.
  - 2D_versatile_he    : H&E histology stained tissue sections.
  - 2D_paper_dsb2018   : DSB 2018 challenge nuclei (fluorescence).

Reference:
  Schmidt U, Weigert M, Broaddus C, Myers G.
  "Cell Detection with Star-convex Polygons."
  MICCAI 2018. https://doi.org/10.1007/978-3-030-00934-2_30

IMPORT HANDLING:
- Standard Python libraries and BioEngine core libraries are imported at module top level.
- stardist, tensorflow, tifffile are NOT part of the BioEngine base environment; they are
  specified in runtime_env.pip and imported inside each method that uses them.
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

# Supported model IDs and their metadata
_AVAILABLE_MODELS = [
    {
        "id": "2D_versatile_fluo",
        "description": "Fluorescence microscopy (DAPI, Hoechst). Most general-purpose.",
        "modality": "fluorescence",
    },
    {
        "id": "2D_versatile_he",
        "description": "H&E histology stained tissue sections.",
        "modality": "brightfield",
    },
    {
        "id": "2D_paper_dsb2018",
        "description": "DSB 2018 challenge nuclei (fluorescence).",
        "modality": "fluorescence",
    },
]

_VALID_MODEL_IDS = {m["id"] for m in _AVAILABLE_MODELS}


@serve.deployment(
    ray_actor_options={
        "num_cpus": 4,
        "num_gpus": 0,
        "memory": 8 * 1024**3,
        "runtime_env": {
            "pip": [
                "stardist>=0.8",
                "tensorflow>=2.8,<2.14",
                "tifffile",
                "numpy",
                "scipy",
            ],
        },
    }
)
class StarDistDeployment:
    def __init__(self) -> None:
        """Initialize the StarDist Nuclei deployment."""
        self.start_time = time.time()

    async def check_health(self) -> None:
        """Periodically checked by Ray Serve to monitor deployment health."""
        pass  # Model loading is lazy via multiplexing; no persistent state to check.

    # === Internal Methods ===

    @serve.multiplexed(max_num_models_per_replica=3)
    async def _get_model(self, model_id: str):
        """Load StarDist pretrained model by ID. Cached up to 3 models per replica.

        Requirements (Ray Serve multiplexed contract):
        - Must be an async method.
        - Must accept exactly one argument: model_id (str).
        - Must return the loaded model.
        - Must not be called with a keyword argument for model_id.
        """
        from stardist.models import StarDist2D

        if model_id not in _VALID_MODEL_IDS:
            raise ValueError(
                f"Unknown model_id '{model_id}'. "
                f"Valid options: {sorted(_VALID_MODEL_IDS)}"
            )

        logger.info(f"Loading StarDist pretrained model: {model_id}")
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None, StarDist2D.from_pretrained, model_id
        )
        logger.info(f"StarDist model '{model_id}' loaded successfully.")
        return model

    # === Exposed API Methods ===

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float]]:
        """
        Ping the deployment to test connectivity.

        Returns:
            Dict with 'status', 'message', 'timestamp', and 'uptime' fields.
        """
        return {
            "status": "ok",
            "message": "StarDist nuclei segmentation deployment is ready.",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
        }

    @schema_method
    async def list_available_models(self) -> Dict[str, Union[List[dict], str]]:
        """
        List all available StarDist pretrained model variants.

        Returns:
            Dict with a 'models' list (each entry has 'id', 'description', 'modality')
            and a 'note' string explaining how to switch models.
        """
        return {
            "models": _AVAILABLE_MODELS,
            "note": (
                "Pass model_id to segment_nuclei() to switch between variants "
                "without redeploying."
            ),
        }

    @schema_method
    async def segment_nuclei(
        self,
        image: list = Field(
            ...,
            description=(
                "2-D grayscale image as a nested Python list "
                "(use numpy_array.tolist()). Expected shape [H, W]. "
                "Raw intensity values; normalisation is applied internally."
            ),
        ),
        model_id: str = Field(
            "2D_versatile_fluo",
            description=(
                "StarDist pretrained model to use. "
                "Call list_available_models() to see all options."
            ),
        ),
        prob_thresh: Optional[float] = Field(
            None,
            description=(
                "Object probability threshold (0–1). Lower values detect more "
                "candidates but may increase false positives. "
                "Defaults to the model's built-in optimised value."
            ),
        ),
        nms_thresh: Optional[float] = Field(
            None,
            description=(
                "Non-maximum suppression (overlap) threshold (0–1). "
                "Lower values suppress more overlapping detections. "
                "Defaults to the model's built-in optimised value."
            ),
        ),
    ) -> Dict[str, Union[list, int, str]]:
        """
        Segment nuclei in a 2-D grayscale image using StarDist.

        The image is passed as a nested Python list; convert with
        ``numpy_array.tolist()`` on the client side. The returned label array
        can be reconstructed with ``numpy.array(result['labels'])``.

        Normalisation is applied internally using the 1st–99.8th percentile
        range, matching the training-time normalisation used for all StarDist
        pretrained models.

        Returns:
            Dict containing:
              - 'labels'  : integer label array (same H×W shape as input) as a
                            nested list. Each non-zero integer is one nucleus instance.
              - 'n_nuclei': total number of detected nuclei.
              - 'model_id': the model variant used for this prediction.
              - 'coord'   : polygon coordinates for each nucleus as a nested list
                            (shape [n_nuclei, n_rays, 2]); empty list if unavailable.
        """
        from csbdeep.utils import normalize

        img = np.array(image, dtype=np.float32)

        if img.ndim != 2:
            raise ValueError(
                f"Expected a 2-D image [H, W], got shape {img.shape}. "
                "For multi-channel images, pass a single channel."
            )

        # Normalise to [0, 1] using 1st–99.8th percentile (StarDist convention)
        img_norm = normalize(img, 1, 99.8)

        # Load (or retrieve cached) model via multiplexed method
        model = await self._get_model(model_id)

        # Build kwargs only for non-None thresholds so model defaults are used otherwise
        predict_kwargs = {}
        if prob_thresh is not None:
            predict_kwargs["prob_thresh"] = prob_thresh
        if nms_thresh is not None:
            predict_kwargs["nms_thresh"] = nms_thresh

        loop = asyncio.get_event_loop()
        labels, details = await loop.run_in_executor(
            None,
            lambda: model.predict_instances(img_norm, **predict_kwargs),
        )

        n_nuclei = int(labels.max())
        coord = details.get("coord", [])
        # coord is a numpy array of shape [n_nuclei, n_rays, 2]; serialise to list
        if hasattr(coord, "tolist"):
            coord = coord.tolist()

        logger.info(
            f"StarDist '{model_id}' detected {n_nuclei} nuclei "
            f"in image of shape {img.shape}."
        )

        return {
            "labels": labels.tolist(),
            "n_nuclei": n_nuclei,
            "model_id": model_id,
            "coord": coord,
        }

"""
StarDist + Cellpose Ensemble BioEngine App

Runs StarDist and Cellpose in parallel, merges predictions by IoU matching
to produce a high-confidence consensus segmentation.

Sources:
  StarDist: https://github.com/stardist/stardist (BSD-3)
  Cellpose: https://github.com/mouseland/cellpose (BSD-3)
License: MIT
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

logger = logging.getLogger("ray.serve")


def _iou_ensemble(mask_a: np.ndarray, mask_b: np.ndarray,
                  iou_threshold: float = 0.5) -> np.ndarray:
    """Return consensus mask: instances present in BOTH masks at IoU >= threshold."""
    ids_a = [i for i in np.unique(mask_a) if i > 0]
    ids_b = [i for i in np.unique(mask_b) if i > 0]
    consensus = np.zeros_like(mask_a)
    cid = 1
    matched_b: set = set()
    for aid in ids_a:
        ma = mask_a == aid
        best_iou, best_bid = 0.0, None
        for bid in ids_b:
            if bid in matched_b:
                continue
            mb = mask_b == bid
            inter = int((ma & mb).sum())
            if inter == 0:
                continue
            iou = inter / int((ma | mb).sum())
            if iou > best_iou:
                best_iou, best_bid = iou, bid
        if best_iou >= iou_threshold and best_bid is not None:
            consensus[ma] = cid
            cid += 1
            matched_b.add(best_bid)
    return consensus


@serve.deployment(
    ray_actor_options={
        "num_cpus": 6,
        "num_gpus": 0,
        "memory": 12 * 1024**3,
        "runtime_env": {
            "pip": [
                "cellpose>=4.0",
                "stardist>=0.8",
                "tensorflow>=2.8,<2.14",
                "tifffile",
                "numpy",
                "scipy",
            ],
        },
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 2,
        "downscale_delay_s": 60,
    },
)
class EnsembleDeployment:
    def __init__(self) -> None:
        self.start_time = time.time()
        self._cellpose = None
        self._stardist = None

    async def async_init(self) -> None:
        """Load both models concurrently at startup."""
        from cellpose import models as cp_models
        from stardist.models import StarDist2D

        loop = asyncio.get_event_loop()
        logger.info("Loading Cellpose cyto3 and StarDist 2D_versatile_fluo in parallel ...")
        self._cellpose, self._stardist = await asyncio.gather(
            loop.run_in_executor(None, lambda: cp_models.CellposeModel(model_type="cyto3")),
            loop.run_in_executor(None, lambda: StarDist2D.from_pretrained("2D_versatile_fluo")),
        )
        logger.info("Both models loaded.")

    async def check_health(self) -> None:
        if self._cellpose is None or self._stardist is None:
            raise RuntimeError("Models not loaded.")

    def _run_cellpose(self, img: np.ndarray, diameter: Optional[float]) -> np.ndarray:
        masks, _, _ = self._cellpose.eval(
            img, diameter=diameter, channels=[0, 0],
            flow_threshold=0.4, cellprob_threshold=-0.5,
        )
        return masks.astype(np.int32)

    def _run_stardist(self, img: np.ndarray) -> np.ndarray:
        from csbdeep.utils import normalize
        img_norm = normalize(img, 1, 99.8)
        labels, _ = self._stardist.predict_instances(img_norm)
        return labels.astype(np.int32)

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float, bool]]:
        """Health check — returns model status and uptime."""
        return {
            "status": "ok",
            "message": "StarDist + Cellpose ensemble ready.",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "models_loaded": self._cellpose is not None and self._stardist is not None,
        }

    @schema_method
    async def segment_ensemble(
        self,
        image: list = Field(..., description="Grayscale fluorescence image as nested list (numpy.tolist())."),
        diameter: Optional[float] = Field(None, description="Cell diameter hint for Cellpose (pixels). None = auto-detect."),
        iou_threshold: float = Field(0.5, description="IoU threshold for consensus matching (0.0-1.0). Higher = stricter consensus."),
    ) -> Dict[str, Union[list, int, str]]:
        """
        Run StarDist and Cellpose in parallel, return IoU-consensus instance masks.

        Only instances detected by BOTH models with IoU >= iou_threshold are kept,
        yielding high-precision predictions. StarDist and Cellpose run concurrently
        using asyncio.gather for maximum throughput.

        Returns:
            labels: consensus instance mask as nested list.
            n_cells: number of consensus instances.
            cellpose_n: raw Cellpose detection count.
            stardist_n: raw StarDist detection count.
            consensus_n: final consensus count (same as n_cells).
            method: description of ensemble strategy.
        """
        img = np.array(image, dtype=np.float32)
        if img.ndim == 3 and img.shape[0] <= 4:
            img = img[0]

        loop = asyncio.get_event_loop()
        cp_mask, sd_mask = await asyncio.gather(
            loop.run_in_executor(None, self._run_cellpose, img, diameter),
            loop.run_in_executor(None, self._run_stardist, img),
        )

        consensus = _iou_ensemble(cp_mask, sd_mask, iou_threshold=iou_threshold)

        return {
            "labels": consensus.tolist(),
            "n_cells": int(consensus.max()),
            "cellpose_n": int(cp_mask.max()),
            "stardist_n": int(sd_mask.max()),
            "consensus_n": int(consensus.max()),
            "method": f"StarDist + Cellpose IoU ensemble (threshold={iou_threshold})",
        }

    @schema_method
    async def segment_cellpose(
        self,
        image: list = Field(..., description="Grayscale image as nested list."),
        diameter: Optional[float] = Field(None, description="Cell diameter hint (pixels)."),
    ) -> Dict[str, Union[list, int, str]]:
        """Run Cellpose only (single-model baseline for comparison)."""
        img = np.array(image, dtype=np.float32)
        if img.ndim == 3 and img.shape[0] <= 4:
            img = img[0]
        loop = asyncio.get_event_loop()
        masks = await loop.run_in_executor(None, self._run_cellpose, img, diameter)
        return {"labels": masks.tolist(), "n_cells": int(masks.max()), "model": "cellpose-cyto3"}

    @schema_method
    async def segment_stardist(
        self,
        image: list = Field(..., description="Grayscale fluorescence image as nested list."),
    ) -> Dict[str, Union[list, int, str]]:
        """Run StarDist only (single-model baseline for comparison)."""
        img = np.array(image, dtype=np.float32)
        if img.ndim == 3 and img.shape[0] <= 4:
            img = img[0]
        loop = asyncio.get_event_loop()
        masks = await loop.run_in_executor(None, self._run_stardist, img)
        return {"labels": masks.tolist(), "n_cells": int(masks.max()), "model": "stardist-2D_versatile_fluo"}

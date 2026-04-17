"""
DeepCell Mesmer — Whole-Cell Segmentation BioEngine App

Whole-cell and nuclear segmentation for spatial proteomics (CODEX, CyCIF, IMC, MIBI-ToF)
using DeepCell's Mesmer model trained on TissueNet (1M+ annotated cells).

Source: https://github.com/vanvalenlab/deepcell-tf (Apache 2.0)
Paper: Greenwald et al., Nature Biotechnology 2022
Weights: auto-downloaded from DeepCell AWS S3 on first Mesmer() instantiation.
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
        "memory": 16 * 1024**3,
        "runtime_env": {
            "pip": [
                "deepcell>=0.12",
                "tensorflow>=2.8,<2.14",
                "tifffile",
                "numpy",
            ],
        },
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 1,
        "downscale_delay_s": 120,
    },
)
class MesmerDeployment:
    def __init__(self) -> None:
        self.start_time = time.time()
        self._app = None

    async def async_init(self) -> None:
        """Download TissueNet weights and load Mesmer model."""
        from deepcell.applications import Mesmer
        loop = asyncio.get_event_loop()
        logger.info("Loading Mesmer model (TissueNet weights from DeepCell S3) ...")
        self._app = await loop.run_in_executor(None, Mesmer)
        logger.info("Mesmer ready.")

    async def check_health(self) -> None:
        if self._app is None:
            raise RuntimeError("Mesmer model not loaded.")

    def _predict(self, nuclear: np.ndarray, membrane: Optional[np.ndarray],
                 image_mpp: float, compartment: str) -> Dict:
        """Blocking predict — run in executor."""
        if nuclear.ndim == 2:
            nuclear = nuclear[np.newaxis, :, :, np.newaxis]   # (1,H,W,1)
        elif nuclear.ndim == 3:
            nuclear = nuclear[np.newaxis, ..., np.newaxis]

        if membrane is not None:
            if membrane.ndim == 2:
                membrane = membrane[np.newaxis, :, :, np.newaxis]
            inp = np.concatenate([nuclear, membrane], axis=-1)
        else:
            inp = np.concatenate([nuclear, np.zeros_like(nuclear)], axis=-1)

        result = self._app.predict(inp, image_mpp=image_mpp, compartment=compartment)
        # result shape: (1,H,W,2) or (1,H,W,1)
        if result.ndim == 4 and result.shape[-1] >= 2:
            wc = result[0, :, :, 0].astype(np.int32)
            nk = result[0, :, :, 1].astype(np.int32)
        else:
            wc = result[0, :, :, 0].astype(np.int32)
            nk = wc.copy()
        return {"whole_cell_masks": wc, "nuclear_masks": nk}

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float, bool]]:
        """Health check."""
        return {
            "status": "ok",
            "message": "DeepCell Mesmer whole-cell segmentation ready.",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "model_loaded": self._app is not None,
        }

    @schema_method
    async def segment_cells(
        self,
        nuclear_image: list = Field(..., description=(
            "Nuclear marker image (DAPI, Hoechst, CD45) as nested list [H, W] or [1, H, W]."
        )),
        membrane_image: Optional[list] = Field(None, description=(
            "Membrane/cytoplasm marker (optional). Same shape as nuclear_image. "
            "If omitted, zero-filled membrane channel is used."
        )),
        image_mpp: float = Field(0.5, description=(
            "Microns per pixel. Default 0.5 for CODEX/CyCIF. "
            "Use 0.2125 for 10x Xenium, 0.138 for 40x objective."
        )),
        compartment: str = Field("whole-cell", description=(
            "Segmentation target: 'whole-cell', 'nuclear', or 'both'."
        )),
    ) -> Dict[str, Union[list, int, str]]:
        """
        Whole-cell and nuclear segmentation for spatial proteomics.

        Uses DeepCell's Mesmer model (TissueNet, Nature Biotechnology 2022) to segment
        cells in multiplexed tissue images from CODEX, CyCIF, IMC, or MIBI-ToF platforms.
        Accepts a nuclear marker channel and an optional membrane/cytoplasm marker.

        Returns:
            whole_cell_masks: whole-cell instance mask as nested list.
            nuclear_masks: nuclear instance mask as nested list.
            n_cells: number of detected cells.
            compartment: compartment used.
            image_mpp: pixel size used.
        """
        nuc = np.array(nuclear_image, dtype=np.float32)
        mem = np.array(membrane_image, dtype=np.float32) if membrane_image is not None else None

        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            None, self._predict, nuc, mem, image_mpp, compartment
        )

        return {
            "whole_cell_masks": out["whole_cell_masks"].tolist(),
            "nuclear_masks":    out["nuclear_masks"].tolist(),
            "n_cells":          int(out["whole_cell_masks"].max()),
            "compartment":      compartment,
            "image_mpp":        image_mpp,
            "platform_note": (
                "Optimised for CODEX/CyCIF at 0.5 µm/px. "
                "Adjust image_mpp: Xenium=0.2125, MERSCOPE=0.108, IMC=1.0."
            ),
        }

    @schema_method
    async def segment_xenium_dapi(
        self,
        dapi_image: list = Field(..., description="DAPI image from 10x Xenium as nested list [H, W]."),
        image_mpp: float = Field(0.2125, description="Xenium pixel size in µm (default 0.2125)."),
    ) -> Dict[str, Union[list, int, str]]:
        """
        Convenience wrapper for 10x Xenium spatial transcriptomics data.

        Runs nuclear segmentation at Xenium's native pixel size (0.2125 µm/px).
        Output nuclear masks are ready for transcript-to-cell assignment.

        Returns:
            nuclear_masks: nucleus instance masks as nested list.
            n_cells: number of nuclei detected.
            platform: "10x Xenium".
        """
        nuc = np.array(dapi_image, dtype=np.float32)
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(
            None, self._predict, nuc, None, image_mpp, "nuclear"
        )
        return {
            "nuclear_masks": out["nuclear_masks"].tolist(),
            "n_cells": int(out["nuclear_masks"].max()),
            "platform": "10x Xenium",
            "image_mpp": image_mpp,
        }

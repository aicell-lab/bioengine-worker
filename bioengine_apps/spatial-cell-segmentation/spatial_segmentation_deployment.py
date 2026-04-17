"""
BioEngine Spatial Cell Segmentation Deployment

Cell segmentation for spatial transcriptomics platforms (Xenium, CosMx, Visium HD, MERSCOPE).
Uses Cellpose with cyto3 model trained on diverse tissue/DAPI images.

Provides two primary services:
1. segment_tissue_section — run Cellpose on a DAPI/nuclear stain image, return cell masks
2. assign_transcripts_to_cells — assign per-transcript (x, y) coordinates to segmented cells

This mimics what platform-native tools (e.g. Xenium Explorer) do but as an open,
re-usable BioEngine service that integrates with AnnData/Squidpy workflows.

IMPORT HANDLING:
- Standard Python and BioEngine-bundled libraries are imported at the top.
- cellpose, tifffile, scipy, skimage are specified in runtime_env and imported inside methods.
"""

import logging
import math
import time
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        "num_cpus": 4,
        "num_gpus": 0,
        "memory": 8 * 1024**3,
        "runtime_env": {
            "pip": [
                "cellpose>=4.0",
                "tifffile",
                "numpy",
                "scipy",
            ],
        },
    }
)
class SpatialSegmentationDeployment:
    def __init__(self) -> None:
        """Initialise the deployment; model is loaded in async_init."""
        self.start_time = time.time()
        self.model = None

    # === Lifecycle methods ===

    async def async_init(self) -> None:
        """
        Load Cellpose cyto3 model at startup.
        cyto3 is the recommended model for tissue sections and DAPI-stained images.
        """
        from cellpose import models

        logger.info("Loading Cellpose cyto3 model...")
        self.model = models.Cellpose(model_type="cyto3", gpu=False)
        logger.info("Cellpose cyto3 model loaded successfully.")

    async def check_health(self) -> None:
        """Ray Serve health check — fails if model was never loaded."""
        if self.model is None:
            raise RuntimeError("Cellpose model is not loaded.")

    # === Exposed API methods ===

    @schema_method
    async def ping(self) -> Dict[str, Union[str, float]]:
        """
        Ping the deployment to test connectivity.

        Returns:
            Dict with status, message, timestamp, and uptime in seconds.
        """
        return {
            "status": "ok",
            "message": "Spatial Cell Segmentation service is running.",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
        }

    @schema_method
    async def segment_tissue_section(
        self,
        image: list = Field(..., description="2-D DAPI/nuclear stain image as a nested list (H x W, uint8 or uint16)."),
        pixel_size_um: float = Field(0.2125, description="Physical pixel size in micrometres. Xenium default: 0.2125 µm/px."),
        min_cell_diameter_um: float = Field(5.0, description="Minimum expected cell diameter in µm. Cells smaller than the corresponding area are removed."),
        max_cell_diameter_um: float = Field(30.0, description="Maximum expected cell diameter in µm. Used for reference only; Cellpose diameter guides detection."),
        platform: str = Field("xenium", description="Source platform: 'xenium', 'cosmx', 'visium_hd', or 'merscope'. Used for metadata only."),
    ) -> Dict[str, Union[list, int, float, str]]:
        """
        Segment cells in a DAPI/nuclear stain image from a spatial transcriptomics section.

        The method runs Cellpose (cyto3) using a diameter derived from the pixel size,
        then removes masks smaller than the minimum cell area.

        Returns:
            Dict with keys:
              - cell_masks: 2-D integer array (H x W) where 0 = background, >0 = cell IDs (nested list)
              - n_cells: number of detected cells after filtering
              - platform: echo of the input platform string
              - pixel_size_um: echo of the input pixel size
              - mean_cell_area_um2: mean cell area in µm² across all detected cells
        """
        from skimage.measure import regionprops

        arr = np.array(image, dtype=np.float32)

        # Derive target diameter in pixels from a reference cell size of 10 µm
        diameter_px = 10.0 / pixel_size_um

        logger.info(
            f"Segmenting {arr.shape} image | platform={platform} | "
            f"pixel_size={pixel_size_um} µm | diameter_px={diameter_px:.1f}"
        )

        # Run Cellpose; channels=[0,0] means grayscale input, no cytoplasm channel
        masks, _, _ = self.model.eval(
            arr,
            diameter=diameter_px,
            channels=[0, 0],
            cellprob_threshold=-0.5,
        )

        # Filter out cells smaller than min_cell_diameter_um^2 * pi/4 pixels
        min_area_px = math.pi / 4.0 * (min_cell_diameter_um / pixel_size_um) ** 2
        props = regionprops(masks)

        small_labels = {p.label for p in props if p.area < min_area_px}
        if small_labels:
            logger.info(f"Removing {len(small_labels)} cells smaller than {min_area_px:.1f} px²")
            remove_mask = np.isin(masks, list(small_labels))
            masks[remove_mask] = 0

        # Re-compute props after filtering for statistics
        props = regionprops(masks)
        n_cells = len(props)
        if n_cells > 0:
            px_to_um2 = pixel_size_um ** 2
            mean_area_um2 = float(np.mean([p.area for p in props])) * px_to_um2
        else:
            mean_area_um2 = 0.0

        logger.info(f"Segmentation complete: {n_cells} cells detected.")

        return {
            "cell_masks": masks.tolist(),
            "n_cells": n_cells,
            "platform": platform,
            "pixel_size_um": pixel_size_um,
            "mean_cell_area_um2": mean_area_um2,
        }

    @schema_method
    async def assign_transcripts_to_cells(
        self,
        cell_masks: list = Field(..., description="2-D integer cell mask array (H x W) as a nested list, as returned by segment_tissue_section."),
        transcript_x: list = Field(..., description="List of transcript x-coordinates in pixels (float or int)."),
        transcript_y: list = Field(..., description="List of transcript y-coordinates in pixels (float or int)."),
        transcript_gene: list = Field(..., description="List of gene names (str), one per transcript. Must be same length as transcript_x/y."),
    ) -> Dict[str, Union[list, int]]:
        """
        Assign transcript detections to segmented cells.

        For each transcript at (x, y), looks up the cell mask ID at that pixel.
        Transcripts that fall in background (mask == 0) are marked as unassigned (-1).

        This replicates the transcript-to-cell assignment step performed by
        Xenium Explorer and similar platform tools, as an open BioEngine service.

        Returns:
            Dict with keys:
              - cell_id_per_transcript: list of cell IDs (int), one per transcript;
                -1 means the transcript is in background
              - n_assigned: number of transcripts assigned to a cell
              - n_background: number of transcripts in background (unassigned)
              - cells_with_transcripts: number of unique cells that received at least one transcript
        """
        masks = np.array(cell_masks, dtype=np.int32)
        xs = np.array(transcript_x, dtype=np.float64)
        ys = np.array(transcript_y, dtype=np.float64)

        h, w = masks.shape

        # Round to nearest pixel and clip to image bounds
        xi = np.clip(np.round(xs).astype(np.int64), 0, w - 1)
        yi = np.clip(np.round(ys).astype(np.int64), 0, h - 1)

        cell_ids = masks[yi, xi].tolist()  # list of ints, 0 = background

        # Convert 0 (background) to -1 for clarity; positive values are cell IDs
        cell_id_per_transcript = [cid if cid > 0 else -1 for cid in cell_ids]

        n_assigned = sum(1 for cid in cell_id_per_transcript if cid > 0)
        n_background = len(cell_id_per_transcript) - n_assigned
        cells_with_transcripts = len({cid for cid in cell_id_per_transcript if cid > 0})

        logger.info(
            f"Transcript assignment: {n_assigned}/{len(cell_id_per_transcript)} assigned "
            f"to {cells_with_transcripts} cells."
        )

        return {
            "cell_id_per_transcript": cell_id_per_transcript,
            "n_assigned": n_assigned,
            "n_background": n_background,
            "cells_with_transcripts": cells_with_transcripts,
        }

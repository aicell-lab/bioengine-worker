from cellpose import models
from typing import List
import numpy as np
import zip_util

async def service(zip_data: bytes, context=None) -> List[np.ndarray]:
    masks_list = []
    model = models.Cellpose(gpu=False, model_type='cyto3')
    for img_data in zip_util.extract_image_data_from_zip(zip_data):
        masks, flows, styles, diams = model.eval([img_data], diameter=None, channels=[0, 0])
        masks_list.append(masks[0])
    return masks_list





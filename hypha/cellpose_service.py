from cellpose import models
import zip_util
import numpy as np
import ray
import asyncio

@ray.remote(num_gpus=1)
def sync_ray_service(encoded_zip: str, context=None) -> str:
    return asyncio.run(service(encoded_zip, context))

async def service(encoded_zip: str, context=None) -> str:
    masks_list = []
    model = models.Cellpose(gpu=False, model_type='cyto3')
    for img_data in zip_util.extract_image_data_from_zip(zip_util.decode_base64_zip(encoded_zip)):
        masks, _, _, _ = model.eval([img_data], diameter=None, channels=[0, 0])
        mask_scaled = (masks[0] * 255).astype(np.uint8)
        masks_list.append(mask_scaled)
    return zip_util.encode_to_base64(zip_util.pack_zip(masks_list))





from cellpose import models
from hypha.zip_util import extract_image_data_from_zip, decode_base64_zip, encode_to_base64, pack_zip
import numpy as np
import ray

def _get_mask(img_data, model):
    masks, _, _, _ = model.eval([img_data], diameter=None, channels=[0, 0])
    mask_scaled = (masks[0] * 255).astype(np.uint8)
    return mask_scaled

def _get_model():
    return models.Cellpose(gpu=True, model_type='cyto3')

@ray.remote(num_gpus=1)
def service(encoded_zip: str, context=None) -> str:
    masks_list = []
    model = _get_model()
    zip_data = decode_base64_zip(encoded_zip)
    for img_data in extract_image_data_from_zip(zip_data):
        masks_list.append(_get_mask(img_data=img_data, model=model))
    return encode_to_base64(pack_zip(masks_list))

@ray.remote(num_gpus=1)
def test_cellpose(img_data, context=None) -> str:
    return _get_mask(img_data=img_data, model=_get_model())

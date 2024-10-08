from cellpose import models
import hypha.zip_util
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
    for img_data in zip_util.extract_image_data_from_zip(zip_util.decode_base64_zip(encoded_zip)):
        masks_list.append(_get_mask(img_data=img_data, model=model))
    return zip_util.encode_to_base64(zip_util.pack_zip(masks_list))

@ray.remote(num_gpus=1)
def test_cellpose(img_data, context=None) -> str:
    #assert len(img_data.shape) == 3, "Image data must be 3-dimensional (H, W, C)"
    #assert img_data.shape[2] == 1 or img_data.shape[2] == 3, "Image must have 1 channel (grayscale) or 3 channels (RGB)"
    #assert img_data.shape[0] > 0 and img_data.shape[1] > 0, "Image dimensions must be positive"

    return _get_mask(img_data=img_data, model=_get_model())





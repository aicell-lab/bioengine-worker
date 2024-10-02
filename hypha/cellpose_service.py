from cellpose import models
import zip_util
import numpy as np
import ray

@ray.remote(num_gpus=1)
def service(encoded_zip: str, context=None) -> str:
    masks_list = []
    model = models.Cellpose(gpu=True, model_type='cyto3')
    for img_data in zip_util.extract_image_data_from_zip(zip_util.decode_base64_zip(encoded_zip)):
        masks, _, _, _ = model.eval([img_data], diameter=None, channels=[0, 0])
        mask_scaled = (masks[0] * 255).astype(np.uint8)
        masks_list.append(mask_scaled)
    return zip_util.encode_to_base64(zip_util.pack_zip(masks_list))

#@ray.remote(num_gpus=1)
#def test_cellpose(img_data, context=None) -> str:
    #assert img_data.shape
#    model = models.Cellpose(gpu=False, model_type='cyto3')
#    masks, _, _, _ = model.eval([img_data], diameter=None, channels=[0, 0])
#    return (masks[0] * 255).astype(np.uint8)





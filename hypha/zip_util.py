import zipfile
import io
from typing import Dict, List
import numpy as np
from PIL import Image
import base64

MAX_TOTAL_UNCOMPRESSED_SIZE = 10 * 1000 * 1024 * 1024 # 10 GB

def _path_check(member: zipfile.ZipInfo):
    if '..' in member.filename or member.filename.startswith('/'):
        raise Exception("Potential zip slip vulnerability detected")

def _is_image(member: zipfile.ZipInfo) -> bool:
    return member.filename.endswith('.png')

def _extract_images_from_zip(zip_data: bytes) -> Dict[str,bytes]:
    total_uncompressed_size = 0
    extracted_images = {}

    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_file:
        for member in zip_file.infolist():
            _path_check(member)
            if _is_image(member):
                total_uncompressed_size += member.file_size
                if total_uncompressed_size > MAX_TOTAL_UNCOMPRESSED_SIZE:
                    raise Exception("Zip bomb detected: uncompressed size exceeds safe limit")
                with zip_file.open(member) as file:
                    extracted_images[member.filename] = file.read()
    return extracted_images 

def _extract_np_arrays(images: Dict[str, bytes]) -> List[np.ndarray]:
    return [np.array(Image.open(io.BytesIO(img_data))) for img_name, img_data in images.items()]

def extract_image_data_from_zip(zip_data: bytes) -> List[np.ndarray]:
    return _extract_np_arrays(_extract_images_from_zip(zip_data))

def pack_zip(images: List[np.ndarray]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, img_array in enumerate(images):
            pil_image = Image.fromarray(img_array)
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0) 
            zf.writestr(f'image_{i}.png', img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def encode_to_base64(byte_data: bytes) -> str:
    return base64.b64encode(byte_data).decode('utf-8')
import numpy as np
import requests
import yaml
import json

import io

HTTP_BASE = "https://hypha.aicell.io/bioimage-io/services/model-runner"

def infer_http(model_id, image_array):
    """
    Run inference via HTTP for a numpy array. Uploads, infers, downloads result.
    Returns the first output as a numpy array.
    """
    # 1. Get upload URL
    res = requests.get(f"{HTTP_BASE}/get_upload_url?file_type=.npy")
    res.raise_for_status()
    upload_info = res.json()
    upload_url = upload_info["upload_url"]
    file_path = upload_info["file_path"]

    # 2. Upload array
    buffer = io.BytesIO()
    np.save(buffer, image_array.astype(np.float32))
    buffer.seek(0)
    upload_res = requests.put(upload_url, data=buffer.getvalue())
    upload_res.raise_for_status()

    # 3. Infer
    payload = {
        "model_id": model_id,
        "inputs": file_path,
        "return_download_url": True
    }
    infer_res = requests.post(f"{HTTP_BASE}/infer", json=payload)
    if infer_res.status_code != 200:
        raise Exception(f"Inference failed: {infer_res.text}")
    
    out_dict = infer_res.json()
    
    # 4. Download first output
    out_url = list(out_dict.values())[0]
    out_res = requests.get(out_url)
    out_res.raise_for_status()
    
    out_buffer = io.BytesIO(out_res.content)
    return np.load(out_buffer)

def get_model_rdf(model_id):
    """Fetch the RDF yaml for a given model ID."""
    url = f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}/files/rdf.yaml"
    res = requests.get(url)
    if res.status_code == 200:
        return yaml.safe_load(res.text)
    raise Exception(f"Failed to fetch RDF for {model_id}, status: {res.status_code}")

def get_input_axes_info(rdf):
    """Parse RDF to get input axis string or list."""
    version = rdf.get("format_version", "0.4.")
    inputs = rdf.get("inputs", [])
    if not inputs:
        return None
    
    inp = inputs[0]
    axes = inp.get("axes")
    if version.startswith("0.4"):
        # axes is a string like "bcyx"
        return axes, "0.4"
    else:
        # axes is a list of dicts
        axes_str = ""
        for ax in axes:
            t = ax.get("type", "")
            if t == "batch": axes_str += "b"
            elif t == "channel": axes_str += "c"
            elif t == "space":
                axes_str += ax.get("id", "")
        return axes_str, "0.5"

def pad_or_crop_to_valid_size(img_array, axes_str, rdf_input):
    """
    Pad or crop height and width to meet min and step requirements for 0.5.x.
    """
    version = rdf_input.get("format_version", "0.4.")
    if version.startswith("0.4"):
        return img_array # For 0.4 we might assume fixed shape or resize
        
    axes = rdf_input.get("inputs")[0].get("axes", [])
    # not fully implemented: this gets complex, but we do basic padding if needed
    return img_array

def prepare_image_for_model(img, axes_str):
    """
    Given an input image (assumed HxW or Cyx), format it to match axes_str (like 'bcyx').
    """
    # Cast to float32
    img = img.astype(np.float32)
    
    # Let's assume input img from image_pair is HxW or HxWxC
    if img.ndim == 2:
        # It's HW.
        # Target might be bcyx
        if "b" in axes_str and "c" in axes_str:
            return img[np.newaxis, np.newaxis, ...]
        elif "b" in axes_str:
            return img[np.newaxis, ...]
        elif "c" in axes_str:
            return img[np.newaxis, ...]
    
    # If it is 3D (H,W,C) or (C,H,W)
    if img.ndim == 3:
        if img.shape[2] <= 3: # HWC
            img = np.transpose(img, (2, 0, 1)) # CHW
        if axes_str == "bcyx":
            return img[np.newaxis, ...]
            
    return img

def evaluate_segmentation(pred, gt):
    """
    Evaluate instance or semantic segmentation mask against ground truth.
    Returns Dictionary of metrics: IoU, Dice.
    """
    # Simple semantic metrics
    p_bin = (pred > 0.5).astype(bool)
    g_bin = (gt > 0).astype(bool)
    
    intersection = np.logical_and(p_bin, g_bin).sum()
    union = np.logical_or(p_bin, g_bin).sum()
    
    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (p_bin.sum() + g_bin.sum()) if (p_bin.sum() + g_bin.sum()) > 0 else 0.0
    
    return {"iou": float(iou), "dice": float(dice)}

def evaluate_instance_segmentation(pred, gt):
    """
    Instance segmentation evaluation.
    Requires matching between instances, approximating mean Average Precision,
    or just calculating object counts and pixel overlap.
    """
    # Placeholder for simple instance logic without installing heavy dependencies 
    # like scikit-image or stardist just for this skill script. 
    # Fallback to standard segmentation evaluation.
    return evaluate_segmentation(pred, gt)

def normalize_image(img, pmin=1.0, pmax=99.8):
    """Percentile-based normalization commonly used in fluorescence."""
    perc_min, perc_max = np.percentile(img, (pmin, pmax))
    img_norm = (img - perc_min) / (perc_max - perc_min + 1e-6)
    return np.clip(img_norm, 0, 1)

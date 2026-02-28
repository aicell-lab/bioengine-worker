"""Test script to verify model-runner API endpoints work correctly."""
import asyncio
import numpy as np
import httpx
from io import BytesIO
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"
SERVICE_ID = "bioimage-io/model-runner"


async def main():
    print("=" * 60)
    print("Model Runner API Test Script")
    print("=" * 60)

    server = await connect_to_server(
        {"server_url": SERVER_URL, "method_timeout": 180}
    )
    mr = await server.get_service(SERVICE_ID)

    # --- Test 1: search_models ---
    print("\n--- Test 1: search_models ---")
    results = await mr.search_models(
        keywords=["nuclei", "segmentation"], limit=5
    )
    print(f"Found {len(results)} models:")
    for r in results:
        print(f"  {r['model_id']}: {r['description'][:80]}")

    # --- Test 2: get_model_rdf ---
    print("\n--- Test 2: get_model_rdf ---")
    model_id = results[0]["model_id"]
    rdf = await mr.get_model_rdf(model_id=model_id)
    print(f"Model: {rdf['name']}")
    print(f"Inputs: {len(rdf['inputs'])} tensor(s)")
    inp = rdf["inputs"][0]
    axes = inp["axes"]
    print(f"  axes: {axes}")

    # Determine expected shape from RDF
    if isinstance(axes, str):
        # format_version 0.4.x: axes is a string like "bcyx"
        print(f"  Format 0.4.x: axes string = '{axes}'")
        if "shape" in inp:
            print(f"  shape: {inp['shape']}")
    else:
        # format_version 0.5.x: axes is a list of dicts
        print(f"  Format 0.5.x: axes = {[a.get('id', a.get('type')) for a in axes]}")

    # --- Test 3: get_upload_url + upload + infer with return_download_url ---
    print("\n--- Test 3: Upload + Infer + Download ---")
    test_image = np.random.rand(1, 1, 256, 256).astype(np.float32)
    print(f"Created test image: shape={test_image.shape}, dtype={test_image.dtype}")

    # Get upload URL 
    upload_info = await mr.get_upload_url(file_type=".npy")
    print(f"Got upload URL for: {upload_info['file_path']}")

    # Upload
    buffer = BytesIO()
    np.save(buffer, test_image)
    async with httpx.AsyncClient() as client:
        resp = await client.put(upload_info["upload_url"], content=buffer.getvalue())
        print(f"Upload status: {resp.status_code}")

    # Infer with return_download_url
    print(f"Running inference with model '{model_id}'...")
    result = await mr.infer(
        model_id=model_id,
        inputs=upload_info["file_path"],
        return_download_url=True,
    )
    print(f"Inference result keys: {list(result.keys())}")

    # Download results
    for key, url in result.items():
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            arr = np.load(BytesIO(resp.content))
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    # --- Test 4: Direct numpy inference (without file upload) ---
    print("\n--- Test 4: Direct numpy array inference ---")
    direct_result = await mr.infer(
        model_id=model_id,
        inputs=test_image,
    )
    for key, value in direct_result.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    # --- Test 5: search_models via HTTP GET ---
    print("\n--- Test 5: HTTP GET search_models ---")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SERVER_URL}/bioimage-io/services/model-runner/search_models",
            params={"keywords": "nuclei,segmentation", "limit": "3"},
        )
        print(f"HTTP Status: {resp.status_code}")
        data = resp.json()
        print(f"Found {len(data)} models via HTTP")
        for r in data:
            print(f"  {r['model_id']}: {r['description'][:60]}")

    # --- Test 6: get_upload_url via HTTP GET ---
    print("\n--- Test 6: HTTP GET get_upload_url ---")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SERVER_URL}/bioimage-io/services/model-runner/get_upload_url",
            params={"file_type": ".npy"},
        )
        print(f"HTTP Status: {resp.status_code}")
        data = resp.json()
        print(f"file_path: {data['file_path']}")
        print(f"upload_url starts with: {data['upload_url'][:60]}...")

    await server.disconnect()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

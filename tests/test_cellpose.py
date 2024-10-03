import pytest
from hypha_rpc import connect_to_server
from dotenv import load_dotenv
import os
import imageio.v2 as imageio
import requests

load_dotenv()

class ImgConfig:
    img_link = "https://www.mathworks.com/help/medical-imaging/ug/cellpose-gs-basic-workflow-result.png"
    img_name = "test_img.png"
    img_out_name = "test_img_out.png"

def download_image():
    if not os.path.exists(ImgConfig.img_name):
        print(f"Image {ImgConfig.img_name} not found. Downloading...")
        response = requests.get(ImgConfig.img_link)
        if response.status_code == 200:
            with open(ImgConfig.img_name, 'wb') as f:
                f.write(response.content)
            print(f"Image downloaded and saved as {ImgConfig.img_name}")
        else:
            pytest.fail(f"Failed to download the image: {response.status_code}")

async def connect_to_berzelius():
    server = None
    try:
        server = await connect_to_server({"token": os.environ.get("HYPHA_TEST_TOKEN"),
                                      "server_url": "https://hypha.aicell.io",
                                       "workspace": "hpc-ray-cluster" })
    except Exception as e:
        pytest.fail(f"Connecting to the server failed: {e}")
    return server

@pytest.mark.asyncio
async def test_cellpose():
    download_image()
    server = await connect_to_berzelius()
    svc = await server.get_service("hpc-ray-cluster/berzelius:ray")
    img_data = imageio.imread(ImgConfig.img_name)
    print(f"Image shape: {img_data.shape}")
    ret = await svc.test_cellpose(img_data)
    print(ret)
    imageio.imwrite('output.png', ret)
    


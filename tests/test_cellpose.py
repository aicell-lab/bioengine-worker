import pytest
import hypha_rpc
from hypha_rpc import connect_to_server
from dotenv import load_dotenv
import os
import imageio.v2 as imageio

import pathlib
script_directory = pathlib.Path(__file__).parent.resolve()
print(script_directory)

load_dotenv()

@pytest.mark.asyncio
async def test_cellpose():
    server = await connect_to_server({"token": os.environ.get("HYPHA_TEST_TOKEN"),
                                      "server_url": "https://hypha.aicell.io",
                                       "workspace": "hpc-ray-cluster" })

    svc = await server.get_service("hpc-ray-cluster/berzelius:ray")
    img_data = imageio.imread(f"{script_directory}/test_img.png")
    print(f"Image shape: {img_data.shape}")
    ret = await svc.test_cellpose(img_data)
    print(ret)
    imageio.imwrite('output.png', ret)
    


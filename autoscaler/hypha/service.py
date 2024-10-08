from typing import List, Callable
import asyncio
from hypha.connection import Hypha
import hypha.cellpose_service
import ray

def create_services() -> List[Callable]:
    def hello_world_task(context=None):
        return "hello world!"
    
    async def test_cellpose(img_data, context=None):
        return await hypha.cellpose_service.test_cellpose.remote(img_data)
    
    async def cellpose_inference(encoded_zip: str = None, context=None):
        if encoded_zip is None:
            return {"success": False, "message": f"Missing argument 'encoded_zip' (type: zip file encoded as base64)"}
        result = await hypha.cellpose_service.service.remote(encoded_zip)
        return {"success": True, "encoded_zip": result}
    
    return [hello_world_task, cellpose_inference, test_cellpose]

async def register_services():
    services = create_services()
    server = await Hypha.authenticate()
    if server:
        service_info = await Hypha.register_service(server_handle=server, callbacks=services)
        Hypha.print_services(service_info=service_info, callbacks=services)



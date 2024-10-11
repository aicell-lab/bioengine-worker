from typing import List, Callable
from hypha.connection import Hypha
import hypha.cellpose_service
from hypha.admin import AdminChecker
from config import Config

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

def apply_admin_check(services: List[Callable]) -> List[Callable]:
    """Apply the admin verification decorator to each service function."""
    admin_checker = AdminChecker(Config.Workspace.admin_emails)
    return [admin_checker.context_verification()(service) for service in services]

async def register_services():
    services = apply_admin_check(create_services())
    server = await Hypha.authenticate()
    if server:
        service_info = await Hypha.register_service(server_handle=server, callbacks=services)
        Hypha.print_services(service_info=service_info, callbacks=services)



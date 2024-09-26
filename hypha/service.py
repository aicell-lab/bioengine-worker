from typing import List, Callable
import asyncio
from connection import Hypha

def create_services() -> List[Callable]:
    def hello_world_task(context=None):
        return "hello world!"
    
    async def cellpose_inference(encoded_zip: str = None, context=None):
        if encoded_zip is None:
            return {"success": False, "message": f"Missing argument 'encoded_zip' (type: zip file encoded as base64)"}
        
        encoded_zip_result = "..."
        return {"success": True, "encoded_zip": encoded_zip_result}
    
    return [hello_world_task, cellpose_inference]

async def register_service():
    services = create_services()
    server = await Hypha.authenticate()
    service_info = await Hypha.register_service(server_handle=server, callbacks=services)
    Hypha.print_services(service_info=service_info, callbacks=services)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(register_service())
    loop.run_forever()


from typing import List, Callable
import asyncio
from connection import Hypha

def create_services() -> List[Callable]:
    def inference_task(context=None):
        return "my_inference_result"
    async def another_task(context=None):
        import numpy as np
        import zip_util
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        zip_bytes = zip_util.pack_zip(images)
        base64_encoded = zip_util.encode_to_base64(zip_bytes)
        return {"success": True, "zip_file_encoded": base64_encoded}

    return [inference_task, another_task]

async def register_service():
    services = create_services()
    server = await Hypha.authenticate()
    service_info = await Hypha.register_service(server_handle=server, callbacks=services)
    Hypha.print_services(service_info=service_info, callbacks=services)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(register_service())
    loop.run_forever()


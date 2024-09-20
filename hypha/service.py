from typing import List, Callable
import asyncio
from connection import Hypha


def create_services() -> List[Callable]:
    def inference_task(context=None):
        return "my_inference_result"
    def another_task(context=None):
        return "another_result"
    return [inference_task, another_task]

async def register_service():
    services = create_services()
    service_info = await Hypha.register_service(server_handle=await Hypha.authenticate(), callbacks=services)
    Hypha.print_services(service_info=service_info, callbacks=services)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(register_service())
    loop.run_forever()

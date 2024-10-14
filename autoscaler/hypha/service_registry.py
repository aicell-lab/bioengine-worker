from typing import List, Callable, Any
from hypha.connection import Hypha
from hypha.admin import AdminChecker
from config import Config
import inspect
from hypha.connection import Hypha
import ray

class RemoteRayMethods:
    @ray.remote(num_gpus=1)
    def exec_str(script: str) -> str:
        local_context = {}
        exec(script, {}, local_context)
        return local_context.get('result', '') 
    
    @ray.remote(num_gpus=1)
    def exec_bytes(script: str) -> str:
        local_context = {}
        exec(script, {}, local_context)
        return local_context.get('result', b'')
    
def service_method(func):
    func._is_service = True
    return func

class ServiceRegistry:
    def __init__(self):
        self.admin_checker = AdminChecker(Config.Workspace.admin_emails)

    @service_method
    def hello_world_task(self, context=None) -> str:
        return "hello world!"
    
    @service_method
    async def exec_str(self, script: str, context=None) -> str:
        return await RemoteRayMethods.exec_str.remote(script)

    @service_method
    async def exec_bytes(self, script: str, context=None) -> bytes:
        return await RemoteRayMethods.exec_bytes.remote(script)

    def _get_service_methods(self) -> List[Callable]:
        """Collect only methods marked as service methods."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        return [method for _, method in methods if getattr(method, '_is_service', False)]

    def apply_admin_check(self, services: List[Callable]) -> List[Callable]:
        """Apply the admin verification decorator to each service method."""
        return [self.admin_checker.context_verification()(service) for service in services]
    
    def get_services(self) -> List[Callable]:
        return self.apply_admin_check(self._get_service_methods())
    

async def register_services():
    services = ServiceRegistry().get_services()
    server = await Hypha.authenticate()
    if server:
        service_info = await Hypha.register_service(server_handle=server, callbacks=services)
        Hypha.print_services(service_info=service_info, callbacks=services)
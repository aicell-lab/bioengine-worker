from datetime import datetime
from status import Status
from config import Config
import terminal

class ZombieTerminator:
    
    def __init__(self, status: Status):
        self.status = status
        self.last_work_time = datetime.now()

    def _terminate_jobs(self):
        #terminal.run_command()
        ...

    def _are_workers_zombies(self) -> bool:
        return not self.status.ray_metrics.is_work_ongoing() and self.status.job_metrics.total_jobs > 0
    
    def _is_zombie_timeout(self) -> bool:
        zombie_time = datetime.now() - self.last_work_time
        return zombie_time > Config.ZOMBIE_TIMEOUT

    def _should_terminate_jobs(self) -> bool:
        return self._are_workers_zombies() and self._is_zombie_timeout()
    
    def update(self):
        if self.status.ray_metrics.is_work_ongoing():
            self.last_work_time = datetime.now()
        if self._should_terminate_jobs():
            self._terminate_jobs()
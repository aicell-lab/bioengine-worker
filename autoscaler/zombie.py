from datetime import datetime
from status import Status
from config import Config
from typing import List
import terminal
import os

class ZombieTerminator:
    
    def __init__(self, status: Status):
        self.status = status
        self.last_work_time = datetime.now()

    @staticmethod
    def _get_job_ids() -> List[str]:
        user = os.getenv("USER")
        job_ids_output = terminal.run_command(["squeue", "-u", user, "-h", "-o", "%A"])    
        return job_ids_output.splitlines()

    def _terminate_jobs(self, job_ids: List[str]):
        for job_id in job_ids:
            terminal.run_command(["scancel", job_id])

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
             self._terminate_jobs(ZombieTerminator._get_job_ids())
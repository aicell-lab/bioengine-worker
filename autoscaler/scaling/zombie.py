from datetime import datetime
from scaling.status import Status
from config import Config
from typing import List
import terminal
import getpass

class ZombieTerminator:
    
    def __init__(self, status: Status):
        self.status = status
        self.last_work_time = datetime.now()

    @staticmethod
    def _get_job_ids() -> List[str]:
        user = getpass.getuser()
        job_ids_output = terminal.run_command(["squeue", "-u", user, "-h", "-o", "%A"])    
        return job_ids_output.splitlines() if job_ids_output else []
    
    @staticmethod
    def _terminate_job(job_id: str):
        terminal.run_command(["scancel", job_id])
    
    @staticmethod
    def _terminate_jobs(job_ids: List[str]):
        for job_id in job_ids:
            ZombieTerminator._terminate_job(job_id)

    @staticmethod
    def terminate_all_jobs():
        ZombieTerminator._terminate_jobs(ZombieTerminator._get_job_ids())

    @staticmethod
    def terminate_jobs(num_jobs: int):
        job_ids = ZombieTerminator._get_job_ids()
        ZombieTerminator._terminate_jobs(job_ids[:num_jobs])

    def _are_workers_zombies(self) -> bool:
        return not self.status.ray_metrics.is_work_ongoing() and self.status.job_metrics.total_jobs > 0
    
    def _is_zombie_timeout(self) -> bool:
        zombie_time = datetime.now() - self.last_work_time
        return zombie_time > Config.Scaling.ZOMBIE_TIMEOUT

    def _should_terminate_jobs(self) -> bool:
        return self._are_workers_zombies() and self._is_zombie_timeout()
    
    def _has_more_than_min_workers(self) -> bool:
        return self.status.ray_metrics.num_workers > Config.Scaling.MIN_NODES

    def _get_num_workers_to_terminate(self) -> int:
        return self.status.ray_metrics.num_workers - Config.Scaling.MIN_NODES

    def update(self):
        if self.status.ray_metrics.is_work_ongoing() or not self._has_more_than_min_workers():
            self.last_work_time = datetime.now()
        if self._should_terminate_jobs():
            self.last_work_time = datetime.now()
            ZombieTerminator.terminate_jobs(self._get_num_workers_to_terminate())

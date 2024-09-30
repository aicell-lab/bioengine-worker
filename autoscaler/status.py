import logging
import terminal
from config import Config
import ray
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class JobMetrics:
    pending_jobs: int = field(default=0)
    running_jobs: int = field(default=0)
    total_jobs: int = field(default=0)

    @staticmethod
    def _get_slurm_jobs_by_state(state: str) -> int:
        output = terminal.run_command(["squeue", "-u", "$USER", f"--state={state}"])
        num_jobs = 0
        if output:
            num_jobs = len(output.strip().split("\n")) - 1
        return num_jobs
    @staticmethod
    def _get_num_slurm_pending_jobs() -> int:
        return JobMetrics._get_slurm_jobs_by_state("PENDING")
    @staticmethod
    def _get_num_slurm_running_jobs() -> int:
        return JobMetrics._get_slurm_jobs_by_state("RUNNING")

    def update(self):
        self._set_jobs(pending_jobs=JobMetrics._get_num_slurm_pending_jobs(), running_jobs=JobMetrics._get_num_slurm_running_jobs())

    def _set_jobs(self, pending_jobs: int, running_jobs: int):
        self.pending_jobs = pending_jobs
        self.running_jobs = running_jobs
        self.total_jobs = self.pending_jobs + self.running_jobs

@dataclass
class RayMetrics:
    num_workers: int = field(default=0)
    num_available_workers: int = field(default=0)
    num_occupied_workers: int = field(default=0)

    def __post_init__(self):
        self.update()

    @staticmethod
    def _get_num_workers() -> int:
        return ray.cluster_resources().get("GPU", 0)

    @staticmethod
    def _get_num_available_workers() -> int:
        return ray.available_resources().get("GPU", 0)

    def update(self):
        self.num_workers = RayMetrics._get_num_workers()
        self.num_available_workers = RayMetrics._get_num_available_workers()
        self.num_occupied_workers = self.num_workers - self.num_available_workers


class Status:
    def __init__(self):
        self.job_metrics = JobMetrics()
        self.ray_metrics = RayMetrics()
        self.update()

    def update(self):
        self.job_metrics.update()
        self.ray_metrics.update()

    def is_worker_queue_full(self) -> bool:
        if self.job_metrics.total_jobs >= Config.MAX_NODES:
            return True
        if self.job_metrics.pending_jobs > 0:
            return True # Avoid spamming slurm jobs
        return False
    
    def need_more_workers(self) -> bool: # TODO: Implement logic
        return False 

    def __str__(self):
        return f"JobMetrics({self.job_metrics}) RayMetrics({self.ray_metrics})"


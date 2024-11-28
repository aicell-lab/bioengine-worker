import terminal
from config import Config
import ray
from dataclasses import dataclass, field
from ray.util.state import list_tasks
import logging
import getpass

@dataclass
class JobMetrics:
    pending_jobs: int = field(default=0)
    running_jobs: int = field(default=0)
    total_jobs: int = field(default=0)

    @staticmethod
    def _get_slurm_jobs_by_state(state: str) -> int:
        user = getpass.getuser()
        output = terminal.run_command(["squeue", "-u", user, f"--state={state}", "--noheader", "--format=%i"])
        return len(output.splitlines())
    
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
    num_pending_tasks: int = field(default=0)

    def __post_init__(self):
        self.update()

    @staticmethod
    def _get_num_workers() -> int:
        return ray.cluster_resources().get("GPU", 0)

    @staticmethod
    def _get_num_available_workers() -> int:
        return ray.available_resources().get("GPU", 0)

    @staticmethod
    def _get_num_pending_tasks() -> int:    # https://docs.ray.io/en/latest/ray-observability/user-guides/cli-sdk.html#state-api-overview-ref
        pending_tasks = list_tasks(filters=[ ("state", "=", "PENDING_NODE_ASSIGNMENT")])
        return len(pending_tasks)

    def update(self):
        self.num_workers = RayMetrics._get_num_workers()
        self.num_available_workers = RayMetrics._get_num_available_workers()
        self.num_occupied_workers = self.num_workers - self.num_available_workers
        self.num_pending_tasks = RayMetrics._get_num_pending_tasks()

    def is_work_ongoing(self):
        return self.num_occupied_workers > 0 or self.num_pending_tasks > 0



class Status:
    def __init__(self):
        self.status_log = ''
        self.prev_status_log = ''
        self.job_metrics = JobMetrics()
        self.ray_metrics = RayMetrics()
        self.update()

    def update(self):
        self.job_metrics.update()
        self.ray_metrics.update()
        self._log_status()

    def _is_job_queue_full(self) -> bool:
        slurm = self.job_metrics
        return any([
            slurm.total_jobs >= Config.Scaling.MAX_NODES,
            slurm.total_jobs > self.ray_metrics.num_workers,
            slurm.pending_jobs > 0
        ])
    
    def _log_status(self):
        self.status_log = str(self)
        if self.prev_status_log != self.status_log:
            logging.info(f"\n{self.status_log}")
            self.prev_status_log = self.status_log

    def is_worker_queue_full(self) -> bool: # Avoid spamming slurm jobs
        return self.ray_metrics.num_available_workers > 0 or self._is_job_queue_full()
    
    def need_more_workers(self) -> bool:
        return self.ray_metrics.num_pending_tasks > self.ray_metrics.num_available_workers

    def __str__(self):
        return (f"SLURM Jobs:\n"
                f"  Total:     {self.job_metrics.total_jobs}\n"
                f"  Running:   {self.job_metrics.running_jobs}\n"
                f"  Pending:   {self.job_metrics.pending_jobs}\n"
                f"\nRay Workers:\n"
                f"  Total:     {self.ray_metrics.num_workers}\n"
                f"  Available: {self.ray_metrics.num_available_workers}\n"
                f"  Occupied:  {self.ray_metrics.num_occupied_workers}\n"
                f"Pending Tasks: {self.ray_metrics.num_pending_tasks}\n")


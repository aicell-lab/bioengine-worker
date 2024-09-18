import subprocess
import logging
import terminal
from config import Config

logger = logging.getLogger(__name__)

## SLURM STATUS
def get_slurm_jobs_by_state(state: str) -> int:
    output = terminal.run_command(["squeue", "-u", "$USER", f"--state={state}"])
    num_jobs = 0
    if output:
        num_jobs = len(output.strip().split("\n")) - 1
    return num_jobs

def get_num_slurm_pending_jobs() -> int:
    return get_slurm_jobs_by_state("PENDING")
def get_num_slurm_running_jobs() -> int:
    return get_slurm_jobs_by_state("RUNNING")
def get_num_slurm_jobs() -> int:
    return get_num_slurm_pending_jobs() + get_num_slurm_running_jobs()

## RAY CLUSTER STATUS
def get_ray_status() -> str:
    return terminal.run_command(args=["ray", "status"])

def get_num_ray_pending_jobs() -> int:
    status_output = get_ray_status()
    if status_output and "(no pending nodes)" not in status_output:
        pending_section = status_output.split("Pending:")[1].split("\n")[0].strip()
        return len(pending_section.split("\n"))
    return 0
def get_num_ray_running_jobs() -> int:
    status_output = get_ray_status()
    if "Healthy:" in status_output:
        healthy_section = status_output.split("Healthy:")[1].split("\n")[0].strip()
        num_active_workers = len(healthy_section.split("\n"))
        return max(0,num_active_workers - 1)  # Subtract 1 to exclude login head-node
    return 0

def get_num_ray_jobs() -> int:
    return get_num_ray_pending_jobs() + get_num_ray_running_jobs()

class JobMetrics:
    def __init__(self):
        self.pending_jobs = 0
        self.running_jobs = 0
        self.total_jobs = 0

    def set_jobs(self, pending_jobs: int, running_jobs: int):
        self.pending_jobs = pending_jobs
        self.running_jobs = running_jobs
        self._set_total_jobs()

    def _set_total_jobs(self):
        self.total_jobs = self.pending_jobs + self.running_jobs

    def __str__(self):
        return (f"JobStatus(pending_jobs={self.pending_jobs}, "
                f"running_jobs={self.running_jobs}, "
                f"total_jobs={self.total_jobs})")

class Status:
    def __init__(self):
        self.slurm = JobMetrics()
        self.ray = JobMetrics()
        self.update()

    def update(self):
        self.slurm.set_jobs(pending_jobs=get_num_slurm_pending_jobs(), running_jobs=get_num_slurm_running_jobs())
        self.ray.set_jobs(pending_jobs=get_num_ray_pending_jobs(), running_jobs=get_num_ray_running_jobs())

    def is_worker_queue_full(self) -> bool:
        if self.slurm.total_jobs >= Config.MAX_NODES:
            return True
        if self.slurm.pending_jobs > 0:
            return True # Avoid spamming slurm jobs
        return False
    
    def need_more_workers(self) -> bool:
        return self.ray.total_jobs > self.slurm.total_jobs

    def __str__(self):
        return f"Slurm({self.slurm}) Ray({self.ray})"


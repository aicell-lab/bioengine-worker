"""Minimal GPU sanity-check deployment for BioEngine SLURM mode validation.

Requests 1 GPU per replica and exposes two methods:
- `ping`: liveness probe; returns a small dict so the deployment can be tested without GPU calls.
- `gpu_info`: shells out to `nvidia-smi -L` inside the replica and reports
  `CUDA_VISIBLE_DEVICES`. No heavy pip deps — uses only the stdlib so the
  runtime env builds in seconds.
"""

import logging
import os
import subprocess
import time
from typing import Any, Dict, List

from hypha_rpc.utils.schema import schema_method
from ray import serve

logger = logging.getLogger("ray.serve")


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 1,
        "memory": 1 * 1024**3,
        "runtime_env": {},
    }
)
class GpuTest:
    def __init__(self) -> None:
        self.start_time = time.time()

    @schema_method
    async def ping(self) -> Dict[str, Any]:
        """Cheap liveness probe; does not invoke nvidia-smi."""
        return {
            "status": "ok",
            "uptime": time.time() - self.start_time,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    @schema_method
    async def gpu_info(self) -> Dict[str, Any]:
        """Run `nvidia-smi -L` inside the replica and report visible GPUs.

        Returns:
            {
                "cuda_visible_devices": str | None,
                "nvidia_smi_list": list of strings (one per visible device),
                "returncode": int,
                "stderr": str (empty on success),
            }
        """
        try:
            proc = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            devices: List[str] = [
                line.strip() for line in proc.stdout.splitlines() if line.strip()
            ]
            return {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvidia_smi_list": devices,
                "returncode": proc.returncode,
                "stderr": proc.stderr.strip(),
            }
        except FileNotFoundError:
            return {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvidia_smi_list": [],
                "returncode": -1,
                "stderr": "nvidia-smi not found on PATH inside the replica",
            }

import logging
import time
from config import Config
from status import Status
from zombie import ZombieTerminator
import terminal


class Scaler:
    def __init__(self):
        self.status = Status()
        self.zombie_terminator = ZombieTerminator(status=self.status)
        self.prev_status_log = ""

    def _allocate_workers(self):
        if not self.status.is_worker_queue_full() and self.status.need_more_workers():
            terminal.launch_worker_node()

    def _log_status(self):
        status_log = str(self.status)
        if self.prev_status_log != status_log:
            logging.info(f"Status: {status_log}")
            self.prev_status_log = status_log

    def loop_step(self):
        self.status.update()
        self._log_status()        
        self._allocate_workers()
        self.zombie_terminator.update()
        time.sleep(Config.Scaling.CHECK_INTERVAL.total_seconds())

    def run(self):
        logging.info("Starting autoscaler...")
        while True:
            self.loop_step()

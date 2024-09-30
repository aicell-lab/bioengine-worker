from config import Config
from status import Status
import terminal
import logging
import time

class Scaler:
    def __init__(self):
        self.status = Status()

    def _allocate_workers(self):
        if not self.status.is_worker_queue_full() and self.status.need_more_workers():
            terminal.launch_worker_node()
    
    def _update_status(self):
        self.status.update()
        logging.info(f"Status: {self.status}")

    def loop_step(self):
        self._update_status()
        self._allocate_workers()
        time.sleep(Config.AUTOSCALER_CHECK_INTERVAL)

    def run(self):
        logging.info(Config.Logging.START_MSG)
        while True:
            self.loop_step()

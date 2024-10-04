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

    def _allocate_workers(self):
        if not self.status.is_worker_queue_full() and self.status.need_more_workers():
            terminal.launch_worker_node()

    def loop_step(self):
        self.status.update()
        logging.info(f"Status: {self.status}")
        self._allocate_workers()
        self.zombie_terminator.update()
        time.sleep(Config.AUTOSCALER_CHECK_INTERVAL)

    def run(self):
        logging.info(Config.Logging.START_MSG)
        while True:
            self.loop_step()

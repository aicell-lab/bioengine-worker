import unittest
import os
from datetime import datetime
from main import Autoscaler, Config

class TestAutoscaler(unittest.TestCase):
    def setUp(self):
        """Set up the environment for testing."""
        # Remove the log file if it exists before each test
        log_file = os.path.join(Config.LOGS_DIR, Config.AUTOSCALER_LOGS_FILENAME)
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_creates_log_file(self):
        """Test that the initialize_logs method creates the log file."""
        Autoscaler()
        log_file = os.path.join(Config.LOGS_DIR, Config.AUTOSCALER_LOGS_FILENAME)
        self.assertTrue(os.path.exists(log_file))

    def test_get_load_status(self):
        "Test if the correct load status is retrieved from mock ray status messages and mock slurm status messages"
        ...

if __name__ == "__main__":
    unittest.main()

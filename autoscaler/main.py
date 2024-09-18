
from scaler import Scaler
import config
import logging

if __name__ == "__main__":
    config.init_singletons()
    autoscaler = Scaler()
    autoscaler.run()
    config.shutdown_singletons()


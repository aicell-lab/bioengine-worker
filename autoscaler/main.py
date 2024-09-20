
from scaler import Scaler
import init
import signal

def signal_break(signum, frame):
    import os
    os.write(1, f"Received signal {signum}. Shutting down...\n".encode())
    init.shutdown()
    exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_break)
    if init.setup():
        Scaler().run()



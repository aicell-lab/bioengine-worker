from scaling.scaler import Scaler
import hypha.service
import init
import signal
import asyncio
import logging

def signal_break(signum, frame):
    logging.info(f"Received signal {signum}. Shutting down...")
    init.shutdown()
    exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_break)
    if init.setup():
        asyncio.run(hypha.service.register_services())
        Scaler().run()



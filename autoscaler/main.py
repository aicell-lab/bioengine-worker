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

async def scaler_loop():
    await asyncio.to_thread(Scaler().run)

async def main_loop():
    await hypha.service.register_services()
    scaler_task = asyncio.create_task(scaler_loop())  
    await scaler_task 

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_break)
    if init.setup():
        asyncio.run(main_loop())

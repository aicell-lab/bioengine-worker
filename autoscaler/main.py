from scaling.scaler import Scaler
import hypha.service
import config
import init
import signal
import asyncio
import logging
import argparse
from config import Config

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

def parse_args():
    parser = argparse.ArgumentParser(description='Autoscaler script.')
    parser.add_argument('head_node_ip', type=str, help='IP address of the head node')
    args = parser.parse_args()
    Config.Head.ip = args.head_node_ip

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_break)
    parse_args()
    if init.setup():
        asyncio.run(main_loop())

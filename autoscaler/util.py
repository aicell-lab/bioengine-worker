from typing import List
import psutil
import socket
from pathlib import Path

def _get_path_n_levels_up(file_path: str, n: int) -> Path:
    return str(Path(file_path).resolve().parents[n])

def get_dir_path(relative_path: str) -> Path:
    return _get_path_n_levels_up(__file__, 2) / relative_path

def get_script_path(script_dir: str, script_filename: str) -> Path:
    return get_dir_path(script_dir) / script_filename

def _get_LAN_IPs(target_netmask: str = '255.255.0.0') -> List[str]:
    return [
        addr.address 
        for _, addrs in psutil.net_if_addrs().items() 
        for addr in addrs 
        if addr.family == socket.AF_INET and addr.netmask == target_netmask
    ]

def get_head_LAN_IP() -> str:
     return _get_LAN_IPs()[0]

from typing import List
import psutil
import socket
from pathlib import Path

def _get_path_n_levels_up(file_path: str, n: int) -> Path:
    return Path(file_path).resolve().parents[n]

def get_dir_path(relative_path: str) -> Path:
    return _get_path_n_levels_up(__file__, 1) / relative_path

def get_script_path(script_dir: str, script_filename: str) -> Path:
    return get_dir_path(script_dir) / script_filename


import socket
import subprocess
from copy import copy
from typing import Optional, Tuple


def get_internal_ip() -> str:
    """
    Find the internal IP address of the system.

    Uses the hostname command to retrieve the system's internal IP address.

    Returns:
        str: The internal IP address of the system
    """
    result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
    return result.stdout.strip().split()[0]  # Take the first IP


def acquire_free_port(
    port: int, step: int = 1, ip: Optional[str] = "localhost", keep_open: bool = False
) -> Tuple[int, socket.socket]:
    """
    Find next free port starting from given port number.

    Checks for port availability by attempting to bind to the port.
    If the port is in use, it increments by the step value until a
    free port is found.

    Args:
        port: Starting port number to check
        step: Increment between port numbers to check
        ip: IP address to bind the socket to
        keep_open: Whether to keep the socket open after finding a free port.
            Useful when searching for multiple free ports.

    Returns:
        Tuple[int, socket.socket]: The free port number and socket object.
    """
    port = copy(port)
    while True:
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((ip, port))
            _, port = s.getsockname()
            if not keep_open:
                s.close()
            return port, s
        except OSError:
            port += step
            if s:
                s.close()


if __name__ == "__main__":
    print("Internal IP:", get_internal_ip())
    ip, port, s = acquire_free_port(8000)
    print("Free port:", port)

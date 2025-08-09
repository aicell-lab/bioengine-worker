import socket
import subprocess
from copy import copy


def get_internal_ip() -> str:
    """
    Find the internal IP address of the system.

    Uses the hostname command to retrieve the system's internal IP address.

    Returns:
        str: The internal IP address of the system
    """
    result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
    return result.stdout.strip().split()[0]  # Take the first IP


def acquire_free_port(port: int, step: int = 1) -> socket.socket:
    """
    Find next free port starting from given port number.

    Checks for port availability by attempting to bind to the port.
    If the port is in use, it increments by the step value until a
    free port is found.

    Args:
        port: Starting port number to check
        step: Increment between port numbers to check

    Returns:
        First free port number found
    """
    port = copy(port)
    while True:
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("localhost", port))
            return s
        except OSError:
            port += step
            if s:
                s.close()


if __name__ == "__main__":
    print("Internal IP:", get_internal_ip())
    s = acquire_free_port(8000)
    ip, port = s.getsockname()
    print("Free port:", port)
    s.close()

import socket
from copy import copy
from typing import Optional, Tuple


def get_internal_ip() -> str:
    """
    Get the internal IP address of the system (cross-platform).

    Works on Linux, macOS, and Windows using the socket module only.

    Returns:
        The internal IP address as a string.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))  # No data is sent
        return s.getsockname()[0]


def acquire_free_port(
    port: int,
    step: int = 1,
    ip: Optional[str] = "localhost",
    keep_open: bool = False,
) -> Tuple[int, Optional[socket.socket]]:
    """
    Find the next free TCP port starting from a given port number.

    Tries to bind to the port; if unavailable, increments by `step`
    until a free port is found.

    Args:
        port: Starting port number to check.
        step: Increment between port numbers to check.
        ip: IP address to bind to (default: 'localhost').
        keep_open: Whether to keep the socket open after finding a free port.
                   Useful when reserving multiple ports to avoid race conditions.

    Returns:
        (port, socket): The free port number and, if `keep_open` is True,
                        the open socket object (otherwise None).
    """
    port = copy(port)

    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # allow quick reuse
        try:
            s.bind((ip, port))
            s.listen(1)  # mark as a passive socket (for good measure)
            _, bound_port = s.getsockname()

            if not keep_open:
                s.close()
                s = None

            return bound_port, s

        except OSError:
            s.close()
            port += step


if __name__ == "__main__":
    print("Internal IP:", get_internal_ip())
    ip, port, s = acquire_free_port(8000)
    print("Free port:", port)

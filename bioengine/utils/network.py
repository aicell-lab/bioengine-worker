import ipaddress
import socket
import struct
from copy import copy
from typing import List, Optional, Tuple


def _enumerate_ipv4_addresses() -> List[str]:
    """Enumerate IPv4 addresses bound to the host's network interfaces.

    Uses ``socket.if_nameindex`` + the ``SIOCGIFADDR`` ioctl, which works
    inside minimal Linux containers (no ``/usr/bin/ip`` required). On
    non-Linux platforms (or if ``fcntl`` is unavailable) returns an empty
    list and the caller falls back to the legacy UDP-connect trick.
    """
    try:
        import fcntl  # Linux/Unix only
    except ImportError:
        return []
    if not hasattr(socket, "if_nameindex"):
        return []

    addresses: List[str] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _, name in socket.if_nameindex():
            if name == "lo":
                continue
            try:
                packed = fcntl.ioctl(
                    sock.fileno(),
                    0x8915,  # SIOCGIFADDR
                    struct.pack("256s", name[:15].encode()),
                )
                addr = socket.inet_ntoa(packed[20:24])
            except OSError:
                continue
            if addr == "127.0.0.1":
                continue
            addresses.append(addr)
    finally:
        sock.close()
    return addresses


def get_internal_ip() -> str:
    """
    Get the cluster-internal IPv4 address of the host.

    On multi-homed hosts (e.g. HPC login nodes with both a public network
    and a private compute network) prefer the RFC 1918 private address that
    other cluster nodes can route to. Falls back to the legacy
    ``connect-to-8.8.8.8`` trick.

    Returns:
        The internal IPv4 address as a string.
    """
    addresses = _enumerate_ipv4_addresses()
    if addresses:
        private = [a for a in addresses if ipaddress.IPv4Address(a).is_private]
        if private:
            # Prefer 10.x and 172.16-31.x over 192.168.x (compute clusters
            # typically use the larger private blocks).
            private.sort(
                key=lambda a: 0 if not a.startswith("192.168.") else 1
            )
            return private[0]
        return addresses[0]

    # Fallback: open a UDP socket to a public address; the kernel picks the
    # source IP via the routing table.
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
    port, s = acquire_free_port(8000)
    print("Free port:", port)

import logging
import socket
import time
from copy import copy
from typing import Dict, Optional, Tuple

import httpx


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


def get_url_with_retry(
    url: str,
    params: Optional[Dict[str, str]] = None,
    raise_for_status: bool = False,
    http_client: Optional[httpx.Client] = None,
    logger: Optional[logging.Logger] = None,
) -> httpx.Response:
    """
    Helper method to fetch a URL with retries.

    Implements a simple retry mechanism for HTTP GET requests to handle
    transient network issues. Retries the request up to 3 times with
    exponential backoff.

    Args:
        url: The URL to fetch
    Returns:
        The HTTP response object
    Raises:
        httpx.HTTPError: If all retry attempts fail
    """
    if http_client is None:
        http_client = httpx.Client(timeout=10.0)  # seconds

    if logger is None:
        logger = logging.getLogger(__name__)

    max_attempts = 4
    backoff = 1.0  # backoff: 1.0s, 2.0s, 4.0s
    backoff_multiplier = 2.0

    for attempt in range(1, max_attempts + 1):
        try:
            response = http_client.get(url, params=params)
            response.raise_for_status()
            return response
        except Exception as e:
            # Don't retry on 4xx client errors (except 429 Too Many Requests)
            if isinstance(e, httpx.HTTPStatusError):
                if (
                    400 <= e.response.status_code < 500
                    and e.response.status_code != 429
                ):
                    if raise_for_status:
                        raise e

                    return response

            if attempt < max_attempts:
                # Sleep with exponential backoff before retrying
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed for URL {url}, "
                    f"params: {params}, error: {e}. Retrying in {backoff:.2f}s..."
                )
                time.sleep(backoff)
                backoff *= backoff_multiplier
            else:
                # If we get here, all retries failed due to errors (network, transport, etc.)
                logger.error(
                    f"Failed to fetch URL {url}, params: {params} after {max_attempts} attempts: {e}"
                )
                if not isinstance(e, httpx.HTTPStatusError) or raise_for_status:
                    raise e

                return response


if __name__ == "__main__":
    print("Internal IP:", get_internal_ip())
    port, s = acquire_free_port(8000)
    print("Free port:", port)

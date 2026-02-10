import asyncio
import logging
from typing import Dict, Optional

import httpx


async def get_url_with_retry(
    url: str,
    params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    raise_for_status: bool = False,
    http_client: Optional[httpx.AsyncClient] = None,
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
        http_client = httpx.AsyncClient(timeout=20.0)  # seconds

    if logger is None:
        logger = logging.getLogger(__name__)

    max_attempts = 4
    backoff = 0.2  # backoff: 0.2s, 0.4s, 0.8s
    backoff_multiplier = 2.0

    for attempt in range(1, max_attempts + 1):
        try:
            response = await http_client.get(url, params=params, headers=headers)
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
                await asyncio.sleep(backoff)
                backoff *= backoff_multiplier
            else:
                # If we get here, all retries failed due to errors (network, transport, etc.)
                logger.error(
                    f"Failed to fetch URL '{url}' after {max_attempts} attempts: {e}"
                )
                if not isinstance(e, httpx.HTTPStatusError) or raise_for_status:
                    raise e

                return response


async def get_presigned_url(
    data_service_url: str,
    dataset_id: str,
    file_path: str,
    token: Optional[str] = None,
    http_client: Optional[httpx.AsyncClient] = None,
    logger: Optional[logging.Logger] = None,
) -> str | None:
    """
    Get a presigned URL for secure access to a specific dataset file.

    Requests a temporary, authenticated URL for accessing the specified file path within
    the dataset from the data service.

    Args:
        data_service_url: URL of the data service endpoint
        dataset_id: Name of the dataset containing the file
        file_path: Path to the file within the dataset
        token: Authentication token for the data service
        http_client: Optional httpx AsyncClient instance for making the request
        logger: Optional logger for logging messages

    Returns:
        str | None: Presigned URL for accessing the file, or None if the file doesn't exist

    Raises:
        httpx.HTTPStatusError: If the request fails for reasons other than file not found
    """
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=20.0)  # seconds

    if logger is None:
        logger = logging.getLogger(__name__)

    # Build query parameters, excluding None values
    params = {
        "dataset_id": dataset_id,
        "file_path": file_path,
    }
    if token is not None:
        params["token"] = token

    query_url = f"{data_service_url}/get_presigned_url"

    response = await get_url_with_retry(
        url=query_url,
        params=params,
        raise_for_status=False,
        http_client=http_client,
        logger=logger,
    )
    if response.status_code == 400 and "FileNotFoundError" in response.text:
        return None
    response.raise_for_status()
    presigned_url = response.json()

    return presigned_url

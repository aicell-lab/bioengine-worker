from typing import Optional

import httpx


async def get_presigned_url(
    data_service_url: str,
    dataset_id: str,
    file_path: str,
    token: Optional[str] = None,
    http_client: Optional[httpx.AsyncClient] = None,
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

    Returns:
        str | None: Presigned URL for accessing the file, or None if the file doesn't exist

    Raises:
        httpx.HTTPStatusError: If the request fails for reasons other than file not found
    """
    # Build query parameters, excluding None values
    params = {
        "dataset_id": dataset_id,
        "file_path": file_path,
    }
    if token is not None:
        params["token"] = token

    query_url = f"{data_service_url}/get_presigned_url"

    if http_client is None:
        http_client = httpx.AsyncClient(timeout=120)  # seconds

    response = await http_client.get(query_url, params=params)
    if response.status_code == 400 and "FileNotFoundError" in response.text:
        return None
    response.raise_for_status()
    presigned_url = response.json()

    return presigned_url

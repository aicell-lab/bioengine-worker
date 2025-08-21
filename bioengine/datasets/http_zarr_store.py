import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterable

import httpx
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype


@dataclass
class HttpZarrStore(Store):
    """
    HTTP-based Zarr store for efficient remote dataset access.

    This class implements the Zarr Store interface for accessing datasets over HTTP,
    providing efficient partial data access for large scientific datasets. It handles
    authentication, presigned URL generation, and range requests to optimize network
    bandwidth while maintaining compatibility with the Zarr API.

    The implementation focuses on read-only access optimized for scientific workflows,
    with clear error handling for operations that would modify data. It integrates with
    the BioEngine Datasets service's authentication and presigned URL mechanisms for
    secure access to protected datasets.

    Key Features:
    - Efficient partial data access through HTTP range requests
    - Authentication via token-based access control
    - Presigned URL management for secure access to S3-backed storage
    - Compatible with standard Zarr API for seamless integration
    - Read-only access with clear error handling for write operations

    Implementation Details:
    The store uses presigned URLs for efficient and secure access to dataset chunks,
    supporting HTTP range requests to minimize data transfer when accessing large arrays.
    It is optimized for read-only access to scientific datasets in Zarr format. The store
    uses HTTP range requests to optimize bandwidth usage and supports parallel chunk
    retrieval for improved performance with array-based data access patterns.

    Attributes:
        dataset_name (str): Name of the dataset being accessed
        service_url (str): Base URL for the dataset service
        token (str): Authentication token for access control
        http_client (httpx.AsyncClient): HTTP client for service communication
        _read_only (bool): Flag indicating read-only access (always True)
        supports_* (bool): Capability flags for the Zarr API
    """

    dataset_name: str
    _read_only: bool = True

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = False

    def __init__(self, service_url: str, dataset_name: str, token: str):
        """
        Initialize the HTTP-based Zarr store for remote dataset access.

        Creates a store instance connected to the specified dataset service and
        configured for authenticated access to the target dataset.

        Args:
            service_url: Base URL for the dataset service API
            dataset_name: Name of the dataset to access through this store
            token: Authentication token for access control
        """
        super().__init__(read_only=True)
        self.service_url = service_url.rstrip("/")
        self.dataset_name = dataset_name
        self.token = token
        self.http_client = httpx.AsyncClient(timeout=120)  # seconds

    async def _get_presigned_url(self, key: str) -> str | None:
        """
        Get a presigned URL for secure access to a specific dataset chunk.

        Requests a temporary, authenticated URL for accessing the specified key within
        the dataset. This method is used internally by the get() and exists() methods
        to securely access data chunks.

        Args:
            key: Path to the chunk within the dataset

        Returns:
            Presigned URL for accessing the chunk, or None if the chunk doesn't exist

        Raises:
            httpx.HTTPStatusError: If the request fails for reasons other than file not found
        """
        query_url = (
            f"{self.service_url}/get_presigned_url?dataset_name={self.dataset_name}&"
            f"file_path={key}&token={self.token}"
        )
        response = await self.http_client.get(query_url)
        if response.status_code == 400 and "FileNotFoundError" in response.text:
            return None
        response.raise_for_status()
        presigned_url = response.json()

        return presigned_url

    def __eq__(self, other: object) -> bool:
        return all(
            isinstance(other, HttpZarrStore)
            and self.service_url == other.service_url
            and self.dataset_name == other.dataset_name
            and self.token == other.token
        )

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest = None
    ) -> Buffer | None:
        """
        Get data from the store for a specific key with optional range specification.

        This is the primary method for data access, implementing the Zarr Store interface
        for retrieving data chunks. It supports efficient partial data access through
        HTTP range requests, optimizing network bandwidth for large datasets.

        Data Access Process:
        1. Obtains a presigned URL for secure access to the chunk
        2. Constructs appropriate HTTP range headers based on byte_range type
        3. Retrieves the requested data chunk with proper error handling
        4. Returns data in the format specified by the prototype buffer

        Args:
            key: Path to the chunk within the dataset
            prototype: Buffer prototype for creating the return buffer
            byte_range: Optional specification for partial data access within the chunk

        Returns:
            Buffer containing the requested data, or None if the key doesn't exist

        Raises:
            httpx.HTTPStatusError: If the HTTP request fails
        """
        url = await self._get_presigned_url(key)
        if url is None:
            return None

        headers = {}
        if byte_range:
            if isinstance(byte_range, RangeByteRequest):
                headers["Range"] = f"bytes={byte_range.start}-{byte_range.end - 1}"
            elif isinstance(byte_range, OffsetByteRequest):
                headers["Range"] = f"bytes={byte_range.offset}-"
            elif isinstance(byte_range, SuffixByteRequest):
                headers["Range"] = f"bytes=-{byte_range.suffix}"

        response = await self.http_client.get(url, headers=headers)
        response.raise_for_status()
        content = response.content
        return prototype.buffer.from_bytes(content)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """
        Get multiple data chunks in parallel with optional range specifications.

        Optimizes data access by retrieving multiple chunks in parallel, reducing
        the impact of network latency when accessing many small chunks. This method
        is critical for efficient array access patterns in scientific computing.

        Args:
            prototype: Buffer prototype for creating the return buffers
            key_ranges: Iterable of (key, byte_range) tuples for parallel retrieval

        Returns:
            List of buffers containing the requested data, with None for missing keys
        """
        return await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges))

    async def exists(self, key: str) -> bool:
        """
        Check if a specific key exists in the store.

        Verifies the existence of a data chunk without retrieving its contents,
        providing an efficient way to check for available data.

        Args:
            key: Path to the chunk to check within the dataset

        Returns:
            True if the key exists, False otherwise
        """
        url = await self._get_presigned_url(key)
        return url is not None

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("Write not supported")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("Delete not supported")

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes]]
    ) -> None:
        raise NotImplementedError("Partial write not supported")

    def list(self) -> AsyncIterator[str]:
        raise NotImplementedError("Listing not supported")

    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("Prefix listing not supported")

    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError("Dir listing not supported")

import asyncio
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterable

import httpx

try:
    import zarr
except ImportError as e:
    raise ImportError("zarr>=3.0.0 is required") from e

if zarr.__version__ < "3.0.0":
    raise ImportError(f"zarr>=3.0.0 is required but found {zarr.__version__}")

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype

from bioengine.datasets.utils import get_presigned_url


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
    - Presigned URL management with 59-minute caching for performance optimization
    - Size-limited chunk data caching (1GB default) with LRU eviction for optimal performance
    - Compatible with standard Zarr API for seamless integration
    - Read-only access with clear error handling for write operations

    Implementation Details:
    The store uses presigned URLs for efficient and secure access to dataset chunks,
    supporting HTTP range requests to minimize data transfer when accessing large arrays.
    It implements a two-tier caching system: presigned URLs are cached for 59 minutes,
    while actual chunk data is cached with a size limit (1GB default) using LRU eviction.
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
    zarr_path: str
    _read_only: bool = True

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = False

    def __init__(
        self,
        service_url: str,
        dataset_name: str,
        zarr_path: str,
        token: str,
        max_chunk_cache_size: int = int(
            os.getenv("BIOENGINE_DATASETS_ZARR_STORE_CACHE_SIZE", 1024 * 1024 * 1024)
        ),  # 1 GiB default
    ):
        """
        Initialize the HTTP-based Zarr store for remote dataset access.

        Creates a store instance connected to the specified dataset service and
        configured for authenticated access to the target dataset.

        Args:
            service_url: Base URL for the dataset service API
            dataset_name: Name of the dataset to access through this store
            zarr_path: Path within the dataset to the Zarr store (must end with .zarr)
            token: Authentication token for access control
        """
        super().__init__(read_only=True)
        self.service_url = service_url.rstrip("/")
        self.dataset_name = dataset_name
        self.zarr_path = zarr_path[1:] if zarr_path.startswith("/") else zarr_path
        self.token = token
        self.http_client = None

        # Presigned URL cache (expires just under 1 hour for safety)
        # Cache expiry is set to 59 minutes to be safe with 60 minute URL expiry
        self._presigned_url_cache = {}  # {cache_key: (url, timestamp)}
        self._cache_expiry_seconds = 59 * 60

        # Chunk data cache (size-limited LRU cache)
        self._chunk_cache = {}  # {cache_key: (buffer, size)}
        self._chunk_cache_order = (
            []
        )  # List of cache_keys in access order (oldest first)
        self._chunk_cache_size = 0  # Current total size in bytes
        self._max_chunk_cache_size = max_chunk_cache_size

        if not self.zarr_path.endswith(".zarr"):
            raise ValueError("zarr_path must end with .zarr")

    def _set_http_client(self) -> None:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=120)  # seconds

    def _get_url_cache_key(self, file_path: str) -> str:
        """Generate a cache key for the presigned URL based on file path."""
        return f"{self.dataset_name}:{file_path}"

    def _is_cache_expired(self, timestamp: float) -> bool:
        """Check if a cached URL has expired."""
        return (time.time() - timestamp) >= self._cache_expiry_seconds

    async def _get_cached_url(self, file_path: str) -> str | None:
        """
        Get a cached presigned URL if available and not expired.

        Args:
            file_path: File path to get the cached URL for

        Returns:
            Cached URL if available and not expired, None otherwise
        """
        cache_key = self._get_url_cache_key(file_path)

        if cache_key in self._presigned_url_cache:
            url, timestamp = self._presigned_url_cache[cache_key]
            if not self._is_cache_expired(timestamp):
                return url
            else:
                # Remove expired entry
                del self._presigned_url_cache[cache_key]

        return None

    async def _get_and_cache_url(self, file_path: str) -> str | None:
        """
        Get a presigned URL, using cache if available, otherwise fetching and caching.

        Args:
            file_path: File path to get the presigned URL for

        Returns:
            Presigned URL or None if file doesn't exist
        """
        # Try cache first
        cached_url = await self._get_cached_url(file_path)
        if cached_url is not None:
            return cached_url

        # Not in cache or expired, fetch new URL
        self._set_http_client()
        url = await get_presigned_url(
            data_service_url=self.service_url,
            dataset_name=self.dataset_name,
            file_path=file_path,
            token=self.token,
            http_client=self.http_client,
        )

        # Cache the URL if we got one
        if url is not None:
            cache_key = self._get_url_cache_key(file_path)
            self._presigned_url_cache[cache_key] = (url, time.time())

        return url

    def _get_chunk_cache_key(self, key: str, byte_range: ByteRequest | None) -> str:
        """Generate a cache key for chunk data based on key and byte range."""
        if byte_range is None:
            return f"chunk:{self.dataset_name}:{key}"
        else:
            # Create a unique representation of the byte range
            if isinstance(byte_range, RangeByteRequest):
                range_str = f"range:{byte_range.start}-{byte_range.end}"
            elif isinstance(byte_range, OffsetByteRequest):
                range_str = f"offset:{byte_range.offset}"
            elif isinstance(byte_range, SuffixByteRequest):
                range_str = f"suffix:{byte_range.suffix}"
            else:
                range_str = f"other:{str(byte_range)}"
            return f"chunk:{self.dataset_name}:{key}:{range_str}"

    def _evict_oldest_chunks(self, required_size: int) -> None:
        """Evict least recently used chunks until we have enough space for required_size bytes."""
        while (
            self._chunk_cache_size + required_size > self._max_chunk_cache_size
            and self._chunk_cache_order
        ):
            # Remove least recently used chunk (first in order list)
            oldest_key = self._chunk_cache_order.pop(0)
            if oldest_key in self._chunk_cache:
                print(f"Evicting chunk from cache: {oldest_key}")
                _, size = self._chunk_cache[oldest_key]
                del self._chunk_cache[oldest_key]
                self._chunk_cache_size -= size

    def _cache_chunk(self, cache_key: str, buffer: Buffer) -> None:
        """Add a chunk to the cache, evicting old chunks if necessary."""
        # Estimate buffer size (this is approximate)
        buffer_size = len(buffer.to_bytes()) if hasattr(buffer, "to_bytes") else 0

        # Don't cache if the chunk is larger than the entire cache
        if buffer_size > self._max_chunk_cache_size:
            return

        # Evict old chunks if necessary
        self._evict_oldest_chunks(buffer_size)

        # Add to cache (most recently used, so goes at end of order list)
        self._chunk_cache[cache_key] = (buffer, buffer_size)
        self._chunk_cache_order.append(cache_key)
        self._chunk_cache_size += buffer_size

    def _get_cached_chunk(self, cache_key: str) -> Buffer | None:
        """Get a chunk from cache and move it to end of LRU order."""
        if cache_key not in self._chunk_cache:
            return None

        buffer, size = self._chunk_cache[cache_key]

        # Move to end of order list (most recently used)
        if cache_key in self._chunk_cache_order:
            self._chunk_cache_order.remove(cache_key)
        self._chunk_cache_order.append(cache_key)

        return buffer

    def __eq__(self, other: object) -> bool:
        return all(
            isinstance(other, HttpZarrStore)
            and self.service_url == other.service_url
            and self.dataset_name == other.dataset_name
            and self.zarr_path == other.zarr_path
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
        1. Checks chunk cache for previously downloaded data
        2. If not cached, obtains a presigned URL for secure access to the chunk
        3. Constructs appropriate HTTP range headers based on byte_range type
        4. Retrieves the requested data chunk with proper error handling
        5. Caches the chunk data for future access
        6. Returns data in the format specified by the prototype buffer

        Args:
            key: Path to the chunk within the dataset
            prototype: Buffer prototype for creating the return buffer
            byte_range: Optional specification for partial data access within the chunk

        Returns:
            Buffer containing the requested data, or None if the key doesn't exist

        Raises:
            httpx.HTTPStatusError: If the HTTP request fails
        """
        # Check chunk cache first
        if self._max_chunk_cache_size > 0:
            chunk_cache_key = self._get_chunk_cache_key(key, byte_range)
            cached_buffer = self._get_cached_chunk(chunk_cache_key)
            if cached_buffer is not None:
                return cached_buffer

        # Not in cache, fetch from remote
        url = await self._get_and_cache_url(f"{self.zarr_path}/{key}")
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
        buffer = prototype.buffer.from_bytes(content)

        # Cache the buffer for future use
        if self._max_chunk_cache_size > 0:
            self._cache_chunk(chunk_cache_key, buffer)

        return buffer

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
        url = await self._get_and_cache_url(f"{self.zarr_path}/{key}")
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

    def close(self):
        self.http_client = None
        # Clear both presigned URL cache and chunk cache when closing
        self._presigned_url_cache.clear()
        self._chunk_cache.clear()
        self._chunk_cache_order.clear()
        self._chunk_cache_size = 0
        return super().close()

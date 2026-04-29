import asyncio
import logging
import os
from asyncio import Semaphore
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

import httpx

try:
    import zarr
except ImportError as e:
    raise ImportError("zarr>=3.0.8 is required") from e

if zarr.__version__ < "3.0.8":
    raise ImportError(f"zarr>=3.0.8 is required but found {zarr.__version__}")

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype

from bioengine.datasets.utils import get_url_with_retry


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
    - Authentication via token passed on each request
    - Size-limited chunk data caching (1GB default) with LRU eviction for optimal performance
    - Compatible with standard Zarr API for seamless integration
    - Read-only access with clear error handling for write operations

    Implementation Details:
    The store constructs direct URLs of the form
    ``{service_url}/data/{dataset_id}/{zarr_path}/{key}?token={token}``
    and uses HTTP range requests to minimise data transfer. Chunk data is cached
    locally with a size limit (1GB default) using LRU eviction.

    Attributes:
        dataset_id (str): Name of the dataset being accessed
        service_url (str): Base URL for the dataset service
        token (str): Authentication token for access control
        http_client (httpx.AsyncClient): HTTP client for service communication
        _read_only (bool): Flag indicating read-only access (always True)
        supports_* (bool): Capability flags for the Zarr API
    """

    dataset_id: str
    zarr_path: str
    _read_only: bool = True

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = False

    def __init__(
        self,
        service_url: str,
        dataset_id: str,
        zarr_path: str,
        token: Optional[str] = None,
        max_chunk_cache_size_gb: int = int(
            os.getenv("BIOENGINE_DATASETS_ZARR_STORE_CACHE_SIZE", 1)
        ),  # 1 GiB default; Note that this is not shared between multiple store instances
        max_concurrent_requests: int = int(
            os.getenv("BIOENGINE_DATASETS_ZARR_STORE_CONCURRENT_REQUESTS", 50)
        ),
        max_connections: int = int(
            os.getenv("BIOENGINE_DATASETS_ZARR_STORE_CONNECTIONS", 100)
        ),
        logger: logging.Logger = logging.getLogger("HttpZarrStore"),
    ):
        """
        Initialize the HTTP-based Zarr store for remote dataset access.

        Creates a store instance connected to the specified dataset service and
        configured for authenticated access to the target dataset.

        Args:
            service_url: Base URL for the dataset service API
            dataset_id: Name of the dataset to access through this store
            zarr_path: Path within the dataset to the Zarr store (must end with .zarr)
            token: Authentication token for access control
            max_chunk_cache_size: Maximum size of chunk cache in bytes (default 1GB)
            max_concurrent_requests: Maximum number of concurrent HTTP requests (default 50)
            max_connections: Maximum number of HTTP connections in pool (default 100)
            logger: Logger instance for logging messages
        """
        super().__init__(read_only=True)
        self.service_url = service_url.rstrip("/")
        self.dataset_id = dataset_id
        self.zarr_path = zarr_path[1:] if zarr_path.startswith("/") else zarr_path
        self.token = token
        self.logger = logger

        # Concurrency control
        self.max_concurrent_requests = max_concurrent_requests
        self._request_semaphore = Semaphore(max_concurrent_requests)

        # Configure HTTP/2 and connection pooling for optimal performance
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections // 2,
        )
        self.http_client = httpx.AsyncClient(
            timeout=60,  # seconds
            limits=limits,
            http2=True,  # Enable HTTP/2 for better multiplexing
        )

        # Chunk data cache (size-limited LRU cache)
        self._chunk_cache = {}  # {cache_key: (buffer, size)}
        self._chunk_cache_order = (
            []
        )  # List of cache_keys in access order (oldest first)
        self._chunk_cache_size = 0  # Current total size in bytes
        self._max_chunk_cache_size = (
            max_chunk_cache_size_gb * 1024 * 1024 * 1024
        )  # Convert GB to bytes
        # Lock to protect operations that read/modify cache size and cache structures
        self._cache_lock = asyncio.Lock()

        if not self.zarr_path.endswith(".zarr"):
            raise ValueError("zarr_path must end with .zarr")

    def _build_url(self, file_path: str) -> str:
        """Build the direct data URL for a file path, appending the token if present."""
        url = f"{self.service_url}/data/{self.dataset_id}/{file_path}"
        if self.token:
            url += f"?token={self.token}"
        return url

    def _get_chunk_cache_key(self, key: str, byte_range: ByteRequest | None) -> str:
        """Generate a cache key for chunk data based on key and byte range."""
        if byte_range is None:
            return f"chunk:{self.dataset_id}:{key}"
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
            return f"chunk:{self.dataset_id}:{key}:{range_str}"

    def _evict_oldest_chunks(self, required_size: int) -> None:
        """Evict least recently used chunks until we have enough space for required_size bytes.

        NOTE: This helper assumes the caller holds ``self._cache_lock`` to make eviction
        atomic with other cache-size touching operations.
        """
        while (
            self._chunk_cache_size + required_size > self._max_chunk_cache_size
            and self._chunk_cache_order
        ):
            # Remove least recently used chunk (first in order list)
            oldest_key = self._chunk_cache_order.pop(0)
            if oldest_key in self._chunk_cache:
                _, size = self._chunk_cache[oldest_key]
                del self._chunk_cache[oldest_key]
                self._chunk_cache_size -= size
                self.logger.info(
                    f"Evicted chunk {oldest_key} (size={size} bytes). Cache "
                    f"{self._chunk_cache_size / self._max_chunk_cache_size * 100:.2f}% full after eviction. "
                    f"(dataset ID={self.dataset_id}, zarr_path={self.zarr_path})"
                )

    async def _cache_chunk(self, cache_key: str, buffer: Buffer) -> None:
        """Add a chunk to the cache, evicting old chunks if necessary.

        This operation is performed under ``self._cache_lock`` so that cache-size
        accounting and eviction are atomic.
        """
        # Estimate buffer size (this is approximate)
        buffer_size = len(buffer.to_bytes()) if hasattr(buffer, "to_bytes") else 0

        # Don't cache if the chunk is larger than the entire cache
        if buffer_size > self._max_chunk_cache_size:
            self.logger.warning(
                f"Chunk {cache_key} size {buffer_size} exceeds cache capacity {self._max_chunk_cache_size}, "
                f"not caching (dataset ID={self.dataset_id}, zarr_path={self.zarr_path})"
            )
            return

        # Perform eviction and insertion while holding the cache lock
        async with self._cache_lock:
            # Evict old chunks if necessary
            self._evict_oldest_chunks(buffer_size)

            # Add to cache (most recently used, so goes at end of order list)
            self._chunk_cache[cache_key] = (buffer, buffer_size)
            self._chunk_cache_order.append(cache_key)
            self._chunk_cache_size += buffer_size
            self.logger.info(
                f"Cached chunk {cache_key} (size={buffer_size} bytes). Cache "
                f"{self._chunk_cache_size / self._max_chunk_cache_size * 100:.2f}% full after caching. "
                f"(dataset ID={self.dataset_id}, zarr_path={self.zarr_path})"
            )

    async def _get_cached_chunk(self, cache_key: str) -> Buffer | None:
        """Get a chunk from cache and move it to end of LRU order.

        This acquires ``self._cache_lock`` briefly to update LRU ordering safely.
        """
        async with self._cache_lock:
            if cache_key not in self._chunk_cache:
                self.logger.debug(f"Chunk cache miss: {cache_key}")
                return None

            buffer, size = self._chunk_cache[cache_key]

            # Move to end of order list (most recently used)
            if cache_key in self._chunk_cache_order:
                self._chunk_cache_order.remove(cache_key)
            self._chunk_cache_order.append(cache_key)

            self.logger.debug(f"Chunk cache hit: {cache_key} (size={size} bytes)")
            return buffer

    def __eq__(self, other: object) -> bool:
        return all(
            isinstance(other, HttpZarrStore)
            and self.service_url == other.service_url
            and self.dataset_id == other.dataset_id
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
        2. If not cached, builds the direct data URL and adds range headers
        3. Retrieves the requested data chunk with proper error handling
        4. Caches the chunk data for future access
        5. Returns data in the format specified by the prototype buffer

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
            cached_buffer = await self._get_cached_chunk(chunk_cache_key)
            if cached_buffer is not None:
                return cached_buffer

        # Use semaphore to limit concurrent requests
        async with self._request_semaphore:
            url = self._build_url(f"{self.zarr_path}/{key}")
            headers = {}
            if byte_range:
                if isinstance(byte_range, RangeByteRequest):
                    headers["Range"] = f"bytes={byte_range.start}-{byte_range.end - 1}"
                elif isinstance(byte_range, OffsetByteRequest):
                    headers["Range"] = f"bytes={byte_range.offset}-"
                elif isinstance(byte_range, SuffixByteRequest):
                    headers["Range"] = f"bytes=-{byte_range.suffix}"

            try:
                response = await get_url_with_retry(
                    url=url,
                    headers=headers,
                    raise_for_status=False,
                    http_client=self.http_client,
                    logger=self.logger,
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                content = response.content
                self.logger.debug(
                    f"Fetched {len(content)} bytes for {key} (range={headers.get('Range')})"
                )
                buffer = prototype.buffer.from_bytes(content)

                # Cache the buffer for future use
                if self._max_chunk_cache_size > 0:
                    await self._cache_chunk(chunk_cache_key, buffer)

                return buffer

            except Exception as e:
                self.logger.error(
                    f"Failed to fetch {key} (range={headers.get('Range')}) from {url}: {e}"
                )
                raise e

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
        url = self._build_url(f"{self.zarr_path}/{key}")
        response = await self.http_client.get(url, headers={"Range": "bytes=0-0"})
        return response.status_code in (200, 206)

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
        self._chunk_cache.clear()
        self._chunk_cache_order.clear()
        self._chunk_cache_size = 0
        return super().close()

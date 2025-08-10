import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterable

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
    base_url: str
    headers: Dict[str, str]
    download_timeout: float = 120.0  # seconds
    _read_only: bool = True

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = False

    def __init__(self, base_url: str, headers: Dict[str, str]):
        super().__init__(read_only=True)
        self.base_url = base_url.rstrip("/")
        self.headers = headers

    def _full_url(self, key: str) -> str:
        return f"{self.base_url}/{key}?use_proxy=true"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HttpZarrStore) and self.base_url == other.base_url

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest = None
    ) -> Buffer | None:
        url = self._full_url(key)
        headers = self.headers.copy()

        if byte_range:
            if isinstance(byte_range, RangeByteRequest):
                headers["Range"] = f"bytes={byte_range.start}-{byte_range.end - 1}"
            elif isinstance(byte_range, OffsetByteRequest):
                headers["Range"] = f"bytes={byte_range.offset}-"
            elif isinstance(byte_range, SuffixByteRequest):
                headers["Range"] = f"bytes=-{byte_range.suffix}"

        async with httpx.AsyncClient(timeout=self.download_timeout) as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            content = response.content
            return prototype.buffer.from_bytes(content)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges))

    async def exists(self, key: str) -> bool:
        url = self._full_url(key)
        async with httpx.AsyncClient(timeout=self.download_timeout) as client:
            response = await client.head(url, headers=self.headers)
            return response.status_code == 200

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

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
    dataset_name: str
    _read_only: bool = True

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = False

    def __init__(self, service_url: str, dataset_name: str, token: str):
        super().__init__(read_only=True)
        self.service_url = service_url.rstrip("/")
        self.dataset_name = dataset_name
        self.token = token
        self.http_client = httpx.AsyncClient(timeout=120)  # seconds

    async def _get_presigned_url(self, key: str) -> str | None:
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
        return await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges))

    async def exists(self, key: str) -> bool:
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

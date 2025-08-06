import asyncio
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import httpx
import numpy as np
import yaml
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from ray import serve
from ray.exceptions import RayTaskError
from ray.serve.handle import DeploymentHandle


class BioimageioPackage:
    """Wrapper for cached bioimage.io model package with access tracking."""

    def __init__(
        self,
        package_path: Path,
        latest_download: float,
        replica_id: str,
    ) -> None:
        self.package_path = package_path
        self.source = str(self.package_path / "rdf.yaml")
        self.latest_download = latest_download
        self.replica_id = replica_id
        self._access_file = None

    async def __aenter__(self):
        """Create access lock when model is being used."""
        self._access_file = self.package_path / ".last_access"
        current_time = time.time()
        try:
            await asyncio.to_thread(self._access_file.write_text, str(current_time))
        except (OSError, IOError) as e:
            print(f"⚠️ [{self.replica_id}] Failed to update access time on enter: {e}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Update access time when model usage is complete."""
        if self._access_file and await asyncio.to_thread(self._access_file.exists):
            current_time = time.time()
            try:
                await asyncio.to_thread(self._access_file.write_text, str(current_time))
            except (OSError, IOError) as e:
                print(
                    f"⚠️ [{self.replica_id}] Failed to update access time on exit: {e}"
                )


class ModelCache:
    def __init__(
        self,
        cache_size_in_gb: float,
        replica_id: str,
    ):
        self.cache_dir = Path().resolve() / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_size_bytes = int(
            cache_size_in_gb * 1024 * 1024 * 1024
        )  # Convert GB to bytes
        self.replica_id = replica_id

        self.per_file_download_timeout = 180.0
        download_timeout = httpx.Timeout(self.per_file_download_timeout)
        self.client = httpx.AsyncClient(timeout=download_timeout, follow_redirects=True)

        self.timeout_threshold = self.per_file_download_timeout + 60.0  # 60s buffer

        num_existing_models = len(list(self.cache_dir.glob("**/rdf.yaml")))
        print(
            f"🔄 [{self.replica_id}] Found {num_existing_models} existing models in cache at "
            f"{self.cache_dir}/. {'Starting model validation in the background.' if num_existing_models > 0 else ''}"
        )
        asyncio.create_task(self._scan_cache_dir())

    async def _remove_package(self, package_path: Path) -> None:
        """Safely remove package directory using atomic operations across replicas."""
        if not await asyncio.to_thread(package_path.exists):
            return

        try:
            # Use atomic rename for safe removal across replicas
            temp_dir = (
                package_path.parent
                / f".removing_{package_path.name}_{int(time.time() * 1000000)}"
            )
            await asyncio.to_thread(package_path.rename, temp_dir)
            print(
                f"🔄 [{self.replica_id}] Atomically moved model for removal: '{package_path.name}'"
            )

            # Remove the temporary directory
            await asyncio.to_thread(shutil.rmtree, temp_dir)
            print(
                f"🗑️ [{self.replica_id}] Successfully removed cached model: '{package_path.name}'"
            )

        except FileNotFoundError:
            # Another replica already removed it
            print(
                f"🔍 [{self.replica_id}] Model '{package_path.name}' already removed by another replica"
            )
        except OSError as e:
            # Package might be in use, log but don't fail
            print(
                f"⚠️ [{self.replica_id}] Could not remove cached model '{package_path.name}': {e}"
            )
        except Exception as e:
            print(
                f"❌ [{self.replica_id}] Unexpected error removing model '{package_path.name}': {e}"
            )

    async def _scan_cache_dir(self) -> None:
        """Scan the cache directory and validate existing models."""
        try:
            all_dirs = await asyncio.to_thread(lambda: list(self.cache_dir.iterdir()))
            local_dirs = []
            for d in all_dirs:
                if await asyncio.to_thread(d.is_dir) and not d.name.startswith("."):
                    local_dirs.append(d)
        except (OSError, IOError) as e:
            print(f"⚠️ [{self.replica_id}] Error reading cache directory: {e}")
            return

        for dir_path in local_dirs:
            try:
                await self._validate_package(dir_path)
            except RuntimeError as e:
                await self._remove_package(dir_path)

        # Check for any stale temporary directories
        all_temp_dirs = await asyncio.to_thread(
            lambda: list(self.cache_dir.glob(".temp_*"))
        )
        temp_dirs = []
        for d in all_temp_dirs:
            if await asyncio.to_thread(d.is_dir) and d.name != ".temp_":
                temp_dirs.append(d)
        for temp_dir in temp_dirs:
            # Remove stale temporary directories if timeout exceeded
            try:
                stat_result = await asyncio.to_thread(temp_dir.stat)
                if time.time() - stat_result.st_ctime > self.timeout_threshold:
                    await asyncio.to_thread(shutil.rmtree, temp_dir)
                    print(
                        f"🧹 [{self.replica_id}] Cleaned up stale temporary directory: {temp_dir}"
                    )
            except (OSError, IOError) as e:
                print(
                    f"⚠️ [{self.replica_id}] Failed to clean up stale temporary directory {temp_dir}: {e}"
                )

    async def _check_model_published_status(self, model_id: str, stage: bool) -> bool:
        """Check if a model is published by looking at its manifest status."""
        stage_param = f"?stage={str(stage).lower()}"
        artifact_url = (
            f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}{stage_param}"
        )

        try:
            response = await self.client.get(artifact_url)
            if response.status_code == 404 and stage:
                # If staged version doesn't exist, try with stage=false
                print(
                    f"⚠️ [{self.replica_id}] Staged version not found for model '{model_id}', trying committed version..."
                )
                stage_param = "?stage=false"
                artifact_url = f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}{stage_param}"
                response = await self.client.get(artifact_url)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to download manifest from {artifact_url}: "
                    f"HTTP {response.status_code} - {response.text}"
                )

            manifest = await asyncio.to_thread(yaml.safe_load, response.text)
            # Model is published if status is NOT "request-review"
            return manifest["manifest"].get("status") != "request-review"
        except Exception as e:
            print(
                f"❌ [{self.replica_id}] Error checking model '{model_id}' status: {e}"
            )
            return False

    async def _wait_for_download_completion(
        self, package_dir: Path, max_wait_time: int = 300
    ) -> bool:
        """Wait for another replica to finish downloading. Returns True if successful."""
        import aiofiles

        start_time = time.time()
        downloading_marker = (
            package_dir.parent / f".downloading_{package_dir.name}.lock"
        )

        print(
            f"⏳ [{self.replica_id}] Waiting for download completion: {package_dir.name}"
        )

        check_interval = 2.0
        while time.time() - start_time < max_wait_time:
            try:
                # Check if download is complete (package exists and no downloading marker)
                if await asyncio.to_thread(
                    package_dir.exists
                ) and not await asyncio.to_thread(downloading_marker.exists):
                    print(
                        f"✅ [{self.replica_id}] Download of model '{package_dir.name}' completed by another replica."
                    )
                    return True

                # Check if download failed (no package and no downloading marker)
                if not await asyncio.to_thread(
                    package_dir.exists
                ) and not await asyncio.to_thread(downloading_marker.exists):
                    print(
                        f"⚠️ [{self.replica_id}] Download of model '{package_dir.name}' appears to have failed on another replica."
                    )
                    return False

                # Check if download has timed out
                if await asyncio.to_thread(downloading_marker.exists):
                    try:
                        async with aiofiles.open(downloading_marker, "r") as f:
                            lock_data = await asyncio.to_thread(
                                json.loads, await f.read()
                            )

                        download_start_time = lock_data.get("start_time", 0)
                        elapsed_time = time.time() - download_start_time

                        if elapsed_time > self.timeout_threshold:

                            print(
                                f"🕒 [{self.replica_id}] Download by replica '{lock_data.get('replica_id', 'unknown')}' has timed out ({elapsed_time:.1f}s > {self.timeout_threshold:.1f}s)"
                            )

                            # Remove stale downloading directory
                            temp_download_dir = (
                                self.cache_dir
                                / f".temp_{package_dir.name}_{int(download_start_time * 1000000)}"
                            )
                            if await asyncio.to_thread(temp_download_dir.exists):
                                try:
                                    await asyncio.to_thread(
                                        shutil.rmtree, temp_download_dir
                                    )
                                    print(
                                        f"🧹 [{self.replica_id}] Cleaned up stale download directory: {temp_download_dir}"
                                    )
                                except Exception as e:
                                    print(
                                        f"⚠️ [{self.replica_id}] Failed to clean up stale download directory: {e}"
                                    )

                            return False

                    except (json.JSONDecodeError, KeyError, OSError, IOError):
                        # Corrupted or unreadable lock file, treat as timed out
                        print(
                            f"⚠️ [{self.replica_id}] Corrupted lock file detected, treating as timed out"
                        )
                        return False

            except (OSError, IOError) as e:
                # Handle filesystem errors gracefully
                print(f"⚠️ [{self.replica_id}] Filesystem error while waiting: {e}")

            await asyncio.sleep(check_interval)

        # Timeout reached
        print(
            f"⏰ [{self.replica_id}] Timeout waiting for '{package_dir.name}' download completion."
        )
        return False

    async def _get_cached_models_info(self) -> List[Dict[str, Union[str, float, bool]]]:
        """Get information about all cached models including access times and locks."""
        import aiofiles

        models_info = []

        try:
            items = await asyncio.to_thread(lambda: list(self.cache_dir.iterdir()))
        except (OSError, IOError) as e:
            print(f"⚠️ [{self.replica_id}] Error reading models directory: {e}")
            return models_info

        for item in items:
            try:
                if not await asyncio.to_thread(item.is_dir) or item.name.startswith(
                    "."
                ):
                    continue

                access_file = item / ".last_access"
                meta_file = item / ".file_metadata.json"
                downloading_marker = self.cache_dir / f".downloading_{item.name}.lock"

                # Check if currently downloading
                is_downloading = await asyncio.to_thread(downloading_marker.exists)

                # Get last access time
                last_access = 0
                is_locked = False
                if await asyncio.to_thread(access_file.exists):
                    try:
                        access_content = await asyncio.to_thread(access_file.read_text)
                        last_access = float(access_content.strip())
                        # Consider locked if accessed very recently (within 10 seconds)
                        is_locked = (time.time() - last_access) < 10
                    except (ValueError, FileNotFoundError, OSError, IOError):
                        last_access = 0

                # Get download time from file metadata
                download_time = 0
                if await asyncio.to_thread(meta_file.exists):
                    try:
                        async with aiofiles.open(meta_file, "r") as f:
                            content = await f.read()
                            metadata = await asyncio.to_thread(json.loads, content)
                        # Get the newest timestamp from all files
                        timestamps = [
                            float(ts)
                            for ts in metadata.values()
                            if isinstance(ts, (int, float, str))
                        ]
                        download_time = max(timestamps) if timestamps else 0
                    except (
                        ValueError,
                        FileNotFoundError,
                        OSError,
                        IOError,
                        json.JSONDecodeError,
                    ):
                        download_time = 0

                # Calculate model size in bytes
                model_size_bytes = await self._calculate_model_size(item)

                models_info.append(
                    {
                        "model_id": item.name,
                        "path": item,
                        "last_access": last_access,
                        "download_time": download_time,
                        "size_bytes": model_size_bytes,
                        "is_locked": is_locked,
                        "is_downloading": is_downloading,
                    }
                )
            except (OSError, IOError) as e:
                # Skip problematic directories but continue processing others
                print(
                    f"⚠️ [{self.replica_id}] Error processing cache directory {item}: {e}"
                )
                continue

        return models_info

    async def _calculate_model_size(self, model_dir: Path) -> int:
        """Calculate the total size of a model directory in bytes."""
        total_size = 0
        try:
            all_items = await asyncio.to_thread(lambda: list(model_dir.rglob("*")))
            for item in all_items:
                if await asyncio.to_thread(item.is_file):
                    try:
                        stat_result = await asyncio.to_thread(item.stat)
                        total_size += stat_result.st_size
                    except (OSError, IOError):
                        # Skip files that can't be accessed
                        continue
        except (OSError, IOError):
            # Return 0 if directory can't be accessed
            pass
        return total_size

    async def _ensure_cache_space(
        self,
        model_id: str,
        model_size_bytes: int,
        max_retries: int = 10,
        retry_delay: float = 5.0,
    ) -> None:
        """Ensure there's space in cache for a new model, evicting old ones if necessary."""
        print(
            f"🔍 [{self.replica_id}] Checking cache space for new model: '{model_id}' ({model_size_bytes / (1024*1024):.1f} MB)"
        )

        for attempt in range(max_retries):
            # Add small random delay to reduce contention between replicas
            if attempt > 0:
                delay = retry_delay + random.uniform(0, 2)
                await asyncio.sleep(delay)

            models_info = await self._get_cached_models_info()

            # Calculate current cache size in bytes
            current_size_bytes = sum(model["size_bytes"] for model in models_info)

            # Check if we need to account for the downloading model
            downloading_marker = self.cache_dir / f".downloading_{model_id}.lock"
            if await asyncio.to_thread(downloading_marker.exists):
                current_size_bytes += (
                    model_size_bytes  # Account for the model we're about to download
                )

            print(
                f"📊 [{self.replica_id}] Current cache usage: {current_size_bytes / (1024*1024*1024):.2f} GB / {self.cache_size_bytes / (1024*1024*1024):.2f} GB"
            )

            if current_size_bytes + model_size_bytes <= self.cache_size_bytes:
                print(
                    f"✅ [{self.replica_id}] Cache space available for model '{model_id}'"
                )
                return

            # Need to evict models - sort by last access time (oldest first)
            evictable_models = [
                model
                for model in models_info
                if not model["is_locked"] and not model["is_downloading"]
            ]

            if not evictable_models:
                if attempt < max_retries - 1:
                    print(
                        f"⚠️ [{self.replica_id}] No evictable models found, retrying..."
                    )
                    continue
                else:
                    print(
                        f"⚠️ [{self.replica_id}] Could not evict any models, proceeding anyway"
                    )
                    return

            # Sort by last access time (oldest first)
            evictable_models.sort(key=lambda x: x["last_access"])

            # Evict models until we have enough space
            space_needed = (
                current_size_bytes + model_size_bytes
            ) - self.cache_size_bytes

            for oldest_model in evictable_models:
                if space_needed <= 0:
                    break

                print(
                    f"🗑️ [{self.replica_id}] Evicting model: {oldest_model['model_id']} ({oldest_model['size_bytes'] / (1024*1024):.1f} MB, last accessed: {oldest_model['last_access']})"
                )

                try:
                    await self._remove_package(oldest_model["path"])
                    print(
                        f"✅ [{self.replica_id}] Successfully evicted model: {oldest_model['model_id']} ({oldest_model['size_bytes'] / (1024*1024):.1f} MB)"
                    )
                    space_needed -= oldest_model["size_bytes"]
                except Exception as e:
                    print(
                        f"❌ [{self.replica_id}] Failed to evict model '{oldest_model['model_id']}': {e}"
                    )

            # Check if we've freed enough space
            if space_needed <= 0:
                print(
                    f"✅ [{self.replica_id}] Successfully freed enough cache space for model '{model_id}'"
                )
                return
            elif attempt < max_retries - 1:
                print(
                    f"⚠️ [{self.replica_id}] Still need {space_needed / (1024*1024):.1f} MB more space, retrying..."
                )
                await asyncio.sleep(retry_delay)
                continue
            else:
                print(
                    f"⚠️ [{self.replica_id}] Could not free enough space, proceeding anyway"
                )
                return

    async def _fetch_file_list(self, model_id: str, stage: bool = False) -> List[dict]:
        """Fetch the list of files for a model from the bioimage.io artifacts API."""
        stage_param = f"?stage={str(stage).lower()}"
        url = f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}/files/{stage_param}"

        download_timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(
            timeout=download_timeout, follow_redirects=True
        ) as client:
            response = await client.get(url)

            if response.status_code == 404 and stage:
                # If staged version doesn't exist, try with stage=false
                print(
                    f"⚠️ [{self.replica_id}] Staged file list not found for model '{model_id}', trying committed version..."
                )
                stage_param = "?stage=false"
                url = f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}/files/{stage_param}"
                response = await client.get(url)

            response.raise_for_status()
            return response.json()

    async def _calculate_remote_model_size(self, file_list: List[dict]) -> int:
        """Calculate the total size of a model from its file list."""
        total_size = 0
        for file_info in file_list:
            if file_info.get("type") == "file" and "size" in file_info:
                total_size += file_info["size"]
        return total_size

    async def _download_file(
        self,
        client: httpx.AsyncClient,
        model_id: str,
        model_dir: Path,
        file_meta: dict,
        stage: bool = False,
    ):
        """Download a single file for a model."""
        import aiofiles

        stage_param = f"?stage={str(stage).lower()}"
        file_url = f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}/files/{file_meta['name']}{stage_param}"
        file_path = model_dir / file_meta["name"]

        # Create parent directories if needed
        await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)

        response = await client.get(file_url)

        if response.status_code == 404 and stage:
            # If staged version doesn't exist, try with stage=false
            stage_param = "?stage=false"
            file_url = f"https://hypha.aicell.io/bioimage-io/artifacts/{model_id}/files/{file_meta['name']}{stage_param}"
            response = await client.get(file_url)

        response.raise_for_status()

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(response.content)

        return file_meta["name"], file_meta["last_modified"]

    async def _download_model_files(
        self,
        model_id: str,
        model_dir: Path,
        stage: bool = False,
        check_newer_files: bool = True,
        file_list: Optional[List[dict]] = None,
    ):
        """Download all files for a model using concurrent downloads."""
        import aiofiles

        await asyncio.to_thread(model_dir.mkdir, parents=True, exist_ok=True)

        meta_path = model_dir / ".file_metadata.json"
        old_meta = {}
        if check_newer_files and await asyncio.to_thread(meta_path.exists):
            async with aiofiles.open(meta_path, "r") as f:
                content = await f.read()
                old_meta = await asyncio.to_thread(json.loads, content)

        # Use provided file list or fetch it if not provided
        if file_list is None:
            file_list = await self._fetch_file_list(model_id, stage=stage)
        remote_files = {f["name"]: f for f in file_list if f["type"] == "file"}

        # Get local files (excluding metadata files)
        all_files = await asyncio.to_thread(lambda: list(model_dir.glob("*")))
        local_files = set()
        for f in all_files:
            if await asyncio.to_thread(f.is_file) and f.name not in [
                ".last_access",
                ".file_metadata.json",
            ]:
                local_files.add(f.name)

        # Determine files to delete
        remote_file_names = set(remote_files.keys())
        files_to_delete = local_files - remote_file_names
        for fname in files_to_delete:
            (model_dir / fname).unlink()

        # Determine files to download
        files_to_download = []
        for name, meta in remote_files.items():
            if not check_newer_files:
                files_to_download.append(meta)
            elif name not in old_meta or meta["last_modified"] > old_meta[name]:
                files_to_download.append(meta)

        download_timeout = httpx.Timeout(180.0)
        async with httpx.AsyncClient(
            timeout=download_timeout, follow_redirects=True
        ) as client:
            tasks = [
                self._download_file(client, model_id, model_dir, f, stage=stage)
                for f in files_to_download
            ]
            results = await asyncio.gather(*tasks)

        # Update metadata
        new_meta = old_meta.copy()
        for name, ts in results:
            new_meta[name] = ts

        async with aiofiles.open(meta_path, "w") as f:
            await f.write(json.dumps(new_meta, indent=2))

        return {
            "downloaded": [name for name, _ in results],
            "deleted": list(files_to_delete),
            "skipped": list(remote_file_names - {name for name, _ in results}),
        }

    async def _validate_package(self, package_path: Path) -> bool:
        """Validate model RDF and return actual package path."""
        from bioimageio.core import load_model_description
        from bioimageio.spec import InvalidDescr

        try:
            # Find the RDF file in the package
            rdf_path = package_path / "rdf.yaml"
            if not await asyncio.to_thread(rdf_path.exists):
                raise FileNotFoundError(f"No rdf.yaml found in {package_path}/")

            # Validate model source
            model_description = await asyncio.to_thread(
                load_model_description,
                rdf_path,
                perform_io_checks=False,
            )
            if isinstance(model_description, InvalidDescr):
                raise ValueError(
                    f"Downloaded model at {package_path}/ is invalid: {model_description}"
                )

            print(
                f"✅ [{self.replica_id}] Model '{package_path.name}' validation successful"
            )

        except Exception as e:
            raise RuntimeError(
                f"Model validation failed for '{package_path.name}': {e}"
            )

    async def _create_package(self, model_id: str, stage: bool) -> None:
        """
        Create or update a model package in the cache directory.

        Downloads all files of a model artifact from the bioimage.io workspace.
        If files already exist, they are updated only if newer versions are available.
        Uses atomic operations to prevent conflicts between replicas.
        """
        import aiofiles

        package_dir = self.cache_dir / model_id
        downloading_marker = self.cache_dir / f".downloading_{model_id}.lock"

        # Fetch file list once at the beginning
        file_list = None
        try:
            file_list = await self._fetch_file_list(model_id, stage=stage)
        except Exception as e:
            print(
                f"❌ [{self.replica_id}] Failed to fetch file list for model '{model_id}': {e}"
            )
            raise RuntimeError(f"Failed to fetch file list for model {model_id}: {e}")

        # Calculate model size from file list
        model_size_bytes = await self._calculate_remote_model_size(file_list)
        print(
            f"📊 [{self.replica_id}] Model '{model_id}' size: {model_size_bytes / (1024*1024):.1f} MB"
        )

        # Check if model already exists
        if await asyncio.to_thread(package_dir.exists):
            print(
                f"💾 [{self.replica_id}] Model '{model_id}' already exists, checking for updates..."
            )

            # Check if files need updating by comparing with remote file list
            try:
                remote_files = {f["name"]: f for f in file_list if f["type"] == "file"}

                # Get local file metadata
                meta_path = package_dir / ".file_metadata.json"
                local_meta = {}
                if await asyncio.to_thread(meta_path.exists):
                    async with aiofiles.open(meta_path, "r") as f:
                        content = await f.read()
                        local_meta = await asyncio.to_thread(json.loads, content)

                # Check for files that need updating
                files_need_update = False
                for name, remote_file in remote_files.items():
                    if (
                        name not in local_meta
                        or remote_file["last_modified"] > local_meta[name]
                    ):
                        files_need_update = True
                        print(
                            f"📄 [{self.replica_id}] File '{name}' needs update (remote: {remote_file['last_modified']}, local: {local_meta.get(name, 'missing')})"
                        )
                        break

                # Check for files that no longer exist remotely
                all_local_files = await asyncio.to_thread(
                    lambda: list(package_dir.glob("*"))
                )
                local_files = set()
                for f in all_local_files:
                    if await asyncio.to_thread(f.is_file) and f.name not in [
                        ".last_access",
                        ".file_metadata.json",
                    ]:
                        local_files.add(f.name)
                remote_file_names = set(remote_files.keys())
                files_to_delete = local_files - remote_file_names
                if files_to_delete:
                    files_need_update = True
                    print(
                        f"🗑️ [{self.replica_id}] Files to delete: {list(files_to_delete)}"
                    )

                if not files_need_update:
                    print(f"✅ [{self.replica_id}] Model '{model_id}' is up to date")
                    # Update access time and return
                    access_file = package_dir / ".last_access"
                    try:
                        await asyncio.to_thread(
                            access_file.write_text, str(time.time())
                        )
                    except (OSError, IOError) as e:
                        print(
                            f"⚠️ [{self.replica_id}] Failed to update access time for existing model: {e}"
                        )
                    return

                print(
                    f"🔄 [{self.replica_id}] Model '{model_id}' needs updates, proceeding with download..."
                )

            except Exception as e:
                print(
                    f"⚠️ [{self.replica_id}] Failed to check for updates: {e}. Proceeding with download..."
                )
                # Continue with download if update check fails

        # Try to claim the download by creating a downloading marker atomically
        try:
            lock_data = {
                "replica_id": self.replica_id,
                "start_time": time.time(),
                "model_id": model_id,
                "stage": stage,
            }
            async with aiofiles.open(downloading_marker, "x") as f:
                await f.write(json.dumps(lock_data, indent=2))
            print(f"🔒 [{self.replica_id}] Claimed download for model '{model_id}'.")
        except FileExistsError:
            # Another replica is downloading, wait for completion
            print(
                f"⏳ [{self.replica_id}] Another replica is downloading '{model_id}', waiting..."
            )
            if await self._wait_for_download_completion(package_dir):
                # Update access time
                access_file = package_dir / ".last_access"
                try:
                    await asyncio.to_thread(access_file.write_text, str(time.time()))
                except (OSError, IOError) as e:
                    print(
                        f"⚠️ [{self.replica_id}] Failed to update access time after waiting: {e}"
                    )
                return
            else:
                # Download failed or timed out, try to claim it ourselves

                # Remove stale marker
                downloading_marker.unlink()

                print(
                    f"🧹 [{self.replica_id}] Cleaned up stale download marker for model '{model_id}'."
                )

                # Retry claiming
                try:
                    # Update lock data with new start time
                    lock_data["start_time"] = time.time()
                    async with aiofiles.open(downloading_marker, "x") as f:
                        await f.write(json.dumps(lock_data, indent=2))
                    print(
                        f"🔄 [{self.replica_id}] Claimed download after timeout for model '{model_id}'."
                    )
                except FileExistsError:
                    raise RuntimeError(
                        f"Failed to claim download for model {model_id} after timeout"
                    )

        try:
            # Ensure cache space AFTER claiming download (so download marker counts towards limit)
            await self._ensure_cache_space(model_id, model_size_bytes)

            # Create temporary download directory
            temp_download_dir = (
                self.cache_dir
                / f".temp_{model_id}_{int(lock_data['start_time'] * 1000000)}"
            )

            await asyncio.to_thread(temp_download_dir.mkdir)
            print(
                f"📁 [{self.replica_id}] Starting download of model '{model_id}' to temporary directory."
            )

            # If updating an existing package, copy existing files to temp directory first
            if await asyncio.to_thread(package_dir.exists):
                print(
                    f"📋 [{self.replica_id}] Copying existing files to temporary directory for update..."
                )
                try:
                    # Copy all existing files except access tracking files
                    for item in await asyncio.to_thread(
                        lambda: list(package_dir.iterdir())
                    ):
                        if await asyncio.to_thread(item.is_file) and item.name not in [
                            ".last_access"
                        ]:
                            dest_file = temp_download_dir / item.name
                            await asyncio.to_thread(
                                dest_file.parent.mkdir, parents=True, exist_ok=True
                            )
                            await asyncio.to_thread(shutil.copy2, item, dest_file)
                        elif await asyncio.to_thread(item.is_dir):
                            dest_dir = temp_download_dir / item.name
                            await asyncio.to_thread(
                                shutil.copytree, item, dest_dir, dirs_exist_ok=True
                            )
                    print(
                        f"✅ [{self.replica_id}] Copied existing files to temporary directory"
                    )
                except Exception as e:
                    print(
                        f"⚠️ [{self.replica_id}] Failed to copy existing files: {e}. Starting fresh download..."
                    )

            download_start = time.time()

            # Use concurrent file download for all models, passing the pre-fetched file list
            download_result = await self._download_model_files(
                model_id=model_id,
                model_dir=temp_download_dir,
                stage=stage,
                check_newer_files=True,  # Always check for newer files when updating
                file_list=file_list,  # Pass the file list we fetched at the beginning
            )

            print(
                f"📦 [{self.replica_id}] Downloaded {len(download_result['downloaded'])} files for model '{model_id}'."
            )

            download_duration = time.time() - download_start
            print(
                f"⚡ [{self.replica_id}] Download completed in {download_duration:.2f}s for model '{model_id}'."
            )

            # Validate the downloaded package
            await self._validate_package(temp_download_dir)

            # Atomically move to final location (handle existing directory)
            if await asyncio.to_thread(package_dir.exists):
                # For updates: move existing dir away, move temp dir to final location, then remove old dir
                backup_dir = (
                    self.cache_dir / f".backup_{model_id}_{int(time.time() * 1000000)}"
                )
                await asyncio.to_thread(package_dir.rename, backup_dir)
                await asyncio.to_thread(temp_download_dir.rename, package_dir)
                await asyncio.to_thread(shutil.rmtree, backup_dir)
                print(
                    f"🔄 [{self.replica_id}] Atomically updated model '{model_id}' in final location."
                )
            else:
                # For new models: simple rename
                temp_download_dir.rename(package_dir)
                print(
                    f"🔄 [{self.replica_id}] Atomically moved model '{model_id}' to final location."
                )

            # Create last access file (file metadata is already created by self._download_model_files)
            current_time = time.time()
            access_file = package_dir / ".last_access"

            try:
                await asyncio.to_thread(access_file.write_text, str(current_time))
            except (OSError, IOError) as e:
                print(
                    f"⚠️ [{self.replica_id}] Failed to create access file for new model: {e}"
                )

        except Exception as e:
            print(f"❌ [{self.replica_id}] Failed to download model '{model_id}': {e}")
            # Clean up temporary directory
            if await asyncio.to_thread(temp_download_dir.exists):
                try:
                    await asyncio.to_thread(shutil.rmtree, temp_download_dir)
                except Exception as cleanup_error:
                    print(
                        f"⚠️ [{self.replica_id}] Failed to cleanup temp directory: {cleanup_error}"
                    )
            raise RuntimeError(f"Failed to download model {model_id}: {e}")
        finally:
            # Remove downloading marker
            try:
                downloading_marker.unlink()
                print(
                    f"🔓 [{self.replica_id}] Released download claim for model '{model_id}'."
                )
            except (FileNotFoundError, OSError) as e:
                print(f"⚠️ [{self.replica_id}] Failed to remove downloading marker: {e}")

        print(
            f"🎉 [{self.replica_id}] Successfully completed download of model '{model_id}'."
        )

    async def _get_latest_download_time(self, package_path: Path) -> float:
        """Get the latest download time from the .file_metadata.json file."""
        import aiofiles

        meta_path = package_path / ".file_metadata.json"
        if await asyncio.to_thread(meta_path.exists):
            async with aiofiles.open(meta_path, "r") as f:
                content = await f.read()
                metadata = await asyncio.to_thread(json.loads, content)
                return max(metadata.values(), default=0.0)
        return 0.0

    async def get_model_package(
        self,
        model_id: str,
        stage: bool,
        allow_unpublished: bool,
        skip_cache: bool,
    ) -> "BioimageioPackage":
        """Get a cached model package or download it if not available."""

        # Check if model is published
        if not allow_unpublished:
            is_published = await self._check_model_published_status(
                model_id, stage=stage
            )
            if not is_published:
                raise ValueError(
                    f"Model '{model_id}' is not published. Only published models are allowed."
                )

        # Force a complete re-download if skip_cache is True
        package_path = self.cache_dir / model_id
        if await asyncio.to_thread(package_path.exists) and skip_cache:
            await self._remove_package(package_path)

        # Create or update the local package
        await self._create_package(model_id, stage=stage)

        # Get the latest download time from .file_metadata.json
        latest_download = await self._get_latest_download_time(package_path)

        return BioimageioPackage(
            package_path=package_path,
            latest_download=latest_download,
            replica_id=self.replica_id,
        )


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        # "memory": 16 * 1024 * 1024 * 1024,  # 16GB RAM limit
        "runtime_env": {
            "pip": [
                "bioimageio.core==0.9.0",
                "numpy==1.26.4",
                "tqdm>=4.64.0",
                "aiofiles>=23.0.0",
            ],
        },
    },
    max_ongoing_requests=10,  # Should be smaller than the number of cached models to prevent race conditions between requests
    max_queued_requests=30,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
        "target_num_ongoing_requests_per_replica": 1,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class ModelRunner:
    """
    Ray Serve deployment for bioimage.io model operations.

    Handles model downloading, caching, validation, testing, and inference
    with cross-replica coordination and atomic filesystem operations.

    Concurrency Design:
    - Uses atomic filesystem operations (mkdir, rename) for replica coordination
    - Download markers prevent duplicate downloads across replicas
    - Access tracking (.last_access files) prevents eviction of active models
    - LRU eviction with retry logic handles cache space management
    - Context managers ensure proper access time tracking during model usage
    - Graceful error handling for filesystem race conditions and I/O errors
    """

    def __init__(
        self,
        runtime_deployment: DeploymentHandle,
        cache_size_in_gb: float = 50.0,
    ) -> None:
        self.runtime_deployment = runtime_deployment

        # Set Hypha server and workspace
        self.server_url = "https://hypha.aicell.io"
        self.workspace = "bioimage-io"

        # Get replica identifier for logging
        try:
            self.replica_id = serve.get_replica_context().replica_tag
        except Exception:
            self.replica_id = "unknown"

        # Set up model cache
        self.model_cache = ModelCache(
            cache_size_in_gb=cache_size_in_gb,
            replica_id=self.replica_id,
        )

        print(
            f"🚀 [{self.replica_id}] ModelRunner initialized with models directory: "
            f"{self.model_cache.cache_dir} (cache_size={self.model_cache.cache_size_bytes / (1024*1024*1024):.1f} GB)"
        )

    # === BioEngine App Method - will be called when the deployment is started ===

    async def test_deployment(
        self,
        model_id: str = "ambitious-ant",
    ) -> Dict[str, Union[bool, str, float, Dict]]:
        """Comprehensive test of all public endpoints using a known working model (that should pass all checks)."""
        print(f"🧪 [{self.replica_id}] Starting deployment test with model: {model_id}")

        # Test 1: Get model RDF for validation
        print(f"🔍 [{self.replica_id}] Test 1/5: Getting model RDF...")
        rdf_start = time.time()
        model_rdf = await self.get_model_rdf(model_id=model_id, stage=False)
        rdf_duration = time.time() - rdf_start
        print(f"✅ [{self.replica_id}] RDF retrieval successful ({rdf_duration:.2f}s)")

        # Test 2: Validate the RDF
        print(f"🔬 [{self.replica_id}] Test 2/5: Validating RDF...")
        val_start = time.time()
        validation_result = await self.validate(rdf_dict=model_rdf)
        val_duration = time.time() - val_start
        print(
            f"✅ [{self.replica_id}] Validation {'passed' if validation_result['success'] else 'failed'} ({val_duration:.2f}s)"
        )

        # Test 3: Test the model
        print(f"🧩 [{self.replica_id}] Test 3/5: Testing model...")
        test1_start = time.time()
        _ = await self.test(model_id=model_id, stage=False)
        test1_duration = time.time() - test1_start
        print(f"✅ [{self.replica_id}] Model test completed ({test1_duration:.2f}s)")

        # Test 4: Test with skip_cache=True
        print(f"🔄 [{self.replica_id}] Test 4/5: Testing with cache skip...")
        test2_start = time.time()
        _ = await self.test(model_id=model_id, stage=False, skip_cache=True)
        test2_duration = time.time() - test2_start
        print(
            f"✅ [{self.replica_id}] Skip cache test completed ({test2_duration:.2f}s)"
        )

        # Test 5: Test inference (published)
        print(f"🤖 [{self.replica_id}] Test 5/5: Running inference...")

        # Get the model package to load test image
        local_package = await self.model_cache.get_model_package(
            model_id=model_id, stage=False, allow_unpublished=False, skip_cache=False
        )
        async with local_package:
            test_input_source = model_rdf["test_inputs"][0]
            test_image_path = local_package.package_path / test_input_source
            test_image = np.load(test_image_path).astype("float32")

        # Run inference test
        infer_start = time.time()
        _ = await self.infer(model_id=model_id, inputs=test_image)
        infer_duration = time.time() - infer_start
        print(f"✅ [{self.replica_id}] Inference completed ({infer_duration:.2f}s)")

    # === Exposed BioEngine App Methods - all methods decorated with @schema_method will be exposed as API endpoints ===
    # Note: Parameter type hints and docstrings will be used to generate the API documentation.

    @schema_method
    async def get_model_rdf(
        self,
        model_id: str = Field(
            ...,
            description="Unique identifier of the bioimage.io model (e.g., 'ambitious-ant')",
        ),
        stage: Optional[bool] = Field(
            False,
            description="Whether to get RDF from the staged version of the model (True) or the committed version (False)",
        ),
    ) -> Dict[str, Union[str, int, float, List, Dict]]:
        """
        Retrieve the Resource Description Framework (RDF) metadata for a bioimage.io model.

        The RDF contains comprehensive model metadata including:
        - Model identification (id, name, description, authors)
        - Input/output tensor specifications (shape, data type, preprocessing)
        - Model architecture details and framework requirements
        - Training information and performance metrics
        - Compatible software versions and dependencies

        Returns:
            Dictionary containing the complete RDF metadata structure with nested
            configuration for inputs, outputs, preprocessing, postprocessing, and model weights

        Raises:
            ValueError: If model_id is invalid or model not found
            RuntimeError: If download fails
        """
        print(
            f"📋 [{self.replica_id}] Downloading RDF for model '{model_id}' (stage={stage})."
        )

        stage_param = f"?stage={str(stage).lower()}"
        download_url = f"{self.server_url}/{self.workspace}/artifacts/{model_id}/files/rdf.yaml{stage_param}"
        response = await self.model_cache.client.get(download_url)

        if response.status_code == 404 and stage:
            # If staged version doesn't exist, try with stage=false
            print(
                f"⚠️ [{self.replica_id}] Staged RDF not found for model '{model_id}', trying committed version..."
            )
            stage_param = "?stage=false"
            download_url = f"{self.server_url}/{self.workspace}/artifacts/{model_id}/files/rdf.yaml{stage_param}"
            response = await self.model_cache.client.get(download_url)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download RDF from {download_url}: "
                f"HTTP {response.status_code} - {response.text}"
            )

        model_rdf = await asyncio.to_thread(yaml.safe_load, response.text)

        print(
            f"✅ [{self.replica_id}] Successfully downloaded RDF for model '{model_id}'."
        )
        return model_rdf

    @schema_method
    async def validate(
        self,
        rdf_dict: Dict[str, Union[str, int, float, List, Dict]] = Field(
            ..., description="Complete RDF dictionary structure to validate"
        ),
        known_files: Optional[Dict[str, str]] = Field(
            None,
            description="Mapping of relative file paths to their content hashes for validating file references within the RDF",
        ),
    ) -> Dict[str, Union[bool, str]]:
        """
        Validate a model Resource Description Framework (RDF) against bioimage.io specifications.

        Performs comprehensive validation including:
        - Schema compliance checking against bioimage.io RDF specification
        - Data type and format validation for all fields
        - Logical consistency verification between related fields
        - Tensor shape and dimension compatibility analysis
        - File reference and path validation (if known_files provided)

        Returns:
            Validation result containing:
            - success: Boolean indicating overall validation status
            - details: Detailed validation report with specific issues or confirmation

        Note:
            This method performs format validation only (perform_io_checks=False).
            File existence is not verified unless known_files mapping is provided.
        """
        from bioimageio.spec import ValidationContext, validate_format

        print(
            f"🔬 [{self.replica_id}] Validating RDF (known_files: {len(known_files or {})} files)..."
        )

        ctx = ValidationContext(perform_io_checks=False, known_files=known_files or {})
        summary = await asyncio.to_thread(validate_format, rdf_dict, context=ctx)

        result = {
            "success": summary.status == "valid-format",
            "details": summary.format(),
        }

        print(
            f"✅ [{self.replica_id}] RDF validation {'passed' if result['success'] else 'failed'}."
        )
        return result

    @schema_method
    async def test(
        self,
        model_id: str = Field(
            ..., description="Unique identifier of the bioimage.io model to test"
        ),
        stage: Optional[bool] = Field(
            True,
            description="Whether to get the staged version of the model (True) or the committed version (False)",
        ),
        additional_requirements: Optional[List[str]] = Field(
            None,
            description='Extra Python packages to install in the test environment (e.g., ["scipy>=1.7.0", "scikit-image"])',
        ),
        skip_cache: Optional[bool] = Field(
            False, description="Force re-download of model package before testing"
        ),
    ) -> Dict[str, Union[str, bool, List, Dict]]:
        """
        Execute comprehensive bioimage.io model testing using the official test suite.

        Performs automated testing including:
        - Model loading and initialization verification
        - Input/output tensor compatibility testing
        - Preprocessing and postprocessing pipeline validation
        - Sample inference execution with synthetic or provided test data
        - Performance benchmarking and memory usage analysis
        - Framework-specific compatibility verification (PyTorch, TensorFlow, ONNX)

        Returns:
            Comprehensive test results including:
            - Test execution status and outcomes for each test component
            - Performance metrics (inference time, memory usage)
            - Compatibility results across different frameworks
            - Detailed error information if any tests fail
            - Model metadata and configuration verification results

        Note:
            This method delegates to the model_evaluation deployment for isolated testing
            in a controlled environment with the specified requirements.
        """
        print(
            f"🧪 [{self.replica_id}] Testing model '{model_id}' (stage={stage}, skip_cache={skip_cache})."
        )

        try:
            # Get model package with access tracking
            package = await self.model_cache.get_model_package(
                model_id=model_id,
                stage=stage,
                allow_unpublished=True,
                skip_cache=skip_cache,
            )

            # Use context manager to track access and prevent eviction during test
            async with package:
                print(
                    f"📍 [{self.replica_id}] Model source for '{model_id}': {package.source}"
                )

                test_result = await self.runtime_deployment.test.remote(
                    rdf_path=package.source,
                    additional_requirements=additional_requirements,
                )
        except RayTaskError as e:
            error_msg = f"Failed to run model test for '{model_id}': {e}"
            print(f"❌ [{self.replica_id}] {error_msg}")
            raise RuntimeError(error_msg)

        print(f"✅ [{self.replica_id}] Model test completed for '{model_id}'.")
        return test_result

    @schema_method(arbitrary_types_allowed=True)
    async def infer(
        self,
        model_id: str = Field(
            ..., description="Unique identifier of the published bioimage.io model"
        ),
        inputs: Union[np.ndarray, Dict[str, np.ndarray]] = Field(
            ...,
            description="Input data as numpy array or dictionary of named arrays. Must match the model's input specification for shape and data type. For single input models, provide np.ndarray. For multi-input models, provide dict with input names as keys.",
        ),
        weights_format: Optional[str] = Field(
            None,
            description='Preferred model weights format ("pytorch_state_dict", "torchscript", "onnx", "tensorflow_saved_model"). If None, automatically selects best available.',
        ),
        device: Optional[Literal["cuda", "cpu"]] = Field(
            None,
            description='Target computation device. "cuda" for GPU acceleration, "cpu" for CPU-only. If None, automatically selects based on availability and model compatibility.',
        ),
        default_blocksize_parameter: Optional[int] = Field(
            None,
            description="Override default tiling block size for memory management. Larger values use more memory but may be faster. Only applicable for models supporting tiled inference.",
        ),
        sample_id: Optional[str] = Field(
            "sample",
            description="Identifier for this inference request, used for logging and debugging",
        ),
        skip_cache: Optional[bool] = Field(
            False, description="Force re-download of model package before inference"
        ),
    ) -> Dict[str, np.ndarray]:
        """
        Execute inference on a bioimage.io model with provided input data.

        Performs end-to-end inference including:
        - Automatic input preprocessing according to model specification
        - Model execution with optimized framework backend
        - Output postprocessing and format standardization
        - Memory-efficient processing for large inputs using tiling if supported

        Returns:
            Dictionary mapping output names to numpy arrays containing the inference results.
            Output shapes and data types match the model's output specification.
            For single-output models, typically returns {"output": result_array}.

        Raises:
            ValueError: If model_id is a URL (only model IDs allowed) or inputs don't match specification
            RuntimeError: If model loading, preprocessing, inference, or postprocessing fails

        Note:
            Only published models from the bioimage.io model zoo are supported for inference.
            This method delegates to the model_inference deployment for optimized execution.
        """
        print(f"🤖 [{self.replica_id}] Running inference for model '{model_id}'...")

        try:
            # Get model package with access tracking
            package = await self.model_cache.get_model_package(
                model_id=model_id,
                stage=False,
                allow_unpublished=False,
                skip_cache=skip_cache,
            )

            # Use context manager to track access and prevent eviction during inference
            async with package:
                print(
                    f"📍 [{self.replica_id}] Model source for '{model_id}': {package.source} (downloaded: {package.latest_download})"
                )

                result = await self.runtime_deployment.predict.remote(
                    rdf_path=package.source,
                    inputs=inputs,
                    weights_format=weights_format,
                    device=device,
                    default_blocksize_parameter=default_blocksize_parameter,
                    sample_id=sample_id,
                    latest_download=package.latest_download,
                )
        except RayTaskError as e:
            error_msg = f"Failed to run inference for model '{model_id}': {e}"
            print(f"❌ [{self.replica_id}] {error_msg}")
            raise RuntimeError(error_msg)

        print(f"✅ [{self.replica_id}] Inference completed for model '{model_id}'.")
        return result


if __name__ == "__main__":

    async def run_deployment_test():

        class MockMethod:
            def __init__(self, name: str):
                self.name = name

            async def remote(self, *args, **kwargs):
                print(
                    f"🎭 Mocked method '{self.name}' called with args={args}, kwargs={kwargs}"
                )
                return {"status": "success", "data": "mocked data"}

        class MockHandle:
            def __getattr__(self, name):
                return MockMethod(name)

        # Set up the environment variables like in the real deployment
        deployment_workdir = (
            Path(__file__).resolve().parent.parent.parent
            / ".bioengine"
            / "apps"
            / "model-runner"
        )
        deployment_workdir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(deployment_workdir / "tmp")
        os.environ["HOME"] = str(deployment_workdir)
        os.chdir(deployment_workdir)

        # Create a mock replica context for standalone testing
        class MockReplicaContext:
            replica_tag = "test-replica-001"

        # Mock serve.get_replica_context for testing
        original_get_replica_context = serve.get_replica_context
        serve.get_replica_context = lambda: MockReplicaContext()

        try:
            model_runner = ModelRunner.func_or_class(
                runtime_deployment=MockHandle(),
                cache_size_in_gb=0.23,  # 230 MB cache for testing
            )

            await model_runner.test_deployment(model_id="ambitious-ant")

            # Simulate newer remote files for testing updates
            file_metadata_path = (
                model_runner.model_cache.cache_dir
                / "ambitious-ant"
                / ".file_metadata.json"
            )
            with open(file_metadata_path, "r") as f:
                metadata = json.load(f)
                metadata["rdf.yaml"] -= 1000  # Make it older

            with open(file_metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Run test again
            await model_runner.test(model_id="ambitious-ant")  # ~18 MB

            # This should exceed the cache size limit of 230 MB
            await model_runner.test(model_id="charismatic-whale")  # ~225 MB
        finally:
            # Restore original function
            serve.get_replica_context = original_get_replica_context

    asyncio.run(run_deployment_test())

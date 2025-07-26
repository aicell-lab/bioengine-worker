import asyncio
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import httpx
import numpy as np
from hypha_rpc.utils.schema import schema_method
from ray import serve
from ray.exceptions import RayTaskError
from ray.serve.handle import DeploymentHandle


def remove_package(package_dir: Path) -> None:
    """Safely remove package directory using atomic operations across replicas."""
    if not package_dir.exists():
        return

    replica_id = serve.get_replica_context().replica_tag

    try:
        # Use atomic rename for safe removal across replicas
        temp_dir = (
            package_dir.parent
            / f".removing_{package_dir.name}_{int(time.time() * 1000000)}"
        )
        package_dir.rename(temp_dir)
        print(
            f"🔄 [{replica_id}] Atomically moved package for removal: {package_dir.name}"
        )

        # Remove the temporary directory
        shutil.rmtree(temp_dir)
        print(
            f"🗑️ [{replica_id}] Successfully removed cached package: {package_dir.name}"
        )

    except FileNotFoundError:
        # Another replica already removed it
        print(
            f"🔍 [{replica_id}] Package {package_dir.name} already removed by another replica"
        )
    except OSError as e:
        # Package might be in use, log but don't fail
        print(
            f"⚠️ [{replica_id}] Could not remove cached package {package_dir.name}: {e}"
        )
    except Exception as e:
        print(
            f"❌ [{replica_id}] Unexpected error removing package {package_dir.name}: {e}"
        )


class LocalBioimageioPackage:
    """Wrapper for cached bioimage.io model package with access tracking."""

    def __init__(self, package_path: Path, replica_id: str) -> None:
        self.package_path = package_path
        self.model_source = package_path / "rdf.yaml"
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
        if self._access_file and self._access_file.exists():
            current_time = time.time()
            try:
                await asyncio.to_thread(self._access_file.write_text, str(current_time))
            except (OSError, IOError) as e:
                print(
                    f"⚠️ [{self.replica_id}] Failed to update access time on exit: {e}"
                )

    async def remove(self) -> None:
        """Remove this package from the cache."""
        await asyncio.to_thread(remove_package, self.package_path)

    async def validate(self) -> Path:
        """Validate model RDF and return actual package path."""
        from bioimageio.core import load_model_description
        from bioimageio.spec import InvalidDescr

        try:
            # Find the RDF file in the package
            rdf_files = list(self.package_path.rglob("rdf.yaml"))
            if not rdf_files:
                error_msg = f"No rdf.yaml found in {self.package_path}"
                print(f"❌ [{self.replica_id}] {error_msg}")
                raise FileNotFoundError(error_msg)

            model_source = rdf_files[0]
            actual_package_path = model_source.parent
            print(f"🔍 [{self.replica_id}] Found model RDF at: {model_source}")

            # Validate model source
            model_description = load_model_description(model_source)
            if isinstance(model_description, InvalidDescr):
                error_msg = f"Downloaded model at {actual_package_path}/ is invalid: {model_description}"
                print(f"❌ [{self.replica_id}] {error_msg}")
                raise ValueError(error_msg)

            model_id = model_description.id
            print(f"✅ [{self.replica_id}] Model '{model_id}' validation successful")

            # Update the model_source path to the validated one
            self.model_source = model_source
            return actual_package_path

        except Exception as e:
            print(f"❌ [{self.replica_id}] Model validation failed: {e}")
            # Clean up invalid package using convenience method
            await self.remove()
            raise


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1 / 3,
        "num_gpus": 0,
        # "memory": 16 * 1024 * 1024 * 1024,  # 16GB RAM limit
        "runtime_env": {
            "pip": [
                "bioimageio.core==0.9.0",
                "xarray==2025.1.2",  # this is needed for bioimageio.core
                "numpy==1.26.4",
                "torch==2.5.1",
                "torchvision==0.20.1",
                "tensorflow==2.16.1",
                "onnxruntime==1.20.1",
                "tqdm>=4.64.0",  # for download progress bars
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
        model_evaluation: DeploymentHandle,
        model_inference: DeploymentHandle,
        max_models: int = 30,
    ) -> None:
        self.model_evaluation = model_evaluation
        self.model_inference = model_inference

        # Set Hypha server and workspace
        self.server_url = "https://hypha.aicell.io"
        self.workspace = "bioimage-io"

        # Set up model directory
        models_dir = Path().resolve() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = models_dir

        # Cache configuration
        self.max_models = max_models

        # Get replica identifier for logging
        try:
            self.replica_id = serve.get_replica_context().replica_tag
        except Exception:
            self.replica_id = "unknown"

        print(
            f"🚀 [{self.replica_id}] ModelRunner initialized with models directory: {self.models_dir} (max_models={self.max_models})"
        )

    # === BioEngine App Methods - will be called when the deployment is started ===

    async def async_init(self) -> None:
        """Initialize the deployment and validate existing cached models."""
        print(
            f"🔄 [{self.replica_id}] Initializing ModelRunner deployment, scanning {self.models_dir}"
        )

        existing_models = 0
        try:
            items = list(self.models_dir.iterdir())
        except (OSError, IOError) as e:
            print(
                f"⚠️ [{self.replica_id}] Error reading models directory during init: {e}"
            )
            return

        for package_dir in items:
            try:
                if not package_dir.is_dir() or package_dir.name.startswith("."):
                    continue

                # Extract cache key from directory name
                cache_key = package_dir.name
                if not (
                    cache_key.startswith("bmz_model_")
                    or cache_key.startswith("unpublished_model_")
                ):
                    print(
                        f"⚠️ [{self.replica_id}] Skipping invalid cache directory: {cache_key}"
                    )
                    continue

                # Create LocalBioimageioPackage and validate existing model
                local_package = LocalBioimageioPackage(
                    package_path=package_dir, replica_id=self.replica_id
                )
                await local_package.validate()
                existing_models += 1

                # Create last access file if it doesn't exist
                access_file = package_dir / ".last_access"
                if not access_file.exists():
                    try:
                        await asyncio.to_thread(
                            access_file.write_text, str(time.time())
                        )
                    except (OSError, IOError) as e:
                        print(
                            f"⚠️ [{self.replica_id}] Failed to create access file for {cache_key}: {e}"
                        )

                print(f"📦 [{self.replica_id}] Validated existing model: {cache_key}")
            except Exception as e:
                print(
                    f"⚠️ [{self.replica_id}] Failed to validate existing model {package_dir.name}: {e}"
                )
                # Model was already removed by the validate method if it failed
                continue

        print(
            f"✅ [{self.replica_id}] Initialization complete. Validated {existing_models} existing models"
        )

    async def test_deployment(
        self, model_id: str = "charismatic-whale", test_skip_cache: bool = False
    ) -> Dict[str, Union[bool, str, float, Dict]]:
        """Comprehensive test of all public endpoints using a known working model (that should pass all checks)."""
        total_tests = 6 if test_skip_cache else 5

        print(
            f"🧪 [{self.replica_id}] Starting comprehensive deployment test with model: '{model_id}'"
        )
        # Test 1: Get model RDF for validation
        print(f"🔍 [{self.replica_id}] Test 1/{total_tests}: Getting model RDF...")
        rdf_start = time.time()
        model_rdf = await self.get_model_rdf(model_id=model_id)
        rdf_duration = time.time() - rdf_start
        print(f"✅ [{self.replica_id}] RDF retrieval successful ({rdf_duration:.2f}s)")

        # Test 2: Validate the RDF
        print(f"🔬 [{self.replica_id}] Test 2/{total_tests}: Validating RDF...")
        val_start = time.time()
        validation_result = await self.validate(rdf_dict=model_rdf)
        val_duration = time.time() - val_start
        print(
            f"✅ [{self.replica_id}] Validation {'passed' if validation_result['success'] else 'failed'} ({val_duration:.2f}s)"
        )

        # Test 3: Test the model (published)
        print(
            f"🧩 [{self.replica_id}] Test 3/{total_tests}: Testing published model..."
        )
        test1_start = time.time()
        test_result1 = await self.test(model_id=model_id, published=True)
        test1_duration = time.time() - test1_start
        print(
            f"✅ [{self.replica_id}] Published model test completed ({test1_duration:.2f}s)"
        )

        # Test 4: Test the model (unpublished)
        print(
            f"🧩 [{self.replica_id}] Test 4/{total_tests}: Testing unpublished model..."
        )
        test2_start = time.time()
        test_result2 = await self.test(model_id=model_id, published=False)
        test2_duration = time.time() - test2_start
        print(
            f"✅ [{self.replica_id}] Unpublished model test completed ({test2_duration:.2f}s)"
        )

        # Test 5: Test with skip_cache=True
        if test_skip_cache:
            print(
                f"🔄 [{self.replica_id}] Test 5/{total_tests}: Testing with cache skip..."
            )
            test3_start = time.time()
            test_result3 = await self.test(
                model_id=model_id, published=False, skip_cache=True
            )
            test3_duration = time.time() - test3_start
            print(
                f"✅ [{self.replica_id}] Skip cache test completed ({test3_duration:.2f}s)"
            )

        # Test 6: Test inference
        current_test = 6 if test_skip_cache else 5
        print(
            f"🤖 [{self.replica_id}] Test {current_test}/{total_tests}: Testing inference..."
        )
        inf_start = time.time()

        # Get the model package to load test image
        local_package = await self._get_local_package_from_cache(
            model_id=model_id, published=False, skip_cache=False
        )

        async with local_package:
            test_image_path = local_package.package_path / "new_test_input.npy"
            test_image = np.load(test_image_path).astype("float32")
            outputs = await self.infer(model_id=model_id, inputs=test_image)

        inf_duration = time.time() - inf_start
        print(f"✅ [{self.replica_id}] Inference test completed ({inf_duration:.2f}s)")

    # === Internal Methods ===

    def _get_cache_key(self, model_id: str, published: bool) -> str:
        """Generate cache key from model ID and publication status."""
        if published:
            return f"bmz_model_{model_id}"
        else:
            return f"unpublished_model_{model_id}"

    async def _wait_for_download_completion(
        self, package_dir: Path, max_wait_time: int = 300
    ) -> bool:
        """Wait for another replica to finish downloading. Returns True if successful."""
        start_time = time.time()
        downloading_marker = package_dir.parent / f".downloading_{package_dir.name}"

        print(
            f"⏳ [{self.replica_id}] Waiting for download completion: {package_dir.name}"
        )

        check_interval = 2.0
        while time.time() - start_time < max_wait_time:
            try:
                # Check if download is complete (package exists and no downloading marker)
                if package_dir.exists() and not downloading_marker.exists():
                    print(
                        f"✅ [{self.replica_id}] Download completed by another replica: {package_dir.name}"
                    )
                    return True

                # Check if download failed (no package and no downloading marker)
                if not package_dir.exists() and not downloading_marker.exists():
                    print(
                        f"⚠️ [{self.replica_id}] Download appears to have failed on another replica: {package_dir.name}"
                    )
                    return False
            except (OSError, IOError) as e:
                # Handle filesystem errors gracefully
                print(f"⚠️ [{self.replica_id}] Filesystem error while waiting: {e}")

            await asyncio.sleep(check_interval)

        # Timeout reached
        print(
            f"⏰ [{self.replica_id}] Timeout waiting for download completion: {package_dir.name}"
        )
        return False

    async def _get_cached_models_info(self) -> List[Dict[str, Union[str, float, bool]]]:
        """Get information about all cached models including access times and locks."""
        models_info = []

        try:
            items = list(self.models_dir.iterdir())
        except (OSError, IOError) as e:
            print(f"⚠️ [{self.replica_id}] Error reading models directory: {e}")
            return models_info

        for item in items:
            try:
                if not item.is_dir() or item.name.startswith("."):
                    continue

                # Check if it's a valid cache directory
                if not (
                    item.name.startswith("bmz_model_")
                    or item.name.startswith("unpublished_model_")
                ):
                    continue

                access_file = item / ".last_access"
                downloading_marker = self.models_dir / f".downloading_{item.name}"

                # Check if currently downloading
                is_downloading = downloading_marker.exists()

                # Get last access time
                last_access = 0
                is_locked = False
                if access_file.exists():
                    try:
                        access_content = await asyncio.to_thread(access_file.read_text)
                        last_access = float(access_content.strip())
                        # Consider locked if accessed very recently (within 10 seconds)
                        is_locked = (time.time() - last_access) < 10
                    except (ValueError, FileNotFoundError, OSError, IOError):
                        last_access = 0

                models_info.append(
                    {
                        "cache_key": item.name,
                        "path": item,
                        "last_access": last_access,
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

    async def _ensure_cache_space(
        self, cache_key: str, max_retries: int = 10, retry_delay: float = 5.0
    ) -> None:
        """Ensure there's space in cache for a new model, evicting old ones if necessary."""
        print(
            f"🔍 [{self.replica_id}] Checking cache space for new model: '{cache_key}'"
        )

        for attempt in range(max_retries):
            # Add small random delay to reduce contention between replicas
            if attempt > 0:
                jitter = asyncio.create_task(asyncio.sleep(0.1 + (attempt * 0.1)))
                await jitter

            models_info = await self._get_cached_models_info()

            # Count current models (including downloading ones)
            current_count = len(models_info)
            downloading_marker = self.models_dir / f".downloading_{cache_key}"
            if downloading_marker.exists():
                current_count += 1  # Count the model we're about to download

            print(
                f"📊 [{self.replica_id}] Current cache usage: {current_count}/{self.max_models}"
            )

            if current_count <= self.max_models:
                print(f"✅ [{self.replica_id}] Cache has space for new model")
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
                        f"⏳ [{self.replica_id}] All models are locked or downloading, waiting {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(
                        f"⚠️ [{self.replica_id}] No models can be evicted after {max_retries} attempts, proceeding anyway"
                    )
                    return

            # Sort by last access time (oldest first)
            evictable_models.sort(key=lambda x: x["last_access"])

            # Evict the oldest model
            oldest_model = evictable_models[0]
            print(
                f"🗑️ [{self.replica_id}] Evicting oldest model: {oldest_model['cache_key']} (last accessed: {oldest_model['last_access']})"
            )

            try:
                await asyncio.to_thread(remove_package, oldest_model["path"])
                print(
                    f"✅ [{self.replica_id}] Successfully evicted model: {oldest_model['cache_key']}"
                )
                return  # Successfully made space
            except Exception as e:
                print(
                    f"❌ [{self.replica_id}] Failed to evict model {oldest_model['cache_key']}: {e}"
                )
                # Check if the model was already removed by another replica
                if not oldest_model["path"].exists():
                    print(
                        f"🔍 [{self.replica_id}] Model was already removed by another replica"
                    )
                    return  # Space was freed by another replica

                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print(
                        f"⚠️ [{self.replica_id}] Could not evict any models, proceeding anyway"
                    )
                    return

    def _unzip_package(self, zip_file: Path) -> None:
        """Extract ZIP file contents and cleanup."""
        package_path = zip_file.parent
        print(f"📦 [{self.replica_id}] Extracting package to {package_path}")

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            files = zip_ref.namelist()
            print(f"📂 [{self.replica_id}] Package contains {len(files)} files")
            zip_ref.extractall(package_path)

        # Clean up the zip file
        zip_file.unlink()
        print(f"✅ [{self.replica_id}] Package extraction completed")

    async def _download_model_from_url(self, model_id: str, package_path: Path) -> None:
        """Download unpublished model as ZIP from Hypha server with progress bar."""
        from tqdm.asyncio import tqdm

        download_url = (
            f"{self.server_url}/{self.workspace}/artifacts/{model_id}/create-zip-file"
        )
        zip_file = package_path / "tmp_model_zip_file.zip"

        print(f"📥 [{self.replica_id}] Downloading model '{model_id}' from URL...")
        download_timeout = httpx.Timeout(120.0)

        async with httpx.AsyncClient(timeout=download_timeout) as client:
            # Stream the download to show progress
            async with client.stream("GET", download_url) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to download model from {download_url}: "
                        f"HTTP {response.status_code} - {await response.aread()}"
                    )

                # Get content length for progress bar
                total_size = int(response.headers.get("content-length", 0))

                # Create progress bar
                progress_desc = f"[{self.replica_id}] Downloading {model_id}"
                progress_kwargs = {
                    "unit": "B",
                    "unit_scale": True,
                    "unit_divisor": 1024,
                    "desc": progress_desc,
                    "ncols": 80,
                    "colour": "green",
                }

                # Only set total if we have content length
                if total_size > 0:
                    progress_kwargs["total"] = total_size

                with tqdm(**progress_kwargs) as pbar:
                    zip_content = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        zip_content.extend(chunk)
                        pbar.update(len(chunk))

        await asyncio.to_thread(zip_file.write_bytes, bytes(zip_content))
        file_size_mb = zip_file.stat().st_size / (1024 * 1024)
        print(
            f"💾 [{self.replica_id}] Downloaded {file_size_mb:.2f}MB ZIP file for model '{model_id}'"
        )

        await asyncio.to_thread(self._unzip_package, zip_file)

    async def _download_model_from_id(self, model_id: str, package_path: Path) -> None:
        """Download published model using bioimageio.spec."""
        from bioimageio.spec import save_bioimageio_package_as_folder

        print(
            f"📥 [{self.replica_id}] Downloading model '{model_id}' using bioimageio.spec..."
        )
        await asyncio.to_thread(
            save_bioimageio_package_as_folder, model_id, output_path=package_path
        )
        print(f"✅ [{self.replica_id}] Model '{model_id}' downloaded successfully")

    async def _download_model(self, cache_key: str) -> Path:
        """Download model with atomic operations to prevent conflicts between replicas."""
        # Parse cache key
        if cache_key.startswith("bmz_model_"):
            model_id = cache_key[len("bmz_model_") :]
            published = True
        elif cache_key.startswith("unpublished_model_"):
            model_id = cache_key[len("unpublished_model_") :]
            published = False
        else:
            raise ValueError(
                f"Invalid cache key format: {cache_key}. "
                "Expected format 'bmz_model_<model_id>' or 'unpublished_model_<model_id>'."
            )

        package_dir = self.models_dir / cache_key
        downloading_marker = self.models_dir / f".downloading_{cache_key}"
        temp_download_dir = (
            self.models_dir / f".temp_{cache_key}_{int(time.time() * 1000000)}"
        )

        # Check if model already exists
        if package_dir.exists():
            print(f"💾 [{self.replica_id}] Model '{model_id}' already exists")
            # Update access time
            access_file = package_dir / ".last_access"
            try:
                await asyncio.to_thread(access_file.write_text, str(time.time()))
            except (OSError, IOError) as e:
                print(
                    f"⚠️ [{self.replica_id}] Failed to update access time for existing model: {e}"
                )
            return package_dir

        # Try to claim the download by creating a downloading marker atomically
        try:
            downloading_marker.mkdir()
            print(f"🔒 [{self.replica_id}] Claimed download for model '{model_id}'")
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
                return package_dir
            else:
                # Download failed or timed out, try to claim it ourselves
                try:
                    downloading_marker.rmdir()  # Remove stale marker
                except (FileNotFoundError, OSError):
                    pass
                # Retry claiming
                try:
                    downloading_marker.mkdir()
                    print(
                        f"🔄 [{self.replica_id}] Claimed download after timeout for model '{model_id}'"
                    )
                except FileExistsError:
                    raise RuntimeError(
                        f"Failed to claim download for model {model_id} after timeout"
                    )

        try:
            # Ensure cache space AFTER claiming download (so download marker counts towards limit)
            await self._ensure_cache_space(cache_key)

            # Create temporary download directory
            temp_download_dir.mkdir()
            print(
                f"📁 [{self.replica_id}] Starting download of model '{model_id}' to temporary directory"
            )

            download_start = time.time()

            # Download model to temporary directory
            if published:
                await self._download_model_from_id(
                    model_id=model_id, package_path=temp_download_dir
                )
            else:
                await self._download_model_from_url(
                    model_id=model_id, package_path=temp_download_dir
                )

            download_duration = time.time() - download_start
            print(
                f"⚡ [{self.replica_id}] Download completed in {download_duration:.2f}s for model '{model_id}'"
            )

            # Atomically move to final location
            temp_download_dir.rename(package_dir)
            print(
                f"🔄 [{self.replica_id}] Atomically moved model '{model_id}' to final location"
            )

            # Create last access file
            access_file = package_dir / ".last_access"
            try:
                await asyncio.to_thread(access_file.write_text, str(time.time()))
            except (OSError, IOError) as e:
                print(
                    f"⚠️ [{self.replica_id}] Failed to create access file for new model: {e}"
                )

        except Exception as e:
            print(f"❌ [{self.replica_id}] Failed to download model '{model_id}': {e}")
            # Clean up temporary directory
            if temp_download_dir.exists():
                try:
                    shutil.rmtree(temp_download_dir)
                except Exception as cleanup_error:
                    print(
                        f"⚠️ [{self.replica_id}] Failed to cleanup temp directory: {cleanup_error}"
                    )
            raise RuntimeError(f"Failed to download model {model_id}: {e}")
        finally:
            # Remove downloading marker
            try:
                downloading_marker.rmdir()
                print(
                    f"🔓 [{self.replica_id}] Released download claim for model '{model_id}'"
                )
            except (FileNotFoundError, OSError) as e:
                print(f"⚠️ [{self.replica_id}] Failed to remove downloading marker: {e}")

        print(
            f"🎉 [{self.replica_id}] Successfully completed download of model '{model_id}'"
        )
        return package_dir

    async def _get_local_package_from_cache(
        self, model_id: str, published: bool = True, skip_cache: bool = False
    ) -> LocalBioimageioPackage:
        """Get model package from cache, downloading and validating if needed."""
        cache_key = self._get_cache_key(model_id=model_id, published=published)
        package_dir = self.models_dir / cache_key

        # Handle cache skipping
        if skip_cache and package_dir.exists():
            print(
                f"🔄 [{self.replica_id}] Skipping cache for model '{model_id}', removing existing package"
            )
            await asyncio.to_thread(remove_package, package_dir)

        # Download the model if it doesn't exist locally
        package_path = await self._download_model(cache_key)

        # Create LocalBioimageioPackage and validate it
        local_package = LocalBioimageioPackage(
            package_path=package_path, replica_id=self.replica_id
        )
        await local_package.validate()

        return local_package

    # === Exposed BioEngine App Methods - all methods decorated with @schema_method will be exposed as API endpoints ===
    # Note: Parameter type hints and docstrings will be used to generate the API documentation.

    @schema_method
    async def get_model_rdf(
        self, model_id: str, skip_cache: bool = False
    ) -> Dict[str, Union[str, int, float, List, Dict]]:
        """
        Retrieve the Resource Description Framework (RDF) metadata for a bioimage.io model.

        The RDF contains comprehensive model metadata including:
        - Model identification (id, name, description, authors)
        - Input/output tensor specifications (shape, data type, preprocessing)
        - Model architecture details and framework requirements
        - Training information and performance metrics
        - Compatible software versions and dependencies

        Args:
            model_id: Unique identifier of the bioimage.io model (e.g., "charismatic-whale")
            skip_cache: Force re-download from source even if model is cached locally

        Returns:
            Dictionary containing the complete RDF metadata structure with nested
            configuration for inputs, outputs, preprocessing, postprocessing, and model weights

        Raises:
            ValueError: If model_id is invalid or model not found
            RuntimeError: If download or validation fails
        """
        from bioimageio.core import load_model_description

        print(
            f"📋 [{self.replica_id}] Getting RDF for model '{model_id}' (skip_cache={skip_cache})"
        )

        # Get model package with access tracking (only published models for RDF)
        local_package = await self._get_local_package_from_cache(
            model_id=model_id, published=True, skip_cache=skip_cache
        )

        # Use context manager to track access and prevent eviction during RDF loading
        async with local_package:
            model_source = str(local_package.model_source)
            print(
                f"📍 [{self.replica_id}] Model source for '{model_id}': {model_source}"
            )

            model_description = load_model_description(model_source)
            print(
                f"✅ [{self.replica_id}] Successfully loaded RDF for model '{model_id}'"
            )

            return json.loads(model_description.model_dump_json())

    @schema_method
    async def validate(
        self,
        rdf_dict: Dict[str, Union[str, int, float, List, Dict]],
        known_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Union[bool, str]]:
        """
        Validate a model Resource Description Framework (RDF) against bioimage.io specifications.

        Performs comprehensive validation including:
        - Schema compliance checking against bioimage.io RDF specification
        - Data type and format validation for all fields
        - Logical consistency verification between related fields
        - Tensor shape and dimension compatibility analysis
        - File reference and path validation (if known_files provided)

        Args:
            rdf_dict: Complete RDF dictionary structure to validate
            known_files: Optional mapping of relative file paths to their content hashes
                        for validating file references within the RDF

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
            f"🔬 [{self.replica_id}] Validating RDF (known_files: {len(known_files or {})} files)"
        )

        ctx = ValidationContext(perform_io_checks=False, known_files=known_files or {})
        summary = await asyncio.to_thread(validate_format, rdf_dict, context=ctx)

        result = {
            "success": summary.status == "valid-format",
            "details": summary.format(),
        }

        print(
            f"✅ [{self.replica_id}] RDF validation {'passed' if result['success'] else 'failed'}"
        )
        return result

    @schema_method
    async def test(
        self,
        model_id: str,
        published: bool = True,
        skip_cache: bool = False,
        additional_requirements: Optional[List[str]] = None,
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

        Args:
            model_id: Unique identifier of the bioimage.io model to test
            published: Whether to test the published version (True) or unpublished/review version (False)
            skip_cache: Force re-download of model package before testing
            additional_requirements: Extra Python packages to install in the test environment
                                   (e.g., ["scipy>=1.7.0", "scikit-image"])

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
            f"🧪 [{self.replica_id}] Testing model '{model_id}' (published={published}, skip_cache={skip_cache})"
        )

        # Get model package with access tracking
        local_package = await self._get_local_package_from_cache(
            model_id=model_id, published=published, skip_cache=skip_cache
        )

        try:
            # Use context manager to track access and prevent eviction during test
            async with local_package:
                model_source = str(local_package.model_source)
                print(
                    f"📍 [{self.replica_id}] Model source for '{model_id}': {model_source}"
                )

                test_result = await self.model_evaluation.test.remote(
                    model_source=model_source,
                    additional_requirements=additional_requirements,
                )
        except RayTaskError as e:
            error_msg = f"Failed to run model test for '{model_id}': {e}"
            print(f"❌ [{self.replica_id}] {error_msg}")
            raise RuntimeError(error_msg)

        print(f"✅ [{self.replica_id}] Model test completed for '{model_id}'")
        return test_result

    @schema_method(arbitrary_types_allowed=True)
    async def infer(
        self,
        model_id: str,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        weights_format: Optional[str] = None,
        device: Optional[Literal["cuda", "cpu"]] = None,
        default_blocksize_parameter: Optional[int] = None,
        sample_id: str = "sample",
        skip_cache: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Execute inference on a bioimage.io model with provided input data.

        Performs end-to-end inference including:
        - Automatic input preprocessing according to model specification
        - Model execution with optimized framework backend
        - Output postprocessing and format standardization
        - Memory-efficient processing for large inputs using tiling if supported

        Args:
            model_id: Unique identifier of the published bioimage.io model
            inputs: Input data as numpy array or dictionary of named arrays.
                   Must match the model's input specification for shape and data type.
                   For single input models, provide np.ndarray.
                   For multi-input models, provide dict with input names as keys.
            weights_format: Preferred model weights format ("pytorch_state_dict", "torchscript",
                          "onnx", "tensorflow_saved_model"). If None, automatically selects best available.
            device: Target computation device. "cuda" for GPU acceleration, "cpu" for CPU-only.
                   If None, automatically selects based on availability and model compatibility.
            default_blocksize_parameter: Override default tiling block size for memory management.
                                        Larger values use more memory but may be faster.
                                        Only applicable for models supporting tiled inference.
            sample_id: Identifier for this inference request, used for logging and debugging
            skip_cache: Force re-download of model package before inference

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
        print(f"🤖 [{self.replica_id}] Running inference for model '{model_id}'")

        # Get model package with access tracking (only published models for inference)
        local_package = await self._get_local_package_from_cache(
            model_id=model_id, published=True, skip_cache=skip_cache
        )

        try:
            # Use context manager to track access and prevent eviction during inference
            async with local_package:
                model_source = str(local_package.model_source)
                print(
                    f"📍 [{self.replica_id}] Model source for '{model_id}': {model_source}"
                )

                result = await self.model_inference.predict.remote(
                    model_source=model_source,
                    inputs=inputs,
                    weights_format=weights_format,
                    device=device,
                    default_blocksize_parameter=default_blocksize_parameter,
                    sample_id=sample_id,
                )
        except RayTaskError as e:
            error_msg = f"Failed to run inference for model '{model_id}': {e}"
            print(f"❌ [{self.replica_id}] {error_msg}")
            raise RuntimeError(error_msg)

        print(f"✅ [{self.replica_id}] Inference completed for model '{model_id}'")
        return result


if __name__ == "__main__":
    import asyncio

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
        / "bioimage-io-model-runner"
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
            model_inference=MockHandle(),
            model_evaluation=MockHandle(),
            max_models=2,
        )
        asyncio.run(model_runner.async_init())

        other_model_ids = ["polite-pig", "ambitious-ant"]
        for model_id in other_model_ids:
            asyncio.run(model_runner.get_model_rdf(model_id=model_id))

        asyncio.run(model_runner.test_deployment(test_skip_cache=True))
    finally:
        # Restore original function
        serve.get_replica_context = original_get_replica_context

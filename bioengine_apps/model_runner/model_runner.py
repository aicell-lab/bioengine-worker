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
    """Wrapper for cached bioimage.io model package with automatic cleanup."""
    
    def __init__(self, package_path: Path) -> None:
        self.package_path = package_path
        self.model_source = package_path / "rdf.yaml"

    def __del__(self) -> None:
        """Clean up package when evicted from cache."""
        remove_package(self.package_path)


# Test the deployment with a model that should pass all checks
TEST_BMZ_MODEL_ID = "charismatic-whale"
TEST_IMAGE_URL = "https://hypha.aicell.io/bioimage-io/artifacts/charismatic-whale/files/new_test_input.npy"


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 16 * 1024 * 1024 * 1024,  # 16GB RAM limit
        "runtime_env": {
            "pip": [
                "bioimageio.core==0.9.0",
                "xarray==2025.1.2",  # this is needed for bioimageio.core
                "numpy==1.26.4",
                "torch==2.5.1",
                "torchvision==0.20.1",
                "tensorflow==2.16.1",
                "onnxruntime==1.20.1",
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
    """

    def __init__(
        self,
        model_inference: DeploymentHandle,
        model_evaluation: DeploymentHandle,
    ) -> None:
        self.model_inference = model_inference
        self.model_evaluation = model_evaluation

        # Set Hypha server and workspace
        self.server_url = "https://hypha.aicell.io"
        self.workspace = "bioimage-io"

        # Set up model directory
        models_dir = Path().resolve() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = models_dir

        # Get replica identifier for logging
        try:
            self.replica_id = serve.get_replica_context().replica_tag
        except Exception:
            self.replica_id = "unknown"

        print(
            f"🚀 [{self.replica_id}] ModelRunner initialized with models directory: {self.models_dir}"
        )

    # === BioEngine App Methods - will be called when the deployment is started ===

    async def async_init(self) -> None:
        """Load existing cached models into Ray Serve multiplexed cache."""
        print(
            f"🔄 [{self.replica_id}] Initializing ModelRunner deployment, scanning {self.models_dir}"
        )

        existing_models = 0
        for package_dir in self.models_dir.iterdir():
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

            try:
                await self._get_local_package_from_cache(cache_key)
                existing_models += 1
                print(
                    f"📦 [{self.replica_id}] Loaded existing model into cache: {cache_key}"
                )
            except Exception as e:
                print(
                    f"⚠️ [{self.replica_id}] Failed to load existing model {cache_key}: {e}"
                )

        print(
            f"✅ [{self.replica_id}] Initialization complete. Loaded {existing_models} existing models into cache"
        )

    async def test_deployment(self) -> Dict[str, Union[bool, str, float, Dict]]:
        """
        Comprehensive test of all public endpoints using a known working model.
        
        Returns detailed test results with timing and success metrics for monitoring.
        """
        print(
            f"🧪 [{self.replica_id}] Starting comprehensive deployment test with model: {TEST_BMZ_MODEL_ID}"
        )
        test_results = {}
        start_time = time.time()

        try:
            # Test 1: Get model RDF for validation
            print(f"🔍 [{self.replica_id}] Test 1/6: Getting model RDF...")
            rdf_start = time.time()
            model_rdf = await self.get_model_rdf(model_id=TEST_BMZ_MODEL_ID)
            test_results["get_model_rdf"] = {
                "success": True,
                "duration_s": time.time() - rdf_start,
                "model_id": model_rdf.get("id", "unknown"),
            }
            print(
                f"✅ [{self.replica_id}] RDF retrieval successful ({test_results['get_model_rdf']['duration_s']:.2f}s)"
            )

            # Test 2: Validate the RDF
            print(f"🔬 [{self.replica_id}] Test 2/6: Validating RDF...")
            val_start = time.time()
            validation_result = await self.validate(rdf_dict=model_rdf)
            test_results["validate"] = {
                "success": validation_result["success"],
                "duration_s": time.time() - val_start,
                "details": (
                    validation_result["details"][:200] + "..."
                    if len(validation_result["details"]) > 200
                    else validation_result["details"]
                ),
            }
            print(
                f"✅ [{self.replica_id}] Validation {'passed' if validation_result['success'] else 'failed'} ({test_results['validate']['duration_s']:.2f}s)"
            )

            # Test 3: Test the model (published)
            print(f"🧩 [{self.replica_id}] Test 3/6: Testing published model...")
            test1_start = time.time()
            test_result1 = await self.test(model_id=TEST_BMZ_MODEL_ID, published=True)
            test_results["test_published"] = {
                "success": isinstance(test_result1, dict),
                "duration_s": time.time() - test1_start,
            }
            print(
                f"✅ [{self.replica_id}] Published model test completed ({test_results['test_published']['duration_s']:.2f}s)"
            )

            # Test 4: Test the model (unpublished)
            print(f"🧩 [{self.replica_id}] Test 4/6: Testing unpublished model...")
            test2_start = time.time()
            test_result2 = await self.test(model_id=TEST_BMZ_MODEL_ID, published=False)
            test_results["test_unpublished"] = {
                "success": isinstance(test_result2, dict),
                "duration_s": time.time() - test2_start,
            }
            print(
                f"✅ [{self.replica_id}] Unpublished model test completed ({test_results['test_unpublished']['duration_s']:.2f}s)"
            )

            # Test 5: Test with skip_cache=True
            print(f"🔄 [{self.replica_id}] Test 5/6: Testing with cache skip...")
            test3_start = time.time()
            test_result3 = await self.test(
                model_id=TEST_BMZ_MODEL_ID, published=False, skip_cache=True
            )
            test_results["test_skip_cache"] = {
                "success": isinstance(test_result3, dict),
                "duration_s": time.time() - test3_start,
            }
            print(
                f"✅ [{self.replica_id}] Skip cache test completed ({test_results['test_skip_cache']['duration_s']:.2f}s)"
            )

            # Test 6: Test inference
            print(f"🤖 [{self.replica_id}] Test 6/6: Testing inference...")
            inf_start = time.time()

            # Download the test image
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                response = await client.get(TEST_IMAGE_URL)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to download test image from {TEST_IMAGE_URL}: "
                        f"HTTP {response.status_code} - {response.text}"
                    )
                image = np.load(response.content)

            image = image.astype("float32")
            # Reshape to match expected format: (batch=1, y, x, channels=1)
            input_image = image[np.newaxis, :, :, np.newaxis]

            outputs = await self.infer(model_id=TEST_BMZ_MODEL_ID, inputs=input_image)
            test_results["inference"] = {
                "success": isinstance(outputs, dict) and len(outputs) > 0,
                "duration_s": time.time() - inf_start,
                "output_keys": (
                    list(outputs.keys()) if isinstance(outputs, dict) else []
                ),
            }
            print(
                f"✅ [{self.replica_id}] Inference test completed ({test_results['inference']['duration_s']:.2f}s)"
            )

            # Overall results
            total_duration = time.time() - start_time
            all_tests_passed = all(
                result["success"] for result in test_results.values()
            )

            test_results["summary"] = {
                "all_tests_passed": all_tests_passed,
                "total_duration_s": total_duration,
                "replica_id": self.replica_id,
            }

            print(
                f"🎉 [{self.replica_id}] All deployment tests {'PASSED' if all_tests_passed else 'FAILED'} (total: {total_duration:.2f}s)"
            )
            return test_results

        except Exception as e:
            error_msg = f"Deployment test failed: {str(e)}"
            print(f"❌ [{self.replica_id}] {error_msg}")
            test_results["error"] = {
                "success": False,
                "error": error_msg,
                "total_duration_s": time.time() - start_time,
            }
            return test_results

    # === Internal Methods ===

    def _get_cache_key(self, model_id: str, published: bool) -> str:
        """Generate cache key from model ID and publication status."""
        if published:
            return f"bmz_model_{model_id}"
        else:
            return f"unpublished_model_{model_id}"

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
        """Download unpublished model as ZIP from Hypha server."""
        download_url = (
            f"{self.server_url}/{self.workspace}/artifacts/{model_id}/create-zip-file"
        )
        zip_file = package_path / "tmp_model_zip_file.zip"

        print(f"📥 [{self.replica_id}] Downloading model '{model_id}' from URL...")
        download_timeout = httpx.Timeout(120.0)

        async with httpx.AsyncClient(timeout=download_timeout) as client:
            response = await client.get(download_url)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to download model from {download_url}: "
                    f"HTTP {response.status_code} - {response.text}"
                )
            zip_content = response.content

        await asyncio.to_thread(zip_file.write_bytes, zip_content)
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

    async def _wait_for_download_completion(
        self, package_dir: Path, max_wait_time: int = 300
    ) -> bool:
        """Wait for another replica to finish downloading. Returns True if successful."""
        start_time = time.time()
        downloading_marker = package_dir.parent / f".downloading_{package_dir.name}"

        print(
            f"⏳ [{self.replica_id}] Waiting for download completion: {package_dir.name}"
        )

        while time.time() - start_time < max_wait_time:
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

            await asyncio.sleep(2)  # Check every 2 seconds

        # Timeout reached
        print(
            f"⏰ [{self.replica_id}] Timeout waiting for download completion: {package_dir.name}"
        )
        return False

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

    async def _validate_model(self, package_path: Path) -> Path:
        """Validate model RDF and return actual package path."""
        from bioimageio.core import load_model_description
        from bioimageio.spec import InvalidDescr

        try:
            # Find the RDF file in the package
            rdf_files = list(package_path.rglob("rdf.yaml"))
            if not rdf_files:
                error_msg = f"No rdf.yaml found in {package_path}"
                print(f"❌ [{self.replica_id}] {error_msg}")
                raise FileNotFoundError(error_msg)

            model_source = rdf_files[0]
            actual_package_path = model_source.parent
            print(f"🔍 [{self.replica_id}] Found model RDF at: {model_source}")

            # Validate model source
            model_description = load_model_description(model_source)
            model_id = model_description.get("id", str(actual_package_path))

            if isinstance(model_description, InvalidDescr):
                error_msg = f"Model '{model_id}' is invalid: {model_description}"
                print(f"❌ [{self.replica_id}] {error_msg}")
                raise ValueError(error_msg)

            print(f"✅ [{self.replica_id}] Model '{model_id}' validation successful")
            return actual_package_path

        except Exception as e:
            print(f"❌ [{self.replica_id}] Model validation failed: {e}")
            # Clean up invalid package
            await asyncio.to_thread(remove_package, package_path)
            raise

    @serve.multiplexed(max_num_models_per_replica=os.environ.get("CACHE_N_MODELS", 30))
    async def _get_local_package_from_cache(
        self, cache_key: str
    ) -> LocalBioimageioPackage:
        """Get model package from cache, downloading and validating if needed."""
        # Download the model if it doesn't exist locally
        package_path = await self._download_model(cache_key)

        # Validate the model package
        package_path = await self._validate_model(package_path)

        # Return as LocalBioimageioPackage (deletes on eviction)
        return LocalBioimageioPackage(package_path=package_path)

    async def _get_model_source(
        self, model_id: str, published: bool, skip_cache: bool
    ) -> str:
        """Get model RDF path, downloading if necessary or skipping cache."""
        cache_key = self._get_cache_key(model_id=model_id, published=published)
        package_dir = self.models_dir / cache_key

        # Handle cache skipping
        if skip_cache and package_dir.exists():
            print(
                f"🔄 [{self.replica_id}] Skipping cache for model '{model_id}', removing existing package"
            )
            await asyncio.to_thread(remove_package, package_dir)

        # Get the package (download if not cached)
        local_package = await self._get_local_package_from_cache(cache_key)

        model_source = str(local_package.model_source)
        print(f"📍 [{self.replica_id}] Model source for '{model_id}': {model_source}")
        return model_source

    # === Exposed BioEngine App Methods - all methods decorated with @schema_method will be exposed as API endpoints ===
    # Note: Parameter type hints and docstrings will be used to generate the API documentation.

    @schema_method
    async def get_model_rdf(self, model_id: str, skip_cache: bool = False) -> Dict[str, Union[str, int, float, List, Dict]]:
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

        # Only allow published BMZ model
        model_source = await self._get_model_source(
            model_id=model_id, published=True, skip_cache=skip_cache
        )

        model_description = load_model_description(model_source)
        print(f"✅ [{self.replica_id}] Successfully loaded RDF for model '{model_id}'")

        return json.loads(model_description.model_dump_json())

    @schema_method
    async def validate(
        self, rdf_dict: Dict[str, Union[str, int, float, List, Dict]], known_files: Optional[Dict[str, str]] = None
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

        # Allow published BMZ models and unpublished models in review
        model_source = await self._get_model_source(
            model_id=model_id, published=published, skip_cache=skip_cache
        )

        test_result = await self.model_evaluation.test.remote(
            model_source=model_source,
            additional_requirements=additional_requirements,
        )

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
        # Only allow BMZ model IDs, not URLs
        if model_id.startswith("http"):
            raise ValueError(
                "Model ID should not be a URL. Use the model ID from the BioImage Model Zoo."
            )

        print(
            f"🤖 [{self.replica_id}] Running inference for model '{model_id}' (sample_id='{sample_id}')"
        )

        # Only allow published BMZ model
        model_source = await self._get_model_source(
            model_id=model_id, published=True, skip_cache=skip_cache
        )

        result = await self.model_inference.predict.remote(
            model_source=model_source,
            inputs=inputs,
            weights_format=weights_format,
            device=device,
            default_blocksize_parameter=default_blocksize_parameter,
            sample_id=sample_id,
        )

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
        / "bioimage_io_model_runner"
    )
    deployment_workdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(deployment_workdir)
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
        )

        asyncio.run(model_runner.test_deployment())
    finally:
        # Restore original function
        serve.get_replica_context = original_get_replica_context

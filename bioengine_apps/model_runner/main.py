import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


class ModelRunner:
    """
    Local version of ModelRunner for testing bioimage.io models.
    This version includes pipeline caching but without Ray dependencies.
    """

    def __init__(
        self,
        cache_n_models: int = 10,
        pipeline_idle_timeout: float = 300.0,
        max_pipeline_cache_size: int = 3,
    ):
        # Set up working directory
        workdir = Path(os.environ["BIOENGINE_WORKDIR"])

        # Set up cache directory for bioimageio
        bioimageio_cache_path = workdir / ".bioimageio_cache"
        bioimageio_cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["BIOIMAGEIO_CACHE_PATH"] = str(bioimageio_cache_path)

        # Change to the cache directory (for keras models which create files in the current directory)
        os.chdir(bioimageio_cache_path)

        # Set up model directory
        model_dir = workdir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = model_dir

        self.cache_n_models = cache_n_models
        self.cached_models = []

        # Enhanced pipeline cache management
        self.pipeline_idle_timeout = (
            pipeline_idle_timeout  # Time before idle models are unloaded
        )
        self.max_pipeline_cache_size = (
            max_pipeline_cache_size  # Max number of models in memory
        )
        self._pipeline_cache = (
            {}
        )  # cache_key -> {"pipeline": pipeline, "ref_count": int, "last_used": float, "last_accessed": float}
        self._cache_lock = asyncio.Lock()

        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()

        print(f"✅ Initialized ModelRunner with working directory: {workdir}")
        print(
            f"📊 Pipeline cache: max_size={max_pipeline_cache_size}, idle_timeout={pipeline_idle_timeout}s"
        )

    def _get_cache_key_for_url(self, url: str) -> str:
        """Generate a consistent cache key for URL-based models."""
        import hashlib

        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"url_model_{url_hash}"

    def _get_pipeline_cache_key(
        self,
        model_id: str,
        weight_format: Optional[str] = None,
        devices: Optional[list[str]] = None,
    ) -> str:
        """Generate a cache key for prediction pipelines."""
        # Handle URL-based models
        if model_id.startswith("http"):
            base_key = self._get_cache_key_for_url(model_id)
        else:
            base_key = model_id

        # Include weight format and devices in cache key
        key_parts = [base_key]
        if weight_format:
            key_parts.append(f"wf_{weight_format}")
        if devices:
            key_parts.append(f"dev_{'-'.join(sorted(devices))}")

        return "_".join(key_parts)

    def _start_cleanup_task(self):
        """Start the background cleanup task for idle pipelines."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())

    async def _background_cleanup(self):
        """Background task that periodically cleans up idle pipelines."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_pipelines()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️  Background cleanup error: {e}")

    async def _cleanup_idle_pipelines(self):
        """Remove pipelines that haven't been used recently and have no active references."""
        current_time = time.time()
        to_remove = []

        async with self._cache_lock:
            for cache_key, cache_entry in self._pipeline_cache.items():
                idle_time = current_time - cache_entry["last_accessed"]
                if (
                    cache_entry["ref_count"] <= 0
                    and idle_time > self.pipeline_idle_timeout
                ):
                    to_remove.append(cache_key)

            for cache_key in to_remove:
                await self._unload_pipeline(cache_key)

    async def _unload_pipeline(self, cache_key: str):
        """Safely unload a pipeline and remove it from cache."""
        if cache_key not in self._pipeline_cache:
            return

        cache_entry = self._pipeline_cache[cache_key]
        pipeline = cache_entry["pipeline"]
        model_id = cache_entry["model_id"]

        try:
            # Properly unload the pipeline to free GPU memory
            pipeline.unload()
            print(f"🗑️  Unloaded idle pipeline: {model_id}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to unload pipeline {model_id}: {e}")

        del self._pipeline_cache[cache_key]

    async def _evict_lru_pipeline(self):
        """Evict the least recently used pipeline to make room for a new one."""
        if not self._pipeline_cache:
            return

        # Find the LRU pipeline (not currently in use)
        lru_key = None
        lru_time = float("inf")

        for cache_key, cache_entry in self._pipeline_cache.items():
            if cache_entry["ref_count"] <= 0:  # Only evict unused pipelines
                if cache_entry["last_accessed"] < lru_time:
                    lru_time = cache_entry["last_accessed"]
                    lru_key = cache_key

        if lru_key:
            print(f"📤 Evicting LRU pipeline to make room")
            await self._unload_pipeline(lru_key)

    async def _download_model_from_url(self, model_url: str, package_path: str) -> str:
        import os
        import zipfile
        from pathlib import Path

        import aiohttp

        package_path = Path(package_path)
        os.makedirs(package_path, exist_ok=True)
        archive_path = str(package_path) + ".zip"

        print(f"📥 Downloading model from {model_url} to {archive_path}")
        async with aiohttp.ClientSession() as session:
            async with session.get(model_url) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to download model from {model_url}: {response.status}"
                    )
                content = await response.read()
                with open(archive_path, "wb") as f:
                    f.write(content)

        print(f"📦 Downloaded zip file size: {os.path.getsize(archive_path)} bytes")

        # Unzip package_path
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            files = zip_ref.namelist()
            print(f"📋 Files in zip: {len(files)} files")
            zip_ref.extractall(package_path)

        # Clean up the zip file
        os.remove(archive_path)

        # Find the correct path with rdf.yaml
        rdf_files = list(package_path.rglob("rdf.yaml"))
        if rdf_files:
            final_path = rdf_files[0].parent
            print(f"✅ Found model at: {final_path}")
        else:
            final_path = package_path
            print(f"⚠️  Using package path (no rdf.yaml found): {final_path}")

        return str(final_path)

    async def _get_url_model_path(
        self, model_url: str, skip_cache: bool = False
    ) -> Path:
        """Get the path to a URL-based model, downloading if necessary."""
        cache_key = self._get_cache_key_for_url(model_url)
        package_path = self.model_dir / cache_key

        # Handle cache skipping
        if skip_cache and package_path.exists():
            print(f"🗑️  Removing cached URL model at {package_path}")
            try:
                shutil.rmtree(str(package_path))
            except Exception as e:
                print(f"⚠️  Warning: Could not remove cache directory: {e}")

        # Download if not exists
        if not package_path.exists():
            package_path = await self._download_model_from_url(
                model_url, str(package_path)
            )
            package_path = Path(package_path)
        else:
            print(f"♻️  Using cached URL model from {package_path}")

        return package_path

    async def _download_model(self, model_id: str) -> str:
        from bioimageio.spec import save_bioimageio_package_as_folder

        print(f"📥 Downloading model {model_id}...")
        model_path = self.model_dir / model_id
        os.makedirs(model_path, exist_ok=True)
        model_path = Path(
            save_bioimageio_package_as_folder(model_id, output_path=str(model_path))
        )
        print(f"✅ Downloaded model to {model_path}")
        return model_path

    async def _get_model(self, model_id: str):
        """Get model description (lightweight operation)."""
        from bioimageio.core import load_model_description
        from bioimageio.spec import InvalidDescr

        # Handle URL-based models
        if model_id.startswith("http"):
            package_path = await self._get_url_model_path(model_id)
            # Find rdf.yaml file
            rdf_path = package_path / "rdf.yaml"
            if not rdf_path.exists():
                rdf_files = list(package_path.rglob("rdf.yaml"))
                if rdf_files:
                    rdf_path = rdf_files[0]
                else:
                    raise FileNotFoundError(f"No rdf.yaml found in {package_path}")
            model_source = rdf_path
        else:
            # Handle regular model IDs
            if model_id in self.cached_models:
                self.cached_models.remove(model_id)
            else:
                # Download model if not in cache
                await self._download_model(model_id)

            # Add model to cache
            self.cached_models.append(model_id)

            # Check cache size
            if len(self.cached_models) > self.cache_n_models:
                remove_model_id = self.cached_models.pop(0)
                remove_model_path = self.model_dir / remove_model_id
                if remove_model_path.exists():
                    try:
                        shutil.rmtree(str(remove_model_path))
                        print(f"🗑️  Removed old cached model: {remove_model_id}")
                    except:
                        pass

            model_source = str(self.model_dir / model_id / "rdf.yaml")

        print(f"📖 Loading model description from: {model_source}")
        model = load_model_description(model_source)
        assert not isinstance(model, InvalidDescr), f"Model {model_id} is invalid"
        return model

    async def _get_prediction_pipeline(
        self,
        model_id: str,
        weight_format: Optional[str] = None,
        devices: Optional[list] = None,
        **pipeline_kwargs,
    ):
        """Get a cached prediction pipeline or create a new one with smart caching."""
        from bioimageio.core import create_prediction_pipeline

        cache_key = self._get_pipeline_cache_key(model_id, weight_format, devices)
        current_time = time.time()

        async with self._cache_lock:
            # Check if pipeline exists in cache
            if cache_key in self._pipeline_cache:
                cache_entry = self._pipeline_cache[cache_key]
                cache_entry["ref_count"] += 1
                cache_entry["last_used"] = current_time
                cache_entry["last_accessed"] = current_time
                print(f"⚡ Using cached prediction pipeline for {model_id}")
                return cache_entry["pipeline"]

            # Check if we need to make room for a new pipeline
            if len(self._pipeline_cache) >= self.max_pipeline_cache_size:
                print(
                    f"📊 Cache full ({len(self._pipeline_cache)}/{self.max_pipeline_cache_size}), evicting LRU pipeline"
                )
                await self._evict_lru_pipeline()

            # Create new pipeline
            print(f"🔄 Creating new prediction pipeline for {model_id}")

            try:
                # Get model description (this goes through Ray multiplex)
                model_description = await self._get_model(model_id)

                # Create pipeline with specified options
                create_kwargs = pipeline_kwargs.copy()
                if weight_format:
                    create_kwargs["weight_format"] = weight_format
                if devices:
                    create_kwargs["devices"] = devices

                pipeline = create_prediction_pipeline(
                    model_description, **create_kwargs
                )

                # Important: Load the pipeline (this loads the model weights into memory/GPU)
                pipeline.load()

                # Cache the pipeline
                self._pipeline_cache[cache_key] = {
                    "pipeline": pipeline,
                    "ref_count": 1,
                    "last_used": current_time,
                    "last_accessed": current_time,
                    "model_id": model_id,
                    "weight_format": weight_format,
                    "devices": devices,
                }

                print(
                    f"✅ Cached new prediction pipeline for {model_id} ({len(self._pipeline_cache)}/{self.max_pipeline_cache_size})"
                )
                return pipeline

            except Exception as e:
                print(f"❌ Failed to create prediction pipeline for {model_id}: {e}")
                raise

    async def _release_pipeline_reference(
        self,
        model_id: str,
        weight_format: Optional[str] = None,
        devices: Optional[list] = None,
    ):
        """Release a reference to a cached pipeline - but keep it loaded for future use."""
        cache_key = self._get_pipeline_cache_key(model_id, weight_format, devices)

        async with self._cache_lock:
            if cache_key in self._pipeline_cache:
                self._pipeline_cache[cache_key]["ref_count"] -= 1
                # Note: We DON'T unload immediately - let background cleanup handle it

    async def get_model_rdf(self, model_id: str) -> dict:
        """
        Get the model RDF description including inputs, preprocessing, postprocessing, and outputs.
        Args:
            model_id (str): The model ID.
        Returns:
            Dict: The model RDF description.
        """
        model = await self._get_model(model_id)
        return json.loads(model.model_dump_json())

    async def infer(
        self,
        model_id: str,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        weight_format: Optional[str] = None,
        devices: Optional[list] = None,
        blockwise: bool = False,
        sample_id: str = "sample",
        **pipeline_kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run inference on the model using cached prediction pipeline."""
        from bioimageio.core.digest_spec import create_sample_for_model

        try:
            # Get cached prediction pipeline
            pipeline = await self._get_prediction_pipeline(
                model_id,
                weight_format=weight_format,
                devices=devices,
                **pipeline_kwargs,
            )

            # Create sample from inputs
            # Handle single array input by creating a proper dictionary
            if isinstance(inputs, np.ndarray):
                # Get the first input tensor ID
                input_id = pipeline.model_description.inputs[0].id
                input_dict = {str(input_id): inputs}
            else:
                input_dict = inputs

            sample = create_sample_for_model(
                pipeline.model_description, inputs=input_dict, sample_id=sample_id
            )

            # Run prediction using the pipeline
            if blockwise:
                result = pipeline.predict_sample_with_blocking(sample)
            else:
                result = pipeline.predict_sample_without_blocking(sample)

            # Convert outputs back to numpy arrays
            outputs = {str(k): v.data.data for k, v in result.members.items()}
            return outputs

        finally:
            # Release the pipeline reference
            await self._release_pipeline_reference(
                model_id, weight_format=weight_format, devices=devices
            )

    async def get_pipeline_cache_stats(self) -> Dict[str, Any]:
        """Get current pipeline cache statistics."""
        return {
            "cached_pipelines": len(self._pipeline_cache),
            "pipelines": {
                cache_key: {
                    "model_id": entry["model_id"],
                    "ref_count": entry["ref_count"],
                    "last_used": entry["last_used"],
                    "idle_time": time.time() - entry["last_used"],
                }
                for cache_key, entry in self._pipeline_cache.items()
            },
        }

    async def validate(self, rdf_dict: dict, known_files: dict = None) -> dict:
        """
        Validate a model RDF description.
        Args:
            rdf_dict (dict): The RDF description to validate.
            known_files (dict, optional): Known files for validation context.
        Returns:
            dict: Validation result with success status and details.
        """
        from bioimageio.spec import ValidationContext, validate_format

        ctx = ValidationContext(perform_io_checks=False, known_files=known_files or {})
        summary = validate_format(rdf_dict, context=ctx)
        return {
            "success": summary.status == "valid-format",
            "details": summary.format(),
        }

    async def test(self, model_id: str, skip_cache: bool = False) -> dict:
        """
        Test a model using bioimageio.core test functionality.
        Args:
            model_id (str): The model ID or URL to test.
            skip_cache (bool): If True, bypass cache and re-download the model.
        Returns:
            dict: Test result from bioimageio.core.test_model.
        """
        from bioimageio.core import test_model

        print(f"Testing model {model_id}... (skip_cache={skip_cache})")

        # Check if model_id is a URL
        if model_id.startswith("http"):
            # Get the model path (this method can be decorated with Ray multiplex)
            package_path = await self._get_url_model_path(model_id, skip_cache)

            # Update local cache tracking for cleanup purposes
            cache_key = self._get_cache_key_for_url(model_id)
            if cache_key not in self.cached_models:
                self.cached_models.append(cache_key)

                # Check cache size and cleanup if needed
                if len(self.cached_models) > self.cache_n_models:
                    remove_cache_key = self.cached_models.pop(0)
                    remove_model_path = self.model_dir / remove_cache_key
                    if remove_model_path.exists():
                        try:
                            shutil.rmtree(str(remove_model_path))
                        except Exception as e:
                            print(f"Warning: Could not remove old cache directory: {e}")

            # Find rdf.yaml file
            rdf_path = package_path / "rdf.yaml"

            if not rdf_path.exists():
                print(f"Looking for rdf.yaml in {package_path}")
                # Try to find rdf.yaml recursively
                rdf_files = list(package_path.rglob("rdf.yaml"))
                if rdf_files:
                    rdf_path = rdf_files[0]
                    print(f"Found rdf.yaml at: {rdf_path}")
                else:
                    raise FileNotFoundError(f"No rdf.yaml found in {package_path}")

            print(f"Testing model with RDF at: {rdf_path}")
            source = rdf_path
        else:
            # Handle regular model IDs - let Ray's multiplex caching handle this
            if skip_cache:
                raise ValueError(
                    "skip_cache=True is not supported for model IDs, only for URLs."
                )

            # Always use _get_model to leverage Ray's multiplex caching
            print(f"Getting model: {model_id} (Ray multiplex will handle caching)")
            model = await self._get_model(model_id)

            source = model

        # Test the model
        result = test_model(source).model_dump(mode="json")

        return result

    def cleanup(self):
        """Cleanup all cached pipelines and stop background tasks."""
        print(f"🧹 Cleaning up {len(self._pipeline_cache)} cached pipelines...")

        # Cancel background cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            print("🛑 Cancelled background cleanup task")

        # Unload all pipelines
        for cache_key, cache_entry in self._pipeline_cache.items():
            try:
                pipeline = cache_entry["pipeline"]
                pipeline.unload()
                print(f"🗑️  Unloaded pipeline: {cache_entry['model_id']}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to unload pipeline {cache_key}: {e}")

        self._pipeline_cache.clear()
        print("✅ Cleanup completed")

    async def get_cache_status(self) -> Dict[str, Any]:
        """Get detailed status of the pipeline cache."""
        current_time = time.time()

        async with self._cache_lock:
            cache_info = {}
            for cache_key, entry in self._pipeline_cache.items():
                cache_info[cache_key] = {
                    "model_id": entry["model_id"],
                    "ref_count": entry["ref_count"],
                    "last_used": entry["last_used"],
                    "last_accessed": entry["last_accessed"],
                    "idle_time": current_time - entry["last_accessed"],
                    "weight_format": entry.get("weight_format"),
                    "devices": entry.get("devices"),
                }

            return {
                "cached_pipelines": len(self._pipeline_cache),
                "max_cache_size": self.max_pipeline_cache_size,
                "idle_timeout": self.pipeline_idle_timeout,
                "cache_utilization": f"{len(self._pipeline_cache)}/{self.max_pipeline_cache_size}",
                "pipelines": cache_info,
            }


if __name__ == "__main__":
    import asyncio

    from kaibu_utils import fetch_image

    async def test_model():
        # Set up the environment variables like in the real deployment
        deployment_workdir = str(
            Path(__file__).resolve().parent.parent.parent
            / ".bioengine"
            / "apps"
            / "bioimage_io_model_runner"
        )
        os.environ["BIOENGINE_WORKDIR"] = deployment_workdir
        os.environ["TMPDIR"] = deployment_workdir
        os.environ["HOME"] = deployment_workdir

        model_runner = ModelRunner()

        # Test the model validation and testing functions
        print("Testing model validation and testing...")

        # Test the model with an URL
        model_id = "https://hypha.aicell.io/bioimage-io/artifacts/affable-shark/create-zip-file"
        print(f"Testing model {model_id}...")
        test_result = await model_runner.test(model_id)
        print(f"Test result: {test_result}")

        # Test the model with an ID
        model_id = "creative-panda"  # choose different bioimage.io model

        print(f"Testing model {model_id}...")
        test_result = await model_runner.test(model_id)
        print(f"Test result: {test_result}")

        # Get model RDF for validation
        model_rdf = await model_runner.get_model_rdf(model_id)

        # Validate the RDF
        print("Validating model RDF...")
        validation_result = await model_runner.validate(model_rdf)
        print(f"Validation result: {validation_result}")

        # Test inference
        image = await fetch_image(
            "https://zenodo.org/api/records/5906839/files/sample_input_0.tif/content"
        )
        image = image.astype("float32")
        print("example image downloaded: ", image.shape)

        input_image_shape = tuple(model_rdf["inputs"][0]["shape"][1:])
        print("Input image shape", input_image_shape)
        input_image = image[..., None]
        assert (
            input_image.shape == input_image_shape
        ), f"Wrong shape ({input_image.shape}), expected: {input_image_shape}"
        print("Valid image shape, ready to go!")

        outputs = await model_runner.infer(model_id, input_image)

        print("Outputs: ", outputs)

    asyncio.run(test_model())

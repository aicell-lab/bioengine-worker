import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import ray
from ray import serve

logger = logging.getLogger("ray.serve")
logger.setLevel("INFO")

# Deployment default runtime environment
requirements = [
    "bioimageio.core==0.9.5",
    "careamics==0.0.16",
    "cellpose==3.1.1.2",
    "nvidia-ml-py==12.555.43",
    "numpy==1.26.4",
    "onnxruntime==1.20.1",
    "psutil==6.1.1",
    "tensorflow==2.16.1",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "xarray==2025.1.2",  # this is needed for bioimageio.core
]


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0.5,
        "memory": 12 * 1024 * 1024 * 1024,  # GB RAM limit
        "runtime_env": {
            "pip": requirements,
        },
    },
    max_ongoing_requests=1,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 3,
        "target_num_ongoing_requests_per_replica": 0.8,
        "metrics_interval_s": 2.0,
        "look_back_period_s": 10.0,
        "downscale_delay_s": 300,
        "upscale_delay_s": 0.0,
    },
    health_check_period_s=30.0,
    health_check_timeout_s=30.0,
    graceful_shutdown_timeout_s=120.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class RuntimeDeployment:
    """Internal deployment for running model inference using bioimageio.core."""

    def __init__(self):
        self._kwargs_cache = {}

    def _get_memory_usage(self) -> tuple:
        """Return current (cpu_bytes, gpu_bytes) memory usage for this process."""
        import psutil

        cpu_mem = psutil.Process().memory_info().rss
        gpu_mem = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem += info.used
            finally:
                pynvml.nvmlShutdown()
        except (ImportError, Exception):
            pass
        return cpu_mem, gpu_mem

    # === Model Testing ===

    def _test(self, rdf_path: str) -> dict:
        """Run bioimageio.core test_model on the given RDF path."""
        from bioimageio.core import test_model

        try:
            if not Path(rdf_path).exists():
                raise FileNotFoundError(f"RDF not found: {rdf_path}")
            validation_summary = test_model(rdf_path)
            test_result = validation_summary.model_dump(mode="json")

            return test_result
        except Exception as e:
            logger.error(f"❌ Model test failed: {str(e)}")
            raise

    async def test(
        self, rdf_path: str, additional_requirements: Optional[List[str]] = None
    ) -> dict:
        """Test model inference using bioimageio.core with optional additional requirements."""
        cpu_before, gpu_before = self._get_memory_usage()
        logger.info(
            f"📊 [test] Memory before: CPU: {cpu_before / (1024 * 1024):.2f} MB, GPU: {gpu_before / (1024 * 1024):.2f} MB"
        )
        additional_packages = []
        if additional_requirements:
            if not isinstance(additional_requirements, list):
                logger.error("❌ additional_requirements must be a list of strings")
                raise ValueError("additional_requirements must be a list of strings.")

            for ad_req in additional_requirements:
                ad_req = ad_req.strip()
                exists = False
                for req in requirements:
                    package, _ = req.split("==")
                    if ad_req.startswith(package):
                        exists = True
                        break
                if not exists:
                    additional_packages.append(ad_req)

        if additional_packages:
            logger.info(
                f"🚀 Running test with additional packages: {additional_packages}"
            )
            # Execute remotely and return the result reference (will be awaited by DeploymentHandle)
            remote_function = ray.remote(self._test.__func__)
            remote_function = remote_function.options(
                num_cpus=1,
                num_gpus=0,
                memory=4 * 1024 * 1024 * 1024,  # 4GB RAM limit
                runtime_env={"pip": requirements + additional_packages},
            )
            result_ref = remote_function.remote(None, rdf_path)
            logger.info(f"📋 Submitted remote test job")
            return result_ref
        else:
            # Run execution in this deployment without additional packages
            test_result = self._test(rdf_path)

        cpu_after, gpu_after = self._get_memory_usage()
        cpu_delta_mb = (cpu_after - cpu_before) / (1024 * 1024)
        gpu_delta_mb = (gpu_after - gpu_before) / (1024 * 1024)
        logger.info(
            f"📊 [test] Memory after:  CPU: {cpu_after / (1024 * 1024):.2f} MB, GPU: {gpu_after / (1024 * 1024):.2f} MB"
        )
        logger.info(
            f"📊 [test] Memory change:  CPU: {cpu_delta_mb:+.2f} MB, GPU: {gpu_delta_mb:+.2f} MB"
        )

        return test_result

    # === Model Prediction ===

    def _set_prediction_kwargs(
        self,
        rdf_path: str,
        weights_format: str,
        device: str,
        default_blocksize_parameter: int,
        latest_download: Optional[float] = None,
    ) -> str:
        """Generate cache key for prediction pipeline configuration."""
        pipeline_kwargs = {
            "rdf_path": rdf_path,
            "latest_download": latest_download,  # Include latest file download time to invalidate cache on re-download
            "create_kwargs": {
                "weights_format": weights_format,
                "device": device,
                "default_blocksize_parameter": default_blocksize_parameter,
            },
        }
        # Generate a unique cache key based on the pipeline configuration
        json_str = json.dumps(pipeline_kwargs, sort_keys=True)
        cache_key = hashlib.md5(json_str.encode()).hexdigest()

        self._kwargs_cache[cache_key] = pipeline_kwargs

        return cache_key

    @serve.multiplexed(
        max_num_models_per_replica=int(os.environ.get("PIPELINE_CACHE_SIZE", 10))
    )
    async def _create_prediction_pipeline(self, cache_key: str):
        """Create and cache prediction pipeline for the given cache key."""
        cpu_before, gpu_before = self._get_memory_usage()

        from bioimageio.core import create_prediction_pipeline, load_model_description

        # Get model source and create_kwargs using the cache key
        pipeline_kwargs = self._kwargs_cache.get(cache_key)
        if not pipeline_kwargs:
            logger.error(f"❌ No pipeline kwargs found for cache key: {cache_key}")
            raise ValueError(f"No pipeline kwargs found for cache key: {cache_key}")

        rdf_path = pipeline_kwargs["rdf_path"]
        create_kwargs = pipeline_kwargs["create_kwargs"]

        try:
            model_description = load_model_description(rdf_path)
            pipeline = create_prediction_pipeline(model_description, **create_kwargs)

            # Load the pipeline (this loads the model weights into memory/GPU)
            pipeline.load()

            cpu_after, gpu_after = self._get_memory_usage()
            cpu_delta_mb = (cpu_after - cpu_before) / (1024 * 1024)
            gpu_delta_mb = (gpu_after - gpu_before) / (1024 * 1024)

            logger.info(
                f"✅ Created and loaded prediction pipeline for model at {rdf_path}"
            )
            logger.info(
                f"📊 [pipeline load] Memory after:  CPU: {cpu_after / (1024 * 1024):.2f} MB, GPU: {gpu_after / (1024 * 1024):.2f} MB"
            )
            logger.info(
                f"📊 [pipeline load] Memory change:  CPU: {cpu_delta_mb:+.2f} MB, GPU: {gpu_delta_mb:+.2f} MB"
            )
            
            return pipeline
        except Exception as e:
            logger.error(f"❌ Failed to create prediction pipeline: {str(e)}")
            raise
        finally:
            # Clean up the cache entry after use
            self._kwargs_cache.pop(cache_key, None)

    async def predict(
        self,
        rdf_path: str,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        weights_format: Optional[str] = None,
        device: Literal["cuda", "cpu"] = None,
        default_blocksize_parameter: Optional[int] = None,
        sample_id: str = "sample",
        latest_download: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Run inference on model using bioimageio.core prediction pipeline."""
        cpu_before, gpu_before = self._get_memory_usage()
        logger.info(
            f"📊 [predict] Memory before: CPU: {cpu_before / (1024 * 1024):.2f} MB, GPU: {gpu_before / (1024 * 1024):.2f} MB"
        )

        from bioimageio.core.digest_spec import create_sample_for_model

        try:
            if not Path(rdf_path).exists():
                raise FileNotFoundError(f"RDF not found: {rdf_path}")

            logger.info(f"🚀 Starting prediction for model at {rdf_path} with device={device} and weights_format={weights_format}")
            cache_key = self._set_prediction_kwargs(
                rdf_path=rdf_path,
                weights_format=weights_format,
                device=device,
                default_blocksize_parameter=default_blocksize_parameter,
                latest_download=latest_download,
            )
            pipeline = await self._create_prediction_pipeline(cache_key)

            # Create sample from inputs
            # Handle single array input by creating a proper dictionary
            sample = create_sample_for_model(
                pipeline.model_description,
                inputs=inputs,
                sample_id=sample_id,
            )

            # Run prediction using the pipeline
            if default_blocksize_parameter:
                result = pipeline.predict_sample_with_blocking(sample)
            else:
                result = pipeline.predict_sample_without_blocking(sample)
            cpu_after, gpu_after = self._get_memory_usage()
            cpu_delta_mb = (cpu_after - cpu_before) / (1024 * 1024)
            gpu_delta_mb = (gpu_after - gpu_before) / (1024 * 1024)
            logger.info(
                f"📊 [predict] Memory after:  CPU: {cpu_after / (1024 * 1024):.2f} MB, GPU: {gpu_after / (1024 * 1024):.2f} MB"
            )
            logger.info(
                f"📊 [predict] Memory change: CPU: {cpu_delta_mb:+.2f} MB, GPU: {gpu_delta_mb:+.2f} MB"
            )

            # Convert outputs back to numpy arrays
            outputs = {str(k): v.data.data for k, v in result.members.items()}

            return outputs

        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            raise e


if __name__ == "__main__":
    import asyncio

    import yaml

    # Set up the environment variables like in the real deployment
    deployment_workdir = Path.home() / ".bioengine" / "apps" / "model-runner"
    deployment_workdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(deployment_workdir)
    os.environ["HOME"] = str(deployment_workdir)
    os.chdir(deployment_workdir)

    model_runner = RuntimeDeployment.func_or_class()

    # Log the memory usage before testing
    cpu_before, gpu_before = model_runner._get_memory_usage()
    logger.info(
        f"📊 [main] Memory before testing: CPU: {cpu_before / (1024 * 1024):.2f} MB, GPU: {gpu_before / (1024 * 1024):.2f} MB"
    )

    # Test the deployment with a model that should pass all checks
    model_id = "ambitious-ant"
    rdf_path = deployment_workdir / "models" / model_id / "rdf.yaml"

    # Run the model test (torch is already in requirements, should automatically be skipped)
    test_result = asyncio.run(
        model_runner.test(str(rdf_path))  # , additional_requirements=["torch==2.5.1"]
    )
    logger.info("Model testing completed successfully")

    # Load the test image from the package
    model_rdf = yaml.safe_load(rdf_path.read_text())
    test_input_source = model_rdf["test_inputs"][0]

    test_image_path = str(
        deployment_workdir / "models" / model_id / test_input_source
    )
    test_image = np.load(test_image_path).astype("float32")

    # Run the prediction
    result = asyncio.run(model_runner.predict(str(rdf_path), inputs=test_image))
    logger.info(f"Model prediction result: {result}")

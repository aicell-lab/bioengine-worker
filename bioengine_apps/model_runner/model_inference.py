import hashlib
import json
import os
from typing import Dict, Literal, Optional, Union

import numpy as np
from ray import serve

CACHE_N_PREDICTION_PIPELINES = int(os.environ.get("CACHE_N_PREDICTION_PIPELINES", 10))


class CacheEntry:
    """Container for cached prediction pipelines."""

    def __init__(self, pipeline, cache_key: str, kwargs_cache: dict):
        self.pipeline = pipeline
        self.cache_key = cache_key
        self.kwargs_cache = kwargs_cache

    def __del__(self):
        # Clean up the kwargs_cache to avoid memory leaks
        print(f"🗑️ Cleaning up cache entry for key: {self.cache_key}")
        self.kwargs_cache.pop(self.cache_key, None)


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 1 / 3,
        "memory": 8 * 1024 * 1024 * 1024,  # 8GB RAM limit
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
    max_ongoing_requests=1,
    max_queued_requests=10,
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
    graceful_shutdown_timeout_s=300.0,
    graceful_shutdown_wait_loop_s=2.0,
)
class ModelInference:
    """Internal deployment for running model inference using bioimageio.core."""

    def __init__(self):
        print("🚀 Initializing ModelInference deployment")
        self._kwargs_cache = {}

    def _set_prediction_kwargs(
        self,
        model_source: str,
        weights_format: str,
        device: str,
        default_blocksize_parameter: int,
    ) -> str:
        """Generate cache key for prediction pipeline configuration."""
        print(f"🔧 Setting prediction kwargs for model: {model_source}")

        pipeline_kwargs = {
            "model_source": model_source,
            "create_kwargs": {
                "weights_format": weights_format,
                "device": device,
                "default_blocksize_parameter": default_blocksize_parameter,
            },
        }
        # Generate a unique cache key based on the pipeline configuration
        json_str = json.dumps(pipeline_kwargs)
        cache_key = hashlib.md5(json_str.encode()).hexdigest()

        self._kwargs_cache[cache_key] = pipeline_kwargs
        print(f"📦 Cached pipeline kwargs with key: {cache_key}")

        return cache_key

    @serve.multiplexed(max_num_models_per_replica=CACHE_N_PREDICTION_PIPELINES)
    def _create_prediction_pipeline(self, cache_key: str) -> CacheEntry:
        """Create and cache prediction pipeline for the given cache key."""
        from bioimageio.core import create_prediction_pipeline, load_model_description

        # Get model source and create_kwargs using the cache key
        pipeline_kwargs = self._kwargs_cache.get(cache_key)
        if not pipeline_kwargs:
            print(f"❌ No pipeline kwargs found for cache key: {cache_key}")
            raise ValueError(f"No pipeline kwargs found for cache key: {cache_key}")

        model_source = pipeline_kwargs["model_source"]
        create_kwargs = pipeline_kwargs["create_kwargs"]

        print(f"🔄 Creating prediction pipeline for model at {model_source}")

        try:
            model_description = load_model_description(model_source)
            pipeline = create_prediction_pipeline(model_description, **create_kwargs)

            # Load the pipeline (this loads the model weights into memory/GPU)
            pipeline.load()

            print(
                f"✅ Created and loaded prediction pipeline for model at {model_source}"
            )

            return CacheEntry(
                pipeline=pipeline, cache_key=cache_key, kwargs_cache=self._kwargs_cache
            )
        except Exception as e:
            print(f"❌ Failed to create prediction pipeline: {str(e)}")
            raise

    def predict(
        self,
        model_source: str,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        weights_format: Optional[str] = None,
        device: Literal["cuda", "cpu"] = None,
        default_blocksize_parameter: Optional[int] = None,
        sample_id: str = "sample",
    ) -> Dict[str, np.ndarray]:
        """Run inference on model using bioimageio.core prediction pipeline."""
        from bioimageio.core.digest_spec import create_sample_for_model

        print(f"🔮 Starting prediction for model: {model_source}")
        print(
            f"📊 Input type: {type(inputs)}, device: {device}, sample_id: {sample_id}"
        )

        try:
            cache_key = self._set_prediction_kwargs(
                model_source=model_source,
                weights_format=weights_format,
                device=device,
                default_blocksize_parameter=default_blocksize_parameter,
            )
            cache_entry = self._create_prediction_pipeline(cache_key)

            print(f"🎯 Creating sample for model prediction")
            # Create sample from inputs
            # Handle single array input by creating a proper dictionary
            sample = create_sample_for_model(
                cache_entry.pipeline.model_description,
                inputs=inputs,
                sample_id=sample_id,
            )

            print(
                f"⚙️ Running prediction (blocking: {bool(default_blocksize_parameter)})"
            )
            # Run prediction using the pipeline
            if default_blocksize_parameter:
                result = cache_entry.pipeline.predict_sample_with_blocking(sample)
            else:
                result = cache_entry.pipeline.predict_sample_without_blocking(sample)

            # Convert outputs back to numpy arrays
            outputs = {str(k): v.data.data for k, v in result.members.items()}

            print(f"✅ Prediction completed, output keys: {list(outputs.keys())}")
            return outputs

        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            raise


if __name__ == "__main__":
    from pathlib import Path

    # Test the deployment with a model that should pass all checks
    TEST_BMZ_MODEL_ID = "charismatic-whale"

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

    model_inference = ModelInference.func_or_class()

    model_source = str(
        deployment_workdir / "models" / f"bmz_model_{TEST_BMZ_MODEL_ID}" / "rdf.yaml"
    )
    test_image_path = str(
        deployment_workdir / "models" / f"unpublished_model_{TEST_BMZ_MODEL_ID}" / "new_test_input.npy"
    )

    # Load the test image from the package
    image = np.load(test_image_path).astype("float32")

    # Reshape to match expected format: (batch=1, y, x, channels=1)
    input_image = image[np.newaxis, :, :, np.newaxis]

    result = model_inference.predict(model_source, inputs=input_image)
    print(f"Model inference result: {result}")

from typing import List, Optional

import ray
from ray import serve

# Deployment default runtime environment
requirements = [
    "bioimageio.core==0.9.0",
    "xarray==2025.1.2",  # this is needed for bioimageio.core
    "numpy==1.26.4",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "tensorflow==2.16.1",
    "onnxruntime==1.20.1",
]


@serve.deployment(
    ray_actor_options={
        "num_cpus": 1 / 3,
        "num_gpus": 1 / 3,
        # "memory": 16 * 1024 * 1024 * 1024,  # 16GB RAM limit
        "runtime_env": {
            "pip": requirements,
        },
    },
    max_ongoing_requests=1,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 5,
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
class ModelEvaluation:
    """Internal deployment for running model testing using bioimageio.core."""

    def _test(self, model_source: str) -> dict:
        """Run bioimageio.core test_model on the given model source."""
        from pathlib import Path

        from bioimageio.core import test_model

        try:
            if not Path(model_source).exists():
                raise FileNotFoundError(f"Model source not found: {model_source}")
            validation_summary = test_model(model_source)
            test_result = validation_summary.model_dump(mode="json")

            return test_result
        except Exception as e:
            print(f"❌ Model test failed: {str(e)}")
            raise

    async def test(
        self, model_source: str, additional_requirements: Optional[List[str]] = None
    ) -> dict:
        """Test model inference using bioimageio.core with optional additional requirements."""
        additional_packages = []
        if additional_requirements:
            if not isinstance(additional_requirements, list):
                print("❌ additional_requirements must be a list of strings")
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
            print(f"🚀 Running test with additional packages: {additional_packages}")
            # Execute remotely and return the result reference (will be awaited by DeploymentHandle)
            remote_function = ray.remote(self._test.__func__)
            remote_function = remote_function.options(
                num_cpus=1,
                num_gpus=0,
                memory=4 * 1024 * 1024 * 1024,  # 4GB RAM limit
                runtime_env={"pip": requirements + additional_packages},
            )
            result_ref = remote_function.remote(None, model_source)
            print(f"📋 Submitted remote test job")
            return result_ref
        else:
            # Run execution in this deployment without additional packages
            test_result = self._test(model_source)

        return test_result


if __name__ == "__main__":
    import asyncio
    import os
    from pathlib import Path

    # Set up the environment variables like in the real deployment
    deployment_workdir = (
        Path(__file__).resolve().parent.parent.parent
        / ".bioengine"
        / "apps"
        / "bioimage-io-model-runner"
    )
    deployment_workdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(deployment_workdir)
    os.environ["HOME"] = str(deployment_workdir)
    os.chdir(deployment_workdir)

    model_evaluation = ModelEvaluation.func_or_class()

    # Test the deployment with a model that should pass all checks
    model_id = "charismatic-whale"
    model_source = str(
        deployment_workdir / "models" / f"bmz_model_{model_id}" / "rdf.yaml"
    )

    # Run the model test
    test_result = asyncio.run(
        model_evaluation.test(model_source, additional_requirements=["torch==2.5.1"])
    )
    print("Model evaluation completed successfully")

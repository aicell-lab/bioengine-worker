import json
import os
import shutil
from pathlib import Path
from typing import Dict, Union

import numpy as np


class ModelRunner:
    def __init__(self, cache_n_models: int = 10):
        self.cache_dir = (
            Path(os.environ["BIOENGINE_CACHE_PATH"]).resolve() / "bioimageio_models"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        bioimageio_cache_path = self.cache_dir / ".cache"
        bioimageio_cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["BIOIMAGEIO_CACHE_PATH"] = str(bioimageio_cache_path)
        self.cache_n_models = cache_n_models
        self.cached_models = []

    def _get_cache_key_for_url(self, url: str) -> str:
        """Generate a consistent cache key for URL-based models."""
        import hashlib

        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"url_model_{url_hash}"

    async def _get_url_model_path(
        self, model_url: str, skip_cache: bool = False
    ) -> Path:
        """
        Get the path to a URL-based model, downloading if necessary.
        This method can be decorated with Ray's multiplex for URL caching.
        """
        cache_key = self._get_cache_key_for_url(model_url)
        package_path = self.cache_dir / cache_key

        # Handle cache skipping
        if skip_cache and package_path.exists():
            print(f"Removing cached URL model at {package_path}")
            try:
                shutil.rmtree(str(package_path))
            except Exception as e:
                print(f"Warning: Could not remove cache directory: {e}")

        # Download if not exists
        if not package_path.exists():
            print(f"Downloading model from URL: {model_url}")
            package_path = await self._download_model_from_url(
                model_url, str(package_path)
            )
            package_path = Path(package_path)
        else:
            print(f"Using cached URL model from {package_path}")

        return package_path

    async def _download_model_from_url(self, model_url: str, package_path: str) -> str:
        import os
        import zipfile
        from pathlib import Path

        import aiohttp

        package_path = Path(package_path)
        os.makedirs(package_path, exist_ok=True)
        archive_path = str(package_path) + ".zip"

        print(f"Downloading model from {model_url} to {archive_path}")
        async with aiohttp.ClientSession() as session:
            async with session.get(model_url) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to download model from {model_url}: {response.status}"
                    )
                content = await response.read()
                with open(archive_path, "wb") as f:
                    f.write(content)

        print(f"Downloaded zip file size: {os.path.getsize(archive_path)} bytes")

        # Unzip package_path
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            # List all files in zip for debugging
            files = zip_ref.namelist()
            print(f"Files in zip: {files}")

            # Get the root directory from the zip file
            root_dirs = [name for name in files if name.endswith("/")]
            root_dir = root_dirs[0] if root_dirs else ""
            print(f"Root directory found: '{root_dir}'")

            zip_ref.extractall(package_path)

        # Clean up the zip file
        os.remove(archive_path)

        # If there's a root directory in the zip, adjust the package path
        final_path = package_path
        if root_dir:
            extracted_path = package_path / root_dir.rstrip("/")
            if extracted_path.exists():
                final_path = extracted_path

        # Debug output
        print(f"Final package path: {final_path}")
        if final_path.exists():
            print(f"Contents of final path: {list(final_path.glob('*'))}")
            if not (final_path / "rdf.yaml").exists():
                print("Warning: rdf.yaml not found in expected location")
                # Try to find rdf.yaml recursively
                rdf_files = list(final_path.rglob("rdf.yaml"))
                if rdf_files:
                    print(f"Found rdf.yaml at: {rdf_files[0]}")
                    final_path = rdf_files[0].parent
        else:
            print(f"Warning: Final path {final_path} does not exist")

        return final_path

    async def _download_model(self, model_id: str) -> str:
        from bioimageio.spec import save_bioimageio_package_as_folder

        # Download new model
        model_path = self.cache_dir / model_id
        os.makedirs(model_path, exist_ok=True)
        model_path = Path(
            save_bioimageio_package_as_folder(model_id, output_path=str(model_path))
        )
        return model_path

    async def _get_model(self, model_id: str):
        from bioimageio.core import load_model_description
        from bioimageio.spec import InvalidDescr

        # Download model if not in cache
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
            remove_model_path = self.cache_dir / remove_model_id
            if remove_model_path.exists():
                try:
                    shutil.rmtree(str(remove_model_path))
                except:
                    pass

        model = load_model_description(str(self.cache_dir / model_id))
        assert not isinstance(model, InvalidDescr), f"Model {model_id} is invalid"
        return model

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
        self, model_id: str, inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on the model.

        Args:
            model_id (str): The model ID.
            inputs (Union[np.ndarray, Dict[str, np.ndarray]]): The inputs to the model.

        Returns:
            Dict[str, np.ndarray]: The outputs of the model.
        """
        from bioimageio.core import predict

        # Get the model
        model = await self._get_model(model_id)

        # Change working directory (tensorflow models unzip to current directory)
        cwd = os.getcwd()
        try:
            os.chdir(self.cache_dir / ".cache")
            prediction = predict(model=model, inputs=inputs)
        except Exception as e:
            raise e
        finally:
            os.chdir(cwd)

        # Convert outputs back to numpy arrays
        outputs = {str(k): v.data.data for k, v in prediction.members.items()}
        return outputs

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
                    remove_model_path = self.cache_dir / remove_cache_key
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

            # Change working directory (some models may need this)
            cwd = os.getcwd()
            os.chdir(self.cache_dir / ".cache")
            try:
                result = test_model(rdf_path).model_dump(mode="json")
            except Exception as e:
                raise e
            finally:
                os.chdir(cwd)
        else:
            # Handle regular model IDs - let Ray's multiplex caching handle this
            if skip_cache:
                raise ValueError(
                    "skip_cache=True is not supported for model IDs, only for URLs."
                )

            # Always use _get_model to leverage Ray's multiplex caching
            print(f"Getting model: {model_id} (Ray multiplex will handle caching)")
            model = await self._get_model(model_id)

            # Change working directory (some models may need this)
            cwd = os.getcwd()
            os.chdir(self.cache_dir / ".cache")
            try:
                result = test_model(model).model_dump(mode="json")
            except Exception as e:
                raise e
            finally:
                os.chdir(cwd)

        return result


if __name__ == "__main__":
    import asyncio

    from kaibu_utils import fetch_image

    async def test_model():
        os.environ["BIOENGINE_CACHE_PATH"] = str(Path(".cache").resolve())

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

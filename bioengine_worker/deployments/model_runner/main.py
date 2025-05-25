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
        return {"success": summary.status == "valid-format", "details": summary.format()}

    async def test(self, model_id: str) -> dict:
        """
        Test a model using bioimageio.core test functionality.

        Args:
            model_id (str): The model ID to test.

        Returns:
            dict: Test result from bioimageio.core.test_model.
        """
        from bioimageio.core import test_model
        
        # Get the loaded model description using the efficient caching system
        model = await self._get_model(model_id)
        
        print(f"Testing model {model_id}...")
        
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

        model_id = "creative-panda"  # choose different bioimage.io model

        # Test the model validation and testing functions
        print("Testing model validation and testing...")
        
        # Test the model
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

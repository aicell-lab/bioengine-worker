from pathlib import Path
from typing import Dict, Union

import numpy as np


class ModelRunner:
    def __init__(self, cache_n_models: int = 10):
        import os

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
        import os
        from pathlib import Path

        from bioimageio.spec import save_bioimageio_package_as_folder

        # Download new model
        model_path = self.cache_dir / model_id
        os.makedirs(model_path, exist_ok=True)
        model_path = Path(
            save_bioimageio_package_as_folder(model_id, output_path=str(model_path))
        )

    async def _get_model(self, model_id: str):
        import shutil

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

    async def get_model_rdf(self, model_id: str) -> Dict:
        """
        Get the model RDF description including inputs, preprocessing, postprocessing, and outputs.

        Args:
            model_id (str): The model ID.

        Returns:
            Dict: The model RDF description.
        """
        import json

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
        import os

        from bioimageio.core import predict

        # Get the model
        model = await self._get_model(model_id)

        # Change working directory (tensorflow models unzip to current directory)
        cwd = os.getcwd()
        os.chdir(self.cache_dir / ".cache")
        prediction = predict(model=model, inputs=inputs)
        os.chdir(cwd)

        # Convert outputs back to numpy arrays
        outputs = {str(k): v.data.data for k, v in prediction.members.items()}
        return outputs


if __name__ == "__main__":
    import asyncio

    from kaibu_utils import fetch_image

    async def test_model():
        model_runner = ModelRunner()

        model_id = "creative-panda"  # choose different bioimage.io model

        image = await fetch_image(
            "https://zenodo.org/api/records/5906839/files/sample_input_0.tif/content"
        )
        image = image.astype("float32")
        print("example image downloaded: ", image.shape)

        model_rdf = await model_runner.get_model_rdf(model_id)

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

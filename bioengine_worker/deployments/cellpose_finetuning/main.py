import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


class CellposeFinetune(object):
    """
    Based on cellpose 2.0 finetune notebook:
    https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb#scrollTo=Q7c7V4yEqDc_
    """

    def __init__(self):
        self.cache_dir = (
            Path(os.environ["BIOENGINE_CACHE_PATH"]).resolve() / "cellpose_finetune"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(self.cache_dir)

    async def _download_data(self, data_dir, download_url):
        import httpx

        # Define the path to save the downloaded zip file
        zip_file_path = data_dir / "data.zip"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            zip_file_path.write_bytes(response.content)

        # Unzip the downloaded file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        # List all folders that do not start with _
        folders = [
            f for f in data_dir.iterdir() if f.is_dir() and not f.name.startswith("_")
        ]
        assert len(folders) == 1, "Expected exactly one folder in the data directory."

        return folders[0]

    def _find_image_annotation_pairs(self, image_dir):
        annotations_dir = image_dir / "annotations"

        # List to hold pairs of image and corresponding annotation masks
        image_annotation_pairs = []

        # Get list of all annotations
        annotation_files = list(annotations_dir.glob("*.tif"))

        # Iterate through each annotation file
        for annotation_file in annotation_files:
            annotation_name = annotation_file.name
            image_name = annotation_name.split("_mask_")[0]
            image_file = image_dir / f"{image_name}.tif"

            image_annotation_pairs.append((image_file, annotation_file))

        return image_annotation_pairs

    def _prepare_training_data(self, image_annotation_pairs):
        # Get all indices of the list
        all_indices = np.arange(len(image_annotation_pairs))

        # Define the split ratio (e.g., 80% train, 20% test)
        train_ratio = 0.8
        train_size = int(len(all_indices) * train_ratio)

        # Randomly shuffle and split indices
        np.random.shuffle(all_indices)
        train_indices = all_indices[:train_size]
        test_indices = all_indices[train_size:]

        # Create train and test splits
        train_files = [image_annotation_pairs[i][0] for i in train_indices]
        train_labels_files = [image_annotation_pairs[i][1] for i in train_indices]
        test_files = [image_annotation_pairs[i][0] for i in test_indices]
        test_labels_files = [image_annotation_pairs[i][1] for i in test_indices]

        return train_files, train_labels_files, test_files, test_labels_files

    def _train_cellpose(
        self,
        train_files,
        train_labels_files,
        test_files,
        test_labels_files,
        initial_model,  # ["cyto", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "None"]
        train_channel="Grayscale",  # "Grayscale", "Red", "Green", "Blue"
        second_train_channel="Grayscale",
        n_epochs=10,
        learning_rate=0.000001,
        weight_decay=0.0001,
    ):
        from cellpose import core, models, train

        model = models.CellposeModel(gpu=core.use_gpu(), model_type=initial_model)

        channels_lut = {
            "Grayscale": 0,
            "Red": 1,
            "Green": 2,
            "Blue": 3,
        }
        channels = [
            channels_lut[train_channel],  # Channel to use for training
            channels_lut[
                second_train_channel
            ],  # Second training channel (if applicable)
        ]

        new_model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_files=train_files,
            train_labels_files=train_labels_files,
            test_files=test_files,
            test_labels_files=test_labels_files,
            channels=channels,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            SGD=True,
            nimg_per_epoch=1,
            save_path=train_files[0].parent,
            model_name=f"finetuned_{initial_model}",
            min_train_masks=1,
        )

        return model, channels, new_model_path, train_losses, test_losses

    def _evaluate_cellpose(self, model, channels, test_files, test_labels_files):
        from cellpose import metrics
        from tifffile import imread

        # get files (during training, test_data is transformed so we will load it again)
        test_data = [imread(image_path) for image_path in test_files]
        test_labels = [imread(image_path) for image_path in test_labels_files]

        # diameter of labels in training images
        # use model diameter if user diameter is 0
        diameter = 0
        diameter = model.diam_labels if diameter == 0 else diameter
        diam_labels = model.diam_labels.item()

        # run model on test images
        masks = model.eval(test_data, channels=channels, diameter=diam_labels)[0]

        # check performance using ground truth labels
        ap = metrics.average_precision(test_labels, masks, threshold=[0.5, 0.75, 0.9])[
            0
        ]

        # precision at different IOU thresholds
        return {str(t): p for t, p in zip([0.5, 0.75, 0.9], ap.mean(axis=0))}

    async def _upload_finetuned_model(self, model_path, model_url):
        import httpx

        model_content = model_path.read_bytes()
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(model_url, content=model_content)
            response.raise_for_status()

    async def train(self, data: dict):
        """
        Runs Cellpose v2 finetuning

        Args:
            data: Dictionary containing the following keys:
                - data_download_url: Presigned URL of the data to download
                - model_upload_url: Presigned URL to upload the finetuned model
                - initial_model: Initial model to use for finetuning (not used if 'model_download_url' is given)

            Additional optional keys:
                - model_download_url: Presigned URL to download checkpoint from (optional)  # TODO: implement
                - train_channel: Channel to use for training, default is "Grayscale"
                - second_train_channel: Second training channel (if applicable)
                - n_epochs: Number of epochs for training, default is 10
                - learning_rate: Learning rate for training, default is 0.000001
                - weight_decay: Weight decay for training, default is 0.0001
        """
        # Check if the required keys are present in the data dictionary
        required_keys = [
            "data_download_url",
            "model_upload_url",
            "initial_model",
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key} in data dictionary.")

        if "model_download_url" in data:
            raise NotImplementedError(
                "Starting from a model checkpoint is not implemented yet."
            )

        # Create a temporary directory to save the downloaded file
        with tempfile.TemporaryDirectory(dir=self.cache_dir) as data_dir:
            data_dir = Path(data_dir)

            image_dir = await self._download_data(
                data_dir, data["data_download_url"]
            )  # TODO: replace this with HttpZarrStore from artifact manager (https://docs.amun.ai/#/artifact-manager?id=endpoint-2-workspaceartifactsartifact_aliaszip-fileszip_file_pathpathpathpath)
            image_annotation_pairs = self._find_image_annotation_pairs(image_dir)
            train_files, train_labels_files, test_files, test_labels_files = (
                self._prepare_training_data(image_annotation_pairs)
            )
            model, channels, model_path, train_losses, test_losses = self._train_cellpose(
                train_files=train_files,
                train_labels_files=train_labels_files,
                test_files=test_files,
                test_labels_files=test_labels_files,
                initial_model=data[
                    "initial_model"
                ],  # ["cyto", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "None"]
                train_channel=data.get(
                    "train_channel", "Grayscale"
                ),  # "Grayscale", "Red", "Green", "Blue"
                second_train_channel=data.get("second_train_channel", "Grayscale"),
                n_epochs=data.get("n_epochs", 10),
                learning_rate=data.get("learning_rate", 0.000001),
                weight_decay=data.get("weight_decay", 0.0001),
            )
            
            ap = self._evaluate_cellpose(model, channels, test_files, test_labels_files)

            await self._upload_finetuned_model(
                Path(model_path),
                data["model_upload_url"],
            )

            return {
                "train_losses": train_losses,
                "test_losses": test_losses,
                "final_average_precision": ap,
            }


if __name__ == "__main__":
    import asyncio

    from hypha_rpc import connect_to_server, login

    async def test_model():
        os.environ["BIOENGINE_CACHE_PATH"] = str(
            Path("__file__").parent.parent.parent.parent / ".cache"
        )
        server_url = "https://hypha.aicell.io"
        token = os.environ["HYPHA_TOKEN"] or await login({"server_url": server_url})
        server = await connect_to_server({"server_url": server_url, "token": token})

        artifact_manager = await server.get_service("public/artifact-manager")

        workspace = server.config.workspace
        collection_id = f"{workspace}/bioimageio-colab"
        data_artifact_alias = "hpa-demo"
        model_artifact_alias = "cellpose-cyto3-hpa-finetuned"

        # Create an artifact for the fine-tuned Cellpose model
        model_manifest = {
            "name": "Finetuned Cellpose model",
            "description": "Finetuned model for Cellpose cyto3",
            "type": "model",
        }

        try:
            model_artifact = await artifact_manager.create(
                alias=model_artifact_alias,
                parent_id=collection_id,
                manifest=model_manifest,
                type=model_manifest["type"],
                version="stage",
            )
        except:
            model_artifact_id = f"{workspace}/{model_artifact_alias}"
            answer = input(
                f"Artifact {model_artifact_id} already exists. Do you want to overwrite it? (y/n): "
            )
            if answer.lower() != "y":
                raise RuntimeError(
                    f"Artifact {model_artifact_id} already exists and will not be overwritten."
                )

            # Overwrite the existing artifact
            model_artifact = await artifact_manager.edit(
                artifact_id=f"{workspace}/{model_artifact_alias}",
                manifest=model_manifest,
                type=model_manifest["type"],
                version="stage",
            )

        # Create presigned URLs for data download and model upload
        data_download_url = await artifact_manager.get_file(
            artifact_id=f"{workspace}/{data_artifact_alias}", file_path="data.zip"
        )

        finetuned_model = model_artifact_alias.replace("-", "_")
        model_upload_url = await artifact_manager.put_file(
            model_artifact.id, file_path=finetuned_model
        )

        data = {
            "data_download_url": data_download_url,
            "model_upload_url": model_upload_url,
            "initial_model": "cyto3",
        }

        # Run the finetuning process
        cellpose_finetuning = CellposeFinetune()

        output = await cellpose_finetuning.train(data)

        print(f"Average precision at iou threshold 0.5: {output['final_average_precision']['0.5']:.3f}")

        # Commit the artifact
        await artifact_manager.commit(
            artifact_id=model_artifact.id,
            version="new",
        )
        print(f"Committed artifact with ID: {model_artifact.id}")

    asyncio.run(test_model())

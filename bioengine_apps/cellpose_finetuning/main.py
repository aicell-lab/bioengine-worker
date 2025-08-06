import os
import tempfile
import zipfile
from pathlib import Path
import json

import numpy as np


class CellposeFinetune(object):
    """
    Based on cellpose 2.0 finetune notebook:
    https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb#scrollTo=Q7c7V4yEqDc_
    """

    def __init__(self):
        # Set up model directory
        models_dir = Path().resolve() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(models_dir)

        # Define available pretrained models
        self.pretrained_models = [
            "cyto",
            "cyto3",
            "nuclei",
            "tissuenet_cp3",
            "livecell_cp3",
            "yeast_PhC_cp3",
            "yeast_BF_cp3",
            "bact_phase_cp3",
            "bact_fluor_cp3",
            "deepbacs_cp3",
        ]

    async def _download_data(self, tmp_dir: Path, download_url: str) -> Path:
        import httpx

        # Define the path to save the downloaded zip file
        zip_file_path = tmp_dir / "data.zip"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            zip_file_path.write_bytes(response.content)

        # Unzip the downloaded file directly to tmp_dir
        # This will create the data/ folder structure inside tmp_dir
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        # Return the path to the extracted data directory
        data_dir = tmp_dir / "data"

        # Verify the data directory exists
        if not data_dir.exists():
            raise FileNotFoundError(f"Expected data directory not found at {data_dir}")

        return data_dir

    async def _find_image_annotation_pairs(self, image_dir):
        annotations_dir = image_dir / "annotations"

        # Verify annotations directory exists
        if not annotations_dir.exists():
            raise FileNotFoundError(
                f"Annotations directory not found at {annotations_dir}"
            )

        # List to hold pairs of image and corresponding annotation masks
        image_annotation_pairs = []

        # Get list of all annotations
        annotation_files = list(annotations_dir.glob("*.tif"))

        if not annotation_files:
            raise ValueError(f"No annotation files found in {annotations_dir}")

        # Iterate through each annotation file
        for annotation_file in annotation_files:
            annotation_name = annotation_file.name
            # Handle both "_mask.tif" and "_mask_1.tif" patterns
            if "_mask.tif" in annotation_name:
                image_name = annotation_name.replace("_mask.tif", ".tif")
            elif "_mask_" in annotation_name:
                image_name = annotation_name.split("_mask_")[0] + ".tif"
            else:
                # Skip files that don't match expected mask pattern
                continue

            image_file = image_dir / image_name

            # Only add the pair if both files exist
            if image_file.exists():
                image_annotation_pairs.append((image_file, annotation_file))
            else:
                print(
                    f"Warning: Image file {image_file} not found for annotation {annotation_file}"
                )

        if not image_annotation_pairs:
            raise ValueError(f"No valid image-annotation pairs found in {image_dir}")

        return image_annotation_pairs

    def _set_channels(self, train_channel, second_train_channel):
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
        return channels

    def _prepare_training_data(self, image_annotation_pairs, train_ratio):
        # Get all indices of the list
        all_indices = np.arange(len(image_annotation_pairs))

        # Define the split ratio (e.g., 80% train, 20% test)
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
        save_dir,
        model,
        initial_model,
        train_files,
        train_labels_files,
        test_files,
        test_labels_files,
        channels,
        n_epochs=10,
        learning_rate=0.000001,
        weight_decay=0.0001,
    ):
        from cellpose import train

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
            save_path=save_dir,
            model_name=f"finetuned_{initial_model}",
            min_train_masks=1,
        )

        return model, new_model_path, train_losses, test_losses

    def _evaluate_cellpose(self, model, channels, test_files, test_labels_files, stage):
        from cellpose import metrics
        from tifffile import imread, imwrite

        # get files (during training, test_data is transformed so we will load it again)
        test_data = [imread(image_path) for image_path in test_files]
        test_labels = [imread(image_path) for image_path in test_labels_files]

        # diameter of labels in training images - use model diameter
        diam_labels = model.diam_labels.item()

        # run model on test images
        masks = model.eval(test_data, channels=channels, diameter=diam_labels)[0]

        # check performance using ground truth labels
        ap = metrics.average_precision(test_labels, masks, threshold=[0.5, 0.75, 0.9])[
            0
        ]

        # TODO: save masks to disk
        prediction_files = []
        predictions_dir = test_files[0].parent / f"predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        for image_file, mask in zip(test_files, masks):
            counter = 1
            prediction_file = (
                predictions_dir
                / f"{image_file.stem}_{stage}_predicted_mask_{counter}.tif"
            )
            while prediction_file.exists():
                counter += 1
                prediction_file = (
                    predictions_dir
                    / f"{image_file.stem}_{stage}_predicted_mask_{counter}.tif"
                )
            prediction_files.append(prediction_file)
            imwrite(prediction_file, mask)

        # precision at different IOU thresholds
        ap = {str(t): p for t, p in zip([0.5, 0.75, 0.9], ap.mean(axis=0))}

        return prediction_files, ap

    async def _upload_data(
        self,
        result_upload_url: str,
        tmp_dir: Path,
        model_path: str,
        test_files: list,
        test_labels_files: list,
        initial_predictions: list,
        finetuned_predictions: list,
        train_losses: list,
        test_losses: list,
        initial_ap: dict,
        finetuned_ap: dict,
    ):
        # Create a zip file with the finetuned model, test_files, test_labels_files, initial_predictions, finetuned_predictions
        # Summarize train_losses, test_losses, initial_ap and finetuned_ap in one json file and add it to the zip file
        import httpx

        # Create a summary file with train_losses, test_losses, initial_ap and finetuned_ap
        summary = {
            "train_losses": list(train_losses),
            "test_losses": list(test_losses),
            "initial_ap": list(initial_ap),
            "finetuned_ap": list(finetuned_ap),
        }

        summary_file = tmp_dir / "summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f)

        # Create a list of all files to include in the zip
        files_to_zip = (
            [
                model_path,
                summary_file,
            ]
            + test_files
            + test_labels_files
            + initial_predictions
            + finetuned_predictions
        )

        # Create a zip file
        zip_file_path = tmp_dir / "finetuned_model.zip"
        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            for file in files_to_zip:
                zip_file.write(file, arcname=file.name)

        # Upload the zip file to the provided upload URL
        zip_content = zip_file_path.read_bytes()
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.put(result_upload_url, content=zip_content)
            response.raise_for_status()

    async def list_pretrained_models(self) -> list:
        return self.pretrained_models

    async def train(self, data: dict) -> dict:
        """
        Runs Cellpose v2 finetuning

        Args:
            data: Dictionary containing the following keys:
                - data_download_url: Presigned URL of the data to download
                - result_upload_url: Presigned URL to upload the model finetuning results
                - initial_model: Initial model to use for finetuning (not used if 'model_download_url' is given)

            Additional optional keys:
                - model_download_url: Presigned URL to download checkpoint from (optional)  # TODO: implement
                - train_channel: Channel to use for training, default is "Grayscale"
                - second_train_channel: Second training channel (if applicable)
                - train_ratio: Ratio of training data, default is 0.8
                - n_epochs: Number of epochs for training, default is 10
                - learning_rate: Learning rate for training, default is 0.000001
                - weight_decay: Weight decay for training, default is 0.0001

        Returns:
            A dictionary with the status of the finetuning process.
        """
        from cellpose import core, models

        # Check if the required keys are present in the data dictionary
        required_keys = [
            "data_download_url",
            "result_upload_url",
            "initial_model",
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key} in data dictionary.")

        # Check if the initial model is valid
        if data["initial_model"] not in self.pretrained_models:
            raise ValueError(
                f"Invalid initial model: {data['initial_model']}. "
                f"Available models: {', '.join(self.pretrained_models)}."
            )

        # If model_download_url is provided, raise NotImplementedError
        if "model_download_url" in data:
            raise NotImplementedError(
                "Starting from a model checkpoint is not implemented yet."
            )

        # Create a temporary directory to save the downloaded file
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
            tmp_dir = Path(tmp)

            # Download the data from the provided URL
            # TODO: replace this with HttpZarrStore from artifact manager (https://docs.amun.ai/#/artifact-manager?id=endpoint-2-workspaceartifactsartifact_aliaszip-fileszip_file_pathpathpathpath)
            image_dir = await self._download_data(tmp_dir, data["data_download_url"])

            # Create pairs of image and annotation masks
            image_annotation_pairs = await self._find_image_annotation_pairs(image_dir)
            train_files, train_labels_files, test_files, test_labels_files = (
                self._prepare_training_data(
                    image_annotation_pairs=image_annotation_pairs,
                    train_ratio=data.get("train_ratio", 0.8),
                )
            )

            # Initialize Cellpose model from one of the available models
            model = models.CellposeModel(
                gpu=core.use_gpu(), model_type=data["initial_model"]
            )

            # Set channels for training
            channels = self._set_channels(
                train_channel=data.get("train_channel", "Grayscale"),
                second_train_channel=data.get("second_train_channel", "Grayscale"),
            )

            # Evaluate initial model
            initial_predictions, initial_ap = self._evaluate_cellpose(
                model=model,
                channels=channels,
                test_files=test_files,
                test_labels_files=test_labels_files,
                stage="initial",
            )

            # Finetune Cellpose model
            model, model_path, train_losses, test_losses = self._train_cellpose(
                save_dir=tmp_dir,
                model=model,
                initial_model=data["initial_model"],
                train_files=train_files,
                train_labels_files=train_labels_files,
                test_files=test_files,
                test_labels_files=test_labels_files,
                channels=channels,
                n_epochs=data.get("n_epochs", 10),
                learning_rate=data.get("learning_rate", 0.000001),
                weight_decay=data.get("weight_decay", 0.0001),
            )

            # Evaluate finetuned model
            finetuned_predictions, finetuned_ap = self._evaluate_cellpose(
                model=model,
                channels=channels,
                test_files=test_files,
                test_labels_files=test_labels_files,
                stage="finetuned",
            )

            # Save the finetuning results in a zip file and upload it
            await self._upload_data(
                result_upload_url=data["result_upload_url"],
                tmp_dir=tmp_dir,
                model_path=model_path,
                test_files=test_files,
                test_labels_files=test_labels_files,
                initial_predictions=initial_predictions,
                finetuned_predictions=finetuned_predictions,
                train_losses=train_losses,
                test_losses=test_losses,
                initial_ap=initial_ap,
                finetuned_ap=finetuned_ap,
            )

            # Clean up the finetuned model `model_path`
            os.remove(model_path)

            return {"status": "success"}


if __name__ == "__main__":
    import asyncio

    from hypha_rpc import connect_to_server, login

    async def test_model():
        os.environ["TMPDIR"] = str(
            Path("__file__").parent.parent.parent / ".bioengine" / "cellpose_finetuning"
        )
        server_url = "https://hypha.aicell.io"
        token = os.environ["HYPHA_TOKEN"] or await login({"server_url": server_url})
        server = await connect_to_server({"server_url": server_url, "token": token})

        artifact_manager = await server.get_service("public/artifact-manager")

        workspace = server.config.workspace
        collection_id = f"{workspace}/bioimageio-colab"
        data_artifact_alias = "hpa-demo"
        model_artifact_alias = "cellpose-cyto3-hpa-finetuned"

        finetuning_result = model_artifact_alias.replace("-", "_") + ".zip"

        # Create an artifact for the fine-tuned Cellpose model
        model_manifest = {
            "name": "Finetuned Cellpose model",
            "description": "Finetuned model for Cellpose cyto3",
            "type": "generic",
        }

        try:
            model_artifact = await artifact_manager.create(
                alias=model_artifact_alias,
                parent_id=collection_id,
                manifest=model_manifest,
                type="application",
                stage=True,
            )
            print(f"Artifact created with ID: {model_artifact.id}")
        except:
            artifact_id = f"{workspace}/{model_artifact_alias}"
            artifact_files = await artifact_manager.list_files(artifact_id)
            for file in artifact_files:
                if file.name == finetuning_result:
                    print(
                        f"The file '{finetuning_result}' already exists in the artifact '{artifact_id}'. Overwriting it."
                    )
                    break

            # Edit the existing artifact with the new manifest
            model_artifact = await artifact_manager.edit(
                artifact_id=f"{workspace}/{model_artifact_alias}",
                manifest=model_manifest,
                type=model_manifest["type"],
                stage=True,
            )

        # Create presigned URLs for data download and model upload
        data_download_url = await artifact_manager.get_file(
            artifact_id=f"{workspace}/{data_artifact_alias}",
            file_path="data.zip",
        )

        result_upload_url = await artifact_manager.put_file(
            artifact_id=model_artifact.id,
            file_path=finetuning_result,
        )

        data = {
            "data_download_url": data_download_url,
            "result_upload_url": result_upload_url,
            "initial_model": "cyto3",
        }

        # Run the finetuning process
        cellpose_finetuning = CellposeFinetune()

        output = await cellpose_finetuning.train(data)

        assert output["status"] == "success", "Finetuning failed"

        # Commit the artifact
        await artifact_manager.commit(artifact_id=model_artifact.id)
        print(f"Committed artifact with ID: {model_artifact.id}")

    asyncio.run(test_model())

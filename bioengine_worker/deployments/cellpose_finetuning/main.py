class CellposeFinetune(object):
    """
    Based on cellpose 2.0 finetune notebook:
    https://colab.research.google.com/github/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb#scrollTo=Q7c7V4yEqDc_
    """
    def __init__(self):
        pass

    async def _download_data(self, download_url):
        """Write data to a zip file"""
        import tempfile
        import httpx
        import zipfile
        from pathlib import Path

        # Create a temporary directory to save the downloaded file
        data_dir = Path(tempfile.mkdtemp())

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
        folders = [f for f in data_dir.iterdir() if f.is_dir() and not f.name.startswith("_")]
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
        """Prepare training data from image and annotation pairs"""
        import numpy as np

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
            channels_lut[second_train_channel],  # Second training channel (if applicable)
        ]

        new_model_path = train.train_seg(
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
            min_train_masks=1
        )[0]

        return model, new_model_path, channels
    
    def _evaluate_cellpose(self, model, channels, test_files, test_labels_files):
        from tifffile import imread
        from cellpose import metrics

        # get files (during training, test_data is transformed so we will load it again)
        test_data = [imread(image_path) for image_path in test_files]
        test_labels = [imread(image_path) for image_path in test_labels_files]

        # diameter of labels in training images
        # use model diameter if user diameter is 0
        diameter=0
        diameter = model.diam_labels if diameter==0 else diameter
        diam_labels = model.diam_labels.item()

        # run model on test images
        masks = model.eval(
            test_data,
            channels=channels,
            diameter=diam_labels
        )[0]

        # check performance using ground truth labels
        ap = metrics.average_precision(test_labels, masks, threshold=[0.5, 0.75, 0.9])[0]

        # precision at different IOU thresholds
        return {str(t): p for t, p in zip([0.5, 0.75, 0.9], ap.mean(axis=0))}
    
    async def _upload_finetuned_model(self, model_path, model_url):
        import httpx
        from pathlib import Path

        model_content = Path(model_path).read_bytes()
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(model_url, content=model_content)
            response.raise_for_status()

    async def __call__(self, data=None):
        """
        Runs Cellpose v2 finetuning

        Args:
            data_url: Presigned URL of the data to download
            finetuned_model_url: Presigned URL to upload the finetuned model
            initial_model: Initial model to use for finetuning (not used if "model_url" is given)
            model_url: Presigned URL to download checkpoint from (optional)  # TODO: implement
            train_channel: Channel to use for training, default is "Grayscale"
            second_train_channel: Second training channel (if applicable)
            n_epochs: Number of epochs for training, default is 10
            learning_rate: Learning rate for training, default is 0.000001
            weight_decay: Weight decay for training, default is 0.0001
        """
        image_dir = await self._download_data(data["data_url"])
        image_annotation_pairs = self._find_image_annotation_pairs(image_dir)
        train_files, train_labels_files, test_files, test_labels_files = self._prepare_training_data(image_annotation_pairs)
        model, model_path, channels = self._train_cellpose(
            train_files=train_files,
            train_labels_files=train_labels_files,
            test_files=test_files,
            test_labels_files=test_labels_files,
            initial_model=data["initial_model"],  # ["cyto", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "None"]
            train_channel=data.get("train_channel", "Grayscale"),  # "Grayscale", "Red", "Green", "Blue"
            second_train_channel=data.get("second_train_channel", "Grayscale"),
            n_epochs=data.get("n_epochs", 10),
            learning_rate=data.get("learning_rate", 0.000001),
            weight_decay=data.get("weight_decay", 0.0001),
        )
        ap = self._evaluate_cellpose(
            model,
            channels,
            test_files,
            test_labels_files
        )
        await self._upload_finetuned_model(
            model_path,
            data["finetuned_model_url"],
        )

        return ap
    

if __name__ == "__main__":
    import os
    from pathlib import Path
    import httpx
    import asyncio
    from hypha_rpc import connect_to_server, login

    async def test_model():
        server_url="https://hypha.aicell.io"
        workspace="chiron-platform"
        token = os.environ["HYPHA_TOKEN"] or await login({"server_url": server_url})
        server = await connect_to_server({"server_url": server_url, "token": token, "workspace": workspace})

        artifact_manager = await server.get_service("public/artifact-manager")

        dataset_manifest = {
            "name": "HPA Demo",
            "description": "An annotated dataset for Cellpose finetuning",
            "type": "data",
        }
        parent_id = "chiron-platform/collection"

        # Check existing deployments
        deployment_alias = dataset_manifest["name"].lower().replace(" ", "-")
        all_artifacts = []
        for artifact in await artifact_manager.list():
            if artifact.type == "collection":
                all_artifacts.extend(await artifact_manager.list(parent_id=artifact.id))
            else:
                all_artifacts.append(artifact)

        exists = False
        for artifact in all_artifacts:
            if artifact.alias == deployment_alias:
                exists = True
                break

        if exists:
            # Edit the existing deployment and stage it for review
            artifact = await artifact_manager.edit(
                artifact_id=artifact.id,
                manifest=dataset_manifest,
                type=dataset_manifest.get("type"),
                version="stage",
            )
            print(f"Artifact edited with ID: {artifact.id}")
        else:
            # Add the deployment to the gallery and stage it for review
            artifact = await artifact_manager.create(
                alias=deployment_alias,
                parent_id=parent_id,
                manifest=dataset_manifest,
                type=dataset_manifest.get("type"),
                version="stage",
            )
            print(f"Artifact created with ID: {artifact.id}")

        # Load the dataset content
        data_path = Path("__file__").parent.parent.parent / "data" / "hpa_demo" / "data.zip"
        dataset_content = data_path.read_bytes()

        # Upload manifest.yaml
        upload_url = await artifact_manager.put_file(artifact.id, file_path="manifest.yaml")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(upload_url, data=dataset_manifest)
            response.raise_for_status()
            print(f"Uploaded manifest.yaml to artifact")

        # Upload the entry point
        upload_url = await artifact_manager.put_file(artifact.id, file_path="data.zip")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(upload_url, data=dataset_content)
            response.raise_for_status()
            print(f"Uploaded data.zip to artifact")

        cellpose_finetuning = CellposeFinetune()

        data_url = await artifact_manager.get_file(
            artifact_id="hpa-demo", file_path="data.zip"
        )

        model_manifest = {
            "name": "Finetuned Cellpose model",
            "description": "Finetuned model for Cellpose cyto3",
            "type": "model"
        }
        model_artifact = await artifact_manager.create(
            parent_id=parent_id,
            manifest=model_manifest,
            type=model_manifest.get("type"),
            version="stage",
        )
        finetuned_model_url = await artifact_manager.put_file(model_artifact.id, file_path="hpa_finetuned_cyto3")

        data = {
            "data_url": data_url,
            "finetuned_model_url": finetuned_model_url,
            "initial_model": "cyto3",
        }

        ap = await cellpose_finetuning(data)

        print(f"Average precision at iou threshold 0.5 = {ap['0.5']:.3f}")

    asyncio.run(test_model())
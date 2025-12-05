"""Simple end-to-end test: Train on existing dataset, then export."""
import asyncio
import os
from hypha_rpc import connect_to_server, login


async def main():
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    print("=" * 70)
    print("EXPORT MODEL END-TO-END TEST")
    print("=" * 70)

    server = await connect_to_server({"server_url": server_url, "token": token})
    print(f"✓ Connected to workspace: {server.config.workspace}")

    cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning", mode="last")  # Get latest version
    print(f"✓ Cellpose service connected")

    # Use the lazy-laborer dataset which should have valid data
    dataset_id = "bioimage-io/lazy-laborer-smell-less"

    print(f"\n{'='*70}")
    print("STEP 1: Training Model")
    print("=" * 70)
    print(f"Dataset: {dataset_id}")

    # First check what's in this dataset
    artifact_manager = await server.get_service("public/artifact-manager")
    files = await artifact_manager.list_files(dataset_id)
    print(f"Dataset files ({len(files)}):")
    for f in files[:10]:  # Show first 10
        name = f.get('name', str(f)) if isinstance(f, dict) else str(f)
        print(f"  - {name}")

    session_status = await cellpose_service.start_training(
        artifact=dataset_id,
        train_images="input_images/*.png",
        train_annotations="masks_nuclei/*_mask_1.png",
        model="cpsam",
        n_epochs=1,
    )

    session_id = session_status["session_id"]
    print(f"✓ Training started: {session_id}")

    # Monitor training
    print("\nMonitoring training...")
    while True:
        await asyncio.sleep(2)
        status = await cellpose_service.get_training_status(session_id)
        print(f"  [{status['status_type']}] {status['message']}")

        if status["status_type"] == "completed":
            print("\n✓ Training completed!")
            break
        elif status["status_type"] == "failed":
            print(f"\n❌ Training failed: {status['message']}")
            return 1

    print(f"\n{'='*70}")
    print("STEP 2: Exporting Model")
    print("=" * 70)

    model_name = f"test-model-{os.urandom(4).hex()}"
    print(f"Model name: {model_name}")

    try:
        export_result = await cellpose_service.export_model(
            session_id=session_id,
            model_name=model_name,
            collection="bioimage-io/colab-annotations",
        )

        print(f"\n✓ Export successful!")
        print(f"  Artifact ID: {export_result['artifact_id']}")
        print(f"  Artifact URL: {export_result['artifact_url']}")
        print(f"  Download URL: {export_result['download_url']}")

        # Validate
        print(f"\n{'='*70}")
        print("STEP 3: Validating Artifact")
        print("=" * 70)
        artifact_info = await artifact_manager.read(export_result['artifact_id'])

        artifact_type = artifact_info.get("type", "unknown")
        print(f"  Artifact type: {artifact_type}")

        if artifact_type == "model":
            print(f"  ✅ CORRECT TYPE: 'model'")
        else:
            print(f"  ❌ WRONG TYPE: expected 'model', got '{artifact_type}'")
            return 1

        parent_id = artifact_info.get("parent_id", "")
        if "colab-annotations" in parent_id:
            print(f"  ✅ CORRECT COLLECTION: bioimage-io/colab-annotations")
        else:
            print(f"  ❌ WRONG COLLECTION: {parent_id}")

        files = await artifact_manager.list_files(export_result['artifact_id'])
        file_list = [f.get('name', str(f)) if isinstance(f, dict) else str(f) for f in files]
        print(f"  Files: {len(file_list)}")
        for f in sorted(file_list):
            print(f"    - {f}")

        required = ["rdf.yaml", "model.py", "model_weights.pth", "input_sample.npy", "output_sample.npy", "cover.png", "doc.md"]
        missing = [r for r in required if r not in file_list]

        if missing:
            print(f"  ❌ MISSING FILES: {missing}")
            return 1
        else:
            print(f"  ✅ ALL 7 REQUIRED FILES PRESENT")

        print(f"\n{'='*70}")
        print("✅ ✅ ✅ ALL TESTS PASSED! ✅ ✅ ✅")
        print("=" * 70)
        print(f"\nModel artifact successfully created:")
        print(f"  Type: model")
        print(f"  Collection: bioimage-io/colab-annotations")
        print(f"  Files: 7/7")
        print(f"  Artifact URL: {export_result['artifact_url']}")
        print(f"  Download URL: {export_result['download_url']}")
        return 0

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

"""Test training with a single sample pair to reproduce the error."""
import asyncio
import os

from hypha_rpc import connect_to_server, login


async def test_single_sample_training():
    """Test training with the single-sample dataset."""
    # Connect to server
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    print(f"Connecting to {server_url}...")
    server = await connect_to_server({"server_url": server_url, "token": token})
    print(f"Connected to workspace: {server.config.workspace}")

    # Get the cellpose service
    print("\nGetting cellpose-finetuning service...")
    cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
    print("✓ Service connected")

    # First, let's check what files are in the artifact
    print("\n" + "="*60)
    print("Inspecting artifact: bioimage-io/arthurian-fastball-added-quickly")
    print("="*60)

    try:
        from hypha_artifact import AsyncHyphaArtifact

        artifact = AsyncHyphaArtifact(
            artifact_id="bioimage-io/arthurian-fastball-added-quickly",
            token=token,
            server_url=server_url,
        )

        # List root directory
        print("\nListing root directory...")
        root_files = await artifact.ls("/")
        print(f"Root contains {len(root_files)} items:")
        for item in root_files[:20]:  # Show first 20
            if isinstance(item, dict):
                print(f"  - {item.get('name', item.get('path', item))}")
            else:
                print(f"  - {item}")

        # List the actual folders found
        print("\nListing contents of detected folders...")
        for folder_name in ["input_images", "masks_nuclei"]:
            try:
                folder_path = f"{folder_name}/"
                files = await artifact.ls(folder_path)
                if files:
                    print(f"\n✓ Folder: {folder_path} ({len(files)} items)")
                    for f in files[:10]:  # Show first 10
                        if isinstance(f, dict):
                            print(f"    - {f.get('name', f.get('path', f))}")
                        else:
                            print(f"    - {f}")
            except Exception as e:
                print(f"  Error listing {folder_name}: {e}")

    except Exception as e:
        print(f"Error inspecting artifact: {e}")
        import traceback
        traceback.print_exc()

    # Now try to start training
    print("\n" + "="*60)
    print("Starting training with single sample")
    print("="*60)

    try:
        # Use pattern matching since filenames are different
        # Image: 47746_1370_G11_1.png -> Mask: 47746_1370_G11_1_mask_1.png
        # Pattern: * -> *_mask_1
        print("\nAttempt 1: Using pattern matching...")
        print("  Images: input_images/*.png")
        print("  Masks:  masks_nuclei/*_mask_1.png")
        session_status = await cellpose_service.start_training(
            artifact="bioimage-io/arthurian-fastball-added-quickly",
            train_images="input_images/*.png",
            train_annotations="masks_nuclei/*_mask_1.png",
            model="cpsam",
            n_epochs=1,
            n_samples=None,  # Use all available samples
        )

        session_id = session_status["session_id"]
        print(f"✓ Training started! Session ID: {session_id}")

        # Monitor training
        print("\nMonitoring training progress...")
        while True:
            await asyncio.sleep(2)
            status = await cellpose_service.get_training_status(session_id)

            msg = f"[{status['status_type']}] {status['message']}"
            if "current_epoch" in status:
                msg += f" | Epoch: {status.get('current_epoch', 0)}/{status.get('total_epochs', 0)}"
            if "elapsed_seconds" in status:
                msg += f" | Time: {status['elapsed_seconds']:.1f}s"

            print(msg)

            if status["status_type"] in ("completed", "failed"):
                if status["status_type"] == "failed":
                    print(f"\n❌ Training failed: {status['message']}")
                else:
                    print(f"\n✓ Training completed successfully!")
                break

        # Print final status
        print("\n" + "="*60)
        print("Final Status:")
        print("="*60)
        import json
        print(json.dumps(status, indent=2))

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

        # Try to get more details about the error
        print("\n" + "="*60)
        print("Error Details:")
        print("="*60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_single_sample_training())

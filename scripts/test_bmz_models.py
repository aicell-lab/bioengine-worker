import asyncio
import json
import os
from pathlib import Path

import httpx
import requests
from hypha_rpc import connect_to_server, login


def analyze_test_results(test_result):
    """Analyze test results and create test_reports field for manifest"""
    test_reports = [
        {"name": "RDF validation", "status": "failed", "runtime": "bioimageio.core"},
        {"name": "Model Test Run", "status": "failed", "runtime": "bioimageio.core"},
        {"name": "Reproduce Outputs", "status": "failed", "runtime": "bioimageio.core"},
    ]

    # If test_result is not a dict with details, assume all failed
    if not isinstance(test_result, dict) or "details" not in test_result:
        return test_reports

    details = test_result["details"]

    # Check RDF validation (bioimageio.spec format validation)
    for detail in details:
        if (
            detail.get("name", "").startswith("bioimageio.spec format validation")
            and detail.get("status") == "passed"
        ):
            test_reports[0]["status"] = "passed"
            break

    # Check if Model Test Run passed (overall status)
    if test_result.get("status") == "passed":
        test_reports[1]["status"] = "passed"

    # Check Reproduce Outputs (all "Reproduce test outputs from test inputs" tests must pass)
    reproduce_tests = [
        detail
        for detail in details
        if detail.get("name", "").startswith("Reproduce test outputs from test inputs")
    ]
    if reproduce_tests and all(
        test.get("status") == "passed" for test in reproduce_tests
    ):
        test_reports[2]["status"] = "passed"

    return test_reports


async def test_bmz_models():
    server_url = "https://hypha.aicell.io"
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})
    server = await connect_to_server(
        {"server_url": server_url, "token": token, "method_timeout": 3000}
    )

    bioengine = await server.get_service("bioimage-io/bioengine-apps")
    artifact_manager = await server.get_service("public/artifact-manager")

    # Fetch all model IDs
    url = (
        "https://hypha.aicell.io/bioimage-io/artifacts/bioimage.io/children?limit=10000"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch model IDs: {response.status_code}")
    model_ids = [
        item["id"].split("/")[1] for item in response.json() if ":" not in item["id"]
    ]
    model_ids = sorted(model_ids)

    # Create test results directory
    result_dir = Path(__file__).resolve().parent.parent / "bmz_model_tests"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Test each model
    for model_id in model_ids:
        test_results_file = result_dir / f"{model_id}.json"
        if not test_results_file.exists():
            try:
                print(f"Testing model: {model_id}")
                test_result = await bioengine.bioimage_io_model_runner.test(
                    model_id=model_id
                )
            except Exception as e:
                test_result = str(e)

            with open(test_results_file, "w") as f:
                json.dump(test_result, f, indent=2)
        else:
            print(f"Skipping test run for already tested model: {model_id} - file exists")
            test_result = json.loads(test_results_file.read_text(encoding="utf-8"))

        # Update artifact with test results
        artifact_id = f"bioimage-io/{model_id}"
        await asyncio.sleep(0.5)  # Avoid rate limiting issues
        try:
            # Get current artifact to read its manifest
            current_artifact = await artifact_manager.read(artifact_id)
            manifest = current_artifact.get("manifest", {})

            if "test_reports" in manifest:
                print(
                    f"Skipping update of artifact: {artifact_id} - already has test reports"
                )
                continue

            print(f"Adding test reports to artifact: {artifact_id}")

            # Analyze test results and add test_reports to manifest
            test_reports = analyze_test_results(test_result)
            manifest["test_reports"] = test_reports

            # Edit the artifact and stage it for review
            artifact = await artifact_manager.edit(
                artifact_id=artifact_id,
                manifest=manifest,
                type=current_artifact.get("type", "model"),
                stage=True,
            )

            # Upload test results file
            upload_url = await artifact_manager.put_file(
                artifact.id, file_path="test_reports.json"
            )

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.put(upload_url, data=json.dumps(test_result))
                response.raise_for_status()

            # Commit the artifact
            await artifact_manager.commit(artifact_id=artifact.id)
            print(f"Updated artifact {artifact_id} with test reports")

        except Exception as e:
            print(f"Failed to update artifact {artifact_id}: {e}")


if __name__ == "__main__":
    asyncio.run(test_bmz_models())
    print("All models tested successfully.")

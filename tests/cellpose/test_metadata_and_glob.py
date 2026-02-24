from __future__ import annotations

import asyncio
from pathlib import Path

from bioengine_apps.cellpose_finetuning.main import (
    CellposeFinetune,
    _extract_bia_accession,
    _pair_key_from_url,
    list_matching_artifact_paths,
    make_training_pairs_from_metadata,
    match_image_annotation_pairs,
    proportional_manual_sample_counts,
    resolve_requested_sample_count,
    sample_pair_lists,
)


def test_match_image_annotation_pairs_with_nested_globs() -> None:
    image_files = [
        "images/a/t0001.tif",
        "images/b/t0002.tif",
        "images/c/extra.tif",
    ]
    annotation_files = [
        "annotations/a/t0001_mask.ome.tif",
        "annotations/b/t0002_mask.ome.tif",
        "annotations/x/other_mask.ome.tif",
    ]

    pairs = match_image_annotation_pairs(
        image_files,
        annotation_files,
        "images/*/*.tif",
        "annotations/*/*_mask.ome.tif",
    )

    assert pairs == [
        ("images/a/t0001.tif", "annotations/a/t0001_mask.ome.tif"),
        ("images/b/t0002.tif", "annotations/b/t0002_mask.ome.tif"),
    ]


def test_match_image_annotation_pairs_with_mixed_ome_suffix_convention() -> None:
    image_files = [
        "images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif",
        "images/108bb69d-2e52-4382-8100-e96173db24ee/t0001.ome.tif",
    ]
    annotation_files = [
        "annotations/108bb69d-2e52-4382-8100-e96173db24ee/t0000_mask.ome.tif",
        "annotations/108bb69d-2e52-4382-8100-e96173db24ee/t0001_mask.ome.tif",
    ]

    pairs = match_image_annotation_pairs(
        image_files,
        annotation_files,
        "images/108bb69d-2e52-4382-8100-e96173db24ee/*.tif",
        "annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif",
    )

    assert pairs == [
        (
            "images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif",
            "annotations/108bb69d-2e52-4382-8100-e96173db24ee/t0000_mask.ome.tif",
        ),
        (
            "images/108bb69d-2e52-4382-8100-e96173db24ee/t0001.ome.tif",
            "annotations/108bb69d-2e52-4382-8100-e96173db24ee/t0001_mask.ome.tif",
        ),
    ]


def test_match_image_annotation_pairs_with_many_nested_folders() -> None:
    image_files = [
        "images/folder1/sub1/img001.tif",
        "images/folder2/sub2/img002.tif",
    ]
    annotation_files = [
        "annotations/folder1/sub1/img001_mask.ome.tif",
        "annotations/folder2/sub2/img002_mask.ome.tif",
    ]

    pairs = match_image_annotation_pairs(
        image_files,
        annotation_files,
        "images/*/*.tif",
        "annotations/*/*_mask.ome.tif",
    )

    assert pairs == [
        (
            "images/folder1/sub1/img001.tif",
            "annotations/folder1/sub1/img001_mask.ome.tif",
        ),
        (
            "images/folder2/sub2/img002.tif",
            "annotations/folder2/sub2/img002_mask.ome.tif",
        ),
    ]


class _FakeArtifact:
    async def ls(self, folder_path: str):
        await asyncio.sleep(0)
        folder = folder_path.rstrip("/") + "/"
        if folder == "metadata/":
            return [{"path": "metadata/sample.json", "type": "file"}]
        if folder == "images/train/":
            return [
                {"path": "images/train/t0001.tif", "type": "file"},
                {"path": "images/train/t0002.tif", "type": "file"},
            ]
        return []

    async def get(self, remote_paths, local_paths, on_error="ignore"):
        await asyncio.sleep(0)
        for remote, local in zip(remote_paths, local_paths):
            local_path = Path(local)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if str(remote).endswith("metadata/sample.json"):
                local_path.write_text(
                    """
[
  {"image_path": "images/train/t0001.tif", "mask_path": "annotations/train/t0001_mask.ome.tif", "split": "train"},
  {"image_path": "images/test/t0002.tif", "mask_path": "annotations/test/t0002_mask.ome.tif", "split": "test"}
]
                    """.strip(),
                    encoding="utf-8",
                )
            else:
                local_path.write_bytes(b"dummy")


class _FakeArtifactNestedMetadata:
    async def ls(self, folder_path: str):
        await asyncio.sleep(0)
        folder = folder_path.rstrip("/") + "/"
        if folder == "metadata/":
            return [{"path": "metadata/records.json", "type": "file"}]
        return []

    async def get(self, remote_paths, local_paths, on_error="ignore"):
        await asyncio.sleep(0)
        for remote, local in zip(remote_paths, local_paths):
            local_path = Path(local)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if str(remote).endswith("metadata/records.json"):
                local_path.write_text(
                    """
{
    "payload": {
        "items": [
            {
                "imagePath": "images/train/t0010.tif",
                "maskPath": "annotations/train/t0010_mask.ome.tif",
                "split": "train"
            },
            {
                "source": {"path": "images/test/t0011.tif"},
                "target": {"path": "annotations/test/t0011_mask.ome.tif"},
                "subset": "validation"
            }
        ]
    }
}
                                        """.strip(),
                    encoding="utf-8",
                )
            else:
                local_path.write_bytes(b"dummy")


class _FakeArtifactAmbiguousDirs:
    async def ls(self, folder_path: str):
        await asyncio.sleep(0)
        folder = folder_path.rstrip("/") + "/"
        if folder == "images/":
            return [{"path": "images/108bb69d-2e52-4382-8100-e96173db24ee"}]
        if folder == "images/108bb69d-2e52-4382-8100-e96173db24ee/":
            return [
                {
                    "path": "images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif",
                    "type": "file",
                },
                {
                    "path": "images/108bb69d-2e52-4382-8100-e96173db24ee/t0001.ome.tif",
                    "type": "file",
                },
            ]
        return []

    async def get(self, remote_paths, local_paths, on_error="ignore"):
        await asyncio.sleep(0)
        for local in local_paths:
            local_path = Path(local)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"dummy")


class _FakeArtifactFolderGlobDirs:
    async def ls(self, folder_path: str):
        await asyncio.sleep(0)
        folder = folder_path.rstrip("/") + "/"
        if folder == "images/":
            return [
                {"path": "images/sample-a"},
                {"path": "images/sample-b"},
                {"path": "images/readme.txt", "type": "file"},
            ]
        if folder in {"images/sample-a/", "images/sample-b/"}:
            return [
                {"path": f"{folder}0", "type": "file"},
                {"path": f"{folder}1", "type": "file"},
            ]
        return []

    async def get(self, remote_paths, local_paths, on_error="ignore"):
        await asyncio.sleep(0)
        for local in local_paths:
            local_path = Path(local)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"dummy")


def test_make_training_pairs_from_metadata(tmp_path: Path) -> None:
    artifact = _FakeArtifact()
    train_pairs, test_pairs = asyncio.run(
        make_training_pairs_from_metadata(
            artifact=artifact,
            metadata_dir="metadata/",
            save_path=tmp_path,
            n_samples=None,
        )
    )

    assert len(train_pairs) == 1
    assert len(test_pairs) == 1
    assert train_pairs[0]["image"].exists()
    assert train_pairs[0]["annotation"].exists()
    assert test_pairs[0]["image"].exists()
    assert test_pairs[0]["annotation"].exists()


def test_list_matching_artifact_paths_directory_without_trailing_slash() -> None:
    artifact = _FakeArtifact()
    matches = asyncio.run(list_matching_artifact_paths(artifact, "images/train"))
    assert matches == [
        "images/train/t0001.tif",
        "images/train/t0002.tif",
    ]


def test_list_matching_artifact_paths_with_ambiguous_directory_entries() -> None:
    artifact = _FakeArtifactAmbiguousDirs()
    matches = asyncio.run(list_matching_artifact_paths(artifact, "images/*/*.tif"))
    assert matches == [
        "images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif",
        "images/108bb69d-2e52-4382-8100-e96173db24ee/t0001.ome.tif",
    ]


def test_list_matching_artifact_paths_folder_glob_returns_directories() -> None:
    artifact = _FakeArtifactFolderGlobDirs()
    matches = asyncio.run(list_matching_artifact_paths(artifact, "images/*/"))
    assert matches == [
        "images/sample-a/",
        "images/sample-b/",
    ]


def test_make_training_pairs_from_nested_metadata_formats(tmp_path: Path) -> None:
    artifact = _FakeArtifactNestedMetadata()
    train_pairs, test_pairs = asyncio.run(
        make_training_pairs_from_metadata(
            artifact=artifact,
            metadata_dir="metadata/",
            save_path=tmp_path,
            n_samples=None,
        )
    )

    assert len(train_pairs) == 1
    assert len(test_pairs) == 1
    assert train_pairs[0]["image"].as_posix().endswith("images/train/t0010.tif")
    assert (
        train_pairs[0]["annotation"]
        .as_posix()
        .endswith("annotations/train/t0010_mask.ome.tif")
    )
    assert test_pairs[0]["image"].as_posix().endswith("images/test/t0011.tif")
    assert (
        test_pairs[0]["annotation"]
        .as_posix()
        .endswith("annotations/test/t0011_mask.ome.tif")
    )


def test_extract_bia_accession_from_gallery_url() -> None:
    url = "https://beta.bioimagearchive.org/bioimage-archive/galleries/ai/ai-ready-study/S-BIAD1392"
    assert _extract_bia_accession(url) == "S-BIAD1392"


def test_pair_key_from_url_handles_mask_suffix() -> None:
    img = "https://example.org/data/images/folder1/t0001.ome.tif"
    ann = "https://example.org/data/annotations/folder1/t0001_mask.ome.tif"
    assert _pair_key_from_url(img, is_mask=False) == _pair_key_from_url(
        ann, is_mask=True
    )


def test_resolve_requested_sample_count_supports_decimal_fraction() -> None:
    assert resolve_requested_sample_count(0.5, 10) == 5
    assert resolve_requested_sample_count(0.01, 10) == 1
    assert resolve_requested_sample_count(1.0, 10) == 10


def test_resolve_requested_sample_count_supports_absolute_count() -> None:
    assert resolve_requested_sample_count(4, 10) == 4
    assert resolve_requested_sample_count(20, 10) == 10


def test_resolve_requested_sample_count_rejects_non_positive_values() -> None:
    try:
        resolve_requested_sample_count(0, 10)
        assert False, "Expected ValueError for n_samples=0"
    except ValueError as e:
        assert "n_samples must be > 0" in str(e)

    try:
        resolve_requested_sample_count(-0.2, 10)
        assert False, "Expected ValueError for n_samples<0"
    except ValueError as e:
        assert "n_samples must be > 0" in str(e)


def test_proportional_manual_sample_counts_basic_allocation() -> None:
    train_count, test_count = proportional_manual_sample_counts(
        train_available=8,
        test_available=2,
        requested_total=5,
    )
    assert train_count + test_count == 5
    assert train_count == 4
    assert test_count == 1


def test_proportional_manual_sample_counts_respects_capacity_limits() -> None:
    train_count, test_count = proportional_manual_sample_counts(
        train_available=1,
        test_available=9,
        requested_total=6,
    )
    assert train_count + test_count == 6
    assert train_count == 1
    assert test_count == 5


def test_sample_pair_lists_auto_mode_subsets_and_splits() -> None:
    train_pairs = [
        {"image": Path(f"images/t{i}.tif"), "annotation": Path(f"annotations/t{i}.tif")}
        for i in range(10)
    ]
    test_pairs = []

    sampled_train, sampled_test = sample_pair_lists(
        train_pairs,
        test_pairs,
        requested_total=4,
        split_mode="auto",
        train_split_ratio=0.75,
    )

    assert len(sampled_train) + len(sampled_test) == 4
    assert len(sampled_train) == 3
    assert len(sampled_test) == 1


def test_sample_pair_lists_manual_mode_proportional() -> None:
    train_pairs = [
        {"image": Path(f"images/t{i}.tif"), "annotation": Path(f"annotations/t{i}.tif")}
        for i in range(8)
    ]
    test_pairs = [
        {"image": Path(f"images/v{i}.tif"), "annotation": Path(f"annotations/v{i}.tif")}
        for i in range(2)
    ]

    sampled_train, sampled_test = sample_pair_lists(
        train_pairs,
        test_pairs,
        requested_total=5,
        split_mode="manual",
        train_split_ratio=0.8,
    )

    assert len(sampled_train) + len(sampled_test) == 5
    assert len(sampled_train) == 4
    assert len(sampled_test) == 1


def test_preflight_training_dataset_reports_pair_counts(monkeypatch) -> None:
    class _FakeArtifactForPreflight:
        async def ls(self, _folder_path: str):
            await asyncio.sleep(0)
            return []

    async def _fake_make_artifact_client(_artifact_id: str, _server_url: str):
        return _FakeArtifactForPreflight()

    async def _fake_list_matching_paths(_artifact, path_pattern: str, max_results=None):
        if "images" in path_pattern:
            return [f"images/t{i}.tif" for i in range(6)]
        if "annotations" in path_pattern:
            return [f"annotations/t{i}_mask.tif" for i in range(5)]
        return []

    monkeypatch.setattr(
        "bioengine_apps.cellpose_finetuning.main.make_artifact_client",
        _fake_make_artifact_client,
    )
    monkeypatch.setattr(
        "bioengine_apps.cellpose_finetuning.main.list_matching_artifact_paths",
        _fake_list_matching_paths,
    )

    service = CellposeFinetune.func_or_class()
    result = asyncio.run(
        service.preflight_training_dataset(
            artifact="ri-scale/zarr-demo",
            train_images="images/*/",
            train_annotations="annotations/*/",
            split_mode="auto",
            n_samples=5,
        )
    )

    assert result["ok"] is True
    assert result["train_image_count"] == 6
    assert result["train_annotation_count"] == 5
    assert result["train_pair_count"] == 5
    assert result["sampled_total_count"] == 5


def test_preflight_training_dataset_metadata_mode(monkeypatch) -> None:
    class _FakeArtifactWithMetadata:
        async def ls(self, folder_path: str):
            await asyncio.sleep(0)
            if folder_path == "metadata/":
                return [
                    {"path": "metadata/records.json", "type": "file"},
                    {"path": "metadata/readme.txt", "type": "file"},
                ]
            return []

    async def _fake_make_artifact_client(_artifact_id: str, _server_url: str):
        return _FakeArtifactWithMetadata()

    monkeypatch.setattr(
        "bioengine_apps.cellpose_finetuning.main.make_artifact_client",
        _fake_make_artifact_client,
    )

    service = CellposeFinetune.func_or_class()
    result = asyncio.run(
        service.preflight_training_dataset(
            artifact="ri-scale/zarr-demo",
            metadata_dir="metadata/",
        )
    )

    assert result["ok"] is True
    assert result["mode"] == "metadata"
    assert result["metadata_file_count"] == 1

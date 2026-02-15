from __future__ import annotations

import asyncio
from pathlib import Path

from bioengine_apps.cellpose_finetuning.main import (
    list_matching_artifact_paths,
    make_training_pairs_from_metadata,
    match_image_annotation_pairs,
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
    assert train_pairs[0]["annotation"].as_posix().endswith("annotations/train/t0010_mask.ome.tif")
    assert test_pairs[0]["image"].as_posix().endswith("images/test/t0011.tif")
    assert test_pairs[0]["annotation"].as_posix().endswith("annotations/test/t0011_mask.ome.tif")

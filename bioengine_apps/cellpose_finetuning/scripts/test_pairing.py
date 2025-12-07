"""Test script to verify that unpaired images/annotations are excluded."""

import sys
from pathlib import Path

# Add parent directory to path to import main module
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import match_image_annotation_pairs


def test_perfect_matching():
    """Test case where all files are paired."""
    print("\n=== Test 1: Perfect matching ===")
    images = ["t0000.ome.tif", "t0001.ome.tif", "t0002.ome.tif"]
    annots = ["t0000_mask.ome.tif", "t0001_mask.ome.tif", "t0002_mask.ome.tif"]

    pairs = match_image_annotation_pairs(
        images, annots, "*.ome.tif", "*_mask.ome.tif"
    )

    print(f"Images: {len(images)}, Annotations: {len(annots)}, Pairs: {len(pairs)}")
    print(f"Expected: 3 pairs, Got: {len(pairs)}")
    assert len(pairs) == 3, "Should match all 3 pairs"
    print("✓ PASSED: All files paired correctly")


def test_missing_annotations():
    """Test case where some images don't have annotations."""
    print("\n=== Test 2: Missing annotations ===")
    images = ["t0000.ome.tif", "t0001.ome.tif", "t0002.ome.tif", "t0003.ome.tif"]
    annots = ["t0000_mask.ome.tif", "t0002_mask.ome.tif"]  # Missing t0001 and t0003

    pairs = match_image_annotation_pairs(
        images, annots, "*.ome.tif", "*_mask.ome.tif"
    )

    print(f"Images: {len(images)}, Annotations: {len(annots)}, Pairs: {len(pairs)}")
    print(f"Unpaired images excluded: {len(images) - len(pairs)}")
    assert len(pairs) == 2, "Should only match 2 pairs (t0000, t0002)"
    print("✓ PASSED: Unpaired images excluded correctly")

    # Verify the correct pairs were matched
    expected = [("t0000.ome.tif", "t0000_mask.ome.tif"),
                ("t0002.ome.tif", "t0002_mask.ome.tif")]
    assert set(pairs) == set(expected), "Should match correct pairs"
    print("✓ PASSED: Correct pairs matched")


def test_missing_images():
    """Test case where some annotations don't have images."""
    print("\n=== Test 3: Missing images ===")
    images = ["t0000.ome.tif", "t0002.ome.tif"]  # Missing t0001 and t0003
    annots = ["t0000_mask.ome.tif", "t0001_mask.ome.tif", "t0002_mask.ome.tif", "t0003_mask.ome.tif"]

    pairs = match_image_annotation_pairs(
        images, annots, "*.ome.tif", "*_mask.ome.tif"
    )

    print(f"Images: {len(images)}, Annotations: {len(annots)}, Pairs: {len(pairs)}")
    print(f"Unpaired annotations excluded: {len(annots) - len(pairs)}")
    assert len(pairs) == 2, "Should only match 2 pairs (t0000, t0002)"
    print("✓ PASSED: Unpaired annotations excluded correctly")


def test_mixed_unpaired():
    """Test case with various unpaired files."""
    print("\n=== Test 4: Mixed unpaired files ===")
    images = ["t0000.ome.tif", "t0001.ome.tif", "t0003.ome.tif", "t0005.ome.tif"]
    annots = ["t0000_mask.ome.tif", "t0002_mask.ome.tif", "t0003_mask.ome.tif", "t0004_mask.ome.tif"]

    pairs = match_image_annotation_pairs(
        images, annots, "*.ome.tif", "*_mask.ome.tif"
    )

    print(f"Images: {len(images)}, Annotations: {len(annots)}, Pairs: {len(pairs)}")
    print(f"Total unpaired files: {len(images) + len(annots) - 2*len(pairs)}")
    assert len(pairs) == 2, "Should only match 2 pairs (t0000, t0003)"
    print("✓ PASSED: Only matching pairs included")

    expected = [("t0000.ome.tif", "t0000_mask.ome.tif"),
                ("t0003.ome.tif", "t0003_mask.ome.tif")]
    assert set(pairs) == set(expected), "Should match correct pairs"
    print("✓ PASSED: Correct pairs matched")


def test_folder_paths():
    """Test case with folder paths (assumes same filenames)."""
    print("\n=== Test 5: Folder paths with same filenames ===")
    images = ["cell_001.tif", "cell_002.tif", "cell_003.tif"]
    annots = ["cell_001.tif", "cell_003.tif"]  # Missing cell_002

    # For folder paths, both patterns are "*"
    pairs = match_image_annotation_pairs(
        images, annots, "*", "*"
    )

    print(f"Images: {len(images)}, Annotations: {len(annots)}, Pairs: {len(pairs)}")
    print(f"Unpaired images excluded: {len(images) - len(pairs)}")
    assert len(pairs) == 2, "Should only match files that exist in both folders"
    print("✓ PASSED: Folder-based pairing works correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing image-annotation pairing logic")
    print("=" * 60)

    test_perfect_matching()
    test_missing_annotations()
    test_missing_images()
    test_mixed_unpaired()
    test_folder_paths()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nConclusion:")
    print("The pairing logic correctly excludes unpaired files.")
    print("Only files that have both an image AND annotation are included.")


if __name__ == "__main__":
    main()

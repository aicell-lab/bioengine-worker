#!/usr/bin/env python3
"""
Utility script to extract version from pyproject.toml
Usage: python scripts/get_version.py
"""

import re
from pathlib import Path


def get_version():
    """Extract version from pyproject.toml"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            return match.group(1)


if __name__ == "__main__":
    print(get_version())

import re
from pathlib import Path


def _get_version():
    """Get version from pyproject.toml"""
    try:
        # Try to read from pyproject.toml
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        return match.group(1)
    except Exception:
        raise RuntimeError("Failed to read version from pyproject.toml")


__version__ = _get_version()
from .worker import BioEngineWorker

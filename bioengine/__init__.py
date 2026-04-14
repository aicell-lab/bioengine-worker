import importlib.metadata as md


def _get_version():
    """Get version from package metadata"""
    try:
        return md.metadata("bioengine")["Version"]
    except Exception:
        return None


__version__ = _get_version()

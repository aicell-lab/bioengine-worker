import importlib.metadata as md


def _get_version():
    """Get version from package metadata"""
    return md.metadata("bioengine")["Version"]


__version__ = _get_version()

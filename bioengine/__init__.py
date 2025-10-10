import importlib.metadata as md


def _get_version():
    """Get version from package metadata"""
    try:
        return md.metadata("bioengine")["Version"]
    except:
        print("Could not get version from package metadata. Is the package installed?")
        return None


__version__ = _get_version()

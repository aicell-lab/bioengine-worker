"""Override root conftest fixtures for app-level tests.

App tests (tests/apps/) connect directly to the live Hypha service and do
not require the full worker environment (Ray, haikunator, etc.).
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def validate_environment():
    """No-op override: app tests don't need worker package validation."""
    pass


@pytest.fixture(scope="session")
def workspace_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("app_tests")

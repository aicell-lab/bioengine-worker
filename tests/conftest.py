"""
Global pytest configuration for BioEngine Worker tests.

This configuration applies to all test modules in the tests/ directory.
"""

import pytest
import asyncio


@pytest.fixture(scope="session") 
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Configure asyncio for pytest
pytest_plugins = ('pytest_asyncio',)

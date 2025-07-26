# BioEngine Worker Testing

This directory contains the complete test suite for BioEngine Worker, including unit tests, integration tests, and end-to-end tests.

## Environment Setup

### 1. Create and Activate Environment
```bash
conda create -n bioengine-worker python=3.11.9
conda activate bioengine-worker
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### 3. Environment Configuration
The `.env` file in the project root contains required environment variables including `HYPHA_TOKEN`. This is automatically loaded by the test configuration.

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Test Categories
```bash
# End-to-end tests only
pytest tests/end_to_end/ -v
```

### Test Options
```bash
# Stop on first failure
pytest tests/ --maxfail=1

# Generate coverage report
pytest tests/ --cov=bioengine_worker --cov-report=html
```

## Test Structure

- `tests/end_to_end/` - Full system tests with Hypha server and Ray cluster
- `tests/conftest.py` - Shared fixtures and configuration

## Environment Requirements

- **Python 3.11.9** - Tested and verified version
- **HYPHA_TOKEN** - Authentication for Hypha server access
- **Network Access** - Required for end-to-end tests
- **System Resources** - 8GB RAM minimum for Ray cluster tests

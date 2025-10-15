# BioEngine Applications - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Application Structure](#application-structure)
4. [Manifest Configuration](#manifest-configuration)
5. [Creating Deployments](#creating-deployments)
6. [Environment Variables and Built-in Classes](#environment-variables-and-built-in-classes)
7. [Creating Applications from Local Files](#creating-applications-from-local-files)
8. [Deployment and Usage](#deployment-and-usage)

## Introduction

BioEngine applications are flexible runtime environments that enable you to build and deploy various types of services such as:

- **Model Inference Services**: Deploy machine learning models for real-time prediction
- **Training Services**: Run distributed model training workflows
- **Data Exploration Tools**: Create interactive data analysis and visualization services
- **Custom Processing Pipelines**: Build specialized data processing workflows

BioEngine applications run on Ray Serve, providing scalable, distributed computing capabilities with automatic resource management. Applications are packaged as Hypha artifacts and can be deployed on various environments from single machines to HPC clusters.

### Key Benefits
- **Lightweight Runtime**: Minimal dependencies required (Python ≥3.11, Ray, Hypha-RPC, Pydantic, HTTPx)
- **Scalable Infrastructure**: Automatic resource allocation and cluster scaling
- **Unified Access**: WebSocket and WebRTC support for real-time communication
- **Data Integration**: Built-in access to BioEngine datasets
- **Multi-deployment Architecture**: Compose multiple services within a single application

## System Requirements

### Runtime Dependencies
BioEngine applications require only a lightweight set of dependencies:

- **Python**: ≥3.11 (must match Ray cluster version)
- **Ray**: ≥2.33.0 with client and serve components
- **Hypha-RPC**: ≥0.20.81
- **Pydantic**: ≥2.11.0  
- **HTTPx**: ≥0.28.1

> **Important**: The Python and Ray versions must be identical between the Ray cluster and your BioEngine applications.

### Additional Dependencies
Any additional Python packages not included in the lightweight runtime must be:
1. Specified in the deployment's `runtime_env` configuration
2. Imported locally within methods where they're used (not at file top level)

## Application Structure

A BioEngine application consists of a folder containing specific files that define its behavior and configuration.

### Required Files

Every BioEngine application must include:

1. **`manifest.yaml`**: Configuration file defining application metadata and deployments
2. **Python deployment file(s)**: Implementation of your application logic

### Optional Files

Recommended but not required:

- **`README.md`**: Documentation for your application
- **`tutorial.ipynb`**: Jupyter notebook demonstrating usage
- Additional Python files for multi-deployment applications

### Example Folder Structure

```
my_bioengine_app/
├── manifest.yaml           # Required: Application configuration
├── my_deployment.py        # Required: Main deployment implementation  
├── helper_deployment.py    # Optional: Additional deployment
├── README.md               # Optional: Documentation
└── tutorial.ipynb          # Optional: Usage tutorial
```

### Reference Examples

Explore these examples in the repository:
- **`tests/demo_app/`**: Simple single-deployment application
- **`tests/composition_app/`**: Multi-deployment composition example
- **`bioengine_apps/model_runner/`**: Production model inference service

## Manifest Configuration

The `manifest.yaml` file defines your application's metadata, permissions, and deployment structure.

### Required Fields

```yaml
# Application identification
id: "my-app-id"                    # Unique identifier for Hypha artifact
name: "My Application Name"         # Human-readable name
description: "Brief description"    # What your application does

# Application type (always ray-serve for BioEngine)
type: ray-serve

# Access control
authorized_users:                   # List of users with access
  - "user@example.com"             # Email addresses
  - "hypha_user_id"                # Hypha user IDs  
  - "*"                            # Wildcard for all users

# Deployment configuration  
deployments:                        # List of deployments in priority order
  - "main_file:MainClass"          # Format: filename:ClassName (no .py extension)
  - "helper_file:HelperClass"      # Additional deployments (optional)
```

### Optional Fields

```yaml
# Display and metadata
id_emoji: "🚀"                     # Emoji for UI display
version: "1.0.0"                   # Application version
documentation: "README.md"         # Path to documentation file
tutorial: "tutorial.ipynb"         # Path to tutorial notebook

# Attribution
authors:
  - name: "Author Name"
    affiliation: "Organization"
    github_user: "username"
maintainers: ["username"]
license: "MIT"

# Repository and links
git_repo: "https://github.com/user/repo"
links: ["https://example.com"]
tags: ["machine-learning", "inference"]
covers: []                         # Cover images
format_version: 0.5.0              # Manifest format version
```

### Deployment List Priority

The `deployments` list order matters:
1. **First deployment**: Main entry point, automatically exposed via service APIs
2. **Subsequent deployments**: Sub-deployments, passed as parameters to main deployment

## Creating Deployments

Deployments are Ray Serve classes that define your application's behavior and resource requirements. For detailed information about Ray Serve deployments, refer to the [Ray Serve Deployment API documentation](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.Deployment.html) and the [deployment decorator parameters](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.deployment_decorator.html).

### Basic Deployment Structure

```python
import asyncio
import os
from typing import Any, Dict
from hypha_rpc.utils.schema import schema_method
from ray import serve
from pydantic import Field

@serve.deployment(
    ray_actor_options={
        "num_cpus": 2,                    # CPU cores required
        "num_gpus": 1,                    # GPU devices (0 for CPU-only)
        "memory": 2 * 1024**3,           # Memory in bytes (2GB)
        "runtime_env": {
            "pip": ["numpy", "scikit-learn"],  # Additional dependencies
            "env_vars": {
                "CUSTOM_VAR": "value"     # Custom environment variables
            }
        }
    },
    max_ongoing_requests=10,              # Concurrent request limit
)
class MyDeployment:
    def __init__(self):
        """Initialize your deployment."""
        self.model = None
        
    async def async_init(self) -> None:
        """Optional async initialization called on startup."""
        # Perform async setup here
        pass
        
    async def test_deployment(self) -> None:
        """Optional deployment health test."""
        # Perform health checks here
        pass

    @schema_method
    async def my_api_method(
        self, 
        input_data: str = Field(..., description="Input data to process")
    ) -> Dict[str, Any]:
        """API method exposed to users."""
        # Your application logic here
        return {"result": f"Processed: {input_data}"}
```

### Import Patterns

**Standard imports** (at file top):
```python
# Built-in Python libraries
import os
import asyncio
from typing import Any, Dict

# BioEngine runtime dependencies
from hypha_rpc.utils.schema import schema_method
from ray import serve
from pydantic import Field
```

**External dependencies** (inside methods):
```python
@schema_method
async def process_data(self, data: str) -> Dict:
    # Import external packages where needed
    import pandas as pd
    
    # Use the imported libraries
    df = pd.DataFrame(data)
    return {"shape": df.shape}
```

### Multi-Deployment Composition

For applications with multiple deployments, use `DeploymentHandle` parameters:

```python
from ray.serve.handle import DeploymentHandle

@serve.deployment(ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class MainDeployment:
    def __init__(
        self, 
        helper_deployment: DeploymentHandle,  # Parameter name must match manifest
        model_deployment: DeploymentHandle,   # Another sub-deployment
    ):
        """Initialize with sub-deployment handles."""
        self.helper = helper_deployment
        self.model = model_deployment
        
    @schema_method
    async def process(self, data: str) -> Dict:
        """Coordinate between sub-deployments."""
        # Call helper deployment
        preprocessed = await self.helper.preprocess.remote(data)
        
        # Call model deployment  
        result = await self.model.predict.remote(preprocessed)
        
        return {"result": result}
```

**Manifest for composition:**
```yaml
deployments:
  - "main_deployment:MainDeployment"        # Main entry point
  - "helper_deployment:HelperDeployment"    # Sub-deployment
  - "model_deployment:ModelDeployment"      # Another sub-deployment
```

### Lifecycle Methods

#### `async_init()` - Optional
```python
async def async_init(self) -> None:
    """Called once during deployment startup."""
    # Load models, initialize connections, etc.
    self.model = await self.load_model()
```

#### `test_deployment()` - Optional  
```python
async def test_deployment(self) -> None:
    """Test deployment health before marking as ready."""
    # Test your deployment functionality
    test_input = "test"
    result = self.model.predict(test_input)
    assert "expected_output" in result
```

### Resource Configuration

#### GPU Usage
```python
@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,           # Request 1 GPU
        "num_cpus": 4,           # 4 CPU cores
        "memory": 8 * 1024**3,   # 8GB memory
    }
)
```

#### CPU-Only
```python
@serve.deployment(
    ray_actor_options={
        "num_gpus": 0,           # No GPU required
        "num_cpus": 2,           # 2 CPU cores
        "memory": 4 * 1024**3,   # 4GB memory
    }
)
```

> **Note**: GPU allocation is managed by BioEngine. The `disable_gpu=True` parameter in `deploy_application()` will override GPU requests to force CPU-only execution.

## Environment Variables and Built-in Classes

### Available Environment Variables

Your deployment has access to these environment variables:

```python
import os

# Application environment
home_dir = os.environ["HOME"]                    # Application working directory  
tmp_dir = os.environ["TMPDIR"]                   # Temporary directory

# Hypha connection details
server_url = os.environ["HYPHA_SERVER_URL"]      # Hypha server URL
workspace = os.environ["HYPHA_WORKSPACE"]        # Current workspace
artifact_id = os.environ["HYPHA_ARTIFACT_ID"]    # Full artifact ID (workspace/name)
worker_service_id = os.environ["BIOENGINE_WORKER_SERVICE_ID"]  # Worker service ID

# Authentication (if provided via deploy_application)
token = os.environ.get("HYPHA_TOKEN")            # User authentication token
```

### Built-in Classes and Methods

Every deployment automatically has access to these built-in attributes:

#### `self.bioengine_datasets`

Access BioEngine datasets with streaming support:

```python
@schema_method
async def load_dataset(self, dataset_name: str) -> Dict:
    """Load data from BioEngine datasets."""
    
    # List available datasets
    datasets = await self.bioengine_datasets.list_datasets()
    
    # List files in a dataset
    files = await self.bioengine_datasets.list_files(
        dataset_name, 
        token=os.environ.get("HYPHA_TOKEN")  # Required for restricted datasets
    )
    
    # Load a specific file
    data = await self.bioengine_datasets.get_file(
        dataset_name, 
        "data.zarr", 
        token=os.environ.get("HYPHA_TOKEN")
    )
    
    return {"shape": data.shape if hasattr(data, 'shape') else len(data)}
```

#### `self.bioengine_hypha_client`

Maintained connection to Hypha server:

```python
@schema_method  
async def call_hypha_service(self, service_id: str) -> Dict:
    """Call other Hypha services."""
    
    # Get service reference
    service = await self.bioengine_hypha_client.get_service(service_id)
    
    # Call service methods
    result = await service.some_method("parameter")
    
    return {"result": result}
```

#### `self.bioengine_worker_service`

Access to the BioEngine worker (if token provided):

```python
@schema_method
async def get_worker_status(self) -> Dict:
    """Get BioEngine worker status."""
    
    if self.bioengine_worker_service:
        status = await self.bioengine_worker_service.get_status()
        return status
    else:
        return {"error": "No worker access - token required"}
```

### Schema Methods

Use `@schema_method` to expose API methods:

```python
@schema_method
async def process_data(
    self,
    input_text: str = Field(..., description="Text to process"),
    max_length: int = Field(100, description="Maximum output length"),
    temperature: float = Field(0.7, description="Sampling temperature")
) -> Dict[str, Any]:
    """
    Process input text with the model.
    
    This method processes the input text and returns structured results.
    """
    # Method implementation
    return {
        "processed_text": "...",
        "confidence": 0.95,
        "metadata": {"length": len(input_text)}
    }
```

The `Field` descriptions and method docstring become part of the API documentation.

## Creating Applications from Local Files

Once you've created your application directory with the required files, you can upload it to Hypha as an artifact using either the command-line script or the web interface.

### Method 1: Using the Command-Line Script

#### Setup Requirements

Before using the `create_application.py` script, you need to clone the BioEngine repository and install it:

```bash
# Clone the repository
git clone https://github.com/aicell-lab/bioengine-worker.git
cd bioengine-worker

# Install BioEngine
pip install .
```

#### Prerequisites

1. **Authentication** (Optional): You can authenticate to Hypha in two ways:
   - **Environment variable**: Export your Hypha authentication token:
     ```bash
     export HYPHA_TOKEN="your-hypha-token-here"
     ```
   - **Interactive login**: If `HYPHA_TOKEN` is not set, you'll be prompted to log in interactively when running the script
   
   You can obtain a token from the Hypha server web interface or use the interactive login flow.

2. **Prepare your application directory**: Ensure your directory contains at minimum:
   - `manifest.yaml` with all required fields
   - Python deployment file(s) referenced in the manifest

#### Basic Usage

```bash
python scripts/create_application.py --directory <path-to-your-app>
```

**Example:**
```bash
python scripts/create_application.py --directory tests/demo_app
```

This command will:
1. Read all files from the specified directory
2. Validate the `manifest.yaml` file
3. Upload the application to Hypha as an artifact
4. Display the created artifact ID (e.g., `your-workspace/my-app`)

#### What Happens During Upload

The script performs the following steps:

1. **Validation**: Checks that your manifest contains all required fields:
   - `name`, `id`, `id_emoji`, `description`, `type`, `deployments`, `authorized_users`
   - Validates that `type` is set to `ray-serve`
   - Verifies deployment and authorized_users lists are non-empty

2. **File Processing**: 
   - Reads all files in your directory (including subdirectories)
   - Text files (`.py`, `.yaml`, `.md`, etc.) are uploaded as-is
   - Binary files are automatically base64-encoded

3. **Artifact Creation**:
   - Creates or updates the artifact in the Hypha artifact manager
   - Places the artifact in the `applications` collection
   - Adds metadata like `created_by` field with your user ID

4. **File Management**: If updating an existing artifact:
   - Uploads all new files
   - Removes files that existed before but are not in the new directory
   - Only removes old files after all new files upload successfully

#### Success Output

Upon successful upload, you'll see:
```
Application created successfully!
Artifact ID: your-workspace/my-app-id
```

You can now deploy this application using the BioEngine worker service.

#### Common Issues

**Error: "Manifest is missing required field"**
- Solution: Check that your `manifest.yaml` includes all required fields listed in the [Manifest Configuration](#manifest-configuration) section

**Error: "Application type must be 'ray-serve'"**
- Solution: Ensure your manifest has `type: ray-serve`

**Error: "Invalid artifact alias"**
- Solution: The `id` field in your manifest must use lowercase letters, numbers, and hyphens only (no underscores or slashes)

**Authentication error or login prompt**
- The script will prompt you to log in interactively if `HYPHA_TOKEN` is not set
- Alternatively, you can export `HYPHA_TOKEN` as an environment variable to skip the interactive login

### Method 2: Using the Hypha Web Interface

You can also create and upload applications directly through the Hypha web interface without cloning the repository or installing BioEngine.

#### Steps

1. **Navigate to Hypha**: Go to [https://hypha.aicell.io/](https://hypha.aicell.io/)

2. **Access your workspace**:
   - Click on your profile picture in the top right corner
   - Select **"My Workspace"**

3. **Open Artifacts section**:
   - Click on **"Artifacts"** in the navigation menu

4. **Create new artifact**:
   - Click the **"+ Create Artifact"** button

5. **Configure and upload**:
   - Fill in the artifact details (name, description, etc.)
   - Set the artifact type to **"application"**
   - Upload your application files (`manifest.yaml`, Python files, etc.)
   - Click **"Create"** to finalize

The web interface provides a user-friendly way to manage your applications without needing command-line tools, and it automatically validates your manifest and files during the upload process.

## Deployment and Usage

### Deploying Applications

Deploy your application using the BioEngine worker service:

```python
# Basic deployment
application_id = await bioengine_worker_service.deploy_application(
    artifact_id="workspace/my-app"
)

# Advanced deployment with configuration
application_id = await bioengine_worker_service.deploy_application(
    artifact_id="workspace/my-app",
    version=None,                       # Latest version
    application_id="custom-id",         # Custom instance ID
    hypha_token="your_token_here",      # User authentication
    disable_gpu=False,                  # Enable GPU usage
    max_ongoing_requests=20,            # Concurrent request limit
    application_kwargs={                # Per-deployment initialization args
        "MainDeployment": {
            "model_path": "/path/to/model"
        }
    },
    application_env_vars={              # Per-deployment environment variables
        "MainDeployment": {
            "DEBUG_MODE": "true"
        }
    }
)
```

### Deploy Application Parameters

The `deploy_application` method supports these parameters:

- **`artifact_id`** *(required)*: Application artifact identifier
- **`version`** *(optional)*: Specific artifact version to deploy
- **`application_id`** *(optional)*: Custom instance ID for deployment
- **`application_kwargs`** *(optional)*: Initialization parameters per deployment
- **`application_env_vars`** *(optional)*: Environment variables per deployment  
- **`hypha_token`** *(optional)*: Authentication token for user permissions
- **`disable_gpu`** *(optional)*: Force CPU-only execution (default: False)
- **`max_ongoing_requests`** *(optional)*: Concurrent request limit (default: 10)

### Accessing Deployed Applications

#### Service Discovery

```python
# Get worker status to find application services
app_id = "my-application-id"  # Replace with your application ID
worker_status = await bioengine_worker_service.get_status()
app_status = worker_status["bioengine_apps"][app_id]

# Extract service IDs
websocket_service_id = app_status["service_ids"]["websocket_service_id"]
webrtc_service_id = app_status["service_ids"]["webrtc_service_id"]
```

#### WebSocket Connection

Websocket connections send and receive their data through the connected Hypha server.

```python
# Connect via WebSocket for real-time communication
websocket_service = await hypha_client.get_service(websocket_service_id)

# Call application methods
result = await websocket_service.process_data(
    input_text="Hello BioEngine!",
    max_length=200
)
```

#### WebRTC Connection

WebRTC connections enable peer-to-peer communication for low-latency applications, larger data transfers, and privacy-sensitive use cases.

```python
# Requires aiortc to be installed
from hypha_rpc import get_rtc_service

# Connect via WebRTC for peer-to-peer communication
peer_connection = await get_rtc_service(hypha_client, webrtc_service_id)

webrtc_service = await peer_connection.get_service(app_id)

# Use WebRTC for low-latency real-time applications
# Call application methods
result = await webrtc_service.process_data(
    input_text="Hello BioEngine!",
    max_length=200
)
```

### Application Lifecycle Management

```python
# Deploy application
app_id = await bioengine_worker_service.deploy_application("workspace/my-app")

# Monitor deployment status
status = await bioengine_worker_service.get_status()
app_status = status["bioengine_apps"][app_id]

# Undeploy application when done
await bioengine_worker_service.undeploy_application(app_id)
```

---

## BioEngine Worker Setup and Usage

For information about setting up and running the BioEngine Worker itself, please refer to the [main README.md](../README.md) in the repository for complete setup instructions, including Docker and Apptainer deployment options.

<div align="center">
  <img src="docs/assets/bioengine-icon.svg" alt="BioEngine Logo" width="200"/>
  
  # BioEngine Worker
  
  **Cloud-powered AI for simplified bioimage analysis**
  
  [![Docker Image](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/orgs/aicell-lab/packages/container/package/bioengine-worker)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  
  [🚀 Try It Now](#-quick-start) • [📚 Documentation](#-documentation) • [🎯 Examples](#-use-cases) • [💬 Community](https://github.com/aicell-lab/bioengine-worker/discussions)
  
</div>

---

## 🌟 What is BioEngine?

BioEngine is a **distributed AI platform** that brings the power of cloud computing to bioimage analysis. It enables researchers to:

- 🔬 **Deploy AI models** for image analysis with automatic scaling
- 📊 **Stream large datasets** efficiently with privacy-preserving access control
- ⚡ **Run compute-intensive workflows** on HPC clusters or cloud infrastructure
- 🔌 **Access resources remotely** through a unified API via [Hypha](https://ha.amun.ai/)

Built on [Ray](https://www.ray.io/) and [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), BioEngine automatically manages resource allocation, scaling, and deployment across various computing environments.

## 🎯 Use Cases

### Try Models on the Public BioEngine

Experience BioEngine instantly by testing AI models on the community instance:

1. **Visit [BioImage.IO Model Zoo](https://bioimage.io/#/models)**
2. **Select any model** from the collection
3. **Click "TEST RUN MODEL"** to execute on the public BioEngine worker (`bioimage-io/bioengine-worker`)
4. **See results** powered by cloud infrastructure—no setup required!

### Deploy Your Own Applications

Create custom AI-powered analysis services:

- **Model Inference Services**: Deploy ML models for real-time predictions
- **Training Pipelines**: Run distributed model training workflows  
- **Data Exploration Tools**: Build interactive analysis and visualization services
- **Custom Workflows**: Design specialized processing pipelines

**👉 Learn more**: [BioEngine Applications Guide](docs/apps-guide.md)

### Share Scientific Datasets

Serve large datasets with streaming and access control:

- **Privacy-preserving**: Fine-grained user permissions
- **Efficient streaming**: Partial data access for Zarr datasets
- **Easy sharing**: HTTP-based access through Hypha

**👉 Learn more**: [BioEngine Datasets Guide](docs/datasets-guide.md)

## 🚀 Quick Start

### Try the BioEngine Dashboard

The easiest way to get started is through the web interface:

**🌐 Visit [bioimage.io/#/bioengine](https://bioimage.io/#/bioengine)**

The dashboard provides:
- 📋 **Interactive Configuration Wizard**: Generate deployment commands for your environment
- 🖥️ **Instance Management**: View and control BioEngine workers
- 📊 **Resource Monitoring**: Track cluster resources and application status
- 🎮 **Application Deployment**: Deploy and manage AI applications

### Run Locally with Docker

Start a local BioEngine worker on your workstation:

```bash
# Clone the repository
git clone https://github.com/aicell-lab/bioengine-worker.git
cd bioengine-worker

# Create required directories
mkdir -p .bioengine data

# Start with Docker Compose
UID=$(id -u) GID=$(id -g) docker compose up
```

**What happens:**
- Starts a local Ray cluster with your machine's resources
- Registers as a Hypha service for remote access
- Mounts `.bioengine/` for workspace and `data/` for datasets

**Configuration:**
- Add `HYPHA_TOKEN` to `.env` file (or you'll be prompted to login)
- Customize resources via `--head-num-cpus` and `--head-num-gpus`
- See all options: `docker run --rm ghcr.io/aicell-lab/bioengine-worker:latest python -m bioengine.worker --help`

**💡 Pro Tip**: Visit the [BioEngine Dashboard](https://bioimage.io/#/bioengine) to generate custom deployment commands for your environment!

## 📚 Documentation

### Core Guides

- **[🚀 BioEngine Applications](docs/apps-guide.md)** - Deploy AI models and create custom services
- **[📊 BioEngine Datasets](docs/datasets-guide.md)** - Share and stream large scientific datasets
- **[🎮 BioEngine Dashboard](https://bioimage.io/#/bioengine)** - Web-based configuration and management

### Deployment Modes

BioEngine supports three deployment modes to fit your infrastructure:

| Mode | Description | Best For |
|------|-------------|----------|
| **single-machine** | Local Ray cluster on one machine | Workstations, development, small-scale analysis |
| **external-cluster** | Connect to existing Ray cluster | Kubernetes, pre-configured HPC environments |
| **slurm** | Auto-scaling via SLURM jobs | HPC clusters with SLURM scheduler |

## 🔧 Advanced Usage

### Programmatic Access

Once a BioEngine worker is running, access it remotely via Python:

```python
from hypha_rpc import connect_to_server, login

# Authenticate and connect
token = await login({"server_url": "https://hypha.aicell.io"})
server = await connect_to_server({
    "server_url": "https://hypha.aicell.io",
    "token": token,
})

# Get the BioEngine worker service
workspace = server.config.workspace
worker_service = await server.get_service(f"{workspace}/bioengine-worker")

# Check worker status
status = await worker_service.get_status()
print(f"Ray cluster: {status['ray_cluster']}")

# Deploy an application
app_id = await worker_service.run_application(
    artifact_id="workspace/my-app",
    application_id="my-instance",
)

# Get application status
app_status = await worker_service.get_application_status(
    application_ids=[app_id]
)
```

### Available Service Methods

The BioEngine worker service provides comprehensive functionality:

**Resource Management:**
- `get_status()` - Get worker and cluster status
- `stop_worker()` - Shut down the worker

**Dataset Operations:**
- `list_datasets()` - List available datasets

**Application Management:**
- `run_application(artifact_id, ...)` - Deploy an application
- `stop_application(application_id)` - Stop a running application
- `get_application_status(application_ids)` - Get application status
- `list_applications()` - List all deployed applications

**Development:**
- `execute_python_code(code, ...)` - Execute Python code remotely
- `check_access()` - Verify user permissions

See the [Applications Guide](docs/apps-guide.md) for detailed usage examples.

## 🏗️ Architecture

BioEngine consists of three main components:

```
┌─────────────────────────────────────────────────────────┐
│                    Hypha Server                         │
│            (RPC & Service Discovery)                    │
└────────────┬────────────────────────────────────────────┘
             │
             │ Remote Access
             │
┌────────────▼────────────────────────────────────────────┐
│              BioEngine Worker                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Ray Cluster (Distributed Computing)             │  │
│  │  • Ray Serve (Model Deployment)                  │  │
│  │  • Auto-scaling Worker Management                │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Applications Manager                            │  │
│  │  • Deploy & manage AI services                   │  │
│  │  • Resource allocation                           │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Datasets Manager                                │  │
│  │  • HTTP streaming with access control            │  │
│  │  • Zarr store for efficient data access          │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 🤝 Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or improving documentation:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

BioEngine is built on top of several excellent open-source projects:
- [Ray](https://www.ray.io/) - Distributed computing framework
- [Hypha](https://ha.amun.ai/) - Service orchestration and RPC
- [Zarr](https://zarr.readthedocs.io/) - Chunked array storage

## 📞 Support

- 💬 [GitHub Discussions](https://github.com/aicell-lab/bioengine-worker/discussions) - Ask questions and share ideas
- 🐛 [Issue Tracker](https://github.com/aicell-lab/bioengine-worker/issues) - Report bugs or request features
- 📧 Contact: [bioimage.io](https://bioimage.io)

---

<div align="center">
  
**Made with ❤️ by the BioImage.IO community**

[Website](https://bioimage.io) • [Dashboard](https://bioimage.io/#/bioengine) • [GitHub](https://github.com/aicell-lab/bioengine-worker)

</div>

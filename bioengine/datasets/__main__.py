#!/usr/bin/env python3
"""
BioEngine Datasets - Privacy-preserved scientific data management system.

Enterprise-grade command-line interface for deploying and managing BioEngine Datasets,
providing privacy-preserved data streaming with per-user file access control. Integrates
with BioEngine Worker to enable secure access to large scientific datasets through a
local Hypha server with MinIO S3 backend.

This module provides a comprehensive system for secure, efficient access to large
scientific datasets with privacy preservation and fine-grained access control. It
implements both client and server components for a complete data management solution
that integrates with the BioEngine distributed AI infrastructure.

Key Components:
- BioEngineDatasets: Client interface for dataset access from applications
- HttpZarrStore: Efficient streaming store for partial data access
- start_proxy_server: Server deployment for dataset management and access control
- Command-line interface for standalone dataset service deployment

Dataset Architecture:
The system uses a manifest-driven architecture where each dataset includes a manifest.yaml
file defining metadata and access permissions. Data is stored in Zarr format for efficient
partial access, with an HTTP-based streaming protocol that minimizes data transfer.

Key Features:
- Privacy-preserved data streaming with fine-grained access control
- Hypha server integration with MinIO S3 backend for secure file storage
- Automatic artifact registration for easy dataset discovery
- HTTP-based Zarr store for efficient partial data access
- Authentication and authorization management for secure multi-user environments
- Structured logging with file output and debug modes

Integration Points:
- Ray Serve for model deployment integration
- Hypha for service registration and authentication
- MinIO S3 for secure object storage backend
- AnnData for scientific data format compatibility

Usage:
    python -m bioengine.datasets --data-dir /shared/data
    python -m bioengine.datasets --data-dir /shared/data --server-ip 0.0.0.0 --server-port 8080
    python -m bioengine.datasets --data-dir /shared/data --workspace-dir /shared/bioengine/workspace

The module can be used both as a library for accessing datasets from within BioEngine
applications and as a standalone service for dataset management and distribution.

Author: BioEngine Development Team
License: MIT
"""

import argparse
import sys

from bioengine.datasets.proxy_server import start_proxy_server


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the comprehensive argument parser for BioEngine datasets.

    Sets up command-line argument parsing with detailed help text, validation,
    and organized argument options for dataset management service. Provides
    configuration options for data directories, server settings, and MinIO
    backend to support privacy-preserved data streaming.

    Returns:
        Configured ArgumentParser instance with all BioEngine Datasets options
    """
    parser = argparse.ArgumentParser(
        description="BioEngine Datasets - Privacy-Preserved Data Streaming Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start datasets service with default settings
  %(prog)s --data-dir /shared/data
  
  # Start with custom server and port configuration
  %(prog)s --data-dir /shared/data --server-ip 0.0.0.0 --server-port 8080
  
  # Configure custom cache directory for logs and temporary files
  %(prog)s --data-dir /shared/data --cache-dir /shared/bioengine/cache

For detailed documentation, visit: https://github.com/aicell-lab/bioengine-worker
""",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Root directory for dataset storage and access by the dataset manager.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        metavar="PATH",
        dest="bioengine_workspace_dir",
        help="Directory for worker workspace, temporary files, and Ray data storage. "
        "Should be accessible across worker nodes in distributed deployments.",
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        metavar="IP_ADDRESS",
        help="IP address for the BioEngine Datasets proxy server.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        metavar="PORT",
        help="Port for the BioEngine Datasets proxy server.",
    )
    parser.add_argument(
        "--minio-port",
        type=int,
        metavar="PORT",
        help="Port for the MinIO server.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        help="Path to the log file. If set to 'off', logging will only go to console. "
        "If not specified (None), a log file will be created in '<workspace_dir>/logs'. ",
    )

    return parser


if __name__ == "__main__":
    """
    Main entry point for the BioEngine Datasets service.

    Parses command-line arguments, configures logging, and starts the datasets proxy server.
    The proxy server initializes a local Hypha server with MinIO S3 backend for secure file storage
    and serves scientific datasets with privacy-preserved access control. It automatically
    registers datasets as artifacts in Hypha for easy discovery and manages authentication
    for secure multi-user environments.

    The BioEngineDatasets class can connect to this service to stream data efficiently using
    the HttpZarrStore, which enables partial data access to large Zarr-formatted datasets.
    """
    try:
        parser = create_parser()
        args = parser.parse_args()
        args = {k: v for k, v in vars(args).items() if v is not None}

        # Start the BioEngine Datasets proxy server with proper exception handling
        start_proxy_server(**args)

    except Exception as e:
        print(f"Failed to start BioEngine Datasets proxy server: {e}")
        sys.exit(1)
